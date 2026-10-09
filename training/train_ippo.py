"""IPPO training entry point (Week 2: runnable RLlib training).

Usage::

    python -m training.train_ippo [--config PATH] [--seed N]
                                  [--iterations N] [--run-id ID]
                                  [--print-config]

Without ``--print-config`` the command runs the RLlib IPPO loop for
``--iterations`` rounds (default: ``train.num_iterations`` from the
YAML) and writes one checkpoint directory::

    <checkpoint_dir>/<run_id>/
        metadata.json          # training.checkpoint.CheckpointMetadata
        policy.pt              # torch state dict of the shared policy
        metrics.json           # episode metrics + RLlib run metrics
        config_snapshot.yaml   # resolved training configuration
        rllib/                 # RLlib's own algorithm checkpoint

Use ``--print-config`` to inspect the configuration without starting
Ray or training.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import yaml

from env import EnvConfig, SwarmEnv
from training import register_swarm_env
from training.checkpoint import CheckpointMetadata, checkpoint_dir, save_metadata
from training.config import TrainingConfig, default_config_path, load_training_config
from training.metrics import MetricsAggregator
from training.rl_config import (
    POLICY_ID,
    build_algorithm_config,
    build_ppo_config,
    num_iterations,
    summarize,
)

WEIGHTS_FILE_NAME = "policy.pt"
METRICS_FILE_NAME = "metrics.json"
CONFIG_SNAPSHOT_FILE_NAME = "config_snapshot.yaml"
RLLIB_DIR_NAME = "rllib"


@dataclass(frozen=True)
class TrainingRun:
    """Outcome of one IPPO training session.

    Attributes:
        run_dir: Checkpoint directory that received the artifacts.
        metadata: Contract metadata written to ``metadata.json``.
        metrics: Contents of ``metrics.json`` (summary, episodes,
            per-iteration env-runner records).
        iterations: Number of ``Algorithm.train()`` calls executed.
        env_runner_results: One record per executed iteration.
    """

    run_dir: Path
    metadata: CheckpointMetadata
    metrics: dict[str, Any]
    iterations: int
    env_runner_results: tuple[dict[str, Any], ...]

    def summary(self) -> dict[str, Any]:
        """JSON-friendly summary of the run for logging and tests."""
        episode_summary = self.metrics["summary"]
        return {
            "run_dir": str(self.run_dir),
            "iterations": self.iterations,
            "episodes": int(episode_summary["episodes"]),
            "mean_reward": episode_summary["mean_reward"],
            "mean_explored_pct": episode_summary["mean_explored_pct"],
            "metadata": self.metadata.to_dict(),
        }


def build_ippo_config(config: TrainingConfig) -> dict:
    """Build the IPPO algorithm configuration for the given settings."""
    return build_algorithm_config(_as_ippo(config))


def _as_ippo(config: TrainingConfig) -> TrainingConfig:
    """Return the config marked as IPPO (other algorithm names are ignored)."""
    if config.algorithm.upper() == "IPPO":
        return config
    return TrainingConfig(
        algorithm="IPPO",
        env_name=config.env_name,
        framework=config.framework,
        seed=config.seed,
        env_config=config.env_config,
        model=config.model,
        multi_agent=config.multi_agent,
        train=config.train,
        checkpoint_dir=config.checkpoint_dir,
    )


def _default_run_id(config: TrainingConfig) -> str:
    """Timestamped run id, e.g. ``ippo_20261005-141516``."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{config.algorithm.lower()}_{stamp}"


def _number(value: Any) -> float | None:
    """Convert a metric value to ``float`` (``None`` when unavailable)."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _policy_state(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    """Convert RLlib's weight arrays into a torch state dict."""
    if not raw:
        raise RuntimeError(f"training produced no weights for policy {POLICY_ID!r}")
    state: dict[str, Any] = {}
    for name, value in raw.items():
        state[name] = value if isinstance(value, torch.Tensor) else torch.tensor(value)
    return state


def _env_runner_record(result: Mapping[str, Any], iteration: int) -> dict[str, Any]:
    """One JSON-friendly record of RLlib's env-runner metrics."""
    runners = result.get("env_runners") or {}
    return {
        "training_iteration": iteration,
        "episode_return_mean": _number(runners.get("episode_return_mean")),
        "episode_len_mean": _number(runners.get("episode_len_mean")),
    }


def run_training(
    config: TrainingConfig,
    *,
    run_id: str | None = None,
    iterations: int | None = None,
    config_path: str | Path | None = None,
) -> TrainingRun:
    """Run an IPPO training session and write its checkpoint directory.

    The environment is registered with RLlib, the algorithm is built
    from ``config``, and ``Algorithm.train()`` is called ``iterations``
    times. Every artifact of the checkpoint contract is written to
    ``<checkpoint_dir>/<run_id>``.

    Args:
        config: Training configuration to run.
        run_id: Checkpoint directory name. Defaults to
            ``<algorithm>_<UTC timestamp>``.
        iterations: Number of training iterations. Defaults to
            ``train.num_iterations`` from the configuration.
        config_path: Source YAML recorded in the metadata.

    Returns:
        The finished :class:`TrainingRun`.

    Raises:
        ValueError: If ``iterations`` is not a positive integer.
    """
    import ray

    config = _as_ippo(config)
    count = num_iterations(config) if iterations is None else int(iterations)
    if count < 1:
        raise ValueError(f"iterations must be >= 1, got {count}")

    env = SwarmEnv(EnvConfig.from_dict(config.env_config))
    observation_dim = int(env.observation_space(env.possible_agents[0]).shape[0])
    action_dim = int(env.action_space(env.possible_agents[0]).shape[0])
    agent_count = len(env.possible_agents)
    env.close()

    started_ray = not ray.is_initialized()
    collector = MetricsAggregator()
    algorithm = None
    try:
        if started_ray:
            ray.init(ignore_reinit_error=True, include_dashboard=False)

        register_swarm_env(config.env_name, collector=collector)
        algorithm = build_ppo_config(config).build()

        records = []
        for index in range(count):
            result = algorithm.train()
            records.append(_env_runner_record(result, index + 1))

        run_dir = checkpoint_dir(
            config.checkpoint_dir, run_id or _default_run_id(config)
        )
        run_dir = run_dir.resolve()
        run_dir.mkdir(parents=True, exist_ok=True)

        rllib_ckpt_dir = (run_dir / RLLIB_DIR_NAME).resolve()
        algorithm.save(str(rllib_ckpt_dir))
        weights = algorithm.get_weights([POLICY_ID]).get(POLICY_ID)
        torch.save(_policy_state(weights), run_dir / WEIGHTS_FILE_NAME)

        metrics = {
            "summary": collector.summary(),
            "episodes": [episode.to_dict() for episode in collector.episodes],
            "iterations": records,
        }
        (run_dir / METRICS_FILE_NAME).write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
        (run_dir / CONFIG_SNAPSHOT_FILE_NAME).write_text(
            yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8"
        )

        metadata = CheckpointMetadata(
            algorithm=config.algorithm,
            step=count,
            observation_dim=observation_dim,
            action_dim=action_dim,
            agent_count=agent_count,
            seed=int(config.seed),
            created_at=datetime.now(timezone.utc).isoformat(),
            config_path=str(config_path) if config_path is not None else None,
            weights_file=WEIGHTS_FILE_NAME,
        )
        save_metadata(run_dir, metadata)

        return TrainingRun(
            run_dir=run_dir,
            metadata=metadata,
            metrics=metrics,
            iterations=count,
            env_runner_results=tuple(records),
        )
    finally:
        if algorithm is not None:
            algorithm.stop()
        if started_ray:
            ray.shutdown()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train the SwarmRL IPPO policy")
    parser.add_argument(
        "--config",
        default=str(default_config_path("ippo")),
        help="Path to a training YAML file",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override the seed")
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Training iterations to run (default: train.num_iterations)",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Checkpoint directory name (default: <algorithm>_<timestamp>)",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print the resolved algorithm configuration as JSON (no training)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict:
    """Load configuration, then print it or run an IPPO training session.

    ``--print-config`` returns the configuration dictionary without
    touching Ray; otherwise training runs and a JSON summary of the
    written checkpoint is returned.
    """
    args = parse_args(argv)
    config = load_training_config(args.config)
    if args.seed is not None:
        config = config.with_seed(args.seed)

    if args.print_config:
        algorithm_config = build_ippo_config(config)
        print(json.dumps(summarize(algorithm_config), indent=2, default=str))
        return algorithm_config

    run = run_training(
        config,
        run_id=args.run_id,
        iterations=args.iterations,
        config_path=args.config,
    )
    summary = run.summary()
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
