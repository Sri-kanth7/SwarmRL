"""IPPO training entry point (Week 1: configuration foundation).

Usage::

    python -m training.train_ippo [--config PATH] [--seed N] [--print-config]

Same structure as the MAPPO entry point: configuration is loaded
and the algorithm dictionary is built, but no training run is
started in Week 1.
"""

from __future__ import annotations

import argparse
import json
from typing import Sequence

from training.config import TrainingConfig, default_config_path, load_training_config
from training.rl_config import build_algorithm_config, summarize


def build_ippo_config(config: TrainingConfig) -> dict:
    """Build the IPPO algorithm configuration for the given settings."""
    if config.algorithm.upper() != "IPPO":
        config = TrainingConfig(
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
    return build_algorithm_config(config)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Build the SwarmRL IPPO config")
    parser.add_argument(
        "--config",
        default=str(default_config_path("ippo")),
        help="Path to a training YAML file",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override the seed")
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print the resolved algorithm configuration as JSON",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict:
    """Load configuration and build the IPPO config (no training)."""
    args = parse_args(argv)
    config = load_training_config(args.config)
    if args.seed is not None:
        config = config.with_seed(args.seed)
    algorithm_config = build_ippo_config(config)
    if args.print_config:
        print(json.dumps(summarize(algorithm_config), indent=2, default=str))
    else:
        print(
            f"[week1] IPPO config ready: env={algorithm_config['env']} "
            f"seed={algorithm_config['seed']} "
            f"agents={config.env_config.get('agent_count')} "
            "(training runs start in Week 2)"
        )
    return algorithm_config


if __name__ == "__main__":
    main()
