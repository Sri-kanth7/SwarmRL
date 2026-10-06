"""MAPPO training entry point (Week 1: configuration foundation).

Usage::

    python -m training.train_mappo [--config PATH] [--seed N] [--print-config]

The entry point loads configuration, registers the environment, and
builds the RLlib-style algorithm dictionary. It deliberately does
not start Ray or run training; that lands in Week 2 together with
the centralized critic.
"""

from __future__ import annotations

import argparse
import json
from typing import Sequence

from training.config import TrainingConfig, default_config_path, load_training_config
from training.rl_config import build_algorithm_config, summarize


def build_mappo_config(config: TrainingConfig) -> dict:
    """Build the MAPPO algorithm configuration for the given settings."""
    if config.algorithm.upper() != "MAPPO":
        config = TrainingConfig(
            algorithm="MAPPO",
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
    parser = argparse.ArgumentParser(description="Build the SwarmRL MAPPO config")
    parser.add_argument(
        "--config",
        default=str(default_config_path("MAPPO")),
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
    """Load configuration and build the MAPPO config (no training)."""
    args = parse_args(argv)
    config = load_training_config(args.config)
    if args.seed is not None:
        config = config.with_seed(args.seed)
    algorithm_config = build_mappo_config(config)
    if args.print_config:
        print(json.dumps(summarize(algorithm_config), indent=2, default=str))
    else:
        print(
            f"[week1] MAPPO config ready: env={algorithm_config['env']} "
            f"seed={algorithm_config['seed']} "
            f"agents={config.env_config.get('agent_count')} "
            "(training runs start in Week 2)"
        )
    return algorithm_config


if __name__ == "__main__":
    main()
