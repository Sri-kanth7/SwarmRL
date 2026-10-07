"""Training configuration loading and reproducibility helpers.

Configuration files live in ``training/configs/*.yaml``. They mirror
the environment configuration (agent count, world size, seeds) so
training and simulation always agree on the same contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from env.config import EnvConfig

CONFIG_DIR = Path(__file__).resolve().parent / "configs"

DEFAULT_ENV_CONFIG: dict[str, Any] = EnvConfig().to_dict()


@dataclass(frozen=True)
class TrainingConfig:
    """RLlib-oriented training configuration (Week 1 foundation).

    Attributes:
        algorithm: Algorithm name (``MAPPO`` or ``IPPO``).
        env_name: Registry name used when registering the environment.
        framework: Deep-learning framework (``torch``).
        seed: Global seed applied to algorithm and environment.
        env_config: Environment constructor keyword arguments.
        model: Network architecture settings.
        multi_agent: Policy layout and mapping function name.
        train: RLlib trainer hyperparameters (executed by the IPPO entry point).
        checkpoint_dir: Directory that will receive future checkpoints.
    """

    algorithm: str = "MAPPO"
    env_name: str = "swarm"
    framework: str = "torch"
    seed: int = 42
    env_config: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_ENV_CONFIG))
    model: dict[str, Any] = field(
        default_factory=lambda: {"fcnet_hiddens": [256, 256], "fcnet_activation": "relu"}
    )
    multi_agent: dict[str, Any] = field(default_factory=dict)
    train: dict[str, Any] = field(default_factory=dict)
    checkpoint_dir: str = "training/checkpoints"

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable representation."""
        return {
            "algorithm": self.algorithm,
            "env": self.env_name,
            "framework": self.framework,
            "seed": self.seed,
            "env_config": self.env_config,
            "model": self.model,
            "multi_agent": self.multi_agent,
            "train": self.train,
            "checkpoint_dir": self.checkpoint_dir,
        }

    def with_seed(self, seed: int) -> "TrainingConfig":
        """Return a copy with the seed (and environment seed) replaced."""
        env_config = dict(self.env_config)
        env_config["seed"] = seed
        return TrainingConfig(
            algorithm=self.algorithm,
            env_name=self.env_name,
            framework=self.framework,
            seed=seed,
            env_config=env_config,
            model=self.model,
            multi_agent=self.multi_agent,
            train=self.train,
            checkpoint_dir=self.checkpoint_dir,
        )


def default_config_path(algorithm: str = "MAPPO") -> Path:
    """Path of the bundled configuration for an algorithm."""
    name = algorithm.strip().lower()
    if name not in {"mappo", "ippo"}:
        raise ValueError(f"unknown algorithm {algorithm!r}; expected MAPPO or IPPO")
    return CONFIG_DIR / f"{name}.yaml"


def load_training_config(path: str | Path | None = None) -> TrainingConfig:
    """Load a training configuration from YAML.

    Args:
        path: Configuration file. Defaults to the bundled MAPPO file.

    Returns:
        A :class:`TrainingConfig` with environment defaults merged in.
    """
    config_path = Path(path) if path is not None else default_config_path("MAPPO")
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"training config {config_path} must contain a mapping")

    env_config = dict(DEFAULT_ENV_CONFIG)
    env_config.update(raw.get("env_config") or {})
    if "seed" in raw:
        env_config["seed"] = int(raw["seed"])

    return TrainingConfig(
        algorithm=str(raw.get("algorithm", "MAPPO")),
        env_name=str(raw.get("env", "swarm")),
        framework=str(raw.get("framework", "torch")),
        seed=int(raw.get("seed", 42)),
        env_config=env_config,
        model=dict(raw.get("model") or {"fcnet_hiddens": [256, 256]}),
        multi_agent=dict(raw.get("multi_agent") or {}),
        train=dict(raw.get("train") or {}),
        checkpoint_dir=str(raw.get("checkpoint_dir", "training/checkpoints")),
    )
