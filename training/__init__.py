"""Training package: RL configuration, environment registration,
entry points, checkpoints, evaluation, and metrics.

Week 1 provides configuration and structure only; actual RLlib
training runs (IPPO/MAPPO) are Week 2 work.
"""

from __future__ import annotations

from training.config import TrainingConfig, load_training_config

ENV_REGISTRY_NAME = "swarm"


def register_swarm_env(env_name: str = ENV_REGISTRY_NAME) -> str:
    """Register :class:`env.SwarmEnv` in the RLlib environment registry.

    Ray is imported lazily so that importing the training package
    never requires a running Ray runtime.

    Args:
        env_name: Registry name; defaults to ``"swarm"``.

    Returns:
        The registered name.
    """
    from ray.tune.registry import register_env

    from env import EnvConfig, SwarmEnv

    def _creator(env_config: dict | None) -> SwarmEnv:
        return SwarmEnv(EnvConfig.from_dict(env_config))

    register_env(env_name, _creator)
    return env_name


__all__ = [
    "ENV_REGISTRY_NAME",
    "TrainingConfig",
    "load_training_config",
    "register_swarm_env",
]
