"""Training package: RL configuration, environment registration,
entry points, checkpoints, evaluation, and metrics.

Week 1 provides configuration and structure; Week 2 adds the IPPO
training run. MAPPO remains configuration-only until the centralized
critic lands.
"""

from __future__ import annotations

from training.config import TrainingConfig, load_training_config
from training.metrics import MetricsAggregator

ENV_REGISTRY_NAME = "swarm"


def register_swarm_env(
    env_name: str = ENV_REGISTRY_NAME,
    *,
    collector: MetricsAggregator | None = None,
) -> str:
    """Register the swarm environment with the RLlib environment registry.

    The registered creator builds a
    :class:`training.rllib_env.SwarmMultiAgentEnv`, the ``MultiAgentEnv``
    adapter RLlib requires. Ray is imported lazily so that importing the
    training package never requires a running Ray runtime.

    Args:
        env_name: Registry name; defaults to ``"swarm"``.
        collector: Optional episode-metrics collector handed to every
            environment built from this registration. Metrics are
            collected in-process, which requires training runs with
            ``num_env_runners: 0``.

    Returns:
        The registered name.
    """
    from ray.tune.registry import register_env

    from training.rllib_env import SwarmMultiAgentEnv

    def _creator(env_config: dict | None) -> SwarmMultiAgentEnv:
        return SwarmMultiAgentEnv(env_config, collector=collector)

    register_env(env_name, _creator)
    return env_name


__all__ = [
    "ENV_REGISTRY_NAME",
    "MetricsAggregator",
    "TrainingConfig",
    "load_training_config",
    "register_swarm_env",
]
