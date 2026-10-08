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

_ACTIVE_COLLECTOR: MetricsAggregator | None = None
"""Collector bound to the most recent :func:`register_swarm_env` call.

Ray's environment registry pickles every registered creator into Ray's
internal KV and deserializes a fresh copy on retrieval
(``ray.tune.registry._Registry``). A creator that captures the collector
in a closure would therefore hand training environments a deserialized
*copy* of the collector, leaving the caller's collector empty. Reading
the collector from this live module state at environment-creation time
keeps every in-process environment on the caller's own collector
instance.
"""


def _swarm_env_creator(env_config: dict | None) -> "SwarmMultiAgentEnv":
    """Build the adapter with the active registration's collector.

    Registered with RLlib by :func:`register_swarm_env`. The collector is
    resolved from :data:`_ACTIVE_COLLECTOR` at call time, so the
    environments still share the caller's collector even though the
    registry returns a deserialized copy of this function.
    """
    from training.rllib_env import SwarmMultiAgentEnv

    return SwarmMultiAgentEnv(env_config, collector=_ACTIVE_COLLECTOR)


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
    global _ACTIVE_COLLECTOR

    from ray.tune.registry import register_env

    _ACTIVE_COLLECTOR = collector
    register_env(env_name, _swarm_env_creator)
    return env_name


__all__ = [
    "ENV_REGISTRY_NAME",
    "MetricsAggregator",
    "TrainingConfig",
    "load_training_config",
    "register_swarm_env",
]
