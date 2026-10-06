"""Shared RLlib configuration builder used by the training entry points.

Week 1 builds a plain configuration dictionary: it validates the
environment contract (observation/action spaces, seeds, multi-agent
layout) without starting Ray or running training. Passing this
dictionary to an RLlib ``Algorithm`` is Week 2 work.
"""

from __future__ import annotations

from typing import Any, Mapping

from env import EnvConfig, SwarmEnv
from training.config import TrainingConfig

POLICY_ID = "swarm_policy"


def map_agent_to_policy(agent_id: str, episode: Any = None, **kwargs: Any) -> str:
    """Map every drone to the shared swarm policy (module-level so the
    mapping stays picklable)."""
    return POLICY_ID


def build_algorithm_config(config: TrainingConfig) -> dict[str, Any]:
    """Assemble an RLlib-style algorithm configuration dictionary.

    The returned mapping contains everything a future training run
    needs: registry name, environment kwargs, spaces for the policy
    spec, model settings, multi-agent layout, trainer settings, and
    the global seed. No Ray objects are created here.
    """
    env = SwarmEnv(EnvConfig.from_dict(config.env_config))
    seed = config.seed
    return {
        "algo_class_name": config.algorithm,
        "env": config.env_name,
        "env_config": dict(config.env_config),
        "framework": config.framework,
        "seed": seed,
        "torch_seed": seed,
        "observation_space": env.observation_space(env.possible_agents[0]),
        "action_space": env.action_space(env.possible_agents[0]),
        "model": dict(config.model),
        "multi_agent": {
            "policies": {POLICY_ID: None},
            "policy_mapping_fn": map_agent_to_policy,
            "policy_ids": [POLICY_ID],
        },
        "train": dict(config.train),
        "checkpoint_dir": config.checkpoint_dir,
    }


def summarize(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return a JSON-friendly summary of an algorithm configuration."""
    summary: dict[str, Any] = {}
    for key, value in config.items():
        if key in {"observation_space", "action_space", "policy_mapping_fn"}:
            summary[key] = str(value)
        else:
            summary[key] = value
    return summary


__all__ = [
    "POLICY_ID",
    "build_algorithm_config",
    "map_agent_to_policy",
    "summarize",
]
