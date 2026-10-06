"""Episode evaluation helpers.

Week 1 provides a policy-agnostic evaluation loop: any callable
that maps an observation array to an action array can be evaluated
against the environment, and the resulting metrics feed the
training/backend metrics contracts. Running evaluations against a
trained checkpoint is Week 2 work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np

from env import SwarmEnv

PolicyFn = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class EvaluationResult:
    """Aggregated result of an evaluation run.

    Attributes:
        episodes: Number of episodes evaluated.
        mean_reward: Mean per-drone total episode reward.
        mean_explored_pct: Mean explored percentage across episodes.
        mean_collisions: Mean collision-event count across episodes.
        mean_steps: Mean episode length in steps.
    """

    episodes: int
    mean_reward: float
    mean_explored_pct: float
    mean_collisions: float
    mean_steps: float

    def to_dict(self) -> dict[str, float]:
        """Serializable representation."""
        return {
            "episodes": float(self.episodes),
            "mean_reward": self.mean_reward,
            "mean_explored_pct": self.mean_explored_pct,
            "mean_collisions": self.mean_collisions,
            "mean_steps": self.mean_steps,
        }


def evaluate_policy(
    env: SwarmEnv,
    policy_fn: PolicyFn,
    episodes: int = 1,
    seed: int | None = None,
) -> EvaluationResult:
    """Evaluate a policy for a number of episodes.

    Args:
        env: Environment to evaluate against.
        policy_fn: Maps an observation vector to an action vector.
        episodes: Number of episodes to run.
        seed: Optional base seed; episode ``i`` uses ``seed + i``.

    Returns:
        Aggregated :class:`EvaluationResult`.
    """
    if episodes < 1:
        raise ValueError("episodes must be >= 1")
    total_rewards: list[float] = []
    explored_values: list[float] = []
    collision_values: list[float] = []
    step_values: list[float] = []

    for episode in range(episodes):
        episode_seed = None if seed is None else seed + episode
        observations, _ = env.reset(seed=episode_seed)
        episode_reward = {agent: 0.0 for agent in observations}
        steps = 0.0
        terminated = False
        truncated = False
        while not (terminated or truncated):
            actions = {
                agent: np.asarray(policy_fn(obs), dtype=np.float32)
                for agent, obs in observations.items()
            }
            observations, rewards, terminations, truncations, infos = env.step(actions)
            for agent, reward in rewards.items():
                episode_reward[agent] = episode_reward.get(agent, 0.0) + reward
            terminated = all(terminations.values()) if terminations else True
            truncated = all(truncations.values()) if truncations else True
            steps += 1.0
            if not observations and not (terminated or truncated):
                break
        final_infos = list(infos.values()) if infos else []
        explored = float(final_infos[0]["explored_pct"]) if final_infos else 0.0
        collisions = float(final_infos[0]["collisions"]) if final_infos else 0.0
        mean_episode_reward = (
            float(np.mean(list(episode_reward.values()))) if episode_reward else 0.0
        )
        total_rewards.append(mean_episode_reward)
        explored_values.append(explored)
        collision_values.append(collisions)
        step_values.append(steps)

    return EvaluationResult(
        episodes=episodes,
        mean_reward=float(np.mean(total_rewards)),
        mean_explored_pct=float(np.mean(explored_values)),
        mean_collisions=float(np.mean(collision_values)),
        mean_steps=float(np.mean(step_values)),
    )


def evaluate_from_config(
    env_config: Mapping[str, Any] | None,
    policy_fn: PolicyFn,
    episodes: int = 1,
    seed: int | None = None,
) -> EvaluationResult:
    """Convenience wrapper that builds an environment from a mapping."""
    from env import EnvConfig

    env = SwarmEnv(EnvConfig.from_dict(env_config))
    return evaluate_policy(env, policy_fn, episodes=episodes, seed=seed)
