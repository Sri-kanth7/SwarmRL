"""Reward and collision-interface foundation for the swarm environment.

Week 1 provides the reward architecture and its basic terms:

- positive reward for newly explored coverage,
- strong negative reward for collisions,
- shaping terms for boundary contact and per-step cost.

Term weights live in :class:`RewardConfig` and are supplied by the
environment configuration. Tuning, curriculum scheduling, and the
full Week 2 reward system are intentionally out of scope here.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Collection, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class RewardConfig:
    """Weights for the Week 1 reward terms.

    Attributes:
        exploration: Reward per newly explored coverage cell.
        collision: Penalty applied to every drone involved in a
            collision during the step.
        boundary: Penalty applied to every drone that contacted the
            world boundary during the step.
        time: Per-step penalty applied to every drone.
        energy: Reserved weight for a speed/effort term (0.0 = inactive).
        spread: Reserved weight for a swarm-spreading term (0.0 = inactive).
    """

    exploration: float = 1.0
    collision: float = -10.0
    boundary: float = -0.1
    time: float = -0.01
    energy: float = 0.0
    spread: float = 0.0

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "RewardConfig":
        """Build from a mapping, ignoring unknown keys."""
        if not data:
            return cls()
        known = {f.name for f in fields(cls)}
        return cls(**{k: float(v) for k, v in data.items() if k in known})

    def to_dict(self) -> dict[str, float]:
        """Return a serializable representation."""
        return {f.name: getattr(self, f.name) for f in fields(self)}


class RewardCalculator:
    """Computes per-drone rewards for a single environment step.

    The calculator is stateless: episode-level bookkeeping belongs to
    the environment, which can sum the returned step rewards.
    """

    def __init__(self, config: RewardConfig | None = None) -> None:
        self.config = config or RewardConfig()

    def compute(
        self,
        agent_ids: Sequence[str],
        *,
        newly_visited: Mapping[str, int] | None = None,
        collision_agents: Collection[str] = (),
        boundary_agents: Collection[str] = (),
        speeds: Mapping[str, float] | None = None,
    ) -> dict[str, float]:
        """Return the reward of every agent for one step.

        Args:
            agent_ids: All drones that acted during the step.
            newly_visited: Number of newly explored cells per drone.
            collision_agents: Drones involved in an agent-agent collision.
            boundary_agents: Drones that contacted the world boundary.
            speeds: Forward speed magnitude per drone, used by the
                reserved energy term.

        Returns:
            Mapping of drone id to scalar reward.
        """
        newly_visited = newly_visited or {}
        speeds = speeds or {}
        cfg = self.config
        rewards: dict[str, float] = {}
        for agent in agent_ids:
            reward = cfg.time
            reward += cfg.exploration * float(newly_visited.get(agent, 0))
            if agent in collision_agents:
                reward += cfg.collision
            if agent in boundary_agents:
                reward += cfg.boundary
            reward += cfg.energy * abs(float(speeds.get(agent, 0.0)))
            rewards[agent] = float(reward)
        return rewards

    def collision_event_count(
        self,
        collision_pairs: Iterable[tuple[str, str]],
        boundary_agents: Collection[str],
    ) -> int:
        """Count collision events for metrics: one per agent-agent pair
        plus one per boundary contact."""
        return len(list(collision_pairs)) + len(set(boundary_agents))
