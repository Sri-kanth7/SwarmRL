"""PettingZoo Parallel-API swarm environment.

A swarm of drones searches a bounded continuous 3D disaster zone
while a shared coverage grid tracks explored area. The environment
is the root of the pipeline:

    environment -> observations -> training -> backend -> frontend

Week 1 establishes the environment contract (reset/step lifecycle,
observation and action spaces, bounded-world kinematics, coverage,
and basic rewards). Advanced rewards, curriculum, dynamic
obstacles, and wind are later work and can be added behind the
existing module boundaries without rewriting this file.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces
from pettingzoo.utils.env import ParallelEnv

from env.actions import action_space, clip_action
from env.config import EnvConfig
from env.coverage import CoverageGrid
from env.observations import observation_space, build_observation
from env.physics import (
    DroneState,
    detect_collision_pairs,
    integrate_state,
    spawn_states,
)
from env.rewards import RewardCalculator


class SwarmEnv(ParallelEnv):
    """Multi-agent drone-search environment (PettingZoo Parallel API).

    Agent identifiers are strings ``"drone_0" .. "drone_{n-1}"``; the
    integer index encoded in the name is the identifier used on the
    WebSocket wire (see ``docs/contracts/websocket_schema.md``).
    """

    metadata = {"name": "swarm_v1", "render_modes": []}

    def __init__(self, config: EnvConfig | None = None, render_mode: str | None = None):
        self.config = config or EnvConfig()
        self.render_mode = render_mode
        self.possible_agents: list[str] = [
            f"drone_{i}" for i in range(self.config.agent_count)
        ]
        self.agents: list[str] = []
        self._observation_space = observation_space(self.config)
        self._action_space = action_space(self.config)
        self._states: dict[str, DroneState] = {}
        self._coverage = CoverageGrid(
            self.config.world_size, self.config.coverage_cell_size
        )
        self._reward_calculator = RewardCalculator(self.config.reward)
        self._rng = np.random.default_rng(self.config.seed)
        self._step_count = 0
        self._last_newly_visited: dict[str, list[int]] = {}
        self._last_collision_agents: set[str] = set()
        self._last_boundary_agents: set[str] = set()
        self._last_collision_events = 0
        self._total_collisions = 0
        self._episode_reward: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Spaces
    # ------------------------------------------------------------------
    def observation_space(self, agent: str) -> spaces.Space:
        """Observation space of one drone."""
        return self._observation_space

    def action_space(self, agent: str) -> spaces.Space:
        """Action space of one drone."""
        return self._action_space

    @property
    def observation_spaces(self) -> dict[str, spaces.Space]:
        """Observation spaces for every possible agent."""
        return {agent: self._observation_space for agent in self.possible_agents}

    @property
    def action_spaces(self) -> dict[str, spaces.Space]:
        """Action spaces for every possible agent."""
        return {agent: self._action_space for agent in self.possible_agents}

    # ------------------------------------------------------------------
    # Convenience accessors used by tests, backend, and documentation
    # ------------------------------------------------------------------
    @property
    def coverage(self) -> CoverageGrid:
        """Shared coverage grid of the current episode."""
        return self._coverage

    @property
    def step_count(self) -> int:
        """Number of steps taken in the current episode."""
        return self._step_count

    @property
    def explored_pct(self) -> float:
        """Explored percentage of the current episode, in ``[0, 100]``."""
        return self._coverage.explored_pct()

    @property
    def last_collision_events(self) -> int:
        """Collision events counted during the most recent step."""
        return self._last_collision_events

    @property
    def total_collisions(self) -> int:
        """Cumulative collision events of the current episode."""
        return self._total_collisions

    @property
    def states(self) -> dict[str, DroneState]:
        """Read-only view of the current drone states."""
        return self._states

    def agent_index(self, agent: str) -> int:
        """Integer index of a drone, matching the wire-format id."""
        return int(agent.rsplit("_", 1)[-1])

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------
    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
        """Start a new episode.

        Args:
            seed: Optional seed that reseeds the spawn generator.
            options: Reserved for future overrides (ignored in Week 1).

        Returns:
            ``(observations, infos)`` keyed by every agent id.
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.agents = list(self.possible_agents)
        self._states = spawn_states(self.agents, self.config, self._rng)
        self._coverage.reset()
        self._step_count = 0
        self._last_newly_visited = {agent: [] for agent in self.agents}
        self._last_collision_agents = set()
        self._last_boundary_agents = set()
        self._last_collision_events = 0
        self._total_collisions = 0
        self._episode_reward = {agent: 0.0 for agent in self.agents}
        observations = {agent: self._observe(agent) for agent in self.agents}
        infos = {agent: self._info(agent) for agent in self.agents}
        return observations, infos

    def step(
        self, actions: dict[str, np.ndarray]
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, Any]],
    ]:
        """Advance every live drone by one simulation step.

        Args:
            actions: Mapping of agent id to its 3-value action vector.

        Returns:
            ``(observations, rewards, terminations, truncations, infos)``.
            Observations cover the agents that remain alive; the other
            mappings cover every agent that acted. Episodes truncate
            when ``max_steps`` is reached.
        """
        acting = list(self.agents)
        expected = set(acting)
        provided = set(actions)
        if provided != expected:
            raise ValueError(
                f"actions must cover exactly the live agents; "
                f"missing={sorted(expected - provided)}, "
                f"unexpected={sorted(provided - expected)}"
            )

        previous_positions = {
            agent: self._states[agent].position.copy() for agent in acting
        }
        boundary_agents: set[str] = set()
        speeds: dict[str, float] = {}
        for agent in acting:
            vector = clip_action(actions[agent], self.config)
            velocity = float(vector[0])
            pitch = float(vector[1])
            yaw = float(vector[2])
            new_state, boundary = integrate_state(
                self._states[agent],
                velocity=velocity,
                pitch=pitch,
                yaw=yaw,
                config=self.config,
            )
            self._states[agent] = new_state
            speeds[agent] = abs(velocity)
            if boundary:
                boundary_agents.add(agent)

        pairs = detect_collision_pairs(self._states, self.config.collision_radius)
        collision_agents = {agent for pair in pairs for agent in pair}

        newly_visited: dict[str, list[int]] = {}
        for agent in acting:
            previous = previous_positions[agent]
            current = self._states[agent].position
            newly_visited[agent] = self._coverage.mark_segment(
                float(previous[0]),
                float(previous[2]),
                float(current[0]),
                float(current[2]),
            )

        rewards = self._reward_calculator.compute(
            acting,
            newly_visited={a: len(cells) for a, cells in newly_visited.items()},
            collision_agents=collision_agents,
            boundary_agents=boundary_agents,
            speeds=speeds,
        )
        collision_events = self._reward_calculator.collision_event_count(
            pairs, boundary_agents
        )

        for agent in acting:
            self._episode_reward[agent] += rewards[agent]

        self._step_count += 1
        truncated = self._step_count >= self.config.max_steps

        self._last_newly_visited = newly_visited
        self._last_collision_agents = collision_agents
        self._last_boundary_agents = boundary_agents
        self._last_collision_events = collision_events
        self._total_collisions += collision_events

        terminations = {agent: False for agent in acting}
        truncations = {agent: truncated for agent in acting}
        infos = {agent: self._info(agent) for agent in acting}

        if truncated:
            self.agents = []
            observations: dict[str, np.ndarray] = {}
        else:
            self.agents = list(acting)
            observations = {agent: self._observe(agent) for agent in acting}
        return observations, rewards, terminations, truncations, infos

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _observe(self, agent: str) -> np.ndarray:
        return build_observation(agent, self._states, self._coverage, self.config)

    def _info(self, agent: str) -> dict[str, Any]:
        return {
            "step": self._step_count,
            "explored_pct": self.explored_pct,
            "collisions": self._total_collisions,
            "step_collisions": self._last_collision_events,
            "collision": agent in self._last_collision_agents,
            "boundary_contact": agent in self._last_boundary_agents,
            "newly_visited_cells": list(self._last_newly_visited.get(agent, [])),
            "episode_reward": self._episode_reward.get(agent, 0.0),
            "agent_index": self.agent_index(agent),
        }

    def render(self) -> None:
        """Rendering is not implemented in Week 1."""

    def close(self) -> None:
        """End the current episode and release its state."""
        self.agents = []
        self._states = {}
