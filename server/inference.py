"""Mock state generator for the backend.

Week 1 ships a deterministic mock generator instead of policy
inference: every connected client receives plausible drone states
without any trained checkpoint, so the frontend and integration
layers can be developed independently of training.

The mock motion model gives each drone a circular patrol path over
the ground plane with a gently oscillating height. Coverage and
collision bookkeeping reuse the environment's primitives
(``env.coverage.CoverageGrid`` and ``env.physics``), so the mock
messages follow the same rules as the real simulation.

Replacing this generator with checkpoint-backed inference is later
work; the interface (``reset`` / ``current`` / ``step`` / ``metrics``)
is already the one inference will implement.
"""

from __future__ import annotations

import math

import numpy as np

from env.coverage import CoverageGrid
from env.physics import DroneState, detect_collision_pairs, wrap_angle
from env.rewards import RewardConfig
from server.config import ServerConfig
from server.metrics import MetricsCollector
from server.schemas import AgentState, MetricsSnapshot, StepMessage


class MockInferenceEngine:
    """Deterministic mock drone-state generator.

    All randomness comes from a seeded generator, so the same seed
    always produces the same stream of messages.
    """

    def __init__(self, config: ServerConfig) -> None:
        self.config = config
        self._rng = np.random.default_rng(config.seed)
        self._coverage = CoverageGrid(config.world_size, config.coverage_cell_size)
        self._collector = MetricsCollector()
        self._step = 0
        self._agent_count = config.agent_count
        self._positions = np.zeros((self._agent_count, 3), dtype=np.float64)
        self._velocities = np.zeros((self._agent_count, 3), dtype=np.float64)
        self._pitches = np.zeros(self._agent_count, dtype=np.float64)
        self._yaws = np.zeros(self._agent_count, dtype=np.float64)
        self._centres = np.zeros((self._agent_count, 2), dtype=np.float64)
        self._radii = np.zeros(self._agent_count, dtype=np.float64)
        self._angular_speeds = np.zeros(self._agent_count, dtype=np.float64)
        self._base_heights = np.zeros(self._agent_count, dtype=np.float64)
        self._height_amplitudes = np.zeros(self._agent_count, dtype=np.float64)
        self._phases = np.zeros(self._agent_count, dtype=np.float64)
        self._newly_visited: list[int] = []
        self._last_step_events = 0
        self._initialize_paths()
        self._update_state(at_time=0.0, record_coverage=False)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def reset(self, seed: int | None = None) -> StepMessage:
        """Start a new mock episode and return its initial message."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        elif self.config.seed is not None:
            self._rng = np.random.default_rng(self.config.seed)
        self._coverage.reset()
        self._collector.reset()
        self._step = 0
        self._last_step_events = 0
        self._initialize_paths()
        self._update_state(at_time=0.0, record_coverage=False)
        return self.current()

    def current(self) -> StepMessage:
        """Message for the current state (does not advance time)."""
        return StepMessage(
            step=self._step,
            agents=self._agent_states(),
            newly_visited_cells=list(self._newly_visited),
            collisions=self._last_step_events,
            explored_pct=self._coverage.explored_pct(),
        )

    def step(self) -> StepMessage:
        """Advance the mock simulation by one tick and return the message."""
        self._step += 1
        tick_seconds = self.config.tick_interval_seconds
        self._update_state(at_time=self._step * tick_seconds, record_coverage=True)
        self._last_step_events = self._collision_events()
        cumulative = self._collector_latest_collisions() + self._last_step_events
        snapshot = self._collector.record(
            step=self._step,
            explored_pct=self._coverage.explored_pct(),
            collisions=cumulative,
            rewards=self._mock_rewards(),
        )
        return StepMessage(
            step=snapshot.step,
            agents=self._agent_states(),
            newly_visited_cells=list(self._newly_visited),
            collisions=self._last_step_events,
            explored_pct=snapshot.explored_pct,
        )

    def metrics(self) -> MetricsSnapshot:
        """Latest metrics snapshot (creates an initial one if needed)."""
        if self._collector.latest is None:
            self._collector.record(
                step=self._step,
                explored_pct=self._coverage.explored_pct(),
                collisions=0,
                rewards=self._mock_rewards(),
            )
        return self._collector.latest or MetricsSnapshot()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _initialize_paths(self) -> None:
        x_size, y_size, z_size = (float(v) for v in self.config.world_size)
        margin = 5.0
        count = self._agent_count
        self._radii = self._rng.uniform(3.0, 12.0, size=count)
        self._angular_speeds = self._rng.uniform(0.1, 0.4, size=count) * self._rng.choice(
            [-1.0, 1.0], size=count
        )
        self._centres[:, 0] = self._rng.uniform(
            margin + 15.0, max(margin + 15.0, x_size - margin - 15.0), size=count
        )
        self._centres[:, 1] = self._rng.uniform(
            margin + 15.0, max(margin + 15.0, z_size - margin - 15.0), size=count
        )
        self._base_heights = self._rng.uniform(0.15 * y_size, 0.7 * y_size, size=count)
        self._height_amplitudes = self._rng.uniform(0.5, 3.0, size=count)
        self._phases = self._rng.uniform(0.0, 2.0 * math.pi, size=count)

    def _update_state(self, at_time: float, record_coverage: bool) -> None:
        x_size, y_size, z_size = (float(v) for v in self.config.world_size)
        previous = self._positions.copy() if record_coverage else None
        newly: list[int] = []
        for i in range(self._agent_count):
            angle = self._phases[i] + self._angular_speeds[i] * at_time
            x = self._centres[i, 0] + self._radii[i] * math.cos(angle)
            z = self._centres[i, 1] + self._radii[i] * math.sin(angle)
            y = self._base_heights[i] + self._height_amplitudes[i] * math.sin(
                0.5 * at_time + self._phases[i]
            )
            x = float(np.clip(x, 0.0, x_size))
            z = float(np.clip(z, 0.0, z_size))
            y = float(np.clip(y, 0.0, y_size))
            direction_sign = 1.0 if self._angular_speeds[i] >= 0 else -1.0
            yaw = wrap_angle(angle + direction_sign * math.pi / 2.0)
            pitch = 0.05 * math.sin(0.7 * at_time + self._phases[i])
            if previous is not None:
                delta = np.array([x, z]) - previous[i][[0, 2]]
                self._velocities[i] = np.array(
                    [delta[0], 0.0, delta[1]]
                ) / max(1e-9, self.config.tick_interval_seconds)
                newly.extend(
                    self._coverage.mark_segment(
                        float(previous[i][0]),
                        float(previous[i][2]),
                        x,
                        z,
                    )
                )
            self._positions[i] = (x, y, z)
            self._pitches[i] = pitch
            self._yaws[i] = yaw
        self._newly_visited = sorted(set(newly))

    def _drone_states(self) -> dict[str, DroneState]:
        states: dict[str, DroneState] = {}
        for i in range(self._agent_count):
            states[str(i)] = DroneState(
                position=self._positions[i].copy(),
                velocity=self._velocities[i].copy(),
                pitch=float(self._pitches[i]),
                yaw=float(self._yaws[i]),
            )
        return states

    def _collision_events(self) -> int:
        return len(
            detect_collision_pairs(self._drone_states(), self.config.collision_radius)
        )

    def _collector_latest_collisions(self) -> int:
        latest = self._collector.latest
        return latest.collisions if latest else 0

    def _mock_rewards(self) -> list[float]:
        """Toy per-drone rewards exercising the metrics contract only."""
        exploration = RewardConfig().exploration
        if self._agent_count == 0:
            return []
        return [exploration * len(self._newly_visited) / self._agent_count] * self._agent_count

    def _agent_states(self) -> list[AgentState]:
        return [
            AgentState(
                id=i,
                x=float(self._positions[i][0]),
                y=float(self._positions[i][1]),
                z=float(self._positions[i][2]),
                pitch=float(self._pitches[i]),
                yaw=float(self._yaws[i]),
            )
            for i in range(self._agent_count)
        ]
