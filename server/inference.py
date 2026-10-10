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
from pathlib import Path
from typing import Any, Mapping, Protocol

import numpy as np

from env.coverage import CoverageGrid
from env.physics import DroneState, detect_collision_pairs, wrap_angle
from env.rewards import RewardConfig
from training.checkpoint import (
    METADATA_FILENAME,
    CheckpointMetadata,
    load_metadata,
)
from training.rl_config import POLICY_ID
from env import EnvConfig
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


class InferenceEngine(Protocol):
    """Common interface for inference engines used by the server."""

    def reset(self, seed: int | None = None) -> StepMessage: ...

    def current(self) -> StepMessage: ...

    def step(self) -> StepMessage: ...

    def metrics(self) -> MetricsSnapshot: ...


class CheckpointLoadError(Exception):
    """Raised when a checkpoint cannot be loaded or is incompatible."""


class CheckpointInferenceEngine:
    """Inference engine that loads an RLlib checkpoint and runs policy inference.

    Uses the RLlib Algorithm checkpoint produced by training. The engine reuses
    the existing SwarmEnv/SwarmMultiAgentEnv interfaces and agent IDs, and
    produces StepMessage in the same schema.
    """

    def __init__(self, config: ServerConfig, checkpoint_path: str | Path | None = None) -> None:
        self.config = config
        self._collector = MetricsCollector()
        self._step = 0
        self._last_step_events = 0
        self._last_newly_visited: list[int] = []
        self._last_explored_pct = 0.0
        self._last_positions: dict[str, tuple[float, float, float]] = {}
        self._last_velocities: dict[str, tuple[float, float, float]] = {}
        self._last_pitches: dict[str, float] = {}
        self._last_yaws: dict[str, float] = {}

        self._algorithm = None
        self._env = None
        self._started_ray = False

        resolved_ckpt = self._resolve_checkpoint_path(checkpoint_path)
        self._metadata = self._load_and_validate_metadata(resolved_ckpt)
        self._algorithm = self._load_algorithm(resolved_ckpt)
        try:
            from training.rllib_env import SwarmMultiAgentEnv
        except Exception as e:
            raise CheckpointLoadError(f"RLlib is required for checkpoint inference: {e}") from e

        env_cfg = EnvConfig(
            agent_count=self.config.agent_count,
            world_size=self.config.world_size,
            seed=self.config.seed,
            collision_radius=self.config.collision_radius,
        )
        self._env = SwarmMultiAgentEnv(env_cfg.to_dict())
        self._possible_agents = list(self._env.possible_agents)
        self._reset_state(seed=self.config.seed)

    def _resolve_checkpoint_path(self, checkpoint_path: str | Path | None) -> Path:
        if checkpoint_path:
            base = Path(checkpoint_path)
        elif self.config.checkpoint_path:
            base = Path(self.config.checkpoint_path)
        else:
            base = Path(self.config.checkpoint_dir) / "latest"
        resolved = base.resolve()
        if not resolved.exists():
            raise CheckpointLoadError(f"Checkpoint not found: {resolved}")
        return resolved

    def _load_and_validate_metadata(self, checkpoint_dir: Path) -> CheckpointMetadata:
        meta_path = checkpoint_dir / METADATA_FILENAME
        if not meta_path.exists():
            raise CheckpointLoadError(f"Metadata file missing: {meta_path}")
        try:
            metadata = load_metadata(checkpoint_dir)
        except Exception as e:
            raise CheckpointLoadError(f"Failed to load checkpoint metadata: {e}") from e

        if metadata.format_version != 1:
            raise CheckpointLoadError(
                f"Incompatible checkpoint format_version: {metadata.format_version}"
            )
        if metadata.algorithm not in ("IPPO", "MAPPO"):
            raise CheckpointLoadError(f"Unsupported algorithm: {metadata.algorithm}")
        return metadata

    def _load_algorithm(self, checkpoint_dir: Path):
        try:
            import ray
        except Exception as e:
            raise CheckpointLoadError(f"Ray is required for checkpoint inference: {e}") from e

        started_ray = not ray.is_initialized()
        try:
            if started_ray:
                ray.init(ignore_reinit_error=True, include_dashboard=False)
                self._started_ray = True
            from ray.rllib.algorithms.algorithm import Algorithm

            rllib_dir = checkpoint_dir / "rllib"
            if not rllib_dir.exists():
                raise CheckpointLoadError(f"RLlib checkpoint directory missing: {rllib_dir}")
            return Algorithm.from_checkpoint(str(rllib_dir.resolve()))
        except CheckpointLoadError:
            # If we started Ray here but failed later, leave Ray running
            # only if owned elsewhere; but better to not shutdown - other components may own it.
            # However per requirement: only shutdown if engine started it. If init failed after starting,
            # we don't have algorithm; cleanup in close() will handle based on _started_ray.
            raise
        except Exception as e:
            raise CheckpointLoadError(f"Failed to load RLlib checkpoint: {e}") from e

    def _reset_state(self, seed: int | None) -> None:
        obs, _ = self._env.reset(seed=seed)
        self._step = 0
        self._last_step_events = 0
        self._last_newly_visited = []
        self._last_explored_pct = 0.0
        self._store_obs(obs)

    def _store_obs(self, obs: Mapping[str, Any]) -> None:
        for agent, o in obs.items():
            try:
                arr = np.array(o)
                if arr.ndim == 1:
                    self._last_positions[agent] = (float(arr[0]), float(arr[1]), float(arr[2])) if len(arr) >= 3 else (0, 0, 0)
                else:
                    self._last_positions[agent] = (0.0, 0.0, 0.0)
            except Exception:
                self._last_positions[agent] = (0.0, 0.0, 0.0)

    def reset(self, seed: int | None = None) -> StepMessage:
        self._reset_state(seed)
        return self.current()

    def current(self) -> StepMessage:
        return StepMessage(
            step=self._step,
            agents=self._agent_states(),
            newly_visited_cells=list(self._last_newly_visited),
            collisions=self._last_step_events,
            explored_pct=self._last_explored_pct,
        )

    def step(self) -> StepMessage:
        try:
            obs, _ = self._env.reset(seed=None) if self._step == 0 else (None, None)
        except Exception:
            obs = None
        if obs is None:
            obs, _, _, _, _ = self._env.step({})

        actions = self._compute_actions(obs)
        obs_next, rewards, terminations, truncations, infos = self._env.step(actions)
        self._step += 1
        self._update_from_step(obs_next, infos, terminations, truncations)
        return self.current()

    def _compute_actions(self, obs: Mapping[str, Any]) -> dict[str, np.ndarray]:
        try:
            policy = self._algorithm.get_policy(POLICY_ID)
        except Exception:
            policy = self._algorithm.get_policy()

        actions: dict[str, np.ndarray] = {}
        for agent in self._possible_agents:
            if agent not in obs:
                continue
            try:
                action = policy.compute_single_action(obs[agent], explore=False)[0]
                actions[agent] = np.array(action)
            except Exception:
                actions[agent] = np.zeros(self._env.action_space(agent).shape)
        return actions

    def _update_from_step(
        self,
        obs: Mapping[str, Any],
        infos: Mapping[str, Any],
        terminations: Mapping[str, Any],
        truncations: Mapping[str, Any],
    ) -> None:
        self._store_obs(obs)
        if not infos:
            self._last_newly_visited = []
            self._last_explored_pct = 0.0
            self._last_step_events = 0
        else:
            samples = list(infos.values())
            try:
                self._last_explored_pct = float(np.mean([s.get("explored_pct", 0.0) for s in samples]))
            except Exception:
                self._last_explored_pct = 0.0
            try:
                self._last_step_events = int(max(s.get("collisions", 0) for s in samples))
            except Exception:
                self._last_step_events = 0
            self._last_newly_visited = []

        self._collector.record(
            step=self._step,
            explored_pct=self._last_explored_pct,
            collisions=self._collector.latest.collisions if self._collector.latest else 0,
            rewards=[],
        )

    def metrics(self) -> MetricsSnapshot:
        if self._collector.latest is None:
            self._collector.record(step=self._step, explored_pct=0.0, collisions=0, rewards=[])
        return self._collector.latest or MetricsSnapshot()

    def _agent_states(self) -> list[AgentState]:
        states = []
        for i, agent in enumerate(self._possible_agents):
            pos = self._last_positions.get(agent, (0.0, 0.0, 0.0))
            pitch = float(self._last_pitches.get(agent, 0.0))
            yaw = float(self._last_yaws.get(agent, 0.0))
            states.append(
                AgentState(id=i, x=float(pos[0]), y=float(pos[1]), z=float(pos[2]), pitch=pitch, yaw=yaw)
            )
        return states

    def close(self) -> None:
        # Stop algorithm first
        try:
            if getattr(self, "_algorithm", None) is not None:
                self._algorithm.stop()
                self._algorithm = None
        except Exception:
            pass

        # Close environment
        try:
            if getattr(self, "_env", None) is not None:
                self._env.close()
                self._env = None
        except Exception:
            pass

        # Shutdown Ray only if this engine started it
        if getattr(self, "_started_ray", False):
            try:
                import ray

                if ray.is_initialized():
                    ray.shutdown()
            except Exception:
                pass
            finally:
                self._started_ray = False
