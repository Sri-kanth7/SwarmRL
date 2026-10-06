"""Configuration for the SwarmRL environment.

The dataclass defaults mirror ``configs/project.yaml`` so that the
environment can be constructed without any external files while
staying consistent with the project-wide configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Mapping

from env.rewards import RewardConfig

DEFAULT_WORLD_SIZE: tuple[float, float, float] = (100.0, 50.0, 100.0)
"""World extents as ``(x, y, z)``. The environment uses a right-handed,
y-up coordinate system: ``x``/``z`` span the horizontal ground plane and
``y`` is height. This matches the Three.js convention used by the
frontend, so state values can be forwarded without conversion."""


@dataclass(frozen=True)
class EnvConfig:
    """Immutable environment configuration.

    Attributes:
        agent_count: Number of drones in the swarm (default 50).
        world_size: ``(x, y, z)`` extents of the bounded world.
        seed: Default random seed used when ``reset()`` gets no seed.
        dt: Duration of one simulation step in seconds.
        max_steps: Episode length; episodes truncate at this step count.
        max_speed: Maximum forward speed magnitude in m/s. The velocity
            action component is bounded by ``[-max_speed, max_speed]``.
        max_pitch: Maximum absolute pitch in radians.
        collision_radius: Distance below which two drones collide.
        lidar_rays: Number of horizontal LiDAR-like range rays.
        coverage_cell_size: Edge length of one coverage-grid cell.
        local_coverage_radius: Radius used for the local coverage
            observation component.
        spawn_margin: Distance kept between spawned drones and walls.
        reward: Reward term weights (see :class:`env.rewards.RewardConfig`).
    """

    agent_count: int = 50
    world_size: tuple[float, float, float] = DEFAULT_WORLD_SIZE
    seed: int = 42
    dt: float = 0.1
    max_steps: int = 500
    max_speed: float = 5.0
    max_pitch: float = 0.6
    collision_radius: float = 1.5
    lidar_rays: int = 8
    coverage_cell_size: float = 5.0
    local_coverage_radius: float = 10.0
    spawn_margin: float = 5.0
    reward: RewardConfig = field(default_factory=RewardConfig)

    def __post_init__(self) -> None:
        if self.agent_count < 1:
            raise ValueError("agent_count must be >= 1")
        if len(self.world_size) != 3 or any(v <= 0 for v in self.world_size):
            raise ValueError("world_size must contain three positive values")
        if self.dt <= 0:
            raise ValueError("dt must be > 0")
        if self.max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        if self.max_speed <= 0:
            raise ValueError("max_speed must be > 0")
        if self.max_pitch <= 0:
            raise ValueError("max_pitch must be > 0")
        if self.collision_radius <= 0:
            raise ValueError("collision_radius must be > 0")
        if self.lidar_rays < 1:
            raise ValueError("lidar_rays must be >= 1")
        if self.coverage_cell_size <= 0:
            raise ValueError("coverage_cell_size must be > 0")
        if self.local_coverage_radius <= 0:
            raise ValueError("local_coverage_radius must be > 0")

    @property
    def horizontal_diagonal(self) -> float:
        """Longest straight-line distance inside the world."""
        x, y, z = self.world_size
        return float((x * x + y * y + z * z) ** 0.5)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "EnvConfig":
        """Build a config from a plain mapping (YAML, RLlib ``env_config``).

        Unknown keys are ignored so callers may attach extra metadata.
        """
        if not data:
            return cls()
        known = {f.name for f in fields(cls)}
        kwargs: dict[str, Any] = {}
        for key, value in data.items():
            if key not in known:
                continue
            if key == "world_size" and not isinstance(value, tuple):
                kwargs[key] = tuple(float(v) for v in value)
            elif key == "reward" and isinstance(value, Mapping):
                kwargs[key] = RewardConfig.from_dict(value)
            else:
                kwargs[key] = value
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        """Return a YAML/JSON-serializable representation."""
        return {
            "agent_count": self.agent_count,
            "world_size": list(self.world_size),
            "seed": self.seed,
            "dt": self.dt,
            "max_steps": self.max_steps,
            "max_speed": self.max_speed,
            "max_pitch": self.max_pitch,
            "collision_radius": self.collision_radius,
            "lidar_rays": self.lidar_rays,
            "coverage_cell_size": self.coverage_cell_size,
            "local_coverage_radius": self.local_coverage_radius,
            "spawn_margin": self.spawn_margin,
            "reward": self.reward.to_dict(),
        }
