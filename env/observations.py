"""Observation-space definition for the swarm environment.

Each drone receives one fixed-length ``float32`` vector. The layout
is defined once here and mirrored in ``docs/contracts/
environment_api.md``. All components are normalized so that values
stay inside the declared ``Box`` bounds:

====== ===================== ============= =========================
Block  Content               Dim           Range
====== ===================== ============= =========================
1      Own position          3             ``[0, 1]`` per axis,
       (x, y, z) / world                   world-normalized
2      Own velocity          3             ``[-1, 1]``,
       (vx, vy, vz) / max_speed            world-normalized
3      Own attitude          2             ``[-1, 1]``,
       pitch / max_pitch, yaw / pi          angle-normalized
4      Nearest neighbour     4             relative offset in
       (dx, dy, dz)/world, dist/diag       ``[-1, 1]``, dist in
                                           ``[0, 1]``
5      LiDAR ranges          K             ``[0, 1]``, distance to
       K horizontal rays                   boundary / world diagonal
6      Local coverage        1             ``[0, 1]``, visited
                                           fraction near the drone
====== ===================== ============= =========================

``K`` is ``EnvConfig.lidar_rays`` (default 8), so the observation
dimension is ``13 + K`` (21 by default).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Mapping

import numpy as np
from gymnasium import spaces

from env.coverage import CoverageGrid
from env.physics import DroneState, ray_distance_to_boundary

if TYPE_CHECKING:
    from env.config import EnvConfig

NEAREST_NEIGHBOUR_DIM = 4
BASE_OBS_DIM = 13


def observation_dim(config: "EnvConfig") -> int:
    """Length of the observation vector for the given configuration."""
    return BASE_OBS_DIM + int(config.lidar_rays)


def observation_bounds(config: "EnvConfig") -> tuple[np.ndarray, np.ndarray]:
    """Return the ``(low, high)`` arrays matching the documented layout."""
    dim = observation_dim(config)
    low = np.zeros(dim, dtype=np.float32)
    high = np.ones(dim, dtype=np.float32)
    offset = 3
    low[offset : offset + 3] = -1.0
    offset += 3
    low[offset : offset + 2] = -1.0
    offset += 2
    low[offset : offset + NEAREST_NEIGHBOUR_DIM - 1] = -1.0
    return low, high


def observation_space(config: "EnvConfig") -> spaces.Box:
    """Continuous ``Box`` observation space of shape ``(observation_dim,)``."""
    low, high = observation_bounds(config)
    return spaces.Box(low=low, high=high, dtype=np.float32)


def _nearest_neighbour(
    agent: str, states: Mapping[str, DroneState], config: "EnvConfig"
) -> np.ndarray:
    own = states[agent].position
    best_offset = np.zeros(3, dtype=np.float64)
    best_distance = math.inf
    for other, state in states.items():
        if other == agent:
            continue
        offset = state.position - own
        distance = float(np.dot(offset, offset))
        if distance < best_distance:
            best_distance = distance
            best_offset = offset
    if math.isinf(best_distance):
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    normalized = best_offset / np.asarray(config.world_size, dtype=np.float64)
    distance = math.sqrt(best_distance)
    diagonal = config.horizontal_diagonal
    return np.array(
        [normalized[0], normalized[1], normalized[2], min(1.0, distance / diagonal)],
        dtype=np.float64,
    )


def _lidar_readings(
    position: np.ndarray, config: "EnvConfig"
) -> np.ndarray:
    readings = np.zeros(int(config.lidar_rays), dtype=np.float64)
    for i in range(int(config.lidar_rays)):
        angle = 2.0 * math.pi * i / int(config.lidar_rays)
        direction = (math.cos(angle), 0.0, math.sin(angle))
        distance = ray_distance_to_boundary(position, direction, config.world_size)
        readings[i] = min(1.0, distance / config.horizontal_diagonal)
    return readings


def build_observation(
    agent: str,
    states: Mapping[str, DroneState],
    coverage: CoverageGrid,
    config: "EnvConfig",
) -> np.ndarray:
    """Assemble the observation vector of one drone.

    Args:
        agent: The drone's identifier; must exist in ``states``.
        states: Kinematic states of every drone.
        coverage: Shared coverage grid of the current episode.
        config: Environment configuration.

    Returns:
        ``float32`` vector following the layout documented above.
    """
    state = states[agent]
    world = np.asarray(config.world_size, dtype=np.float64)

    own_position = state.position / world
    own_velocity = state.velocity / config.max_speed
    attitude = np.array(
        [state.pitch / config.max_pitch, state.yaw / math.pi], dtype=np.float64
    )
    neighbour = _nearest_neighbour(agent, states, config)
    lidar = _lidar_readings(state.position, config)
    local_coverage = np.array(
        [
            coverage.local_fraction(
                float(state.position[0]),
                float(state.position[2]),
                config.local_coverage_radius,
            )
        ],
        dtype=np.float64,
    )

    observation = np.concatenate(
        [own_position, own_velocity, attitude, neighbour, lidar, local_coverage]
    ).astype(np.float32)
    low, high = observation_bounds(config)
    return np.clip(observation, low, high)
