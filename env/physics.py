"""Deterministic drone kinematics and collision primitives.

The physics layer is intentionally lightweight: constant-speed
translation along the drone's heading, bounded by the world box.
It is not a full rigid-body engine; aerodynamics, wind, and
obstacles are outside the Week 1 scope.

Coordinate convention: right-handed, y-up. ``x``/``z`` span the
horizontal plane, ``y`` is height. Heading ``yaw`` is measured from
the ``+x`` axis toward ``+z``; ``pitch`` lifts the nose toward ``+y``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Mapping, Sequence

import numpy as np

if TYPE_CHECKING:
    from env.config import EnvConfig

Vector3 = np.ndarray


@dataclass
class DroneState:
    """Kinematic state of a single drone.

    Attributes:
        position: ``(x, y, z)`` position in world units.
        velocity: ``(vx, vy, vz)`` velocity of the current step.
        pitch: Nose pitch in radians.
        yaw: Heading in radians, wrapped to ``(-pi, pi]``.
    """

    position: Vector3
    velocity: Vector3 = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    pitch: float = 0.0
    yaw: float = 0.0

    def copy(self) -> "DroneState":
        """Return an independent copy of this state."""
        return DroneState(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            pitch=float(self.pitch),
            yaw=float(self.yaw),
        )


def wrap_angle(angle: float) -> float:
    """Wrap an angle to the range ``(-pi, pi]``."""
    wrapped = (float(angle) + math.pi) % (2.0 * math.pi) - math.pi
    return math.pi if wrapped == -math.pi else wrapped


def heading_vector(pitch: float, yaw: float) -> Vector3:
    """Unit forward vector for the given orientation."""
    cos_pitch = math.cos(pitch)
    return np.array(
        [cos_pitch * math.cos(yaw), math.sin(pitch), cos_pitch * math.sin(yaw)],
        dtype=np.float64,
    )


def is_inside_world(position: Sequence[float], world_size: Sequence[float]) -> bool:
    """True when the position lies within (or on) the world box."""
    return all(
        0.0 <= float(position[i]) <= float(world_size[i]) for i in range(3)
    )


def integrate_state(
    state: DroneState,
    *,
    velocity: float,
    pitch: float,
    yaw: float,
    config: "EnvConfig",
    dt: float | None = None,
) -> tuple[DroneState, bool]:
    """Advance one drone by a commanded velocity/pitch/yaw.

    Commands are absolute: pitch is clipped to ``+-max_pitch`` and yaw
    is wrapped. The drone translates along its commanded heading for
    ``dt`` seconds. If the move leaves the world box, the position is
    clipped back inside and the boundary flag is returned.

    Returns:
        ``(new_state, boundary_contact)``
    """
    step_dt = config.dt if dt is None else dt
    clipped_pitch = float(np.clip(pitch, -config.max_pitch, config.max_pitch))
    clipped_yaw = wrap_angle(yaw)
    velocity_vector = heading_vector(clipped_pitch, clipped_yaw) * float(velocity)
    new_position = state.position + velocity_vector * step_dt
    boundary_contact = not is_inside_world(new_position, config.world_size)
    if boundary_contact:
        new_position = np.clip(
            new_position,
            a_min=np.zeros(3),
            a_max=np.asarray(config.world_size, dtype=np.float64),
        )
    return (
        DroneState(
            position=new_position,
            velocity=velocity_vector,
            pitch=clipped_pitch,
            yaw=clipped_yaw,
        ),
        boundary_contact,
    )


def detect_collision_pairs(
    states: Mapping[str, DroneState], radius: float
) -> list[tuple[str, str]]:
    """Return every pair of drones closer than ``radius``.

    The comparison is symmetric: each pair appears once, ordered by
    iteration order of ``states``. The cost is O(n^2), which is fine
    for the Week 1 swarm size.
    """
    ids = list(states)
    radius_squared = float(radius) * float(radius)
    pairs: list[tuple[str, str]] = []
    for i, first in enumerate(ids):
        first_position = states[first].position
        for second in ids[i + 1 :]:
            delta = first_position - states[second].position
            if float(np.dot(delta, delta)) < radius_squared:
                pairs.append((first, second))
    return pairs


def ray_distance_to_boundary(
    position: Sequence[float], direction: Sequence[float], world_size: Sequence[float]
) -> float:
    """Distance from ``position`` to the world box along ``direction``.

    The direction need not be normalized; only its orientation matters.
    Returns 0.0 when the ray points out of the box from a point that is
    already on its boundary.
    """
    distance = math.inf
    for axis in range(3):
        component = float(direction[axis])
        if abs(component) < 1e-12:
            continue
        if component > 0.0:
            to_boundary = (float(world_size[axis]) - float(position[axis])) / component
        else:
            to_boundary = (0.0 - float(position[axis])) / component
        if 0.0 <= to_boundary < distance:
            distance = to_boundary
    return 0.0 if math.isinf(distance) else float(distance)


def spawn_states(
    agent_ids: Sequence[str], config: "EnvConfig", rng: np.random.Generator
) -> dict[str, DroneState]:
    """Create deterministic initial states drawn from ``rng``.

    Positions are uniform inside the world minus ``spawn_margin``,
    headings are uniform, and drones start at rest with zero pitch.
    """
    x_size, y_size, z_size = (float(v) for v in config.world_size)
    margin = min(config.spawn_margin, 0.25 * min(x_size, y_size, z_size))
    states: dict[str, DroneState] = {}
    for agent in agent_ids:
        states[agent] = DroneState(
            position=np.array(
                [
                    rng.uniform(margin, max(margin, x_size - margin)),
                    rng.uniform(margin, max(margin, y_size - margin)),
                    rng.uniform(margin, max(margin, z_size - margin)),
                ],
                dtype=np.float64,
            ),
            velocity=np.zeros(3, dtype=np.float64),
            pitch=0.0,
            yaw=float(rng.uniform(-math.pi, math.pi)),
        )
    return states
