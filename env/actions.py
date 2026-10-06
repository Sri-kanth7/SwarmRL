"""Action-space definition for the swarm environment.

Every drone receives one continuous action vector of three values:

======== ========== ==========================================
Index    Name       Meaning
======== ========== ==========================================
0        velocity   Forward speed in m/s along the current
                    heading, bounded by ``+-max_speed``.
1        pitch      Absolute nose pitch in radians, bounded by
                    ``+-max_pitch``.
2        yaw        Absolute heading in radians, wrapped to
                    ``(-pi, pi]``.
======== ========== ==========================================

Commands are absolute rather than incremental, which keeps the
kinematics deterministic and the semantics easy to inspect. The
bounds are exposed through :func:`action_bounds` and
:func:`action_space`; RLlib sees the same bounds through the
environment's ``action_space(agent)``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from gymnasium import spaces

if TYPE_CHECKING:
    from env.config import EnvConfig

ACTION_DIM = 3
ACTION_COMPONENTS: tuple[str, str, str] = ("velocity", "pitch", "yaw")
VELOCITY_INDEX = 0
PITCH_INDEX = 1
YAW_INDEX = 2


def action_bounds(config: "EnvConfig") -> tuple[np.ndarray, np.ndarray]:
    """Return the ``(low, high)`` bound arrays of the action vector."""
    low = np.array([-config.max_speed, -config.max_pitch, -math.pi], dtype=np.float32)
    high = np.array([config.max_speed, config.max_pitch, math.pi], dtype=np.float32)
    return low, high


def action_space(config: "EnvConfig") -> spaces.Box:
    """Continuous ``Box`` action space of shape ``(3,)``."""
    low, high = action_bounds(config)
    return spaces.Box(low=low, high=high, dtype=np.float32)


def clip_action(action: np.ndarray, config: "EnvConfig") -> np.ndarray:
    """Sanitize a raw action: flatten, replace non-finite values, clip."""
    vector = np.asarray(action, dtype=np.float64).reshape(-1)
    if vector.shape[0] != ACTION_DIM:
        raise ValueError(f"expected an action of {ACTION_DIM} values, got {vector.size}")
    vector = np.nan_to_num(vector, nan=0.0, posinf=1.0, neginf=-1.0)
    low, high = action_bounds(config)
    return np.clip(vector, low, high).astype(np.float32)


def decode_action(action: np.ndarray, config: "EnvConfig") -> dict[str, float]:
    """Map an action vector onto its named components."""
    vector = clip_action(action, config)
    return {
        name: float(vector[index]) for index, name in enumerate(ACTION_COMPONENTS)
    }


def encode_action(velocity: float, pitch: float, yaw: float) -> np.ndarray:
    """Build an action vector from named components (unclipped)."""
    return np.array([velocity, pitch, yaw], dtype=np.float32)
