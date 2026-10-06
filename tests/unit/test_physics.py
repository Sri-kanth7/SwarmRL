"""Unit tests for the physics foundation: heading math, bounded
integration, collision detection, spawning, and ray distances."""

import math

import numpy as np
import pytest

from env.config import EnvConfig
from env.physics import (
    DroneState,
    detect_collision_pairs,
    heading_vector,
    integrate_state,
    is_inside_world,
    ray_distance_to_boundary,
    spawn_states,
    wrap_angle,
)

CONFIG = EnvConfig()


def test_wrap_angle_wraps_into_pi_range():
    assert wrap_angle(0.5) == pytest.approx(0.5)
    assert wrap_angle(math.pi) == pytest.approx(math.pi)
    assert wrap_angle(3.0 * math.pi) == pytest.approx(math.pi)
    assert wrap_angle(-3.0 * math.pi) == pytest.approx(math.pi)
    assert wrap_angle(-math.pi / 2.0) == pytest.approx(-math.pi / 2.0)
    assert -math.pi < wrap_angle(7.5 * math.pi) <= math.pi


@pytest.mark.parametrize(
    ("pitch", "yaw"),
    [(0.0, 0.0), (0.3, 1.2), (-0.4, -2.0), (0.6, math.pi)],
)
def test_heading_vector_is_unit_length(pitch, yaw):
    vector = heading_vector(pitch, yaw)
    assert np.linalg.norm(vector) == pytest.approx(1.0)


def test_heading_vector_cardinal_directions():
    assert heading_vector(0.0, 0.0) == pytest.approx([1.0, 0.0, 0.0])
    assert heading_vector(0.0, math.pi / 2.0) == pytest.approx([0.0, 0.0, 1.0])
    assert heading_vector(math.pi / 2.0, 0.0) == pytest.approx([0.0, 1.0, 0.0])


def test_integrate_moves_along_heading():
    state = DroneState(position=np.array([50.0, 25.0, 50.0]))
    new_state, boundary = integrate_state(
        state, velocity=5.0, pitch=0.0, yaw=0.0, config=CONFIG
    )
    assert boundary is False
    assert new_state.position == pytest.approx([50.5, 25.0, 50.0])
    assert new_state.velocity == pytest.approx([5.0, 0.0, 0.0])


def test_integrate_clips_at_boundary_and_flags_contact():
    state = DroneState(position=np.array([99.9, 25.0, 50.0]))
    new_state, boundary = integrate_state(
        state, velocity=5.0, pitch=0.0, yaw=0.0, config=CONFIG
    )
    assert boundary is True
    assert new_state.position[0] == pytest.approx(100.0)
    assert is_inside_world(new_state.position, CONFIG.world_size)


def test_integrate_clips_pitch_and_wraps_yaw():
    state = DroneState(position=np.array([50.0, 25.0, 50.0]))
    new_state, _ = integrate_state(
        state, velocity=0.0, pitch=10.0, yaw=3.0 * math.pi, config=CONFIG
    )
    assert new_state.pitch == pytest.approx(CONFIG.max_pitch)
    assert new_state.yaw == pytest.approx(math.pi)


def test_detect_collision_pairs_reports_close_pairs_once():
    states = {
        "a": DroneState(position=np.array([0.0, 0.0, 0.0])),
        "b": DroneState(position=np.array([1.0, 0.0, 0.0])),
        "c": DroneState(position=np.array([50.0, 0.0, 0.0])),
    }
    pairs = detect_collision_pairs(states, radius=1.5)
    assert pairs == [("a", "b")]
    assert detect_collision_pairs(states, radius=0.5) == []


def test_spawn_states_is_deterministic_and_bounded():
    ids = ["drone_0", "drone_1", "drone_2"]
    first = spawn_states(ids, CONFIG, np.random.default_rng(7))
    second = spawn_states(ids, CONFIG, np.random.default_rng(7))
    for agent in ids:
        np.testing.assert_array_equal(first[agent].position, second[agent].position)
        assert is_inside_world(first[agent].position, CONFIG.world_size)
        np.testing.assert_array_equal(first[agent].velocity, np.zeros(3))
        assert first[agent].pitch == 0.0
        assert -math.pi <= first[agent].yaw <= math.pi


def test_ray_distance_to_boundary():
    centre = [50.0, 25.0, 50.0]
    assert ray_distance_to_boundary(centre, (1, 0, 0), CONFIG.world_size) == pytest.approx(
        50.0
    )
    assert ray_distance_to_boundary(centre, (-1, 0, 0), CONFIG.world_size) == pytest.approx(
        50.0
    )
    assert ray_distance_to_boundary(centre, (0, 1, 0), CONFIG.world_size) == pytest.approx(
        25.0
    )
    at_wall = [0.0, 25.0, 50.0]
    assert ray_distance_to_boundary(at_wall, (-1, 0, 0), CONFIG.world_size) == 0.0
