"""Unit tests for the PettingZoo environment lifecycle, spaces,
shapes, and bounds."""

import numpy as np
import pytest

from env import EnvConfig, SwarmEnv
from env.observations import observation_dim


def test_default_agent_count_is_fifty():
    env = SwarmEnv()
    assert env.config.agent_count == 50
    assert len(env.possible_agents) == 50
    assert env.possible_agents[0] == "drone_0"
    assert env.possible_agents[-1] == "drone_49"


def test_reset_returns_observations_and_infos_for_all_agents():
    env = SwarmEnv(EnvConfig(agent_count=4, max_steps=3))
    observations, infos = env.reset(seed=0)
    assert set(observations) == set(env.possible_agents)
    assert set(infos) == set(env.possible_agents)
    assert env.agents == env.possible_agents


def test_spaces_match_configuration():
    config = EnvConfig(agent_count=3, lidar_rays=6)
    env = SwarmEnv(config)
    agent = env.possible_agents[0]
    observation_space = env.observation_space(agent)
    action_space = env.action_space(agent)
    assert observation_space.shape == (observation_dim(config),)
    assert observation_space.shape == (19,)
    assert observation_space.dtype == np.float32
    assert action_space.shape == (3,)
    assert action_space.dtype == np.float32


def test_observations_are_in_bounds_after_reset_and_step():
    env = SwarmEnv(EnvConfig(agent_count=3, max_steps=5))
    observations, _ = env.reset(seed=1)
    space = env.observation_space(env.possible_agents[0])
    for value in observations.values():
        assert value.dtype == np.float32
        assert space.contains(value)

    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    observations, _, _, _, _ = env.step(actions)
    for value in observations.values():
        assert space.contains(value)


def test_step_accepts_zero_actions_and_returns_full_key_sets():
    env = SwarmEnv(EnvConfig(agent_count=3, max_steps=5))
    env.reset(seed=0)
    actions = {agent: np.zeros(3, dtype=np.float32) for agent in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)
    assert set(rewards) == set(env.agents)
    assert set(terminations) == set(env.agents)
    assert set(truncations) == set(env.agents)
    assert set(infos) == set(env.agents)
    assert set(observations) == set(env.agents)
    assert not any(terminations.values())
    assert not any(truncations.values())


def test_step_rejects_incomplete_action_sets():
    env = SwarmEnv(EnvConfig(agent_count=2, max_steps=3))
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step({env.agents[0]: np.zeros(3)})


def test_episode_truncates_at_max_steps():
    config = EnvConfig(agent_count=2, max_steps=3)
    env = SwarmEnv(config)
    env.reset(seed=0)
    observations = None
    truncated = False
    for _ in range(config.max_steps):
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        observations, rewards, terminations, truncations, infos = env.step(actions)
        assert len(rewards) == len(infos)
        truncated = any(truncations.values())
    assert truncated
    assert env.agents == []
    assert observations == {}


def test_reset_with_seed_is_deterministic():
    env = SwarmEnv(EnvConfig(agent_count=5))
    first, _ = env.reset(seed=99)
    second, _ = env.reset(seed=99)
    for agent in env.possible_agents:
        np.testing.assert_array_equal(first[agent], second[agent])


def test_infos_expose_contract_keys():
    env = SwarmEnv(EnvConfig(agent_count=2, max_steps=3))
    _, infos = env.reset(seed=0)
    info = infos[env.agents[0]]
    for key in (
        "step",
        "explored_pct",
        "collisions",
        "step_collisions",
        "collision",
        "boundary_contact",
        "newly_visited_cells",
        "episode_reward",
        "agent_index",
    ):
        assert key in info
    assert info["agent_index"] == 0
    assert 0.0 <= info["explored_pct"] <= 100.0
