"""Environment-to-training integration tests (Week 2).

Coverage (see ``docs/testing/test_strategy.md``):

* the RLlib ``MultiAgentEnv`` adapter passes RLlib's own environment
  check and records one :class:`~training.metrics.EpisodeMetrics` per
  finished episode,
* the environment registry creator builds the adapter,
* the YAML-to-RLlib configuration translation validates its settings,
* ``--print-config`` never starts Ray or training,
* a short IPPO run writes a complete checkpoint directory.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import ray
import torch
import yaml
from gymnasium import spaces
from ray import is_initialized
from ray.rllib.utils.pre_checks.env import check_multiagent_environments
from ray.tune.registry import ENV_CREATOR, _global_registry

from env import EnvConfig, SwarmEnv
from training import (
    ENV_REGISTRY_NAME,
    MetricsAggregator,
    TrainingConfig,
    load_training_config,
    register_swarm_env,
)
from training.checkpoint import load_metadata
from training.config import default_config_path
from training.rllib_env import ALL_AGENTS_KEY, SwarmMultiAgentEnv
from training.rl_config import POLICY_ID, build_ppo_config, translate_train_settings
from training.train_ippo import TrainingRun, main as train_ippo_main, run_training

TINY_ENV_CONFIG = {
    "agent_count": 4,
    "world_size": [40.0, 20.0, 40.0],
    "seed": 7,
    "max_steps": 8,
    "lidar_rays": 4,
    "coverage_cell_size": 5.0,
}

TINY_TRAIN = {
    "num_env_runners": 0,
    "train_batch_size": 64,
    "minibatch_size": 32,
    "num_epochs": 1,
    "num_iterations": 1,
    "lr": 0.001,
    "gamma": 0.99,
    "lambda_": 0.95,
}

CHECKPOINT_FILES = (
    "metadata.json",
    "policy.pt",
    "metrics.json",
    "config_snapshot.yaml",
)


def _tiny_config(checkpoint_dir: Path) -> TrainingConfig:
    """Small configuration that keeps a training run fast."""
    return TrainingConfig(
        algorithm="IPPO",
        env_name=ENV_REGISTRY_NAME,
        framework="torch",
        seed=7,
        env_config=dict(TINY_ENV_CONFIG),
        model={"fcnet_hiddens": [32, 32], "fcnet_activation": "relu"},
        train=dict(TINY_TRAIN),
        checkpoint_dir=str(checkpoint_dir),
    )


def _env_dims(env_config: dict) -> tuple[int, int]:
    """Observation and action dimensions of a configured environment."""
    env = SwarmEnv(EnvConfig.from_dict(env_config))
    agent = env.possible_agents[0]
    observation_dim = int(env.observation_space(agent).shape[0])
    action_dim = int(env.action_space(agent).shape[0])
    env.close()
    return observation_dim, action_dim


def test_adapter_passes_rllib_env_check():
    env = SwarmMultiAgentEnv(dict(TINY_ENV_CONFIG))
    try:
        check_multiagent_environments(env)
        assert isinstance(env.observation_space, spaces.Box)
        assert isinstance(env.action_space, spaces.Box)
        assert env.get_observation_space("drone_0") == env.observation_space

        observations, infos = env.reset(seed=7)
        assert set(observations) == set(env.agents) == set(env.possible_agents)
        assert set(infos) == set(observations)
        assert observations["drone_0"].shape == env.observation_space.shape
    finally:
        env.close()


def test_adapter_records_one_episode_metric_per_episode():
    collector = MetricsAggregator()
    env = SwarmMultiAgentEnv(dict(TINY_ENV_CONFIG), collector=collector)
    try:
        for seed in (7, 8):
            env.reset(seed=seed)
            for _ in range(TINY_ENV_CONFIG["max_steps"]):
                actions = {a: env.get_action_space(a).sample() for a in env.agents}
                observations, rewards, terminations, truncations, infos = env.step(
                    actions
                )
            assert truncations[ALL_AGENTS_KEY] is True
            assert terminations[ALL_AGENTS_KEY] is False
            assert set(observations) == set(rewards) == set(infos)
            assert env.agents == []
    finally:
        env.close()

    assert len(collector) == 2
    assert [episode.episode for episode in collector.episodes] == [0, 1]
    for episode in collector.episodes:
        assert episode.steps == TINY_ENV_CONFIG["max_steps"]
        assert episode.explored_pct >= 0.0
        assert episode.collisions >= 0


def test_registry_creator_builds_adapter():
    name = register_swarm_env()
    assert name == ENV_REGISTRY_NAME
    creator = _global_registry.get(ENV_CREATOR, name)
    env = creator({"agent_count": 2, "max_steps": 4})
    try:
        assert isinstance(env, SwarmMultiAgentEnv)
        assert len(env.possible_agents) == 2
    finally:
        env.close()


def test_translate_train_settings_renames_legacy_keys():
    settings = translate_train_settings(
        {
            "num_workers": 0,
            "sgd_minibatch_size": 64,
            "gae_lambda": 0.95,
            "num_iterations": 3,
            "lr": 0.001,
        }
    )
    assert settings == {
        "num_env_runners": 0,
        "minibatch_size": 64,
        "lambda_": 0.95,
        "lr": 0.001,
    }


def test_translate_train_settings_rejects_unknown_settings():
    with pytest.raises(ValueError, match="unknown training setting"):
        translate_train_settings({"mystery_setting": 1})


def test_build_ppo_config_translates_training_config():
    ppo = build_ppo_config(_tiny_config(Path("training/checkpoints")))

    assert ppo.env == ENV_REGISTRY_NAME
    assert ppo.env_config["agent_count"] == TINY_ENV_CONFIG["agent_count"]
    assert ppo.framework_str == "torch"
    assert ppo.seed == 7
    assert ppo.num_env_runners == 0
    assert ppo.rollout_fragment_length == "auto"
    assert ppo.train_batch_size == TINY_TRAIN["train_batch_size"]
    assert ppo.minibatch_size == TINY_TRAIN["minibatch_size"]
    assert ppo.lambda_ == TINY_TRAIN["lambda_"]
    assert ppo.model_config["fcnet_hiddens"] == [32, 32]
    assert ppo.model_config["fcnet_activation"] == "relu"

    assert POLICY_ID in ppo.policies
    spec = ppo.policies[POLICY_ID]
    observation_dim, action_dim = _env_dims(TINY_ENV_CONFIG)
    assert spec.observation_space.shape == (observation_dim,)
    assert spec.action_space.shape == (action_dim,)
    assert ppo.policy_mapping_fn("drone_0") == POLICY_ID


def test_print_config_never_starts_training(capsys):
    config_path = default_config_path("ippo")
    summary = train_ippo_main(["--config", str(config_path), "--print-config"])
    printed = json.loads(capsys.readouterr().out)

    assert printed["env"] == ENV_REGISTRY_NAME
    assert printed["algo_class_name"] == "IPPO"
    assert printed["model"]["fcnet_activation"] == "relu"
    assert summary["algo_class_name"] == "IPPO"
    assert not is_initialized()


def test_run_training_rejects_non_positive_iterations():
    config = _tiny_config(Path("training/checkpoints"))
    with pytest.raises(ValueError, match="iterations must be >= 1"):
        run_training(config, iterations=0)


@pytest.fixture(scope="module")
def ippo_run(tmp_path_factory) -> TrainingRun:
    """One short IPPO run shared by the checkpoint-contract tests."""
    config_path = default_config_path("ippo")
    base = load_training_config(config_path)
    config = TrainingConfig(
        algorithm=base.algorithm,
        env_name=base.env_name,
        framework=base.framework,
        seed=base.seed,
        env_config={**base.env_config, **TINY_ENV_CONFIG},
        model={"fcnet_hiddens": [32, 32], "fcnet_activation": "relu"},
        multi_agent=base.multi_agent,
        train=dict(TINY_TRAIN),
        checkpoint_dir=str(tmp_path_factory.mktemp("checkpoints")),
    )
    return run_training(config, run_id="ippotest", config_path=config_path)


def test_run_training_writes_checkpoint_contract(ippo_run: TrainingRun):
    run_dir = ippo_run.run_dir
    missing = [name for name in CHECKPOINT_FILES if not (run_dir / name).exists()]
    assert missing == []

    # RLlib's own algorithm checkpoint: on the new API stack, `Algorithm.save()`
    # goes through the Checkpointable API and overrides METADATA_FILE_NAME to
    # "rllib_checkpoint.json", so the RLlib metadata lives under that name
    # (the "metadata.json" name belongs to the SwarmRL contract above).
    rllib_dir = run_dir / "rllib"
    rllib_metadata_file = rllib_dir / "rllib_checkpoint.json"
    assert rllib_metadata_file.exists()
    rllib_metadata = json.loads(rllib_metadata_file.read_text("utf-8"))
    assert rllib_metadata["ray_version"] == ray.__version__
    assert (rllib_dir / rllib_metadata["class_and_ctor_args_file"]).exists()
    rllib_state_file = rllib_dir / rllib_metadata["state_file"]
    assert (
        rllib_state_file.with_suffix(".pkl").exists()
        or rllib_state_file.with_suffix(".msgpack").exists()
    )

    metadata = load_metadata(run_dir)
    observation_dim, action_dim = _env_dims(TINY_ENV_CONFIG)
    assert metadata.format_version == 1
    assert metadata.algorithm == "IPPO"
    assert metadata.step == 1
    assert metadata.seed == 42
    assert metadata.agent_count == TINY_ENV_CONFIG["agent_count"]
    assert metadata.observation_dim == observation_dim
    assert metadata.action_dim == action_dim
    assert metadata.config_path is not None
    assert metadata.weights_file == "policy.pt"
    assert (run_dir / metadata.weights_file).exists()

    state = torch.load(run_dir / "policy.pt", weights_only=True)
    assert state
    assert all(isinstance(value, torch.Tensor) for value in state.values())

    metrics = json.loads((run_dir / "metrics.json").read_text("utf-8"))
    assert metrics["summary"]["episodes"] >= 1
    assert metrics["episodes"][0]["steps"] == TINY_ENV_CONFIG["max_steps"]
    assert metrics["iterations"][0]["training_iteration"] == 1
    assert metrics["iterations"][0]["episode_return_mean"] is not None

    snapshot = yaml.safe_load((run_dir / "config_snapshot.yaml").read_text("utf-8"))
    assert snapshot["algorithm"] == "IPPO"
    assert snapshot["train"]["train_batch_size"] == TINY_TRAIN["train_batch_size"]

    assert ippo_run.iterations == 1
    assert ippo_run.metadata == metadata
    assert len(ippo_run.env_runner_results) == 1
    assert ippo_run.summary()["episodes"] >= 1
    assert not is_initialized()
