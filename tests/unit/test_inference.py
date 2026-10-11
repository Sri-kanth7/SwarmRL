"""Unit tests for inference engine selection and checkpoint handling."""
import sys
import types

import pytest

from server import config
from server.app import create_app
from server.config import ServerConfig
from server.inference import (
    CheckpointInferenceEngine,
    CheckpointLoadError,
    MockInferenceEngine,
)


def test_mock_engine_selected_by_default():
    app = create_app(ServerConfig())
    assert isinstance(app.state.engine, MockInferenceEngine)


def test_mock_engine_explicit():
    app = create_app(ServerConfig(inference_mode="mock"))
    assert isinstance(app.state.engine, MockInferenceEngine)


def test_checkpoint_mode_requires_existing_path():
    try:
        create_app(ServerConfig(inference_mode="checkpoint", checkpoint_path="/no/such/path"))
    except CheckpointLoadError:
        pass
    else:
        assert False


def test_config_overrides():
    c = config.load_server_config(
        environ={
            "SWARMRL_INFERENCE_MODE": "mock",
            "SWARMRL_CHECKPOINT_PATH": "training/checkpoints/latest",
        }
    )
    assert c.inference_mode == "mock"
    assert c.checkpoint_path == "training/checkpoints/latest"


def test_public_config_includes_mode_and_checkpoint():
    c = ServerConfig(inference_mode="checkpoint", checkpoint_path="training/checkpoints/latest")
    pub = c.public_dict()
    assert pub["mode"] == "checkpoint"
    assert pub["checkpoint_path"] == "training/checkpoints/latest"


# ----------------------------------------------------------------------
# CheckpointInferenceEngine lifecycle regression tests.
#
# These tests never touch a real Ray cluster, checkpoint, or network:
# the ``ray`` module and RLlib's ``Algorithm`` are replaced with fakes
# that record lifecycle calls, and the RLlib environment class is
# replaced with a lightweight fake.
# ----------------------------------------------------------------------


class _FakeAlgorithm:
    """Minimal stand-in for an RLlib Algorithm."""

    def __init__(self, *, fail_stop: bool = False) -> None:
        self.stop_calls = 0
        self.fail_stop = fail_stop

    def stop(self) -> None:
        self.stop_calls += 1
        if self.fail_stop:
            raise RuntimeError("algorithm stop failed")


def _install_fake_ray(
    monkeypatch,
    *,
    initialized: bool = False,
    algorithm=None,
    load_error=None,
    fail_shutdown: bool = False,
):
    """Install a fake ``ray`` module (and RLlib import path) into sys.modules.

    Returns a mutable ``state`` dict recording ``init``/``shutdown`` calls
    and the current ``initialized`` flag.
    """
    fake_ray = types.ModuleType("ray")
    state = {"initialized": initialized, "init_calls": 0, "shutdown_calls": 0}

    def is_initialized() -> bool:
        return state["initialized"]

    def init(*args, **kwargs) -> None:
        state["init_calls"] += 1
        state["initialized"] = True

    def shutdown() -> None:
        state["shutdown_calls"] += 1
        if fail_shutdown:
            raise RuntimeError("ray shutdown failed")
        state["initialized"] = False

    fake_ray.is_initialized = is_initialized
    fake_ray.init = init
    fake_ray.shutdown = shutdown

    rllib_pkg = types.ModuleType("ray.rllib")
    algorithms_pkg = types.ModuleType("ray.rllib.algorithms")
    algorithm_mod = types.ModuleType("ray.rllib.algorithms.algorithm")

    class _Algorithm:
        @staticmethod
        def from_checkpoint(path):
            if load_error is not None:
                raise load_error
            return algorithm

    algorithm_mod.Algorithm = _Algorithm
    fake_ray.rllib = rllib_pkg
    rllib_pkg.algorithms = algorithms_pkg
    algorithms_pkg.algorithm = algorithm_mod

    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.setitem(sys.modules, "ray.rllib", rllib_pkg)
    monkeypatch.setitem(sys.modules, "ray.rllib.algorithms", algorithms_pkg)
    monkeypatch.setitem(sys.modules, "ray.rllib.algorithms.algorithm", algorithm_mod)
    return state


def _bare_engine() -> CheckpointInferenceEngine:
    """Build an engine without running ``__init__`` (no Ray/checkpoint needed)."""
    engine = CheckpointInferenceEngine.__new__(CheckpointInferenceEngine)
    engine.config = ServerConfig(agent_count=2)
    engine._algorithm = None
    engine._env = None
    engine._started_ray = False
    return engine


def _install_fake_env_module(monkeypatch, env_cls) -> None:
    fake_env_mod = types.ModuleType("training.rllib_env")
    fake_env_mod.SwarmMultiAgentEnv = env_cls
    monkeypatch.setitem(sys.modules, "training.rllib_env", fake_env_mod)


def _patch_checkpoint_selection(monkeypatch, tmp_path) -> None:
    (tmp_path / "rllib").mkdir(exist_ok=True)
    monkeypatch.setattr(
        CheckpointInferenceEngine,
        "_resolve_checkpoint_path",
        lambda self, path: tmp_path,
    )
    monkeypatch.setattr(
        CheckpointInferenceEngine,
        "_load_and_validate_metadata",
        lambda self, path: object(),
    )


def test_load_algorithm_failure_shuts_down_owned_ray(monkeypatch, tmp_path):
    (tmp_path / "rllib").mkdir()
    state = _install_fake_ray(
        monkeypatch, initialized=False, load_error=RuntimeError("checkpoint boom")
    )
    engine = _bare_engine()

    with pytest.raises(CheckpointLoadError) as excinfo:
        engine._load_algorithm(tmp_path)

    assert "checkpoint boom" in str(excinfo.value)
    assert state["init_calls"] == 1
    assert state["shutdown_calls"] == 1
    assert state["initialized"] is False
    assert engine._started_ray is False


def test_load_algorithm_failure_keeps_external_ray(monkeypatch, tmp_path):
    (tmp_path / "rllib").mkdir()
    state = _install_fake_ray(
        monkeypatch, initialized=True, load_error=RuntimeError("checkpoint boom")
    )
    engine = _bare_engine()

    with pytest.raises(CheckpointLoadError):
        engine._load_algorithm(tmp_path)

    assert state["init_calls"] == 0
    assert state["shutdown_calls"] == 0
    assert state["initialized"] is True
    assert engine._started_ray is False


def test_init_env_construction_failure_cleans_up(monkeypatch, tmp_path):
    algorithm = _FakeAlgorithm()
    state = _install_fake_ray(monkeypatch, initialized=False, algorithm=algorithm)
    _patch_checkpoint_selection(monkeypatch, tmp_path)

    class _ExplodingEnv:
        def __init__(self, env_config):
            raise RuntimeError("env construction boom")

    _install_fake_env_module(monkeypatch, _ExplodingEnv)

    with pytest.raises(RuntimeError, match="env construction boom"):
        CheckpointInferenceEngine(ServerConfig(agent_count=2))

    assert algorithm.stop_calls == 1
    assert state["shutdown_calls"] == 1
    assert state["initialized"] is False


def test_init_state_failure_cleans_up_algorithm_env_and_ray(monkeypatch, tmp_path):
    algorithm = _FakeAlgorithm()
    state = _install_fake_ray(monkeypatch, initialized=False, algorithm=algorithm)
    _patch_checkpoint_selection(monkeypatch, tmp_path)

    events: list[str] = []

    class _FakeEnv:
        def __init__(self, env_config):
            self.possible_agents = ["drone_0"]
            events.append("init")

        def reset(self, seed=None):
            events.append("reset")
            raise RuntimeError("state init boom")

        def close(self):
            events.append("close")

    _install_fake_env_module(monkeypatch, _FakeEnv)

    with pytest.raises(RuntimeError, match="state init boom"):
        CheckpointInferenceEngine(ServerConfig(agent_count=2))

    assert algorithm.stop_calls == 1
    assert "close" in events
    assert state["shutdown_calls"] == 1
    assert state["initialized"] is False


def test_init_env_import_failure_cleans_up(monkeypatch, tmp_path):
    algorithm = _FakeAlgorithm()
    state = _install_fake_ray(monkeypatch, initialized=False, algorithm=algorithm)
    _patch_checkpoint_selection(monkeypatch, tmp_path)

    # A ``None`` entry makes ``from training.rllib_env import ...`` fail.
    monkeypatch.setitem(sys.modules, "training.rllib_env", None)

    with pytest.raises(CheckpointLoadError, match="RLlib is required"):
        CheckpointInferenceEngine(ServerConfig(agent_count=2))

    assert algorithm.stop_calls == 1
    assert state["shutdown_calls"] == 1
    assert state["initialized"] is False


def test_load_failure_not_masked_by_shutdown_error(monkeypatch, tmp_path):
    (tmp_path / "rllib").mkdir()
    state = _install_fake_ray(
        monkeypatch,
        initialized=False,
        load_error=RuntimeError("checkpoint boom"),
        fail_shutdown=True,
    )
    engine = _bare_engine()

    with pytest.raises(CheckpointLoadError) as excinfo:
        engine._load_algorithm(tmp_path)

    assert "checkpoint boom" in str(excinfo.value)
    assert "ray shutdown failed" not in str(excinfo.value)
    assert engine._started_ray is False


def test_init_error_not_masked_by_algorithm_stop_error(monkeypatch, tmp_path):
    algorithm = _FakeAlgorithm(fail_stop=True)
    state = _install_fake_ray(monkeypatch, initialized=False, algorithm=algorithm)
    _patch_checkpoint_selection(monkeypatch, tmp_path)

    class _ExplodingEnv:
        def __init__(self, env_config):
            raise RuntimeError("env construction boom")

    _install_fake_env_module(monkeypatch, _ExplodingEnv)

    with pytest.raises(RuntimeError, match="env construction boom"):
        CheckpointInferenceEngine(ServerConfig(agent_count=2))

    assert algorithm.stop_calls == 1
    assert state["shutdown_calls"] == 1


def test_close_is_idempotent(monkeypatch):
    algorithm = _FakeAlgorithm()
    state = _install_fake_ray(monkeypatch, initialized=True)
    engine = _bare_engine()
    events: list[str] = []

    class _FakeEnv:
        def close(self):
            events.append("close")

    engine._algorithm = algorithm
    engine._env = _FakeEnv()
    engine._started_ray = True

    engine.close()
    engine.close()
    engine.close()

    assert algorithm.stop_calls == 1
    assert events == ["close"]
    assert state["shutdown_calls"] == 1
    assert state["initialized"] is False
    assert engine._algorithm is None
    assert engine._env is None
    assert engine._started_ray is False


def test_close_keeps_external_ray(monkeypatch):
    algorithm = _FakeAlgorithm()
    state = _install_fake_ray(monkeypatch, initialized=True)
    engine = _bare_engine()

    class _FakeEnv:
        def __init__(self):
            self.close_calls = 0

        def close(self):
            self.close_calls += 1

    env = _FakeEnv()
    engine._algorithm = algorithm
    engine._env = env
    engine._started_ray = False

    engine.close()

    assert algorithm.stop_calls == 1
    assert env.close_calls == 1
    assert state["shutdown_calls"] == 0
    assert state["initialized"] is True


def test_mock_engine_close_is_safe_and_semantics_unchanged():
    engine = MockInferenceEngine(ServerConfig(agent_count=3, seed=7))
    first = engine.reset(seed=7).to_dict()

    engine.close()
    engine.close()

    engine.reset(seed=7)
    assert engine.current().to_dict() == first
