"""Unit tests for inference engine selection and checkpoint handling."""
from server import config
from server.app import create_app
from server.config import ServerConfig
from server.inference import CheckpointLoadError, MockInferenceEngine


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
