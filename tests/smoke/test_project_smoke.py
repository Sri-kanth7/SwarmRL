"""Smoke tests: repository layout, configuration parsing, and
imports of every layer."""

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]

REQUIRED_PATHS = [
    "configs/project.yaml",
    "docs/architecture/system_architecture.md",
    "docs/contracts/environment_api.md",
    "docs/contracts/checkpoint_contract.md",
    "docs/contracts/metrics_schema.md",
    "docs/contracts/websocket_schema.md",
    "docs/testing/test_strategy.md",
    "env/swarm_env.py",
    "env/physics.py",
    "env/observations.py",
    "env/actions.py",
    "env/coverage.py",
    "env/rewards.py",
    "training/train_mappo.py",
    "training/train_ippo.py",
    "training/configs/mappo.yaml",
    "training/configs/ippo.yaml",
    "training/checkpoint.py",
    "server/app.py",
    "server/schemas.py",
    "server/inference.py",
    "frontend/package.json",
    "frontend/src/App.tsx",
    "frontend/src/scenes/SwarmScene.tsx",
    "scripts/run_tests.sh",
    "scripts/start_server.sh",
]


def test_required_layout_exists():
    missing = [path for path in REQUIRED_PATHS if not (ROOT / path).exists()]
    assert missing == []


def test_project_yaml_parses_and_carries_key_values():
    data = yaml.safe_load((ROOT / "configs" / "project.yaml").read_text("utf-8"))
    assert data["project"]["name"] == "SwarmRL"
    assert data["environment"]["agent_count"] == 50
    assert data["environment"]["world_size"] == [100.0, 50.0, 100.0]
    assert data["server"]["port"] == 8000
    assert data["server"]["websocket_path"] == "/ws"
    assert data["frontend"]["development_port"] == 5173


def test_training_configs_parse():
    for name, algorithm in (("mappo.yaml", "MAPPO"), ("ippo.yaml", "IPPO")):
        data = yaml.safe_load((ROOT / "training" / "configs" / name).read_text("utf-8"))
        assert data["algorithm"] == algorithm
        assert data["env_config"]["agent_count"] == 50
        assert data["seed"] == 42


def test_pyproject_parses_with_declared_tooling():
    import tomllib

    data = tomllib.loads((ROOT / "pyproject.toml").read_text("utf-8"))
    assert data["project"]["name"] == "swarmrl"
    assert data["project"]["requires-python"] == ">=3.11"
    dependencies = data["project"]["dependencies"]
    for package in ("numpy", "pettingzoo", "gymnasium", "fastapi", "pydantic", "pyyaml"):
        assert package in dependencies
    assert "pytest" in data["project"]["optional-dependencies"]["dev"]
    assert data["tool"]["pytest"]["ini_options"]["testpaths"] == ["tests"]


def test_core_modules_import():
    from env import EnvConfig, SwarmEnv
    import env.actions
    import env.coverage
    import env.observations
    import env.physics
    import env.rewards
    import training.checkpoint
    import training.config
    import training.mappo_model
    import training.metrics
    import training.rl_config
    import server.config
    import server.inference
    import server.metrics
    import server.schemas

    assert EnvConfig().agent_count == 50
    env = SwarmEnv(EnvConfig(agent_count=2))
    assert len(env.possible_agents) == 2


def test_backend_application_builds():
    from server.app import create_app

    app = create_app()
    routes = {route.path for route in app.routes}
    assert "/api/health" in routes
    assert "/api/state" in routes
    assert "/ws" in routes


def test_frontend_package_declares_expected_stack():
    data = json.loads((ROOT / "frontend" / "package.json").read_text("utf-8"))
    dependencies = data["dependencies"]
    for package in ("react", "react-dom", "three", "@react-three/fiber"):
        assert package in dependencies
    assert "vite" in data["devDependencies"]
    assert "typescript" in data["devDependencies"]


def test_contract_documents_are_non_empty():
    documents = [
        "docs/architecture/system_architecture.md",
        "docs/contracts/environment_api.md",
        "docs/contracts/checkpoint_contract.md",
        "docs/contracts/metrics_schema.md",
        "docs/contracts/websocket_schema.md",
        "docs/testing/test_strategy.md",
        "README.md",
        "CONTRIBUTING.md",
    ]
    for relative in documents:
        text = (ROOT / relative).read_text("utf-8")
        assert text.strip(), f"{relative} is empty"


def test_gitignore_has_no_shell_artifact():
    text = (ROOT / ".gitignore").read_text("utf-8")
    assert "<<" not in text
    assert "__pycache__/" in text
