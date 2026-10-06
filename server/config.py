"""Backend configuration loading.

Precedence (highest first):

1. Environment variables (``SWARMRL_*``, see ``.env.example``),
2. ``configs/project.yaml``,
3. built-in defaults.

Only one configuration file is used (``configs/project.yaml``); this
module adds no competing configuration system.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_YAML = REPO_ROOT / "configs" / "project.yaml"


@dataclass(frozen=True)
class ServerConfig:
    """Backend server configuration.

    Attributes:
        host: Bind address for the HTTP/WebSocket server.
        port: Bind port.
        websocket_path: WebSocket route path (default ``/ws``).
        env_name: Runtime environment name (``development``/``production``).
        agent_count: Number of drones streamed to clients.
        world_size: ``(x, y, z)`` world extents (y-up).
        seed: Seed for the mock state generator.
        mock_tick_interval_ms: Milliseconds between streamed steps.
        coverage_cell_size: Coverage-grid cell size of the mock world.
        collision_radius: Collision radius used by the mock generator.
        checkpoint_dir: Directory reserved for future checkpoints.
        cors_origins: Allowed browser origins for the dev frontend.
    """

    host: str = "0.0.0.0"
    port: int = 8000
    websocket_path: str = "/ws"
    env_name: str = "development"
    agent_count: int = 50
    world_size: tuple[float, float, float] = (100.0, 50.0, 100.0)
    seed: int = 42
    mock_tick_interval_ms: int = 100
    coverage_cell_size: float = 5.0
    collision_radius: float = 1.5
    checkpoint_dir: str = "training/checkpoints"
    cors_origins: tuple[str, ...] = ("http://localhost:5173",)

    @property
    def tick_interval_seconds(self) -> float:
        """Mock stream tick interval in seconds."""
        return self.mock_tick_interval_ms / 1000.0

    def public_dict(self) -> dict[str, Any]:
        """Configuration subset exposed through the HTTP API."""
        return {
            "env": self.env_name,
            "agent_count": self.agent_count,
            "world_size": list(self.world_size),
            "seed": self.seed,
            "websocket_path": self.websocket_path,
            "tick_interval_ms": self.mock_tick_interval_ms,
            "mode": "mock",
        }


def _read_project_yaml(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data if isinstance(data, Mapping) else {}


def _section(raw: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = raw.get(key)
    return value if isinstance(value, Mapping) else {}


def _coerce_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def load_server_config(
    project_yaml: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> ServerConfig:
    """Resolve the backend configuration.

    Args:
        project_yaml: Explicit path to ``project.yaml``. When omitted,
            the repository copy is used if present.
        environ: Environment-variable mapping (defaults to ``os.environ``).

    Returns:
        A fully resolved :class:`ServerConfig`.
    """
    env = os.environ if environ is None else environ

    if project_yaml is not None:
        yaml_path: Path | None = Path(project_yaml)
    elif PROJECT_YAML.is_file():
        yaml_path = PROJECT_YAML
    else:
        fallback = Path.cwd() / "configs" / "project.yaml"
        yaml_path = fallback if fallback.is_file() else None

    raw = _read_project_yaml(yaml_path) if yaml_path else {}
    project = _section(raw, "project")
    environment = _section(raw, "environment")
    server = _section(raw, "server")
    coverage = _section(raw, "coverage")
    training = _section(raw, "training")
    frontend = _section(raw, "frontend")

    world_size = tuple(float(v) for v in (environment.get("world_size") or [100.0, 50.0, 100.0]))
    if len(world_size) != 3:
        world_size = (100.0, 50.0, 100.0)

    values: dict[str, Any] = {
        "host": str(server.get("host", "0.0.0.0")),
        "port": _coerce_int(server.get("port", 8000)),
        "websocket_path": str(server.get("websocket_path", "/ws")),
        "env_name": str(project.get("env", "development")),
        "agent_count": _coerce_int(environment.get("agent_count", 50)),
        "world_size": world_size,
        "seed": _coerce_int(environment.get("seed", 42)),
        "mock_tick_interval_ms": _coerce_int(server.get("mock_tick_interval_ms", 100)),
        "coverage_cell_size": float(coverage.get("cell_size", 5.0)),
        "collision_radius": float(environment.get("collision_radius", 1.5)),
        "checkpoint_dir": str(training.get("checkpoint_dir", "training/checkpoints")),
        "cors_origins": tuple(
            str(o)
            for o in (
                server.get("cors_origins")
                or [
                    "http://localhost:{}".format(
                        _coerce_int(frontend.get("development_port", 5173))
                    )
                ]
            )
        ),
    }

    env_overrides = {
        "SWARMRL_ENV": "env_name",
        "SWARMRL_HOST": "host",
        "SWARMRL_PORT": "port",
        "SWARMRL_AGENT_COUNT": "agent_count",
        "SWARMRL_SEED": "seed",
        "SWARMRL_MOCK_TICK_MS": "mock_tick_interval_ms",
        "SWARMRL_CHECKPOINT_DIR": "checkpoint_dir",
        "SWARMRL_WEBSOCKET_PATH": "websocket_path",
    }
    for env_key, field_name in env_overrides.items():
        if env_key in env and env[env_key] != "":
            raw_value = env[env_key]
            current = values[field_name]
            if isinstance(current, bool):
                values[field_name] = str(raw_value).lower() in {"1", "true", "yes"}
            elif isinstance(current, int):
                values[field_name] = _coerce_int(raw_value)
            else:
                values[field_name] = str(raw_value)

    return ServerConfig(**values)


def server_config_fields() -> tuple[str, ...]:
    """Names of the configuration fields (useful for tooling)."""
    return tuple(f.name for f in fields(ServerConfig))
