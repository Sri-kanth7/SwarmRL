"""Backend-to-frontend integration tests.

These tests exercise the WebSocket stream without a real network,
Ray runtime, or trained checkpoint:

* ``ws_server.handle_connection`` is driven with lightweight fake
  sockets so the producer -> serialization -> handler path is checked
  directly (the FastAPI ``TestClient`` is avoided because it would add
  an ``httpx`` test dependency the project does not declare),
* the mock engine is checked for stable ids, finite world coordinates,
  and changing positions across steps,
* the application shutdown lifecycle, engine selection, and HTTP
  endpoints are checked through the application object, and
* the frontend TypeScript contract files are compared field-for-field
  with the backend Pydantic schemas.

See ``docs/contracts/websocket_schema.md`` and
``docs/testing/test_strategy.md``.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
from pathlib import Path

import pytest
from fastapi import WebSocketDisconnect

from server import ws_server
from server.app import _lifespan, create_app
from server.config import ServerConfig
from server.inference import CheckpointLoadError, MockInferenceEngine
from server.schemas import AgentState, MetricsSnapshot, StepMessage

ROOT = Path(__file__).resolve().parents[2]
FRONTEND_TYPES = ROOT / "frontend" / "src" / "types"
FIXTURE = ROOT / "tests" / "fixtures" / "mock_agent_data.json"

REQUIRED_STEP_FIELDS = {"step", "agents", "newly_visited_cells", "collisions", "explored_pct"}
REQUIRED_AGENT_FIELDS = {"id", "x", "y", "z", "pitch", "yaw"}
REQUIRED_METRICS_FIELDS = {"step", "explored_pct", "collisions", "mean_reward"}


# ----------------------------------------------------------------------
# Fake sockets and engines (no network, no external services)
# ----------------------------------------------------------------------
class _RecordingWebSocket:
    """Captures accepted/sent/closed calls from the handler."""

    def __init__(self) -> None:
        self.accepted = False
        self.sent: list[dict[str, object]] = []
        self.closed = False

    async def accept(self) -> None:
        self.accepted = True

    async def send_json(self, payload: dict[str, object]) -> None:
        self.sent.append(payload)

    async def close(self) -> None:
        self.closed = True


class _DisconnectingWebSocket:
    """Raises the disconnect error on the first send (client gone)."""

    def __init__(self) -> None:
        self.closed = False

    async def accept(self) -> None:
        return None

    async def send_json(self, _payload: dict[str, object]) -> None:
        raise WebSocketDisconnect()

    async def close(self) -> None:
        self.closed = True


class _ErroringWebSocket:
    """Raises an unexpected error on the first send."""

    def __init__(self) -> None:
        self.closed = False

    async def accept(self) -> None:
        return None

    async def send_json(self, _payload: dict[str, object]) -> None:
        raise RuntimeError("stream boom")

    async def close(self) -> None:
        self.closed = True


class _StubEngine:
    """Minimal non-mock engine proving the handler is engine-agnostic."""

    def __init__(self) -> None:
        self.calls = 0

    def reset(self, seed: int | None = None) -> StepMessage:
        return self.step()

    def current(self) -> StepMessage:
        return StepMessage()

    def step(self) -> StepMessage:
        self.calls += 1
        return StepMessage(
            step=self.calls,
            agents=[AgentState(id=7, x=1.0, y=2.0, z=3.0, pitch=0.1, yaw=0.2)],
        )

    def metrics(self) -> MetricsSnapshot:
        return MetricsSnapshot()


def _endpoint(app, path: str):
    for route in app.routes:
        if getattr(route, "path", None) == path:
            return route.endpoint
    raise AssertionError(f"route {path!r} is not registered")


# ----------------------------------------------------------------------
# Handler: producer -> serialization -> stream
# ----------------------------------------------------------------------
def test_handler_streams_contract_messages_from_selected_engine():
    engine = MockInferenceEngine(ServerConfig(agent_count=4))
    websocket = _RecordingWebSocket()

    asyncio.run(ws_server.handle_connection(websocket, engine, tick_seconds=0.0, max_messages=3))

    assert websocket.accepted is True
    assert len(websocket.sent) == 3
    assert [message["step"] for message in websocket.sent] == [1, 2, 3]
    for message in websocket.sent:
        assert set(message) == REQUIRED_STEP_FIELDS
        assert len(message["agents"]) == 4


def test_handler_streams_changing_positions_with_stable_ids():
    engine = MockInferenceEngine(ServerConfig(agent_count=5, seed=3))
    websocket = _RecordingWebSocket()

    asyncio.run(ws_server.handle_connection(websocket, engine, tick_seconds=0.0, max_messages=2))

    first, second = websocket.sent
    assert second["step"] == first["step"] + 1
    assert [agent["id"] for agent in second["agents"]] == [agent["id"] for agent in first["agents"]]
    moved = any(
        (a["x"], a["y"], a["z"]) != (b["x"], b["y"], b["z"])
        for a, b in zip(first["agents"], second["agents"])
    )
    assert moved


def test_handler_uses_the_configured_engine_not_a_hard_coded_generator():
    engine = _StubEngine()
    websocket = _RecordingWebSocket()

    asyncio.run(ws_server.handle_connection(websocket, engine, tick_seconds=0.0, max_messages=1))

    assert engine.calls == 1
    assert websocket.sent[0]["agents"] == [
        {"id": 7, "x": 1.0, "y": 2.0, "z": 3.0, "pitch": 0.1, "yaw": 0.2}
    ]


def test_handler_returns_cleanly_on_client_disconnect():
    engine = MockInferenceEngine(ServerConfig(agent_count=2))
    websocket = _DisconnectingWebSocket()

    asyncio.run(ws_server.handle_connection(websocket, engine, tick_seconds=0.0))

    assert websocket.closed is False


def test_handler_closes_socket_on_unexpected_error():
    engine = MockInferenceEngine(ServerConfig(agent_count=2))
    websocket = _ErroringWebSocket()

    asyncio.run(ws_server.handle_connection(websocket, engine, tick_seconds=0.0))

    assert websocket.closed is True


# ----------------------------------------------------------------------
# Engine snapshots
# ----------------------------------------------------------------------
def test_mock_engine_reports_fifty_agents_within_world_bounds():
    engine = MockInferenceEngine(ServerConfig())
    message = engine.step().to_dict()

    assert len(message["agents"]) == 50
    assert [agent["id"] for agent in message["agents"]] == list(range(50))
    x_max, y_max, z_max = 100.0, 50.0, 100.0
    for agent in message["agents"]:
        assert set(agent) == REQUIRED_AGENT_FIELDS
        assert all(math.isfinite(agent[axis]) for axis in ("x", "y", "z"))
        assert 0.0 <= agent["x"] <= x_max
        assert 0.0 <= agent["y"] <= y_max
        assert 0.0 <= agent["z"] <= z_max


# ----------------------------------------------------------------------
# Application lifecycle, selection, and HTTP compatibility
# ----------------------------------------------------------------------
def test_application_shutdown_closes_engine(monkeypatch):
    closed: list[MockInferenceEngine] = []
    monkeypatch.setattr(MockInferenceEngine, "close", lambda self: closed.append(self))
    app = create_app(ServerConfig(agent_count=2))

    async def run_lifespan() -> None:
        async with _lifespan(app):
            assert closed == []

    asyncio.run(run_lifespan())
    assert len(closed) == 1


def test_websocket_route_uses_configured_path():
    app = create_app(ServerConfig(agent_count=2, websocket_path="/stream"))
    paths = {getattr(route, "path", None) for route in app.routes}
    assert "/stream" in paths


def test_checkpoint_failure_is_not_silently_mocked():
    config = ServerConfig(inference_mode="checkpoint", checkpoint_path="/no/such/checkpoint")
    with pytest.raises(CheckpointLoadError):
        create_app(config)


def test_root_reports_configured_mode():
    app = create_app(ServerConfig(agent_count=2))
    payload = _endpoint(app, "/")()
    assert payload["status"] == "ok"
    assert payload["mode"] == "mock"


def test_http_endpoints_remain_compatible():
    app = create_app(ServerConfig(agent_count=3))

    assert _endpoint(app, "/api/health")() == {"status": "ok"}

    public_config = _endpoint(app, "/api/config")()
    assert public_config["agent_count"] == 3
    assert public_config["websocket_path"] == "/ws"
    assert public_config["mode"] == "mock"

    state = _endpoint(app, "/api/state")()
    assert set(state) == REQUIRED_STEP_FIELDS

    metrics = _endpoint(app, "/api/metrics")()
    assert set(metrics) == REQUIRED_METRICS_FIELDS

    reset = _endpoint(app, "/api/reset")()
    assert reset["step"] == 0
    assert len(reset["agents"]) == 3


# ----------------------------------------------------------------------
# Backend <-> frontend contract parity
# ----------------------------------------------------------------------
def _model_fields(model: type) -> set[str]:
    fields = getattr(model, "model_fields", None)
    if fields is None:  # pydantic v1 compatibility
        fields = model.__fields__
    return set(fields)


def _typescript_interface_fields(path: Path, interface: str) -> set[str]:
    text = path.read_text(encoding="utf-8")
    match = re.search(rf"interface\s+{interface}\s*\{{(.*?)\}}", text, re.DOTALL)
    assert match is not None, f"interface {interface} not found in {path}"
    fields: set[str] = set()
    for raw_line in match.group(1).splitlines():
        line = raw_line.split("//", 1)[0].strip()
        declaration = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\s*\??\s*:", line)
        if declaration is not None:
            fields.add(declaration.group(1))
    return fields


def test_frontend_step_message_matches_backend_schema():
    frontend = _typescript_interface_fields(FRONTEND_TYPES / "websocket.ts", "StepMessage")
    assert frontend == _model_fields(StepMessage)


def test_frontend_agent_state_matches_backend_schema():
    frontend = _typescript_interface_fields(FRONTEND_TYPES / "agent.ts", "AgentState")
    assert frontend == _model_fields(AgentState)


def test_frontend_metrics_matches_backend_schema():
    frontend = _typescript_interface_fields(FRONTEND_TYPES / "metrics.ts", "MetricsSnapshot")
    assert frontend == _model_fields(MetricsSnapshot)


def test_shared_fixture_matches_step_message_schema():
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    message = StepMessage(**data)
    assert message.step == data["step"]
    assert [agent.id for agent in message.agents] == [agent["id"] for agent in data["agents"]]
