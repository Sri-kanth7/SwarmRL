"""FastAPI application for the SwarmRL backend.

Run with::

    python -m uvicorn server.app:app --host 0.0.0.0 --port 8000

Week 1 exposes a small HTTP surface (health, public configuration,
current mock state, reset) plus the WebSocket stream defined in
``docs/contracts/websocket_schema.md``. Inference from trained
checkpoints, replay, and control endpoints are later work.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from server import ws_server
from server.config import ServerConfig, load_server_config
from server.inference import MockInferenceEngine

API_VERSION = "0.1.0"


def create_app(config: ServerConfig | None = None) -> FastAPI:
    """Build the backend application.

    Args:
        config: Optional configuration; defaults to
            :func:`server.config.load_server_config`.

    Returns:
        A configured FastAPI instance.
    """
    resolved = config or load_server_config()
    app = FastAPI(title="SwarmRL Backend", version=API_VERSION)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(resolved.cors_origins),
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    engine = MockInferenceEngine(resolved)
    app.state.config = resolved
    app.state.engine = engine

    @app.get("/")
    def root() -> dict[str, Any]:
        return {
            "service": "swarmrl-backend",
            "status": "ok",
            "version": API_VERSION,
            "mode": "mock",
        }

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/config")
    def public_config() -> dict[str, Any]:
        return resolved.public_dict()

    @app.get("/api/state")
    def current_state() -> dict[str, Any]:
        return engine.current().to_dict()

    @app.get("/api/metrics")
    def current_metrics() -> dict[str, Any]:
        return engine.metrics().to_dict()

    @app.post("/api/reset")
    def reset() -> dict[str, Any]:
        return engine.reset().to_dict()

    @app.websocket(resolved.websocket_path)
    async def websocket_endpoint(websocket: WebSocket) -> None:
        await ws_server.handle_connection(
            websocket,
            engine,
            tick_seconds=resolved.tick_interval_seconds,
        )

    return app


app = create_app()
