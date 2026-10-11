"""FastAPI application for the SwarmRL backend.

Run with::

    python -m uvicorn server.app:app --host 0.0.0.0 --port 8000

Week 1 exposes a small HTTP surface (health, public configuration,
current mock state, reset) plus the WebSocket stream defined in
``docs/contracts/websocket_schema.md``. Inference from trained
checkpoints, replay, and control endpoints are later work.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from server import ws_server
from server.config import ServerConfig, load_server_config
from server.inference import (
    CheckpointInferenceEngine,
    CheckpointLoadError,
    MockInferenceEngine,
)

API_VERSION = "0.1.0"


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Release the inference engine when the application shuts down.

    Checkpoint-backed engines own an RLlib algorithm, an environment,
    and possibly an engine-started Ray runtime, so closing on shutdown
    keeps process teardown clean. Closing is idempotent, and the mock
    engine's ``close`` is a no-op.
    """
    try:
        yield
    finally:
        engine = getattr(app.state, "engine", None)
        close = getattr(engine, "close", None)
        if callable(close):
            close()


def create_app(config: ServerConfig | None = None) -> FastAPI:
    """Build the backend application.

    Args:
        config: Optional configuration; defaults to
            :func:`server.config.load_server_config`.

    Returns:
        A configured FastAPI instance.

    Raises:
        CheckpointLoadError: If checkpoint mode is selected but the
            checkpoint cannot be loaded. The error is never silently
            downgraded to the mock engine.
    """
    resolved = config or load_server_config()
    engine = _create_engine(resolved)
    app = FastAPI(title="SwarmRL Backend", version=API_VERSION, lifespan=_lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(resolved.cors_origins),
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.config = resolved
    app.state.engine = engine

    @app.get("/")
    def root() -> dict[str, Any]:
        return {
            "service": "swarmrl-backend",
            "status": "ok",
            "version": API_VERSION,
            "mode": resolved.public_dict()["mode"],
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


def _create_engine(config: ServerConfig):
    mode = (config.inference_mode or "mock").lower()
    if mode not in ("mock", "checkpoint"):
        mode = "mock"
    if mode == "mock":
        return MockInferenceEngine(config)
    try:
        return CheckpointInferenceEngine(config)
    except CheckpointLoadError as e:
        raise
    except Exception as e:
        raise CheckpointLoadError(f"Failed to initialize checkpoint inference: {e}") from e


app = create_app()
