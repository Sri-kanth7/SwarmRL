"""WebSocket streaming handler.

Provides the connection handler used by the FastAPI application. The
engine is the one selected for the application (``server.app`` chooses
the mock or checkpoint-backed ``InferenceEngine``), so the same
protocol logic streams either source without change.

Protocol: on connect the server accepts the socket and pushes one
``StepMessage`` JSON document per simulation step at the configured
tick interval (server push). Client messages are ignored; the control
protocol is reserved for later work.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from fastapi import WebSocket, WebSocketDisconnect

if TYPE_CHECKING:
    from server.inference import InferenceEngine


async def handle_connection(
    websocket: WebSocket,
    engine: "InferenceEngine",
    *,
    tick_seconds: float,
    max_messages: int | None = None,
) -> None:
    """Stream state messages from the selected engine to one client.

    Args:
        websocket: Accepted-or-not WebSocket connection; this function
            performs the handshake.
        engine: State generator shared by all clients.
        tick_seconds: Delay between consecutive messages.
        max_messages: Optional cap (useful for tests); ``None`` streams
            until the client disconnects.
    """
    await websocket.accept()
    sent = 0
    try:
        while max_messages is None or sent < max_messages:
            message = engine.step()
            await websocket.send_json(message.to_dict())
            sent += 1
            if max_messages is None or sent < max_messages:
                await asyncio.sleep(max(tick_seconds, 0.0))
    except WebSocketDisconnect:
        return
    # Best-effort cleanup: a failed stream must still close the socket
    # without replacing the original disconnect error.
    except Exception:  # noqa: BLE001
        try:
            await websocket.close()
        except Exception:  # noqa: BLE001, S110
            pass
