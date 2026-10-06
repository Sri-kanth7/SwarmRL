"""WebSocket streaming foundation.

Provides the connection handler used by the FastAPI application.
During Week 1 every client receives the deterministic mock stream
produced by ``server.inference.MockInferenceEngine``; switching the
handler to checkpoint-backed inference later only requires swapping
the engine, not this protocol logic.

Protocol: on connect the server accepts the socket and sends
``StepMessage`` JSON documents at the configured tick interval.
Client messages are ignored in Week 1 (control protocol is later
work).
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from fastapi import WebSocket, WebSocketDisconnect

if TYPE_CHECKING:
    from server.inference import MockInferenceEngine


async def handle_connection(
    websocket: WebSocket,
    engine: "MockInferenceEngine",
    *,
    tick_seconds: float,
    max_messages: int | None = None,
) -> None:
    """Stream mock state messages to one connected client.

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
    except Exception:
        try:
            await websocket.close()
        except Exception:
            pass
