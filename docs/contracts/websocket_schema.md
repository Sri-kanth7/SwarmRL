# WebSocket Schema Contract

Interface between the **backend** and the **frontend**.

Implementation: `server/schemas.py` (source of truth),
TypeScript mirrors in `frontend/src/types/websocket.ts` and
`frontend/src/types/agent.ts`.

## Transport

| Property | Value |
|---|---|
| Endpoint | `ws://<host>:<port>/ws` (configurable via `websocket_path`) |
| Direction | Server → client only in Week 1 |
| Encoding | JSON, one message per streamed step |
| Cadence | `mock_tick_interval_ms` (default 100 ms) |
| Handshake | Standard WebSocket upgrade; no auth in Week 1 |

During frontend development the Vite dev server proxies `/ws` to
the backend (`frontend/vite.config.js`).

## Step Message

```json
{
  "step": 0,
  "agents": [
    {"id": 0, "x": 10.5, "y": 12.0, "z": 44.2, "pitch": 0.01, "yaw": 1.57},
    {"id": 1, "x": 62.3, "y": 18.4, "z": 30.7, "pitch": -0.02, "yaw": -0.4}
  ],
  "newly_visited_cells": [124, 125, 144],
  "collisions": 0,
  "explored_pct": 3.75
}
```

### Fields

| Field | Type | Constraints | Meaning |
|---|---|---|---|
| `step` | `integer` | `>= 0` | Monotonic step counter of the current episode |
| `agents` | `array<AgentState>` | one entry per drone | Current swarm state |
| `agents[].id` | `integer` | `>= 0` | Drone index `0..agent_count-1` (matches `drone_<id>` in the environment) |
| `agents[].x` | `number` | world bounds | x position, ground plane, y-up world |
| `agents[].y` | `number` | world bounds | Height |
| `agents[].z` | `number` | world bounds | z position, ground plane |
| `agents[].pitch` | `number` | radians | Nose pitch |
| `agents[].yaw` | `number` | radians, `(-pi, pi]` | Heading |
| `newly_visited_cells` | `array<integer>` | flat cell indices | Cells explored during this step; index = `row * cols + col` of the coverage grid (`env/coverage.py`); `[]` when nothing new |
| `collisions` | `integer` | `>= 0` | Collision events during this step (agent-agent pairs + boundary contacts) |
| `explored_pct` | `number` | `[0, 100]` | Explored coverage percentage of the episode |

Notes:

- `agents` preserves `id` order (`0, 1, ...`) so clients can index
  directly into arrays.
- The message describes **one step**; clients store the latest
  message and may accumulate `newly_visited_cells` into their own
  coverage state.
- `collisions` is per-step; cumulative values are reported through
  the metrics endpoint (`docs/contracts/metrics_schema.md`).

## Client Messages

Week 1: clients may send nothing; any received payload is ignored.
A control protocol (pause, resume, speed, reset) is reserved for a
later week and must be additive to this document.

## Frontend Integration

The React frontend consumes this stream in one place:

| Step | Module | Responsibility |
|---|---|---|
| Endpoint | `frontend/src/services/websocket.ts` | Resolve the URL: explicit argument → `VITE_SWARMRL_WS_URL` → same-origin `<protocol>//<host>/ws` |
| Transport | `frontend/src/hooks/useWebSocket.ts` | Open one socket, expose status, clean up handlers/timers, bounded reconnect |
| Decode | `frontend/src/services/messages.ts` | `decodeStepMessage` / `parseStepMessage` validate a frame into a `StepMessage` or `null` |
| State | `frontend/src/hooks/useSwarmState.ts` | Map the latest valid message to agents + metrics for the scene/panels |
| Render | `frontend/src/scenes/SwarmScene.tsx` → `DroneSwarm` → `Drone` | Position the drone meshes from the received coordinates |

Notes:

- Invalid JSON, a non-object payload, a missing/invalid `id`, or a
  non-finite `x`/`y`/`z` rejects the **whole** frame; the UI keeps the
  last valid snapshot instead of rendering partial data.
- The deterministic placeholder swarm (`generateDummyAgents`) only
  fills the scene **before** the first valid frame. It never animates;
  once a frame arrives the scene shows only streamed agents.
- Metrics come from the frame (`step`, `explored_pct`, per-step
  `collisions`). `mean_reward` is not part of this message and is
  rendered as unavailable, not fabricated.
- `VITE_SWARMRL_WS_URL` is optional. Set it (for example
  `ws://localhost:8000/ws`) when the frontend is served from a
  different origin than the backend; otherwise the same-origin `/ws`
  path is used and the Vite dev proxy forwards it.

## Coordinate Mapping

No conversion is applied. The environment and Three.js share the same
right-handed, y-up convention, so received world units map directly:

| Wire field | Three.js use |
|---|---|
| `x`, `y`, `z` | `position=[x, y, z]` (world units) |
| `yaw` | `rotation=[0, -yaw, 0]` (radians) |
| `pitch` | child-group `rotation=[0, 0, pitch]` (radians) |

`agents[].id` is the React key, so each snapshot updates the same
drone mesh rather than recreating the scene.

## Versioning

The schema is versioned with the project (`0.1.0`). Removing or
renaming a field is a breaking change and requires updating this
document, `server/schemas.py`, and the frontend types together.

## HTTP Endpoints (related, Week 1)

| Endpoint | Method | Payload |
|---|---|---|
| `/` | GET | Service info (`service`, `status`, `version`, `mode`) |
| `/api/health` | GET | `{"status": "ok"}` |
| `/api/config` | GET | Public server configuration subset |
| `/api/state` | GET | Current `StepMessage` (does not advance time) |
| `/api/metrics` | GET | Current `MetricsSnapshot` |
| `/api/reset` | POST | Resets the mock generator, returns a fresh `StepMessage` |

## End-To-End Smoke Test

Confirm that **real** positions (not placeholder movement) reach the
browser:

1. Start the backend (mock mode is the default):
   `python -m uvicorn server.app:app --host 0.0.0.0 --port 8000`.
2. Start the frontend: `npm --prefix frontend run dev`, then open
   `http://localhost:5173`.
3. The header badge reads `live stream (streaming)` and the drones
   move between steps; the Control panel shows `Stream: connected` and
   a rising `Step`.
4. Stop the backend: the badge switches to `live stream
   (disconnected)` (the last frame stays on screen; the placeholder
   never returns) and the Control panel shows the disconnected state.

Automated cover for the same path lives in
`tests/integration/test_server_frontend.py`: it asserts two successive
snapshots have stable ids and changing coordinates, that the message
shape is the contract shape, and that shutdown closes the engine.
