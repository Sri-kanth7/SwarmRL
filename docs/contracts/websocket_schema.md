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
