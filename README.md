# SwarmRL

## Multi-Agent Deep Reinforcement Learning Simulator

SwarmRL is a multi-agent deep reinforcement learning simulator
designed to coordinate a swarm of 50 autonomous drones in a
simulated disaster zone.

The objective is to train the drones to search and cover the
environment cooperatively while minimizing collisions and
producing an emergent, decentralized search pattern.

> **Current stage: Week 2 — IPPO training runs and live streaming.**
> The repository contains working foundations (environment,
> configuration, backend with a mock state stream, frontend/3D scene,
> interface contracts, and tests) plus a runnable IPPO entry point
> that produces an RLlib checkpoint with contract metadata. The
> frontend now connects to the backend WebSocket stream and renders
> live drone positions in mock mode; MAPPO training and
> checkpoint-backed backend inference remain works in progress.
> Runtime verification is expected in GitHub Codespaces.

---

## Project Overview

Training a single autonomous agent in an environment is a
well-established problem. Coordinating a large swarm of
autonomous agents introduces additional challenges such as:

- Multi-agent coordination
- Collision avoidance
- Collective area coverage
- Decentralized decision making
- Continuous 3D navigation
- Scalable reinforcement learning

SwarmRL addresses these challenges using Multi-Agent
Proximal Policy Optimization (MAPPO) with a centralized
critic during training and decentralized actors during
inference (the IPPO baseline runs today; MAPPO arrives later).

---

## System Architecture

```text
                   SwarmRL
                      |
        +-------------+-------------+
        |             |             |
        v             v             v
   Environment     RL Engine     Backend
        |             |             |
   PettingZoo     Ray RLlib      FastAPI
   NumPy          PyTorch        WebSocket
        |             |             |
        +-------------+-------------+
                      |
                      v
                3D Frontend
                      |
              React + Three.js
```

Data flow:

```text
Environment -> Observations -> RL Training -> Policy/Checkpoint
            -> Backend -> WebSocket Stream -> React / Three.js UI
```

Full details, including which components are Week 1 foundations
versus future work, live in
`docs/architecture/system_architecture.md`.

---

## Technology Stack

| Layer | Technologies |
|---|---|
| Environment | Python, PettingZoo (Parallel API), Gymnasium, NumPy |
| Training | Ray RLlib, PyTorch, YAML configuration |
| Backend | Python, FastAPI, WebSockets, Pydantic |
| Frontend | React, TypeScript, Vite, react-three-fiber, Three.js |
| Testing | pytest, pytest-cov, ruff |

---

## Repository Structure

```text
configs/            project-wide configuration (project.yaml)
docs/
  architecture/     system architecture document
  contracts/        interface contracts (environment, checkpoint,
                    metrics, websocket)
  reviews/          project reviews (reserved for later weeks)
  testing/          test strategy
env/                PettingZoo environment: physics, actions,
                    observations, coverage, rewards
frontend/           React + TypeScript + Vite + react-three-fiber
scripts/            simple runner scripts (bash)
server/             FastAPI backend: schemas, WebSocket stream,
                    mock state generator, metrics
tests/              unit / smoke / integration (pytest)
training/           RL configuration, entry points, checkpoint
                    metadata, evaluation, metrics
```

---

## Environment Conventions

- **Coordinate system:** right-handed, y-up; `x`/`z` span the
  ground plane, `y` is height (matches Three.js).
- **World:** bounded box, default `(100, 50, 100)` world units.
- **Agents:** `"drone_0" .. "drone_49"` (default 50); the integer
  index is the `id` used on the wire.
- **Action:** `[velocity, pitch, yaw]` (forward speed m/s, absolute
  pitch rad, absolute heading rad).
- **Coverage:** 2D grid over the ground plane; cells identified by
  flat index `row * cols + col`.
- **Metrics:** `explored_pct` (`0..100`), `collisions`, `mean_reward`.

---

## Current Foundation

Implemented:

- PettingZoo parallel environment with reset/step lifecycle,
  bounded 3D kinematics, collision and coverage bookkeeping
- Action/observation definitions and reward-term interface
- RL configuration files (MAPPO/IPPO), config loader, environment
  registration, and a runnable IPPO entry point that trains with
  RLlib
- Checkpoint contract helpers plus a checkpoint directory written by
  every IPPO run (`metadata.json`, `policy.pt`, `metrics.json`,
  `config_snapshot.yaml`, and the RLlib checkpoint under `rllib/`)
- FastAPI backend with HTTP endpoints, WebSocket stream, message
  schemas, a deterministic **mock** drone-state generator, and a
  selectable checkpoint-backed inference engine
- React/Vite/TypeScript frontend with a 3D scene, drone
  prototypes, ground/coverage map, and metrics panels that render the
  live WebSocket stream (with a deterministic placeholder before the
  first frame)
- Interface contracts and architecture documentation
- Unit, smoke, environment↔training, and backend↔frontend
  integration tests

Intentionally deferred (later weeks):

- MAPPO training and the centralized critic, tuned hyperparameters
- Checkpoint-backed backend inference hardening (world-coordinate
  extraction and end-to-end validation with a trained checkpoint)
- Replay, control endpoints, production streaming
- Flight trails, coverage heatmap, analytics charts, final control
  panel, interpolation
- Curriculum, dynamic obstacles, wind, performance optimization

---

## Getting Started

### Inference Configuration (Server)

The backend supports mock and checkpoint-backed inference:

- SWARMRL_INFERENCE_MODE or server.inference_mode in configs/project.yaml: mock (default) or checkpoint
- SWARMRL_CHECKPOINT_PATH or 	raining.checkpoint_path / SWARMRL_CHECKPOINT_DIR: path to checkpoint directory containing metadata.json and RLlib checkpoint

If checkpoint mode is selected and the checkpoint is missing/incompatible, the server fails with a clear error.

### Streaming Configuration (Frontend)

The frontend resolves the stream URL in this order: an explicit
argument, then `VITE_SWARMRL_WS_URL`, then the same-origin `ws(s)://<host>/ws`.
In development the Vite proxy forwards `/ws` to `http://localhost:8000`,
so no variable is required. Set `VITE_SWARMRL_WS_URL` (for example
`ws://localhost:8000/ws`) only when the frontend is served from a
different origin than the backend.

Dependencies are declared in `pyproject.toml` and
`requirements.txt` (install them in your own environment; this
repository does not install anything automatically).

```bash
# backend
pip install -r requirements.txt

# frontend
cd frontend && npm install && cd ..
```

Common commands (run from the repository root):

```bash
# tests
python -m pytest
python -m pytest tests/smoke

# lint
ruff check .

# backend server (state stream on ws://localhost:8000/ws; mock by default)
python -m uvicorn server.app:app --host 0.0.0.0 --port 8000

# backend server with checkpoint-backed inference (fails if the checkpoint is missing)
SWARMRL_INFERENCE_MODE=checkpoint SWARMRL_CHECKPOINT_PATH=training/checkpoints/latest \
  python -m uvicorn server.app:app --host 0.0.0.0 --port 8000

# build a training configuration without training
python -m training.train_mappo --print-config
python -m training.train_ippo --print-config

# run a short IPPO training session (writes training/checkpoints/<run_id>/)
python -m training.train_ippo --config training/configs/ippo.yaml --iterations 2

# frontend dev server (http://localhost:5173)
npm --prefix frontend run dev

# backend-to-frontend streaming contract (in-process, no server needed)
python -m pytest tests/integration/test_server_frontend.py -v
```

To verify the live path in a browser, run the backend and frontend
above, open `http://localhost:5173`, and confirm the header badge
reads `live stream (streaming)` while the drones move and the step
rises. Stopping the backend switches the badge to a disconnected
state. Full details are in `docs/contracts/websocket_schema.md`
("End-To-End Smoke Test").

On Windows, run the `python ...` / `npm ...` commands directly in
PowerShell; the files in `scripts/` are bash helpers.

Configuration comes from `configs/project.yaml` and
`.env.example` (`SWARMRL_*` environment variables take
precedence for the backend).

---

## Interface Contracts

| Document | Interface |
|---|---|
| `docs/contracts/environment_api.md` | environment ↔ training |
| `docs/contracts/checkpoint_contract.md` | training ↔ backend |
| `docs/contracts/metrics_schema.md` | metrics definitions |
| `docs/contracts/websocket_schema.md` | backend ↔ frontend stream |

---

## Contributing

See `CONTRIBUTING.md` for the workflow, code conventions, and the
contract-first rule for cross-layer changes.
