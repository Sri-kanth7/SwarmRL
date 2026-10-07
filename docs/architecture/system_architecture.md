# System Architecture

This document describes the complete intended SwarmRL architecture
and marks which parts exist as Week 1 foundations versus which parts
are future work.

## Overview

```text
                    ENVIRONMENT
                         |
                         v
                    OBSERVATIONS
                         |
                         v
                  RL TRAINING LAYER
                         |
                         v
                 POLICY / CHECKPOINT
                         |
                         v
                      BACKEND
                         |
                         v
                  WEBSOCKET STREAM
                         |
                         v
              REACT / THREE.JS UI
```

SwarmRL trains a swarm of up to 50 drones to search and cover a
bounded 3D disaster zone. Multi-agent training uses PPO-family
algorithms (IPPO baseline, MAPPO with a centralized critic later)
on top of a PettingZoo parallel environment. The backend streams
state and metrics over WebSocket; the frontend renders the swarm in
a react-three-fiber scene.

## Coordinate And Unit Conventions

- Right-handed, **y-up** world: `x`/`z` span the ground plane,
  `y` is height. This matches Three.js, so state values move from
  environment to frontend without conversion.
- `world_size = (x, y, z)`, default `(100, 50, 100)`.
- Angles are radians. `yaw` is measured from `+x` toward `+z`;
  `pitch` lifts toward `+y`.
- Positions are world units, speeds are units/second, time step
  `dt = 0.1` s by default.
- Coverage is a 2D grid over the ground plane; cells are identified
  by the flat index `row * cols + col`.

## Layer Responsibilities

### Environment (`env/`) — Week 1 foundation: **implemented**

| Module | Responsibility |
|---|---|
| `config.py` | `EnvConfig` (agent count, world, seeds, limits) |
| `physics.py` | Kinematics, bounded world, collisions, spawning |
| `actions.py` | Action vector definition and bounds |
| `observations.py` | Observation vector layout and spaces |
| `coverage.py` | Coverage grid, explored percentage |
| `rewards.py` | Reward term interface and basic terms |
| `swarm_env.py` | PettingZoo `ParallelEnv` lifecycle |

Future work: richer rewards and tuning, curriculum, dynamic
obstacles, wind, performance optimization.

### Training (`training/`) — Weeks 1–2: **IPPO training runs; MAPPO configuration only**

| Module | Responsibility |
|---|---|
| `config.py` | YAML loading, seed handling (`TrainingConfig`) |
| `rl_config.py` | RLlib configuration dictionary builder and `PPOConfig` translation |
| `train_mappo.py` | MAPPO entry point (configuration build, no runs yet) |
| `train_ippo.py` | IPPO entry point: runs RLlib training, writes checkpoints |
| `rllib_env.py` | RLlib `MultiAgentEnv` adapter and episode-metric collection |
| `mappo_model.py` | Actor/critic network specifications |
| `checkpoint.py` | Checkpoint metadata contract |
| `evaluate.py` | Policy-agnostic evaluation loop |
| `metrics.py` | Episode-metric records and aggregation |
| `configs/*.yaml` | MAPPO/IPPO configuration files |

Future work: MAPPO training runs with the centralized critic,
decentralized actors at inference, tuned hyperparameters,
checkpoint-backed backend inference.

### Backend (`server/`) — Week 1 foundation: **implemented (mock mode)**

| Module | Responsibility |
|---|---|
| `app.py` | FastAPI application and routes |
| `ws_server.py` | WebSocket streaming handler |
| `schemas.py` | `StepMessage` / `AgentState` / `MetricsSnapshot` |
| `config.py` | Configuration resolution (env vars > YAML > defaults) |
| `metrics.py` | Metrics history collection |
| `inference.py` | **Mock** deterministic state generator |
| `replay.py` | Reserved (deferred) |

Future work: checkpoint-backed inference, replay, control
endpoints, production streaming performance.

### Frontend (`frontend/`) — Week 1 foundation: **implemented (dummy data)**

| Area | Responsibility |
|---|---|
| `scenes/SwarmScene.tsx` | react-three-fiber canvas and camera |
| `components/Drone`, `DroneSwarm` | Drone mesh prototypes |
| `components/VisionCone` | Vision-cone prototype (optional) |
| `components/CoverageMap` | Ground, grid, debris blocks |
| `components/ControlPanel`, `AnalyticsPanel` | Status/metrics panels |
| `hooks/useSwarmState` | Deterministic dummy swarm snapshot |
| `hooks/useWebSocket` | Reserved (no connection yet) |
| `services/websocket.ts` | Endpoint URL resolution |
| `types/*` | Wire-contract TypeScript types |

Future work: live WebSocket animation, interpolation, flight
trails, coverage heatmap, analytics charts, final control panel,
performance tuning.

## Data Contracts

The interfaces between layers are frozen as documents in
`docs/contracts/`:

| Contract | Between |
|---|---|
| `environment_api.md` | environment ↔ training |
| `checkpoint_contract.md` | training ↔ backend |
| `metrics_schema.md` | backend ↔ frontend (metrics) |
| `websocket_schema.md` | backend ↔ frontend (state stream) |

Contract changes must update the document and every implementation
in the same change.

## Configuration Sources

One configuration file plus environment variables:

1. Environment variables (`SWARMRL_*`, see `.env.example`) — highest
   precedence for the backend.
2. `configs/project.yaml` — shared source of truth for environment,
   training, server, and frontend values.
3. Code defaults (`env/config.py`, `server/config.py`) — mirror the
   YAML values so components work without files.

Training additionally reads `training/configs/*.yaml`; these files
duplicate only the environment block and must stay aligned with
`configs/project.yaml`.

## Development Stage Map

| Stage | Content | Status |
|---|---|---|
| Week 1 | Environment, physics, coverage, reward interface, RL configuration, backend skeleton, mock stream, frontend/3D skeleton, contracts, docs, tests layout | Done — foundations |
| Week 2 | RL training runs, checkpoint production, real inference, full backend integration, live frontend data flow | **In progress — IPPO runs and checkpoints done** |
| Week 3 | Visualization depth (trails, heatmap), analytics, replay, control surface | Future |
| Week 4 | Performance optimization, end-to-end hardening, final demo, review | Future |
