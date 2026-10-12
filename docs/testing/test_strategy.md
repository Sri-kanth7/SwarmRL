# Test Strategy

Testing conventions for SwarmRL. Week 1 established the layout,
the tooling configuration, and a minimal set of contract tests.
Week 2 adds the environment↔training integration suite; the
remaining integration suites belong to later weeks.

## Layout

```text
tests/
├── unit/           # pure logic, no network, no processes
├── integration/    # cross-layer contract checks
├── smoke/          # repository-wide sanity checks
└── fixtures/       # shared sample data (JSON)
```

Tooling is configured in `pyproject.toml`:

- framework: `pytest` (`testpaths = ["tests"]`, `pythonpath = ["."]`)
- coverage: `pytest-cov`
- lint: `ruff` (line length 100, target py311)

## Commands

```bash
# full suite
python -m pytest

# by layer
python -m pytest tests/unit
python -m pytest tests/smoke
python -m pytest tests/integration

# with coverage
python -m pytest --cov=env --cov=server --cov=training
```

Shell helpers exist in `scripts/run_tests.sh` and
`scripts/run_smoke_test.sh`; on Windows run the `python -m pytest`
commands directly (PowerShell-compatible).

## What Week 1 Covers

| Test file | Scope |
|---|---|
| `tests/unit/test_physics.py` | Heading math, bounded integration, collision pairs, spawning |
| `tests/unit/test_environment.py` | PettingZoo lifecycle, spaces, shapes, truncation, bounds |
| `tests/unit/test_schemas.py` | WebSocket/metrics schema validation and serialization |
| `tests/unit/test_metrics.py` | Episode aggregation and backend metrics collection |
| `tests/unit/test_coverage.py` | Coverage-grid geometry, `explored_pct` formula, duplicate handling, `mark_segment` sweeping, boundary clamping, reset, and mock-stream coverage (cumulative, then saturating) |
| `tests/smoke/test_project_smoke.py` | Files, configs, and imports of every layer exist and parse |
| `tests/fixtures/mock_agent_data.json` | Sample `StepMessage` for frontend/tooling use |

## What Week 2 Covers

| Test file | Scope |
|---|---|
| `tests/integration/test_env_training.py` | RLlib environment adapter, YAML→RLlib configuration translation, `--print-config`, short IPPO run writing the checkpoint contract |
| `tests/integration/test_server_frontend.py` | WebSocket handler streaming the selected engine's messages, stable ids / finite world coordinates / changing positions, disconnect and error cleanup, engine close on application shutdown, checkpoint-failure-is-not-mocked, HTTP endpoint compatibility, and Python↔TypeScript field parity |

The environment↔training suite runs one short IPPO session, so that
file is the only place a Ray runtime is started. The adapter,
registry, and configuration checks inside it do not need a running
Ray runtime; `tests/unit/*` still never touches Ray.

The backend↔frontend suite drives `server.ws_server.handle_connection`
with in-process fake sockets (no network, no Ray, no checkpoint) and
compares `frontend/src/types/*.ts` against the Pydantic schemas. It
deliberately avoids `fastapi.testclient`, which would require `httpx`
(a package the project does not declare).

`tests/integration/test_end_to_end.py` remains a reserved placeholder.

## Frontend Tests

There is no JavaScript/TypeScript test runner in the repository
(`frontend/package.json` has no `test` script and no vitest/jest
dependency). Adding one would change dependencies, which this change
set is not allowed to do, so frontend behavior is covered indirectly:

- the pure decoder/validator (`frontend/src/services/messages.ts`) is
  written dependency-free so it can be unit tested once a runner is
  added,
- the wire contract is enforced field-for-field from pytest by
  `tests/integration/test_server_frontend.py`, and
- the manual smoke test below exercises the browser path.

Adding a frontend runner (for example vitest) is a recommended
follow-up that requires an explicit dependency decision.

## Manual Streaming Verification

See the "End-To-End Smoke Test" section of
`docs/contracts/websocket_schema.md`. In short: start the backend and
frontend, confirm the header badge reads `live stream (streaming)`
with moving drones and a rising step, then stop the backend and
confirm the badge switches to a disconnected state.

## Conventions

- Tests assert against the contracts in `docs/contracts/`; when a
  contract changes, its tests change in the same pull request.
- No network access, no trained-model artifacts, no Ray runtime in
  unit tests.
- Deterministic only: fixed seeds, no wall-clock assertions.
- Keep tests independent: each test constructs its own environment
  or configuration; no shared mutable state across modules.

## Future Extensions (later weeks)

- Training↔backend: checkpoint metadata handoff.
- Backend↔frontend: schema parity between Python models and
  TypeScript types.
- End-to-end: streamed mock state renders without contract errors.
