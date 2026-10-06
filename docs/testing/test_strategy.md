# Test Strategy

Testing conventions for SwarmRL. Week 1 establishes the layout,
the tooling configuration, and a minimal set of contract tests.
The full integration suites belong to later weeks.

## Layout

```text
tests/
├── unit/           # pure logic, no network, no processes
├── integration/    # cross-layer contract checks (later weeks)
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
| `tests/smoke/test_project_smoke.py` | Files, configs, and imports of every layer exist and parse |
| `tests/fixtures/mock_agent_data.json` | Sample `StepMessage` for frontend/tooling use |

`tests/integration/*` files are reserved placeholders: their
contract-level suites are written in Week 2 (environment↔training,
training↔backend, backend↔frontend, end-to-end).

## Conventions

- Tests assert against the contracts in `docs/contracts/`; when a
  contract changes, its tests change in the same pull request.
- No network access, no trained-model artifacts, no Ray runtime in
  unit tests.
- Deterministic only: fixed seeds, no wall-clock assertions.
- Keep tests independent: each test constructs its own environment
  or configuration; no shared mutable state across modules.

## Future Extensions (later weeks)

- Environment↔training integration: config round-trips, RLlib env
  registration.
- Training↔backend: checkpoint metadata handoff.
- Backend↔frontend: schema parity between Python models and
  TypeScript types.
- End-to-end: streamed mock state renders without contract errors.
