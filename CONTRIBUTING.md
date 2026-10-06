# Contributing to SwarmRL

## Workflow

1. Create a feature branch from `main`.
2. Keep each branch focused on one change (environment, training,
   backend, frontend, tests, or documentation).
3. Follow the interface contracts in `docs/contracts/` before
   writing code that crosses a layer boundary. If a contract must
   change, update the contract document in the same change.
4. Open a pull request describing what changed and which layer it
   affects.
5. Merge only after review; do not force-push shared branches.

## Code Conventions

- Python: target 3.11+, use type hints, module and class docstrings.
- Line length: 100 (configured in `pyproject.toml`).
- Lint with `ruff check .` and format with `ruff format .`.
- Configuration lives in `configs/project.yaml` and `.env.example`.
  Do not add competing configuration systems; prefer configuration
  values over magic numbers.
- Frontend: React + TypeScript with Vite; keep components small and
  typed against `frontend/src/types/`.

## Component Naming

Refer to components only by technical responsibility:

- environment, reward, training, backend, frontend, visualization,
  integration, testing

Do not add individual names to code, documentation, schemas, or APIs.

## Testing

- Test layout: `tests/unit`, `tests/integration`, `tests/smoke`.
- Write unit tests for pure logic (physics, coverage, schemas,
  metrics) and integration tests for cross-layer contracts.
- Run the suite with `python -m pytest` (see `docs/testing/
  test_strategy.md` for the full command set).
- Do not commit tests that depend on network access or trained
  model artifacts.

## Documentation

- Interface contracts live in `docs/contracts/`.
- The system overview lives in `docs/architecture/`.
- Keep `README.md` aligned with the current development stage;
  do not describe foundations as finished features.
