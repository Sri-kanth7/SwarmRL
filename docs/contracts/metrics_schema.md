# Metrics Schema Contract

Interface for the project's shared metric names. The same names are
produced by the **training** layer, the **backend**, and displayed
by the **frontend**, so dashboards and tests can rely on one
definition.

Implementation: `server/schemas.py` (`MetricsSnapshot`),
`training/metrics.py` (`EpisodeMetrics`), TypeScript mirror in
`frontend/src/types/metrics.ts`.

## Metric Definitions

| Metric | Type | Range | Meaning |
|---|---|---|---|
| `explored_pct` | `float` | `[0, 100]` | Percentage of coverage-grid cells visited so far. `100.0` = fully explored. Produced by `env/coverage.py: CoverageGrid.explored_pct()`. |
| `collisions` | `integer` | `>= 0` | Collision events: one per agent-agent pair closer than `collision_radius`, plus one per world-boundary contact. Episode-level values are cumulative; step-level values are per-step. |
| `mean_reward` | `float` | unbounded | Mean per-drone reward. At episode level: mean of the agents' total episode rewards. At step level: mean of the latest per-step rewards. |

## Snapshot Shape

`MetricsSnapshot` (`server/schemas.py`):

```json
{
  "step": 120,
  "explored_pct": 41.25,
  "collisions": 3,
  "mean_reward": 0.42
}
```

| Field | Type | Meaning |
|---|---|---|
| `step` | `integer >= 0` | Step counter of the snapshot |
| `explored_pct` | `float [0, 100]` | Current explored coverage |
| `collisions` | `integer >= 0` | Cumulative collision events |
| `mean_reward` | `float` | Mean reward of the latest step |

## Aggregation Rules

Training-side records (`training/metrics.py: EpisodeMetrics`) add
per-episode context:

| Field | Type | Meaning |
|---|---|---|
| `episode` | `integer` | 0-based episode counter |
| `steps` | `integer` | Episode length |
| `mean_reward` | `float` | Mean per-drone total reward of the episode |
| `explored_pct` | `float` | Coverage at episode end |
| `collisions` | `integer` | Collision events during the episode |

`MetricsAggregator.summary()` reports run-level means plus
`best_explored_pct`.

## Producers And Consumers

| Producer | Consumer | Channel |
|---|---|---|
| Environment (`infos`) | Training metrics, tests | In-process |
| Training aggregator | Experiment tooling | In-process / logs |
| Backend collector | Frontend panels, HTTP `/api/metrics` | HTTP (Week 1), WebSocket (later) |

## Reserved Fields

Future metrics (episode duration in wall time, throughput, FPS,
energy totals) must be added additively with documented types; no
existing field may change meaning without updating this document.
