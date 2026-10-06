/**
 * Metrics contract.
 *
 * Mirrors `MetricsSnapshot` in `server/schemas.py` and
 * `docs/contracts/metrics_schema.md`.
 */
export interface MetricsSnapshot {
  step: number
  /** Explored coverage percentage in [0, 100]. */
  explored_pct: number
  /** Collision event count (>= 0). */
  collisions: number
  /** Mean per-drone reward of the latest step. */
  mean_reward: number
}
