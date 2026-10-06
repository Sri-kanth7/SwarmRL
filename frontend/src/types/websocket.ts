/**
 * WebSocket message contract.
 *
 * Mirrors `StepMessage` in `server/schemas.py` and
 * `docs/contracts/websocket_schema.md`.
 */
import type { AgentState } from './agent'

export interface StepMessage {
  /** Monotonically increasing step counter. */
  step: number
  /** One entry per drone. */
  agents: AgentState[]
  /** Flat coverage-cell indices (`row * cols + col`) explored this step. */
  newly_visited_cells: number[]
  /** Collision events during this step. */
  collisions: number
  /** Explored coverage percentage in [0, 100]. */
  explored_pct: number
}
