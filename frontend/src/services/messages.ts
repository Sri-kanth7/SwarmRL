/**
 * WebSocket message decoding and validation.
 *
 * The backend streams one `StepMessage` JSON document per step
 * (`server/schemas.py`, `docs/contracts/websocket_schema.md`). The
 * functions here are pure and dependency-free so they can be unit
 * tested in isolation: they turn an untrusted frame into a validated
 * `StepMessage` or `null`, never throwing.
 *
 * Validation rejects a whole message when its structure or any drone
 * coordinate is invalid, so the UI keeps the last good snapshot rather
 * than rendering partial or non-finite data.
 */
import type { AgentState } from '../types/agent'
import type { MetricsSnapshot } from '../types/metrics'
import type { StepMessage } from '../types/websocket'

function isFiniteNumber(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value)
}

function isNonNegativeInteger(value: unknown): value is number {
  return typeof value === 'number' && Number.isInteger(value) && value >= 0
}

/** Validate one drone entry; returns `null` if any required field is bad. */
export function parseAgentState(raw: unknown): AgentState | null {
  if (typeof raw !== 'object' || raw === null) {
    return null
  }
  const { id, x, y, z, pitch, yaw } = raw as Record<string, unknown>
  if (!isNonNegativeInteger(id)) {
    return null
  }
  if (!isFiniteNumber(x) || !isFiniteNumber(y) || !isFiniteNumber(z)) {
    return null
  }
  return {
    id,
    x,
    y,
    z,
    pitch: isFiniteNumber(pitch) ? pitch : 0,
    yaw: isFiniteNumber(yaw) ? yaw : 0,
  }
}

/**
 * Validate an already-parsed payload against the `StepMessage`
 * contract. Returns `null` for any malformed message.
 */
export function parseStepMessage(raw: unknown): StepMessage | null {
  if (typeof raw !== 'object' || raw === null) {
    return null
  }
  const {
    step,
    agents: rawAgents,
    newly_visited_cells: rawCells,
    collisions,
    explored_pct,
  } = raw as Record<string, unknown>
  if (!isNonNegativeInteger(step)) {
    return null
  }
  if (!Array.isArray(rawAgents)) {
    return null
  }
  const agents: AgentState[] = []
  for (const entry of rawAgents) {
    const agent = parseAgentState(entry)
    if (agent === null) {
      return null
    }
    agents.push(agent)
  }
  return {
    step,
    agents,
    newly_visited_cells: Array.isArray(rawCells) ? rawCells.filter(isNonNegativeInteger) : [],
    collisions: isNonNegativeInteger(collisions) ? collisions : 0,
    explored_pct: isFiniteNumber(explored_pct) ? explored_pct : 0,
  }
}

/** Decode a raw WebSocket text frame; returns `null` on bad JSON/shape. */
export function decodeStepMessage(data: unknown): StepMessage | null {
  if (typeof data !== 'string') {
    return null
  }
  try {
    return parseStepMessage(JSON.parse(data))
  } catch {
    return null
  }
}

/**
 * Project a `StepMessage` onto the metrics panel contract.
 *
 * The stream carries `step`, `explored_pct`, and per-step `collisions`;
 * `mean_reward` is produced only by the HTTP metrics endpoint, so it is
 * reported as `NaN` (rendered as unavailable) rather than fabricated.
 */
export function messageToMetrics(message: StepMessage): MetricsSnapshot {
  return {
    step: message.step,
    explored_pct: message.explored_pct,
    collisions: message.collisions,
    mean_reward: Number.NaN,
  }
}
