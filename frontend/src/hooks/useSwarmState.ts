/**
 * Swarm state for the visualization.
 *
 * Composes the WebSocket hook with a deterministic placeholder swarm:
 * live drone positions from the backend stream are used as soon as the
 * first valid `StepMessage` arrives, and the placeholder set only fills
 * the scene before the first frame (it never animates or masks a
 * disconnect). Metrics come straight from the stream; values the
 * stream does not carry are reported as unavailable.
 */
import { useMemo } from 'react'
import { DEFAULT_AGENT_COUNT, WORLD_SIZE } from '../config'
import { messageToMetrics } from '../services/messages'
import type { AgentState } from '../types/agent'
import type { MetricsSnapshot } from '../types/metrics'
import { describeWebSocketStatus, useWebSocket, type WebSocketStatusKind } from './useWebSocket'

export interface SwarmSnapshot {
  step: number
  agents: AgentState[]
  metrics: MetricsSnapshot
  connected: boolean
  status: WebSocketStatusKind
  statusLabel: string
  /** True once at least one valid streamed snapshot has been received. */
  hasLiveData: boolean
  agentCount: number
}

function mulberry32(seed: number): () => number {
  let state = seed >>> 0
  return () => {
    state = (state + 0x6d2b79f5) >>> 0
    let t = state
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

export function generateDummyAgents(
  count: number = DEFAULT_AGENT_COUNT,
  seed: number = 42
): AgentState[] {
  const random = mulberry32(seed)
  const [xSize, ySize, zSize] = WORLD_SIZE
  const margin = 5
  const agents: AgentState[] = []
  for (let i = 0; i < count; i += 1) {
    agents.push({
      id: i,
      x: margin + random() * (xSize - 2 * margin),
      y: margin + random() * (ySize - 2 * margin),
      z: margin + random() * (zSize - 2 * margin),
      pitch: (random() - 0.5) * 0.4,
      yaw: random() * Math.PI * 2 - Math.PI,
    })
  }
  return agents
}

export function useSwarmState(
  agentCount: number = DEFAULT_AGENT_COUNT,
  seed: number = 42
): SwarmSnapshot {
  const stream = useWebSocket()
  const placeholder = useMemo(
    () => generateDummyAgents(agentCount, seed),
    [agentCount, seed]
  )

  const liveAgents = stream.message?.agents ?? null
  const agents = liveAgents ?? placeholder
  const metrics: MetricsSnapshot = stream.message
    ? messageToMetrics(stream.message)
    : { step: 0, explored_pct: 0, collisions: 0, mean_reward: 0 }

  return {
    step: stream.message?.step ?? 0,
    agents,
    metrics,
    connected: stream.connected,
    status: stream.status,
    statusLabel: describeWebSocketStatus(stream.status),
    hasLiveData: stream.message !== null,
    agentCount: agents.length,
  }
}
