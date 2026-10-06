/**
 * Dummy swarm state for the Week 1 foundation.
 *
 * Produces a deterministic set of drones so the 3D scene and panels
 * can be developed without a backend connection. Live WebSocket
 * state replaces this hook once integration lands.
 */
import { useMemo } from 'react'
import { DEFAULT_AGENT_COUNT, WORLD_SIZE } from '../config'
import type { AgentState } from '../types/agent'
import type { MetricsSnapshot } from '../types/metrics'

export interface SwarmSnapshot {
  step: number
  agents: AgentState[]
  metrics: MetricsSnapshot
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
  return useMemo(() => {
    const agents = generateDummyAgents(agentCount, seed)
    return {
      step: 0,
      agents,
      metrics: {
        step: 0,
        explored_pct: 0,
        collisions: 0,
        mean_reward: 0,
      },
    }
  }, [agentCount, seed])
}
