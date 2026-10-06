/**
 * DroneSwarm: renders every drone of the current snapshot.
 */
import type { AgentState } from '../types/agent'
import Drone from './Drone'

export interface DroneSwarmProps {
  agents: AgentState[]
  color?: string
}

export default function DroneSwarm({ agents, color }: DroneSwarmProps) {
  return (
    <group name="drone-swarm">
      {agents.map((agent) => (
        <Drone key={agent.id} agent={agent} color={color} />
      ))}
    </group>
  )
}
