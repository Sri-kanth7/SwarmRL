/**
 * Single drone prototype.
 *
 * The drone body is a cone pointing along local `+x`; yaw is applied
 * as a negative rotation about `+y` and pitch as a rotation about
 * local `+z`, which matches the environment's y-up heading formula
 * `forward = (cos(pitch)cos(yaw), sin(pitch), cos(pitch)sin(yaw))`.
 */
import type { AgentState } from '../types/agent'

export interface DroneProps {
  agent: AgentState
  color?: string
  scale?: number
}

export default function Drone({
  agent,
  color = '#4fd1c5',
  scale = 1.5,
}: DroneProps) {
  return (
    <group
      name={`drone-${agent.id}`}
      position={[agent.x, agent.y, agent.z]}
      rotation={[0, -agent.yaw, 0]}
      scale={scale}
    >
      <group rotation={[0, 0, agent.pitch]}>
        <mesh rotation={[0, 0, -Math.PI / 2]}>
          <coneGeometry args={[0.5, 1.8, 12]} />
          <meshStandardMaterial color={color} metalness={0.2} roughness={0.5} />
        </mesh>
        <mesh>
          <boxGeometry args={[0.15, 1.6, 0.15]} />
          <meshStandardMaterial color="#cbd5e1" metalness={0.4} roughness={0.4} />
        </mesh>
      </group>
    </group>
  )
}
