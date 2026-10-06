/**
 * VisionCone: minimal prototype of a drone's sensor footprint.
 *
 * An open cone extends from the drone's position along its heading.
 * This is a placeholder for the fuller sensor visualization of
 * later weeks (LiDAR rays, detection highlighting).
 */
import { DoubleSide } from 'three'
import type { AgentState } from '../types/agent'

export interface VisionConeProps {
  agent: AgentState
  /** Cone length in world units. */
  range?: number
  /** Full cone angle in radians. */
  angle?: number
  color?: string
}

export default function VisionCone({
  agent,
  range = 12,
  angle = 0.7,
  color = '#81e6d9',
}: VisionConeProps) {
  const radius = Math.tan(angle / 2) * range
  return (
    <group
      name={`vision-cone-${agent.id}`}
      position={[agent.x, agent.y, agent.z]}
      rotation={[0, -agent.yaw, 0]}
    >
      <group rotation={[0, 0, agent.pitch]}>
        <mesh position={[range / 2, 0, 0]} rotation={[0, 0, -Math.PI / 2]}>
          <coneGeometry args={[radius, range, 20, 1, true]} />
          <meshStandardMaterial
            color={color}
            transparent
            opacity={0.18}
            side={DoubleSide}
            depthWrite={false}
          />
        </mesh>
      </group>
    </group>
  )
}
