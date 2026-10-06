/**
 * SwarmScene: the react-three-fiber canvas hosting the disaster
 * zone, the drone swarm, and optional vision-cone prototypes.
 *
 * Week 1 renders a static snapshot of dummy drones. Live WebSocket
 * animation, trails, and heatmap overlays arrive in later weeks.
 */
import { Canvas } from '@react-three/fiber'
import CoverageMap from '../components/CoverageMap'
import DroneSwarm from '../components/DroneSwarm'
import VisionCone from '../components/VisionCone'
import { CAMERA_POSITION, CAMERA_TARGET, WORLD_SIZE } from '../config'
import type { AgentState } from '../types/agent'

export interface SwarmSceneProps {
  agents: AgentState[]
  /** Render vision-cone prototypes (off by default in Week 1). */
  showVisionCones?: boolean
}

export default function SwarmScene({
  agents,
  showVisionCones = false,
}: SwarmSceneProps) {
  return (
    <Canvas
      camera={{ position: CAMERA_POSITION, fov: 50, near: 0.1, far: 2000 }}
      gl={{ antialias: true }}
      onCreated={({ camera }) => {
        camera.lookAt(CAMERA_TARGET[0], CAMERA_TARGET[1], CAMERA_TARGET[2])
        camera.updateProjectionMatrix()
      }}
    >
      <color attach="background" args={['#0b0f14']} />
      <fog attach="fog" args={['#0b0f14', 250, 700]} />
      <ambientLight intensity={0.65} />
      <directionalLight position={[120, 180, 60]} intensity={1.1} />
      <CoverageMap worldSize={WORLD_SIZE} />
      <DroneSwarm agents={agents} />
      {showVisionCones
        ? agents.map((agent) => <VisionCone key={`cone-${agent.id}`} agent={agent} />)
        : null}
    </Canvas>
  )
}
