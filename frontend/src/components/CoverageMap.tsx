/**
 * CoverageMap: ground-plane representation of the disaster zone.
 *
 * Week 1 renders the bounded ground, a cell-sized grid, and a few
 * static debris blocks. The coverage heatmap (fed by
 * `newly_visited_cells`) is later-week work.
 */
import { COVERAGE_CELL_SIZE } from '../config'

export interface CoverageMapProps {
  worldSize: [number, number, number]
  cellSize?: number
}

const DEBRIS: { position: [number, number, number]; size: [number, number, number] }[] = [
  { position: [25, 1.5, 30], size: [8, 3, 6] },
  { position: [70, 2, 40], size: [10, 4, 5] },
  { position: [45, 1, 75], size: [6, 2, 9] },
  { position: [85, 2.5, 80], size: [7, 5, 7] },
  { position: [15, 1.5, 70], size: [5, 3, 5] },
]

export default function CoverageMap({
  worldSize,
  cellSize = COVERAGE_CELL_SIZE,
}: CoverageMapProps) {
  const [xSize, , zSize] = worldSize
  const divisions = Math.max(1, Math.round(xSize / cellSize))
  return (
    <group name="coverage-map">
      <mesh
        rotation={[-Math.PI / 2, 0, 0]}
        position={[xSize / 2, 0, zSize / 2]}
        receiveShadow
      >
        <planeGeometry args={[xSize, zSize]} />
        <meshStandardMaterial color="#1b2431" />
      </mesh>
      <gridHelper
        args={[xSize, divisions, '#334155', '#24344a']}
        position={[xSize / 2, 0.02, zSize / 2]}
      />
      <mesh position={[xSize / 2, 0.01, zSize / 2]} rotation={[-Math.PI / 2, 0, 0]}>
        <planeGeometry args={[xSize, zSize]} />
        <meshBasicMaterial color="#38b2ac" transparent opacity={0.05} />
      </mesh>
      {DEBRIS.map((block) => (
        <mesh key={`${block.position[0]}-${block.position[2]}`} position={block.position}>
          <boxGeometry args={block.size} />
          <meshStandardMaterial color="#4a5568" roughness={0.9} />
        </mesh>
      ))}
    </group>
  )
}
