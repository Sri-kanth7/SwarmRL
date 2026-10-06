/**
 * Shared frontend constants.
 *
 * Values mirror `configs/project.yaml` and the environment defaults:
 * the world is a y-up box of (x, y, z) = (100, 50, 100) containing
 * 50 drones, streamed/mock-stepped at the backend tick rate.
 */

export const WORLD_SIZE: [number, number, number] = [100, 50, 100]

export const DEFAULT_AGENT_COUNT = 50

export const COVERAGE_CELL_SIZE = 5

export const CAMERA_POSITION: [number, number, number] = [150, 120, 150]

export const CAMERA_TARGET: [number, number, number] = [50, 0, 50]
