/**
 * Drone state contract.
 *
 * Mirrors `AgentState` in `server/schemas.py` / `docs/contracts/
 * websocket_schema.md`. The coordinate system is right-handed and
 * y-up: `x`/`z` span the ground plane, `y` is height, angles are
 * radians. Field names intentionally match the JSON wire format.
 */
export interface AgentState {
  id: number
  x: number
  y: number
  z: number
  pitch: number
  yaw: number
}
