/**
 * WebSocket subscription hook (deferred).
 *
 * Week 1 reserves this module: no connection is opened yet. Later
 * weeks will subscribe to the stream defined in
 * `docs/contracts/websocket_schema.md` and return the latest
 * `StepMessage` from this hook.
 */
import type { StepMessage } from '../types/websocket'

export interface WebSocketStatus {
  connected: boolean
  message: StepMessage | null
}

export function useWebSocket(): WebSocketStatus {
  return { connected: false, message: null }
}
