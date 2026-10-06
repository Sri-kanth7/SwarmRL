/**
 * WebSocket endpoint resolution for the backend stream.
 *
 * Week 1 only defines where the stream lives (see
 * `docs/contracts/websocket_schema.md`); subscribing to live state
 * is later work. During development the Vite proxy forwards `/ws`
 * to the backend (see `vite.config.js`).
 */

export const DEFAULT_WS_PATH = '/ws'

export function resolveWebSocketUrl(explicit?: string): string {
  if (explicit) {
    return explicit
  }
  const fromEnv = import.meta.env.VITE_SWARMRL_WS_URL as string | undefined
  if (fromEnv) {
    return fromEnv
  }
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  return `${protocol}//${window.location.host}${DEFAULT_WS_PATH}`
}
