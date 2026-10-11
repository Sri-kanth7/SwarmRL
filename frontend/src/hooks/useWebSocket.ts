/**
 * WebSocket subscription hook for the backend state stream.
 *
 * Opens exactly one connection while the owning component is mounted,
 * decodes incoming frames with `decodeStepMessage`, and exposes the
 * latest valid `StepMessage` plus the connection status. The socket,
 * event handlers, and any pending reconnect timer are cleaned up when
 * the component unmounts, and a bounded reconnect policy avoids an
 * unbounded retry loop.
 */
import { useEffect, useMemo, useRef, useState } from 'react'
import { decodeStepMessage } from '../services/messages'
import { resolveWebSocketUrl } from '../services/websocket'
import type { StepMessage } from '../types/websocket'

export type WebSocketStatusKind = 'connecting' | 'open' | 'closed' | 'error'

export interface WebSocketState {
  /** True while the socket is open. */
  connected: boolean
  status: WebSocketStatusKind
  /** Latest frame that passed validation, or `null`. */
  message: StepMessage | null
  error: string | null
  /** Resolved endpoint the hook connects to. */
  url: string
}

export interface UseWebSocketOptions {
  /** Explicit endpoint; defaults to `resolveWebSocketUrl()`. */
  url?: string
  /** When false, no connection is opened (useful for tests/SSR). */
  enabled?: boolean
  /** Maximum reconnect attempts after an unexpected close. */
  maxReconnectAttempts?: number
  /** Base delay between reconnect attempts, in milliseconds. */
  reconnectDelayMs?: number
}

const DEFAULT_MAX_RECONNECT_ATTEMPTS = 5
const DEFAULT_RECONNECT_DELAY_MS = 1000

/** Human-readable label for a connection status. */
export function describeWebSocketStatus(status: WebSocketStatusKind): string {
  switch (status) {
    case 'open':
      return 'streaming'
    case 'connecting':
      return 'connecting'
    case 'error':
      return 'connection error'
    case 'closed':
      return 'disconnected'
    default:
      return 'disconnected'
  }
}

export function useWebSocket(options: UseWebSocketOptions = {}): WebSocketState {
  const {
    url: explicitUrl,
    enabled = true,
    maxReconnectAttempts = DEFAULT_MAX_RECONNECT_ATTEMPTS,
    reconnectDelayMs = DEFAULT_RECONNECT_DELAY_MS,
  } = options

  const url = useMemo(() => resolveWebSocketUrl(explicitUrl), [explicitUrl])
  const [message, setMessage] = useState<StepMessage | null>(null)
  const [status, setStatus] = useState<WebSocketStatusKind>('connecting')
  const [error, setError] = useState<string | null>(null)

  const socketRef = useRef<WebSocket | null>(null)
  const attemptsRef = useRef(0)
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const cancelledRef = useRef(false)

  useEffect(() => {
    if (!enabled || typeof window === 'undefined') {
      return undefined
    }
    if (typeof WebSocket === 'undefined') {
      setStatus('error')
      setError('WebSocket is not available in this environment')
      return undefined
    }

    cancelledRef.current = false
    attemptsRef.current = 0

    function clearTimer(): void {
      if (timerRef.current !== null) {
        clearTimeout(timerRef.current)
        timerRef.current = null
      }
    }

    function scheduleReconnect(): void {
      if (cancelledRef.current || attemptsRef.current >= maxReconnectAttempts) {
        return
      }
      attemptsRef.current += 1
      clearTimer()
      timerRef.current = setTimeout(connect, reconnectDelayMs * attemptsRef.current)
    }

    function connect(): void {
      if (cancelledRef.current) {
        return
      }
      setStatus('connecting')
      let socket: WebSocket
      try {
        socket = new WebSocket(url)
      } catch (caught) {
        setStatus('error')
        setError(caught instanceof Error ? caught.message : 'failed to open WebSocket')
        scheduleReconnect()
        return
      }
      socketRef.current = socket

      socket.onopen = () => {
        attemptsRef.current = 0
        setStatus('open')
        setError(null)
      }

      socket.onmessage = (event: MessageEvent) => {
        const decoded = decodeStepMessage(event.data)
        if (decoded !== null) {
          setMessage(decoded)
        }
      }

      socket.onerror = () => {
        setStatus('error')
        setError('WebSocket connection error')
      }

      socket.onclose = (event: CloseEvent) => {
        socketRef.current = null
        if (cancelledRef.current) {
          return
        }
        setStatus('closed')
        if (!event.wasClean) {
          setError('WebSocket closed unexpectedly')
        }
        scheduleReconnect()
      }
    }

    connect()

    return () => {
      cancelledRef.current = true
      clearTimer()
      const socket = socketRef.current
      socketRef.current = null
      if (socket !== null) {
        socket.onopen = null
        socket.onmessage = null
        socket.onerror = null
        socket.onclose = null
        if (
          socket.readyState === WebSocket.OPEN ||
          socket.readyState === WebSocket.CONNECTING
        ) {
          socket.close()
        }
      }
    }
  }, [url, enabled, maxReconnectAttempts, reconnectDelayMs])

  return { connected: status === 'open', status, message, error, url }
}
