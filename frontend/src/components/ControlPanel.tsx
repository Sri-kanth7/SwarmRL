/**
 * ControlPanel: basic runtime status and reserved controls.
 *
 * Week 1 shows configuration/state read-outs only; buttons are
 * placeholders because streaming controls (pause, speed, reset)
 * are wired in a later week.
 */
import type { MetricsSnapshot } from '../types/metrics'

export interface ControlPanelProps {
  agentCount: number
  metrics: MetricsSnapshot
  mode?: string
  connected?: boolean
  statusLabel?: string
}

export default function ControlPanel({
  agentCount,
  metrics,
  mode = 'mock',
  connected = false,
  statusLabel = 'disconnected',
}: ControlPanelProps) {
  return (
    <section className="panel" aria-label="control panel">
      <h2>Control</h2>
      <dl>
        <dt>Mode</dt>
        <dd>{mode}</dd>
        <dt>Agents</dt>
        <dd>{agentCount}</dd>
        <dt>Step</dt>
        <dd>{Number.isFinite(metrics.step) ? metrics.step : '—'}</dd>
        <dt>Stream</dt>
        <dd>{connected ? 'connected' : statusLabel}</dd>
      </dl>
      <div className="panel-actions">
        <button type="button" disabled>
          Pause
        </button>
        <button type="button" disabled>
          Reset
        </button>
        <button type="button" disabled>
          Connect
        </button>
      </div>
      <p className="hint">Live streaming is read-only; controls arrive in a later week.</p>
    </section>
  )
}
