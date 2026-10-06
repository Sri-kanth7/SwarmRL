/**
 * AnalyticsPanel: text read-out of the metrics contract.
 *
 * Charts and historical plots are later-week work; Week 1 only
 * displays the current values of `explored_pct`, `collisions`,
 * and `mean_reward`.
 */
import type { MetricsSnapshot } from '../types/metrics'

export interface AnalyticsPanelProps {
  metrics: MetricsSnapshot
}

function format(value: number, digits = 2): string {
  return Number.isFinite(value) ? value.toFixed(digits) : '—'
}

export default function AnalyticsPanel({ metrics }: AnalyticsPanelProps) {
  return (
    <section className="panel" aria-label="analytics panel">
      <h2>Metrics</h2>
      <dl>
        <dt>Explored %</dt>
        <dd>{format(metrics.explored_pct)}</dd>
        <dt>Collisions</dt>
        <dd>{Math.max(0, Math.round(metrics.collisions))}</dd>
        <dt>Mean reward</dt>
        <dd>{format(metrics.mean_reward)}</dd>
        <dt>Step</dt>
        <dd>{Math.max(0, Math.round(metrics.step))}</dd>
      </dl>
      <p className="hint">Metrics source: backend snapshot contract.</p>
    </section>
  )
}
