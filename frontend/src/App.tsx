/**
 * Application shell: 3D scene plus side panels.
 *
 * `useSwarmState` feeds the scene from the live backend WebSocket
 * stream; the deterministic placeholder swarm only fills the scene
 * before the first valid snapshot arrives.
 */
import AnalyticsPanel from './components/AnalyticsPanel'
import ControlPanel from './components/ControlPanel'
import { DEFAULT_AGENT_COUNT } from './config'
import { useSwarmState } from './hooks/useSwarmState'
import SwarmScene from './scenes/SwarmScene'

export default function App() {
  const swarm = useSwarmState(DEFAULT_AGENT_COUNT)
  const source = swarm.hasLiveData ? 'live stream' : 'placeholder'
  return (
    <div className="app">
      <header className="app-header">
        <h1>SwarmRL</h1>
        <span className="badge">
          {`Week 2 — ${source} (${swarm.statusLabel})`}
        </span>
      </header>
      <main className="app-main">
        <section className="scene-panel" aria-label="3D swarm scene">
          <SwarmScene agents={swarm.agents} />
        </section>
        <aside className="side-panel">
          <ControlPanel
            agentCount={swarm.agentCount}
            metrics={swarm.metrics}
            connected={swarm.connected}
            statusLabel={swarm.statusLabel}
          />
          <AnalyticsPanel metrics={swarm.metrics} />
        </aside>
      </main>
    </div>
  )
}
