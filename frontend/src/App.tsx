/**
 * Application shell: 3D scene plus side panels.
 *
 * Week 1 renders the dummy swarm snapshot produced by
 * `useSwarmState`. Connecting the WebSocket stream swaps the data
 * source without changing this layout.
 */
import AnalyticsPanel from './components/AnalyticsPanel'
import ControlPanel from './components/ControlPanel'
import { DEFAULT_AGENT_COUNT } from './config'
import { useSwarmState } from './hooks/useSwarmState'
import SwarmScene from './scenes/SwarmScene'

export default function App() {
  const swarm = useSwarmState(DEFAULT_AGENT_COUNT)
  return (
    <div className="app">
      <header className="app-header">
        <h1>SwarmRL</h1>
        <span className="badge">Week 1 foundation — mock state</span>
      </header>
      <main className="app-main">
        <section className="scene-panel" aria-label="3D swarm scene">
          <SwarmScene agents={swarm.agents} />
        </section>
        <aside className="side-panel">
          <ControlPanel agentCount={swarm.agents.length} metrics={swarm.metrics} />
          <AnalyticsPanel metrics={swarm.metrics} />
        </aside>
      </main>
    </div>
  )
}
