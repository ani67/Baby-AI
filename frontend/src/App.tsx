import { StatusBar } from './components/StatusBar'
import { LatentSpace } from './components/LatentSpace'
import { DialogueFeed } from './components/DialogueFeed'
import { HumanChat } from './components/HumanChat'
import { Controls } from './components/Controls'
import { MetricsPanel } from './components/MetricsPanel'
import { useWebSocket } from './hooks/useWebSocket'
import { WS_URL } from './lib/constants'

export function App() {
  useWebSocket(WS_URL)

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      {/* Top bar: status left, controls right */}
      <div style={{ display: 'flex', borderBottom: '1px solid var(--border)' }}>
        <div style={{ flex: 1 }}>
          <StatusBar />
        </div>
        <Controls />
      </div>

      {/* Main content: latent space left, dialogue + chat right */}
      <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
        {/* Left panel: 60% */}
        <div style={{ width: '60%', height: '100%', position: 'relative' }}>
          <LatentSpace />
          <MetricsPanel />
        </div>

        {/* Right panel: 40%, split 60/40 vertically */}
        <div style={{
          width: '40%',
          height: '100%',
          display: 'flex',
          flexDirection: 'column',
          borderLeft: '1px solid var(--border)',
        }}>
          <div style={{ height: '60%', overflow: 'hidden' }}>
            <DialogueFeed />
          </div>
          <div style={{
            height: '40%',
            overflow: 'hidden',
            borderTop: '1px solid var(--border)',
          }}>
            <HumanChat />
          </div>
        </div>
      </div>
    </div>
  )
}
