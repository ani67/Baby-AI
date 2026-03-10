import { useLoopStore } from '../store/loopStore'

export function StatusBar() {
  const status = useLoopStore(s => s.status)
  const graphStats = status?.graph_summary ?? {}

  const stateColor: Record<string, string> = {
    idle:    'var(--text-dim)',
    running: 'var(--accent)',
    paused:  'var(--warning)',
    error:   'var(--negative)',
  }

  return (
    <div className="status-bar">
      <span style={{ color: stateColor[status?.state ?? 'idle'] }}>
        ● {(status?.state ?? 'IDLE').toUpperCase()}
      </span>
      <span>STEP {status?.step ?? 0}</span>
      <span>STAGE {status?.stage ?? 0}</span>
      <span>CLUSTERS {graphStats.cluster_count ?? 0}</span>
      <span>NODES {graphStats.node_count ?? 0}</span>
      <span>EDGES {graphStats.edge_count ?? 0}</span>
      <span>LAYERS {graphStats.layer_count ?? 0}</span>
      {(graphStats.dormant_count ?? 0) > 0 && (
        <span style={{ color: 'var(--text-dim)' }}>
          {graphStats.dormant_count} dormant
        </span>
      )}
      {status?.state === 'error' && (
        <span style={{ color: 'var(--negative)' }}>
          {status.error_message}
        </span>
      )}
    </div>
  )
}
