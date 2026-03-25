import { useEffect, useState } from 'react'
import { api } from '../lib/api'

type Dashboard = Awaited<ReturnType<typeof api.dashboard>>

export function MetricsPanel() {
  const [data, setData] = useState<Dashboard | null>(null)
  const [collapsed, setCollapsed] = useState(false)

  useEffect(() => {
    let active = true
    const poll = () => {
      api.dashboard()
        .then(d => { if (active) setData(d) })
        .catch(() => {})
    }
    poll()
    const id = setInterval(poll, 5000)
    return () => { active = false; clearInterval(id) }
  }, [])

  if (!data) return null

  if (collapsed) {
    return (
      <button className="metrics-toggle" onClick={() => setCollapsed(false)}>
        M
      </button>
    )
  }

  const s = data.structure
  const b = data.memory_buffer
  const bestCat = data.categories.best?.[data.categories.best.length - 1]
  const worstCat = data.categories.worst?.[0]

  return (
    <div className="metrics-panel">
      <div className="legend-header">
        <span>METRICS</span>
        <button className="legend-close" onClick={() => setCollapsed(true)}>x</button>
      </div>

      <div className="legend-section">
        <div className="legend-title">Structure</div>
        <div className="metrics-row">
          <span className="metrics-label">spatial</span>
          <span className={`metrics-value ${(s.spatial_score ?? 0) > 0.15 ? 'good' : ''}`}>
            {s.spatial_score?.toFixed(3) ?? '---'}
          </span>
        </div>
        <div className="metrics-row">
          <span className="metrics-label">communities</span>
          <span className={`metrics-value ${s.cofiring_communities > 1 ? 'good' : ''}`}>
            {s.cofiring_communities}
          </span>
        </div>
        {s.community_sizes.length > 0 && (
          <div className="metrics-row">
            <span className="metrics-label">sizes</span>
            <span className="metrics-value dim">
              {s.community_sizes.slice(0, 4).join(', ')}
            </span>
          </div>
        )}
        <div className="metrics-row">
          <span className="metrics-label">sibling coh.</span>
          <span className="metrics-value">{s.sibling_coherence.toFixed(1)}</span>
        </div>
      </div>

      <div className="legend-section">
        <div className="legend-title">Buffer</div>
        <div className="metrics-row">
          <span className="metrics-label">norm</span>
          <span className="metrics-value">{b.norm.toFixed(1)}</span>
        </div>
        <div className="metrics-row">
          <span className="metrics-label">decay / weight</span>
          <span className="metrics-value dim">{b.decay} / {b.weight}</span>
        </div>
      </div>

      <div className="legend-section">
        <div className="legend-title">Categories ({data.categories.total_tracked})</div>
        {bestCat && (
          <div className="metrics-row">
            <span className="metrics-label best">{bestCat.category}</span>
            <span className="metrics-value">{bestCat.avg_sim.toFixed(3)}</span>
          </div>
        )}
        {worstCat && (
          <div className="metrics-row">
            <span className="metrics-label worst">{worstCat.category}</span>
            <span className="metrics-value">{worstCat.avg_sim.toFixed(3)}</span>
          </div>
        )}
      </div>

      <div className="legend-section">
        <div className="legend-title">Growth</div>
        <div className="metrics-row">
          <span className="metrics-label">rate</span>
          <span className="metrics-value">{data.growth_rate.toFixed(1)}/1K</span>
        </div>
        <div className="metrics-row">
          <span className="metrics-label">edge ratio</span>
          <span className="metrics-value">{data.edge_ratio.toFixed(1)}/cluster</span>
        </div>
      </div>
    </div>
  )
}
