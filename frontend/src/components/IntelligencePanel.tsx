import { useEffect, useState } from 'react'
import { api } from '../lib/api'

type Metrics = Awaited<ReturnType<typeof api.metrics>>

function pct(v: number | null): string {
  return v != null ? `${(v * 100).toFixed(0)}%` : '---'
}

function sim(v: number | null): string {
  return v != null ? v.toFixed(3) : '---'
}

function bar(v: number | null, max: number = 1): string {
  if (v == null) return '░░░░░░░░░░'
  const filled = Math.round((v / max) * 10)
  return '█'.repeat(Math.max(0, filled)) + '░'.repeat(Math.max(0, 10 - filled))
}

function trendArrow(v: number | null): string {
  if (v == null) return ''
  if (v > 0.01) return ' ↑'
  if (v < -0.01) return ' ↓'
  return ' →'
}

export function IntelligencePanel() {
  const [data, setData] = useState<Metrics | null>(null)
  const [collapsed, setCollapsed] = useState(false)

  useEffect(() => {
    let active = true
    const poll = () => {
      api.metrics()
        .then(d => { if (active) setData(d) })
        .catch(() => {})
    }
    poll()
    const id = setInterval(poll, 3000)
    return () => { active = false; clearInterval(id) }
  }, [])

  if (collapsed) {
    return (
      <button
        className="metrics-toggle"
        style={{ right: 'auto', left: 8, bottom: 8 }}
        onClick={() => setCollapsed(false)}
      >
        I
      </button>
    )
  }

  const d = data?.distillation
  const r = data?.reasoning
  const g = data?.generation

  return (
    <div className="metrics-panel" style={{ right: 'auto', left: 8, bottom: 8 }}>
      <div className="legend-header">
        <span>INTELLIGENCE</span>
        <button className="legend-close" onClick={() => setCollapsed(true)}>x</button>
      </div>

      <div className="legend-section">
        <div className="legend-title">Distillation (CLIP → Native)</div>
        <div className="metrics-row">
          <span className="metrics-label">text</span>
          <span className={`metrics-value ${(d?.text_cosine_sim ?? 0) > 0.5 ? 'good' : ''}`}>
            {sim(d?.text_cosine_sim ?? null)}{trendArrow(d?.text_cosine_sim_trend ?? null)}
          </span>
        </div>
        <div className="metrics-row">
          <span className="metrics-label"></span>
          <span className="metrics-value dim" style={{ fontFamily: 'monospace', letterSpacing: 1 }}>
            {bar(d?.text_cosine_sim ?? null)} {d?.text_samples ?? 0}x
          </span>
        </div>
        <div className="metrics-row">
          <span className="metrics-label">vision</span>
          <span className={`metrics-value ${(d?.vision_cosine_sim ?? 0) > 0.5 ? 'good' : ''}`}>
            {sim(d?.vision_cosine_sim ?? null)}
          </span>
        </div>
        <div className="metrics-row">
          <span className="metrics-label"></span>
          <span className="metrics-value dim" style={{ fontFamily: 'monospace', letterSpacing: 1 }}>
            {bar(d?.vision_cosine_sim ?? null)} {d?.vision_samples ?? 0}x
          </span>
        </div>
      </div>

      <div className="legend-section">
        <div className="legend-title">Reasoning ({r?.total_tasks ?? 0} tasks)</div>
        {r && r.total_tasks > 0 ? (
          <>
            <div className="metrics-row">
              <span className="metrics-label">overall</span>
              <span className={`metrics-value ${(r.overall_accuracy ?? 0) > 0.6 ? 'good' : ''}`}>
                {pct(r.overall_accuracy)}
              </span>
            </div>
            {r.comparison_accuracy != null && (
              <div className="metrics-row">
                <span className="metrics-label">compare</span>
                <span className="metrics-value dim">{pct(r.comparison_accuracy)}</span>
              </div>
            )}
            {r.sequence_accuracy != null && (
              <div className="metrics-row">
                <span className="metrics-label">sequence</span>
                <span className="metrics-value dim">{pct(r.sequence_accuracy)}</span>
              </div>
            )}
            {r.analogy_accuracy != null && (
              <div className="metrics-row">
                <span className="metrics-label">analogy</span>
                <span className="metrics-value dim">{pct(r.analogy_accuracy)}</span>
              </div>
            )}
            {r.memory_retrieval_accuracy != null && (
              <div className="metrics-row">
                <span className="metrics-label">memory</span>
                <span className="metrics-value dim">{pct(r.memory_retrieval_accuracy)}</span>
              </div>
            )}
            {r.odd_one_out_accuracy != null && (
              <div className="metrics-row">
                <span className="metrics-label">odd one out</span>
                <span className="metrics-value dim">{pct(r.odd_one_out_accuracy)}</span>
              </div>
            )}
          </>
        ) : (
          <div className="metrics-row">
            <span className="metrics-label dim">waiting for tasks...</span>
          </div>
        )}
      </div>

      <div className="legend-section">
        <div className="legend-title">Generation</div>
        <div className="metrics-row">
          <span className="metrics-label">vocab</span>
          <span className="metrics-value">{g?.vocab_size ?? 0} words</span>
        </div>
        <div className="metrics-row">
          <span className="metrics-label">relevance</span>
          <span className={`metrics-value ${(g?.response_relevance ?? 0) > 0.3 ? 'good' : ''}`}>
            {sim(g?.response_relevance ?? null)}
          </span>
        </div>
      </div>
    </div>
  )
}
