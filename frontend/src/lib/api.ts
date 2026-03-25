import { BASE_URL } from './constants'

export const api = {
  start:    () => fetch(`${BASE_URL}/start`, { method: 'POST' }).then(r => r.json()),
  pause:    () => fetch(`${BASE_URL}/pause`, { method: 'POST' }).then(r => r.json()),
  resume:   () => fetch(`${BASE_URL}/resume`, { method: 'POST' }).then(r => r.json()),
  step:     () => fetch(`${BASE_URL}/step`, { method: 'POST' }).then(r => r.json()),
  reset: (notes: {
    architecture_state: string
    signal_quality: string
    why_reset: string
    what_was_learned: string
  }) =>
    fetch(`${BASE_URL}/reset`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(notes),
    }).then(r => {
      if (!r.ok) throw new Error(`HTTP ${r.status}`)
      return r.json()
    }),
  status:   () => fetch(`${BASE_URL}/status`).then(r => r.json()),
  snapshot: () => fetch(`${BASE_URL}/snapshot`).then(r => r.json()),

  setStage: (stage: number) =>
    fetch(`${BASE_URL}/stage`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ stage }),
    }).then(r => r.json()),

  setSpeed: (delay_ms: number) =>
    fetch(`${BASE_URL}/speed`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ delay_ms }),
    }).then(r => r.json()),

  chat: (message: string) =>
    fetch(`${BASE_URL}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message }),
    }).then(r => r.json()),

  uploadImage: (file: File, label?: string) => {
    const form = new FormData()
    form.append('file', file)
    if (label) form.append('label', label)
    return fetch(`${BASE_URL}/image`, { method: 'POST', body: form }).then(r => r.json())
  },

  uploadImageUrl: (url: string, label?: string) =>
    fetch(`${BASE_URL}/image-url`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url, label: label || null }),
    }).then(r => {
      if (!r.ok) throw new Error(`HTTP ${r.status}`)
      return r.json()
    }),

  uploadImagesBulk: (urls: string[]) =>
    fetch(`${BASE_URL}/images-bulk`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ urls }),
    }).then(r => {
      if (!r.ok) throw new Error(`HTTP ${r.status}`)
      return r.json()
    }),

  clusterLabels: () =>
    fetch(`${BASE_URL}/clusters/labels`).then(r => r.json()) as Promise<{ labels: Record<string, string[]> }>,

  clusterCofiring: () =>
    fetch(`${BASE_URL}/clusters/cofiring`).then(r => r.json()) as Promise<{
      pairs: Array<{ a: string; b: string; count: number; strength: number }>
    }>,

  clusterTree: () =>
    fetch(`${BASE_URL}/clusters/tree`).then(r => r.json()) as Promise<{
      nodes: Array<{
        id: string
        parent: string | null
        depth: number
        dormant: boolean
        cluster_type: string | null
        labels: string[]
        phantom: boolean
      }>
      edges: Array<{ source: string; target: string }>
      max_depth: number
    }>,

  dashboard: () =>
    fetch(`${BASE_URL}/dashboard`).then(r => r.json()) as Promise<{
      step: number
      clusters: number
      layers: number
      edges: number
      nodes: number
      growth_rate: number
      edge_ratio: number
      categories: {
        best: Array<{ category: string; total: number; positive: number; avg_sim: number; last_step: number }>
        worst: Array<{ category: string; total: number; positive: number; avg_sim: number; last_step: number }>
        total_tracked: number
      }
      structure: {
        spatial_score: number | null
        sibling_coherence: number
        layer_diversity: Record<number, number>
        cofiring_communities: number
        community_sizes: number[]
      }
      memory_buffer: { norm: number; decay: number; weight: number; top_k: number }
    }>,
}
