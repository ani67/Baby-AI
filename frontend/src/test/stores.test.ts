import { describe, it, expect, beforeEach } from 'vitest'
import { useGraphStore } from '../store/graphStore'
import { useDialogueStore } from '../store/dialogueStore'
import { useLoopStore } from '../store/loopStore'

describe('graphStore', () => {
  beforeEach(() => {
    useGraphStore.setState({
      nodes: [],
      clusters: [],
      edges: [],
      activations: {},
      growthQueue: [],
    })
  })

  it('setSnapshot replaces all state', () => {
    const store = useGraphStore.getState()
    store.setSnapshot({
      nodes: [
        { id: 'n1', cluster: 'c1', pos: [1, 2, 3], activation_mean: 0.5, alive: true },
      ],
      clusters: [
        { id: 'c1', cluster_type: 'integration', layer_index: 0, dormant: false },
      ],
      edges: [
        { from: 'c1', to: 'c2', strength: 0.8 },
      ],
    })

    const state = useGraphStore.getState()
    expect(state.nodes).toHaveLength(1)
    expect(state.nodes[0].id).toBe('n1')
    expect(state.clusters).toHaveLength(1)
    expect(state.edges).toHaveLength(1)
  })

  it('applyDiff adds and removes nodes', () => {
    const store = useGraphStore.getState()
    store.setSnapshot({
      nodes: [
        { id: 'n1', cluster: 'c1', pos: [0, 0, 0], activation_mean: 0.3, alive: true },
        { id: 'n2', cluster: 'c1', pos: [1, 1, 1], activation_mean: 0.4, alive: true },
      ],
      clusters: [
        { id: 'c1', cluster_type: 'routing', layer_index: 0, dormant: false },
      ],
      edges: [],
    })

    store.applyDiff({
      nodes_added: [
        { id: 'n3', cluster: 'c1', pos: [2, 2, 2], activation_mean: 0.1, alive: true },
      ],
      nodes_removed: [{ id: 'n1' }],
      clusters_added: [],
      clusters_removed: [],
      clusters_updated: [],
      edges_added: [],
      edges_removed: [],
      edges_updated: [],
    }, false)

    const state = useGraphStore.getState()
    expect(state.nodes).toHaveLength(2)
    expect(state.nodes.find(n => n.id === 'n1')).toBeUndefined()
    expect(state.nodes.find(n => n.id === 'n3')).toBeDefined()
  })

  it('applyDiff updates edge strengths', () => {
    const store = useGraphStore.getState()
    store.setSnapshot({
      nodes: [],
      clusters: [],
      edges: [{ from: 'c1', to: 'c2', strength: 0.5 }],
    })

    store.applyDiff({
      nodes_added: [],
      nodes_removed: [],
      clusters_added: [],
      clusters_removed: [],
      clusters_updated: [],
      edges_added: [],
      edges_removed: [],
      edges_updated: [{ from: 'c1', to: 'c2', strength: 0.9 }],
    }, false)

    const state = useGraphStore.getState()
    expect(state.edges[0].strength).toBe(0.9)
  })

  it('setActivations replaces activations', () => {
    const store = useGraphStore.getState()
    store.setActivations({ c1: 0.7, c2: 0.3 })
    expect(useGraphStore.getState().activations).toEqual({ c1: 0.7, c2: 0.3 })
  })

  it('triggerGrowthEvents appends to queue', () => {
    const store = useGraphStore.getState()
    store.triggerGrowthEvents([{ type: 'BUD', from: 'c1' }])
    store.triggerGrowthEvents([{ type: 'PRUNE', from: 'c2', to: 'c3' }])
    expect(useGraphStore.getState().growthQueue).toHaveLength(2)

    store.consumeGrowthQueue()
    expect(useGraphStore.getState().growthQueue).toHaveLength(0)
  })
})

describe('dialogueStore', () => {
  beforeEach(() => {
    useDialogueStore.setState({ entries: [] })
  })

  it('adds dialogue entries', () => {
    const store = useDialogueStore.getState()
    store.add({
      step: 1,
      question: 'What is this?',
      answer: 'A dog.',
      curiosity_score: 0.8,
      is_positive: true,
      stage: 0,
      timestamp: Date.now(),
    })
    expect(useDialogueStore.getState().entries).toHaveLength(1)
    expect(useDialogueStore.getState().entries[0].question).toBe('What is this?')
  })

  it('caps entries at MAX_DIALOGUE_ENTRIES', () => {
    const store = useDialogueStore.getState()
    for (let i = 0; i < 210; i++) {
      store.add({
        step: i,
        question: `q${i}`,
        answer: `a${i}`,
        curiosity_score: 0.5,
        is_positive: true,
        stage: 0,
        timestamp: Date.now(),
      })
    }
    const entries = useDialogueStore.getState().entries
    expect(entries.length).toBe(200)
    expect(entries[0].step).toBe(10)
    expect(entries[entries.length - 1].step).toBe(209)
  })

  it('clears entries', () => {
    const store = useDialogueStore.getState()
    store.add({
      step: 1, question: 'q', answer: 'a',
      curiosity_score: 0.5, is_positive: true, stage: 0, timestamp: 0,
    })
    store.clear()
    expect(useDialogueStore.getState().entries).toHaveLength(0)
  })
})

describe('loopStore', () => {
  beforeEach(() => {
    useLoopStore.setState({ status: null })
  })

  it('stores loop status', () => {
    const store = useLoopStore.getState()
    store.setStatus({
      state: 'running',
      step: 42,
      stage: 1,
      delay_ms: 500,
      error_message: null,
      graph_summary: { cluster_count: 4, node_count: 16, edge_count: 6, layer_count: 2 },
      teacher_healthy: true,
    })

    const status = useLoopStore.getState().status
    expect(status).not.toBeNull()
    expect(status!.state).toBe('running')
    expect(status!.step).toBe(42)
    expect(status!.graph_summary.cluster_count).toBe(4)
  })

  it('starts with null status', () => {
    expect(useLoopStore.getState().status).toBeNull()
  })
})
