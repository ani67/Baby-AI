import { create } from 'zustand'

export interface GraphNode {
  id: string
  cluster: string
  pos: [number, number, number]
  activation_mean: number
  alive: boolean
}

export interface GraphCluster {
  id: string
  cluster_type: string
  layer_index: number
  dormant: boolean
}

export interface GraphEdge {
  from: string
  to: string
  strength: number
}

export interface GrowthEvent {
  type: 'BUD' | 'CONNECT' | 'PRUNE' | 'INSERT' | 'EXTEND' | 'DORMANT'
  from?: string
  to?: string
  new_cluster?: string
  cluster?: string
}

interface GraphDiff {
  nodes_added: GraphNode[]
  nodes_removed: { id: string }[]
  clusters_added: GraphCluster[]
  clusters_removed: { id: string }[]
  clusters_updated: Partial<GraphCluster & { id: string }>[]
  edges_added: GraphEdge[]
  edges_removed: { from: string; to: string }[]
  edges_updated: GraphEdge[]
}

interface SnapshotMessage {
  nodes: GraphNode[]
  clusters: GraphCluster[]
  edges: GraphEdge[]
  model_stats?: Record<string, unknown>
}

interface GraphState {
  nodes: GraphNode[]
  clusters: GraphCluster[]
  edges: GraphEdge[]
  activations: Record<string, number>
  growthQueue: GrowthEvent[]

  setSnapshot: (snapshot: SnapshotMessage) => void
  applyDiff: (diff: GraphDiff, positionsStale: boolean) => void
  setActivations: (activations: Record<string, number>) => void
  updatePositions: (positions: Record<string, [number, number, number]>) => void
  triggerGrowthEvents: (events: GrowthEvent[]) => void
  consumeGrowthQueue: () => void
}

export const useGraphStore = create<GraphState>((set) => ({
  nodes: [],
  clusters: [],
  edges: [],
  activations: {},
  growthQueue: [],

  setSnapshot: (msg) => set({
    nodes: msg.nodes,
    clusters: msg.clusters,
    edges: msg.edges,
  }),

  applyDiff: (diff, positionsStale) => set((state) => {
    let nodes = [...state.nodes]
    let clusters = [...state.clusters]
    let edges = [...state.edges]

    // Add (deduplicate against existing)
    const existingNodeIds = new Set(nodes.map(n => n.id))
    const existingClusterIds = new Set(clusters.map(c => c.id))
    const existingEdgeKeys = new Set(edges.map(e => `${e.from}-${e.to}`))
    nodes = [...nodes, ...diff.nodes_added.filter(n => !existingNodeIds.has(n.id))]
    clusters = [...clusters, ...diff.clusters_added.filter(c => !existingClusterIds.has(c.id))]
    edges = [...edges, ...diff.edges_added.filter(e => !existingEdgeKeys.has(`${e.from}-${e.to}`))]

    // Remove
    const removedNodeIds = new Set(diff.nodes_removed.map(n => n.id))
    const removedClusterIds = new Set(diff.clusters_removed.map(c => c.id))
    const removedEdgeKeys = new Set(
      diff.edges_removed.map(e => `${e.from}-${e.to}`)
    )
    nodes = nodes.filter(n => !removedNodeIds.has(n.id))
    clusters = clusters.filter(c => !removedClusterIds.has(c.id))
    edges = edges.filter(e => !removedEdgeKeys.has(`${e.from}-${e.to}`))

    // Update clusters
    const updatedClusters = new Map(
      diff.clusters_updated.map(c => [c.id, c])
    )
    clusters = clusters.map(c => updatedClusters.has(c.id)
      ? { ...c, ...updatedClusters.get(c.id)! }
      : c
    )

    // Update edge strengths
    const updatedEdges = new Map(
      diff.edges_updated.map(e => [`${e.from}-${e.to}`, e])
    )
    edges = edges.map(e => {
      const key = `${e.from}-${e.to}`
      return updatedEdges.has(key)
        ? { ...e, strength: updatedEdges.get(key)!.strength }
        : e
    })

    // Update node positions from added nodes (may have fresh positions)
    if (diff.nodes_added.length > 0) {
      const updatedNodes = new Map(diff.nodes_added.map(n => [n.id, n]))
      nodes = nodes.map(n => {
        const updated = updatedNodes.get(n.id)
        return updated?.pos ? { ...n, pos: updated.pos } : n
      })
    }

    return { nodes, clusters, edges }
  }),

  setActivations: (activations) => set({ activations }),

  updatePositions: (positions) => set((state) => ({
    nodes: state.nodes.map(n =>
      positions[n.id] ? { ...n, pos: positions[n.id] } : n
    ),
  })),

  triggerGrowthEvents: (events) => set((state) => ({
    growthQueue: [...state.growthQueue, ...events],
  })),

  consumeGrowthQueue: () => set({ growthQueue: [] }),
}))
