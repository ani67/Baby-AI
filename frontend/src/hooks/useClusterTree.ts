import { useEffect, useRef, useState } from 'react'
import { api } from '../lib/api'

export interface TreeNode {
  id: string            // cluster ID (e.g. "c_00", "c_00a")
  parent: string | null // parent cluster ID, null for roots
  depth: number         // BUD depth (0 = original, 1 = first split, etc.)
  dormant: boolean
  cluster_type: string | null
  labels: string[]
  phantom: boolean      // true if cluster was removed by BUD but is needed as a tree ancestor
}

export interface TreeEdge {
  source: string
  target: string
}

export interface TreeData {
  nodes: TreeNode[]
  edges: TreeEdge[]
  max_depth: number
}

export type TreeStatus = 'loading' | 'ok' | 'error' | 'empty'

const REFETCH_INTERVAL_MS = 30_000
const EMPTY: TreeData = { nodes: [], edges: [], max_depth: 0 }

function isValidTreeData(data: unknown): data is TreeData {
  if (!data || typeof data !== 'object') return false
  const d = data as Record<string, unknown>
  return Array.isArray(d.nodes) && Array.isArray(d.edges) && typeof d.max_depth === 'number'
}

/**
 * Fetches /clusters/tree on mount and every 30 seconds.
 * Returns tree data + status so the renderer can show fallback UI.
 */
export function useClusterTree(): { tree: TreeData; status: TreeStatus } {
  const [tree, setTree] = useState<TreeData>(EMPTY)
  const [status, setStatus] = useState<TreeStatus>('loading')
  const timerRef = useRef<ReturnType<typeof setInterval>>()

  useEffect(() => {
    let cancelled = false

    const fetchTree = () => {
      api.clusterTree()
        .then(data => {
          if (cancelled) return
          if (!isValidTreeData(data)) {
            setStatus('error')
            return
          }
          if (data.nodes.length === 0) {
            setTree(EMPTY)
            setStatus('empty')
            return
          }
          setTree(data)
          setStatus('ok')
        })
        .catch(() => {
          if (!cancelled) setStatus('error')
        })
    }

    fetchTree()
    timerRef.current = setInterval(fetchTree, REFETCH_INTERVAL_MS)

    return () => {
      cancelled = true
      clearInterval(timerRef.current)
    }
  }, [])

  return { tree, status }
}
