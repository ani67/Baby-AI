import { useMemo } from 'react'
import { useGraphStore, GraphNode } from '../store/graphStore'

export function useClusterCentroid(clusterId: string): [number, number, number] {
  const nodes = useGraphStore(s => s.nodes)

  return useMemo(() => {
    const clusterNodes = nodes.filter(n => n.cluster === clusterId)
    if (clusterNodes.length === 0) return [0, 0, 0] as [number, number, number]

    const sum: [number, number, number] = [0, 0, 0]
    for (const n of clusterNodes) {
      sum[0] += n.pos[0]
      sum[1] += n.pos[1]
      sum[2] += n.pos[2]
    }
    const count = clusterNodes.length
    return [sum[0] / count, sum[1] / count, sum[2] / count] as [number, number, number]
  }, [nodes, clusterId])
}

export function clusterCentroid(
  nodes: GraphNode[],
  clusterId: string
): [number, number, number] {
  const clusterNodes = nodes.filter(n => n.cluster === clusterId && n.pos)
  if (clusterNodes.length === 0) return [0, 0, 0]

  const sum: [number, number, number] = [0, 0, 0]
  for (const n of clusterNodes) {
    sum[0] += (n.pos[0] ?? 0)
    sum[1] += (n.pos[1] ?? 0)
    sum[2] += (n.pos[2] ?? 0)
  }
  const count = clusterNodes.length
  return [sum[0] / count, sum[1] / count, sum[2] / count]
}
