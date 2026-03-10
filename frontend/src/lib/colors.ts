export const CLUSTER_COLORS: Record<string, string> = {
  integration:    '#3b7fff',
  transformation: '#9b59b6',
  arbitration:    '#e67e22',
  routing:        '#1abc9c',
  dormant:        '#2a2a2a',
}

export function clusterColor(clusterType: string): string {
  return CLUSTER_COLORS[clusterType] ?? '#888888'
}
