import { useEffect, useRef, useState } from 'react'
import { api } from '../lib/api'

const REFETCH_INTERVAL_MS = 30_000

/**
 * Fetches /clusters/labels on mount and every 30 seconds.
 * Returns a cached map of cluster_id → string[] labels.
 */
export function useClusterLabels() {
  const [labels, setLabels] = useState<Record<string, string[]>>({})
  const timerRef = useRef<ReturnType<typeof setInterval>>()

  useEffect(() => {
    let cancelled = false

    const fetchLabels = () => {
      api.clusterLabels()
        .then(data => {
          if (!cancelled) setLabels(data.labels ?? {})
        })
        .catch(() => {})  // silently ignore fetch errors
    }

    fetchLabels()
    timerRef.current = setInterval(fetchLabels, REFETCH_INTERVAL_MS)

    return () => {
      cancelled = true
      clearInterval(timerRef.current)
    }
  }, [])

  return labels
}
