import { create } from 'zustand'

export interface LoopStatus {
  state: 'idle' | 'running' | 'paused' | 'error'
  step: number
  stage: number
  delay_ms: number
  error_message: string | null
  graph_summary: {
    cluster_count?: number
    node_count?: number
    edge_count?: number
    layer_count?: number
    dormant_count?: number
  }
  teacher_healthy: boolean
}

interface LoopState {
  status: LoopStatus | null
  setStatus: (status: LoopStatus) => void
}

export const useLoopStore = create<LoopState>((set) => ({
  status: null,
  setStatus: (status) => set({ status }),
}))
