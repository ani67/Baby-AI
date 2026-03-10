import { create } from 'zustand'
import { MAX_DIALOGUE_ENTRIES } from '../lib/constants'

export interface DialogueEntry {
  step: number
  question: string
  answer: string
  model_answer?: string | null
  curiosity_score: number
  is_positive: boolean
  stage: number
  timestamp: number
  image_url?: string | null
}

interface DialogueState {
  entries: DialogueEntry[]
  add: (entry: DialogueEntry) => void
  clear: () => void
}

export const useDialogueStore = create<DialogueState>((set) => ({
  entries: [],

  add: (entry) => set((state) => {
    const next = [...state.entries, entry]
    if (next.length > MAX_DIALOGUE_ENTRIES) {
      return { entries: next.slice(next.length - MAX_DIALOGUE_ENTRIES) }
    }
    return { entries: next }
  }),

  clear: () => set({ entries: [] }),
}))
