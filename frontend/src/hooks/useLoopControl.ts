import { useCallback } from 'react'
import { api } from '../lib/api'
import { useLoopStore } from '../store/loopStore'

export function useLoopControl() {
  const setStatus = useLoopStore(s => s.setStatus)

  const start = useCallback(async () => {
    const status = await api.start()
    setStatus(status)
  }, [setStatus])

  const pause = useCallback(async () => {
    const status = await api.pause()
    setStatus(status)
  }, [setStatus])

  const resume = useCallback(async () => {
    const status = await api.resume()
    setStatus(status)
  }, [setStatus])

  const step = useCallback(async () => {
    await api.step()
    const status = await api.status()
    setStatus(status)
  }, [setStatus])

  const reset = useCallback(async (notes: {
    architecture_state: string
    signal_quality: string
    why_reset: string
    what_was_learned: string
  }) => {
    const status = await api.reset(notes)
    setStatus(status)
  }, [setStatus])

  const setStage = useCallback(async (stage: number) => {
    const status = await api.setStage(stage)
    setStatus(status)
  }, [setStatus])

  const setSpeed = useCallback(async (delay_ms: number) => {
    const status = await api.setSpeed(delay_ms)
    setStatus(status)
  }, [setStatus])

  return { start, pause, resume, step, reset, setStage, setSpeed }
}
