import { useEffect } from 'react'
import { useGraphStore } from '../store/graphStore'
import { useDialogueStore } from '../store/dialogueStore'
import { useLoopStore } from '../store/loopStore'

export function useWebSocket(url: string) {
  const applyDelta = useGraphStore(s => s.applyDelta)
  const setSnapshot = useGraphStore(s => s.setSnapshot)
  const triggerGrowthEvents = useGraphStore(s => s.triggerGrowthEvents)
  const addDialogue = useDialogueStore(s => s.add)
  const setStatus = useLoopStore(s => s.setStatus)

  useEffect(() => {
    let ws: WebSocket
    let reconnectTimer: ReturnType<typeof setTimeout>

    function connect() {
      ws = new WebSocket(url)

      ws.onopen = () => {
        console.log('WS connected')
      }

      ws.onmessage = (event) => {
        const msg = JSON.parse(event.data)

        if (msg.type === 'snapshot') {
          setSnapshot(msg)
        }

        if (msg.type === 'delta') {
          applyDelta(msg)

          if (msg.growth_events?.length) {
            triggerGrowthEvents(msg.growth_events)
          }
          if (msg.dialogue) {
            addDialogue({
              step: msg.step,
              ...msg.dialogue,
              stage: msg.stage,
              timestamp: Date.now(),
            })
          }
          // Update status bar
          const prev = useLoopStore.getState().status
          setStatus({
            state: 'running',
            step: msg.step,
            stage: msg.stage,
            delay_ms: prev?.delay_ms ?? 500,
            error_message: null,
            graph_summary: msg.graph_summary ?? prev?.graph_summary ?? {},
            teacher_healthy: prev?.teacher_healthy ?? true,
          })
        }

        if (msg.type === 'status') {
          setStatus(msg)
        }
      }

      ws.onerror = () => {}

      ws.onclose = () => {
        reconnectTimer = setTimeout(connect, 2000)
      }
    }

    connect()

    return () => {
      clearTimeout(reconnectTimer)
      ws?.close()
    }
  }, [url, applyDelta, setSnapshot, triggerGrowthEvents, addDialogue, setStatus])
}
