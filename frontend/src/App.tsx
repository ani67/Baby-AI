import { useEffect, useRef, useState } from "react";
import { fetchGraph, fetchState, idle, subscribe } from "./lib/api";
import type {
  AnyEvent,
  CycleEvent,
  GraphPayload,
  Stats,
  StateSnapshot,
} from "./lib/types";
import { AffectBars } from "./components/AffectBars";
import { ConversationEntry, ConversationLog, entryFromCycle } from "./components/ConversationLog";
import { Controls } from "./components/Controls";
import { InputPanel } from "./components/InputPanel";
import { MindGraph } from "./components/MindGraph";
import { StatsPanel } from "./components/StatsPanel";

const GRAPH_REFETCH_INTERVAL_MS = 2000;
const AUTO_IDLE_INTERVAL_MS     = 10_000;

export function App() {
  const [graph, setGraph] = useState<GraphPayload | null>(null);
  const [state, setState] = useState<StateSnapshot | null>(null);
  const [stats, setStats] = useState<Stats | null>(null);
  const [lastCycle, setLastCycle] = useState<CycleEvent | null>(null);
  const [conversation, setConversation] = useState<ConversationEntry[]>([]);
  const lastInteractionRef = useRef<number>(Date.now());

  // Initial fetch of state + graph.
  useEffect(() => {
    fetchState().then((s) => { setState(s); setStats(s); }).catch(console.error);
    fetchGraph().then(setGraph).catch(console.error);
  }, []);

  // WebSocket subscription.
  useEffect(() => {
    return subscribe((ev: AnyEvent) => {
      if (ev.type === "state") {
        setState(ev);
        setStats(ev);
        return;
      }
      if (ev.type === "cycle") {
        setLastCycle(ev);
        setStats(ev.stats);
        lastInteractionRef.current = Date.now();
        // Note: we no longer write to a separate emissions list here.
        // The user-initiated path captures (prompt, cycle) via the
        // InputPanel's onResponse callback below — that's what populates
        // the conversation log. Auto-idle cycles pass through here too
        // but they have no prompt, so they don't end up in the log.
        return;
      }
      if (ev.type === "idle" || ev.type === "sleep") {
        setStats(ev.stats);
        return;
      }
    });
  }, []);

  // Periodic graph re-fetch — the WS pushes per-cycle deltas but the full
  // graph (with new nodes / new edges) is cheaper to re-pull on a slow timer
  // than to incrementally diff over the wire.
  useEffect(() => {
    const id = window.setInterval(() => {
      fetchGraph().then(setGraph).catch(() => {});
      fetchState().then(setState).catch(() => {});
    }, GRAPH_REFETCH_INTERVAL_MS);
    return () => window.clearInterval(id);
  }, []);

  // Auto-idle every 10s when no manual input. Toggle is broadcast from Controls
  // via a custom event so this doesn't need prop-drilling.
  useEffect(() => {
    let enabled = true;
    const onToggle = (e: Event) => {
      enabled = (e as CustomEvent<boolean>).detail;
    };
    window.addEventListener("mind:auto-idle", onToggle);
    const id = window.setInterval(() => {
      if (!enabled) return;
      if (Date.now() - lastInteractionRef.current < AUTO_IDLE_INTERVAL_MS) return;
      idle(1).catch(() => {});
    }, AUTO_IDLE_INTERVAL_MS);
    return () => {
      window.removeEventListener("mind:auto-idle", onToggle);
      window.clearInterval(id);
    };
  }, []);

  return (
    <div className="relative w-full h-full">
      <MindGraph
        graph={graph}
        lastCycle={lastCycle}
        consolidationActive={!!stats?.consolidation_active}
      />

      {/* Top-left: stats */}
      <div className="absolute top-4 left-4 z-10">
        <StatsPanel stats={stats ?? defaultStats()} full={state} />
      </div>

      {/* Top-right: controls */}
      <div className="absolute top-4 right-4 z-10">
        <Controls />
      </div>

      {/* Bottom-left: affect bars */}
      <div className="absolute bottom-4 left-4 z-10">
        <AffectBars
          composite={stats?.composite ?? new Array(12).fill(0)}
          arousal={stats?.arousal ?? 0}
          consolidationActive={!!stats?.consolidation_active}
        />
      </div>

      {/* Bottom-center: input + last action/expression */}
      <div className="absolute bottom-4 left-1/2 -translate-x-1/2 z-10">
        <InputPanel
          lastCycle={lastCycle}
          onResponse={(prompt, cycle) => {
            setConversation((prev) => [...prev, entryFromCycle(prompt, cycle)]);
          }}
        />
      </div>

      {/* Bottom-right: full conversation log (every question, every outcome) */}
      <div className="absolute bottom-4 right-4 z-10">
        <ConversationLog entries={conversation} />
      </div>
    </div>
  );
}

function defaultStats(): Stats {
  return {
    cycle_count: 0, node_count: 0, edge_count: 0, pin_count: 0,
    replay_buffer_size: 0, consolidation_active: false,
    composite: new Array(12).fill(0), arousal: 0,
  };
}
