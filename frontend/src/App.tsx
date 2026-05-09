import { useEffect, useRef, useState } from "react";
import { fetchGraphBinary, fetchState, idle, subscribe } from "./lib/api";
import type { BinaryGraph } from "./lib/api";
import type {
  AnyEvent,
  CycleEvent,
  Stats,
  StateSnapshot,
} from "./lib/types";
import { AffectBars } from "./components/AffectBars";
import { ConversationEntry, ConversationLog, entryFromCycle } from "./components/ConversationLog";
import { Controls } from "./components/Controls";
import { InputPanel } from "./components/InputPanel";
import { MindGraph } from "./components/MindGraph";
import { StatsPanel } from "./components/StatsPanel";

// Graph re-fetch is now a heavy event (~8 MB binary + texture rebuild +
// worker re-init). Bump the interval up. The WS still pushes per-cycle
// state to the existing stateTex; we only need the full graph back when
// nodes/edges actually grew.
const GRAPH_REFETCH_INTERVAL_MS = 30_000;
const AUTO_IDLE_INTERVAL_MS     = 10_000;

export function App() {
  const [graph, setGraph] = useState<BinaryGraph | null>(null);
  const [state, setState] = useState<StateSnapshot | null>(null);
  const [stats, setStats] = useState<Stats | null>(null);
  const [lastCycle, setLastCycle] = useState<CycleEvent | null>(null);
  const [conversation, setConversation] = useState<ConversationEntry[]>([]);
  const lastInteractionRef = useRef<number>(Date.now());
  const lastNodeCountRef = useRef<number>(0);

  // Initial fetch of state + graph.
  useEffect(() => {
    fetchState().then((s) => { setState(s); setStats(s); }).catch(console.error);
    fetchGraphBinary().then((g) => {
      setGraph(g);
      lastNodeCountRef.current = g.nodes.length;
    }).catch(console.error);
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

  // Periodic graph re-fetch. Cheaper than diffing on the wire, but
  // costly enough (worker re-init + texture rebuild) that we only do it
  // when the node count actually moved. /state gives us the count
  // every couple seconds; the binary fetch only fires when it changes
  // (or every GRAPH_REFETCH_INTERVAL_MS as a sanity poll).
  useEffect(() => {
    const stateId = window.setInterval(() => {
      fetchState().then((s) => {
        setState(s);
        if (Math.abs(s.node_count - lastNodeCountRef.current) >= 200) {
          fetchGraphBinary().then((g) => {
            setGraph(g);
            lastNodeCountRef.current = g.nodes.length;
          }).catch(() => {});
        }
      }).catch(() => {});
    }, 2000);
    const graphId = window.setInterval(() => {
      fetchGraphBinary().then((g) => {
        setGraph(g);
        lastNodeCountRef.current = g.nodes.length;
      }).catch(() => {});
    }, GRAPH_REFETCH_INTERVAL_MS);
    return () => {
      window.clearInterval(stateId);
      window.clearInterval(graphId);
    };
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
