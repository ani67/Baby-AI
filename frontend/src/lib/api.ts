import type { GraphPayload, StateSnapshot, AnyEvent } from "./types";

const json = { "Content-Type": "application/json" };

export async function ingest(text: string, agent_handle?: string) {
  const res = await fetch("/ingest", {
    method: "POST",
    headers: json,
    body: JSON.stringify({ text, agent_handle }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

/** v1.1 wave-field path. Returns the runtime's emitted surface text
 *  (or `null` + status="thinking" if the wave hasn't settled enough
 *  to pick word-concepts within the 15 s receive timeout). */
export type RuntimeResponse = {
  status: "ok" | "thinking";
  response: string | null;
  gap?: number;
  active_concepts?: number;
  arousal?: number;
  top_concepts?: string[];
  generator?: string;
};

export async function ingestRuntime(
  text: string, person_id?: string,
): Promise<RuntimeResponse> {
  const res = await fetch("/ingest_runtime", {
    method: "POST",
    headers: json,
    body: JSON.stringify({ text, person_id }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export type RuntimeStatus = {
  available: boolean;
  running?: boolean;
  step_count?: number;
  total_inputs?: number;
  total_outputs?: number;
  wave_energy?: number;
  peak_activation?: number;
  arousal?: number;
  active_concepts?: number;
  node_count?: number;
  contradiction_buffer?: number;
  self_prediction_loss?: number;
};

export async function getRuntimeStatus(): Promise<RuntimeStatus> {
  const res = await fetch("/runtime_status");
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function idle(max_replays = 1) {
  const res = await fetch("/idle", {
    method: "POST",
    headers: json,
    body: JSON.stringify({ max_replays }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function sleep(duration_seconds = 2.0) {
  const res = await fetch("/sleep", {
    method: "POST",
    headers: json,
    body: JSON.stringify({ duration_seconds }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function fetchState(): Promise<StateSnapshot> {
  const res = await fetch("/state");
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function fetchGraph(): Promise<GraphPayload> {
  const res = await fetch("/graph");
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

/** Compact binary representation of the graph used by the GPU
 *  renderer. Layout matches backend.api.get_graph_binary:
 *    header:   i32 node_count, i32 edge_count
 *    node:     i32 id, f32 ex, f32 ey, f32 ez, f32 act, f32 surp, f32 arousal,
 *              f32 wave_activation        (v1.1 — live wave field activation)
 *    edge:     i32 src_idx, i32 tgt_idx, f32 weight, f32 type_idx_norm
 *  source/target on edges are *indices into the node array*, not
 *  concept_ids. The caller can use them directly as texture coordinates.
 */
export type BinaryNode = {
  id: number;
  ex: number; ey: number; ez: number;   // embedding[0..2]
  activation: number;                    // [0,1] normalized historical
  surprise: number;
  arousal: number;
  waveActivation: number;                // v1.1 — live wave-field activation
};
export type BinaryEdge = {
  sourceIdx: number;
  targetIdx: number;
  weight: number;
  typeNorm: number;                      // EdgeType ordinal / 10
};
export type BinaryGraph = {
  nodes: BinaryNode[];
  edges: BinaryEdge[];
};

export async function fetchGraphBinary(): Promise<BinaryGraph> {
  const res = await fetch("/graph/binary");
  if (!res.ok) throw new Error(await res.text());
  const buf = await res.arrayBuffer();
  const view = new DataView(buf);

  const nodeCount = view.getInt32(0, true);
  const edgeCount = view.getInt32(4, true);

  // Per-node stride: 7×i32/f32 = 28 bytes (v1.0) + 1 f32 wave_activation
  // (v1.1) = 32 bytes. Backend writes 32 when runtime is wired;
  // gracefully falls back to 28 if header indicates the legacy layout.
  // We detect by checking total buffer size.
  const totalBytes = buf.byteLength - 8;            // minus header
  const edgeBytes  = 16;                            // unchanged
  const stride28   = 28 * nodeCount + edgeBytes * edgeCount;
  const isV11Layout = totalBytes !== stride28;      // assume new layout if not legacy
  const nodeStride = isV11Layout ? 32 : 28;

  const nodes: BinaryNode[] = new Array(nodeCount);
  for (let i = 0; i < nodeCount; i++) {
    const o = 8 + i * nodeStride;
    nodes[i] = {
      id:             view.getInt32(o, true),
      ex:             view.getFloat32(o + 4, true),
      ey:             view.getFloat32(o + 8, true),
      ez:             view.getFloat32(o + 12, true),
      activation:     view.getFloat32(o + 16, true),
      surprise:       view.getFloat32(o + 20, true),
      arousal:        view.getFloat32(o + 24, true),
      waveActivation: isV11Layout ? view.getFloat32(o + 28, true) : 0,
    };
  }

  const edgeStart = 8 + nodeCount * nodeStride;
  const edges: BinaryEdge[] = new Array(edgeCount);
  for (let i = 0; i < edgeCount; i++) {
    const o = edgeStart + i * 16;
    edges[i] = {
      sourceIdx: view.getInt32(o, true),
      targetIdx: view.getInt32(o + 4, true),
      weight:    view.getFloat32(o + 8, true),
      typeNorm:  view.getFloat32(o + 12, true),
    };
  }
  return { nodes, edges };
}

export async function save() {
  const res = await fetch("/save", { method: "POST" });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

/** Open a WebSocket to /ws and call the handler on each event.
 *  Returns a function that closes the socket. Auto-reconnects every 2s.
 */
export function subscribe(onEvent: (ev: AnyEvent) => void): () => void {
  let closed = false;
  let ws: WebSocket | null = null;
  let reconnectTimer: number | null = null;

  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  const url = `${proto}//${location.host}/ws`;

  function open() {
    if (closed) return;
    ws = new WebSocket(url);
    ws.onmessage = (m) => {
      try {
        onEvent(JSON.parse(m.data));
      } catch (e) {
        // ignore malformed
      }
    };
    ws.onclose = () => {
      if (closed) return;
      reconnectTimer = window.setTimeout(open, 2000);
    };
    ws.onerror = () => {
      try { ws?.close(); } catch {}
    };
  }
  open();

  return () => {
    closed = true;
    if (reconnectTimer) window.clearTimeout(reconnectTimer);
    try { ws?.close(); } catch {}
  };
}
