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
 *    node:     i32 id, f32 ex, f32 ey, f32 ez, f32 act, f32 surp, f32 arousal
 *    edge:     i32 src_idx, i32 tgt_idx, f32 weight, f32 type_idx_norm
 *  source/target on edges are *indices into the node array*, not
 *  concept_ids. The caller can use them directly as texture coordinates.
 */
export type BinaryNode = {
  id: number;
  ex: number; ey: number; ez: number;   // embedding[0..2]
  activation: number;                    // [0,1] normalized
  surprise: number;
  arousal: number;
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

  const nodes: BinaryNode[] = new Array(nodeCount);
  for (let i = 0; i < nodeCount; i++) {
    const o = 8 + i * 28;
    nodes[i] = {
      id:         view.getInt32(o, true),
      ex:         view.getFloat32(o + 4, true),
      ey:         view.getFloat32(o + 8, true),
      ez:         view.getFloat32(o + 12, true),
      activation: view.getFloat32(o + 16, true),
      surprise:   view.getFloat32(o + 20, true),
      arousal:    view.getFloat32(o + 24, true),
    };
  }

  const edgeStart = 8 + nodeCount * 28;
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
