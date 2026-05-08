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
