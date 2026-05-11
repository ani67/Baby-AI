import { useState } from "react";
import { ingest, ingestRuntime } from "../lib/api";
import type { RuntimeResponse } from "../lib/api";
import type { CycleEvent } from "../lib/types";

type Props = {
  lastCycle: CycleEvent | null;
  /** Called with (prompt, cycle) after each successful user-initiated
   *  ingest via the v0.9 native-head path. */
  onResponse?: (prompt: string, cycle: CycleEvent) => void;
  /** Called with (prompt, response) after each successful user-initiated
   *  ingest via the v1.1 wave-field path. */
  onWaveResponse?: (prompt: string, response: RuntimeResponse) => void;
};

export function InputPanel({ lastCycle, onResponse, onWaveResponse }: Props) {
  const [text, setText] = useState("");
  const [agent, setAgent] = useState("alice");
  const [sending, setSending] = useState(false);
  const [useWaveField, setUseWaveField] = useState(true);

  async function send(e: React.FormEvent) {
    e.preventDefault();
    const prompt = text.trim();
    if (!prompt || sending) return;
    setSending(true);
    try {
      if (useWaveField) {
        const result = await ingestRuntime(prompt, agent.trim() || undefined);
        onWaveResponse?.(prompt, result);
      } else {
        const result = (await ingest(prompt, agent.trim() || undefined)) as CycleEvent;
        onResponse?.(prompt, result);
      }
      setText("");
    } catch (err) {
      console.error("ingest failed:", err);
    } finally {
      setSending(false);
    }
  }

  const expr = lastCycle?.expression;
  const action = lastCycle?.action;

  return (
    <div className="glass rounded-xl p-3 w-[420px]">
      <form onSubmit={send} className="flex gap-2">
        <input
          type="text"
          className="flex-1 bg-black/40 border border-white/10 rounded px-3 py-1.5 text-sm focus:outline-none focus:border-white/30"
          placeholder={useWaveField ? "say something to the wave…" : "say something to the mind…"}
          value={text}
          onChange={(e) => setText(e.target.value)}
        />
        <input
          type="text"
          className="w-20 bg-black/40 border border-white/10 rounded px-2 py-1.5 text-xs text-zinc-400 focus:outline-none focus:border-white/30"
          placeholder="agent"
          value={agent}
          onChange={(e) => setAgent(e.target.value)}
        />
        <button
          type="button"
          title={useWaveField
            ? "v1.1 wave-field path — toggle to v0.9 native-head"
            : "v0.9 native-head path — toggle to v1.1 wave-field"}
          onClick={() => setUseWaveField((v) => !v)}
          className={
            "px-2 py-1.5 rounded text-xs font-mono border " +
            (useWaveField
              ? "bg-sky-500/20 border-sky-400/40 text-sky-200"
              : "bg-amber-500/20 border-amber-400/40 text-amber-200")
          }
        >
          {useWaveField ? "〰 wave" : "◉ native"}
        </button>
        <button
          type="submit"
          disabled={sending || !text.trim()}
          className="px-3 py-1.5 bg-white/10 hover:bg-white/20 disabled:opacity-40 disabled:cursor-not-allowed rounded text-sm"
        >
          send
        </button>
      </form>

      <div className="mt-3 text-xs text-zinc-400 space-y-1 min-h-[3rem]">
        {action && (
          <div>
            <span className="text-zinc-500">action </span>
            <span className="text-zinc-100">{action.kind}</span>
            <span className="text-zinc-600 ml-1">· score {action.score.toFixed(3)}</span>
          </div>
        )}
        {expr && (
          <ExpressionLine expr={expr} emitted={lastCycle?.emitted_surface ?? null} />
        )}
        {!action && (
          <div className="text-zinc-600 italic">awaiting first input</div>
        )}
      </div>
    </div>
  );
}

/** Phase 7 workstream B4: when the budget produced multiple sentences,
 *  generate_extended joined them with ". " before the world delivery.
 *  Split here for display only — each sentence on its own line. The
 *  underlying CycleEvent is unchanged.
 *
 *  Splits are made on the literal ". " boundary, with the period kept
 *  on the preceding sentence. Empty fragments are filtered. A single-
 *  sentence response just renders as one line.
 */
function splitForDisplay(s: string): string[] {
  const parts: string[] = [];
  let buf = "";
  for (let i = 0; i < s.length; i++) {
    buf += s[i];
    if (s[i] === "." && i + 1 < s.length && s[i + 1] === " ") {
      parts.push(buf);
      buf = "";
      i++; // skip the separator space
    }
  }
  if (buf) parts.push(buf);
  return parts.map((t) => t.trim()).filter(Boolean);
}

function ExpressionLine({
  expr,
  emitted,
}: {
  expr: NonNullable<CycleEvent["expression"]>;
  emitted: string | null;
}) {
  if (expr.type === "chosen") {
    const text = emitted ?? expr.surface;
    const sentences = splitForDisplay(text);
    return (
      <div>
        <span className="text-emerald-400">said </span>
        {sentences.length <= 1 ? (
          <span className="text-zinc-100">"{text}"</span>
        ) : (
          <div className="mt-1 space-y-1 pl-4 border-l border-emerald-500/20">
            {sentences.map((sent, i) => (
              <div key={i} className="text-zinc-100">
                "{sent}"
              </div>
            ))}
          </div>
        )}
        <span className="text-zinc-600 ml-1">· gap {expr.expression_gap.toFixed(3)}</span>
      </div>
    );
  }
  if (expr.type === "revision") {
    return (
      <div>
        <span className="text-amber-400">revised </span>
        <span className="text-zinc-500">{expr.reason}</span>
      </div>
    );
  }
  return (
    <div>
      <span className="text-rose-400">suppressed </span>
      <span className="text-zinc-500">{expr.reason}</span>
    </div>
  );
}
