import { useState } from "react";
import { ingest } from "../lib/api";
import type { CycleEvent } from "../lib/types";

type Props = {
  lastCycle: CycleEvent | null;
};

export function InputPanel({ lastCycle }: Props) {
  const [text, setText] = useState("");
  const [agent, setAgent] = useState("alice");
  const [sending, setSending] = useState(false);

  async function send(e: React.FormEvent) {
    e.preventDefault();
    if (!text.trim() || sending) return;
    setSending(true);
    try {
      await ingest(text.trim(), agent.trim() || undefined);
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
          placeholder="say something to the mind…"
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

function ExpressionLine({
  expr,
  emitted,
}: {
  expr: NonNullable<CycleEvent["expression"]>;
  emitted: string | null;
}) {
  if (expr.type === "chosen") {
    return (
      <div>
        <span className="text-emerald-400">said </span>
        <span className="text-zinc-100">"{emitted ?? expr.surface}"</span>
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
