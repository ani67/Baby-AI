import type { RuntimeResponse } from "../lib/api";
import type { CycleEvent } from "../lib/types";

export type ConversationEntry = {
  id: number;             // stimulus_id from the cycle (or sequential for wave)
  t: number;
  prompt: string;
  decisionType: "chosen" | "revision" | "suppression" | null;
  surface: string | null;     // the committed text, when ChosenCandidate
  reason: string | null;      // why revised / suppressed
  gap: number | null;
  generator?: "native_head" | "wave_field";
};

type Props = {
  entries: ConversationEntry[];
};

/** Per-input running log. Every question the user asks lands here, with
 *  whatever the mind decided — chosen / revised / suppressed — visible.
 *  Newest at the top. */
export function ConversationLog({ entries }: Props) {
  const recent = entries.slice(-12).reverse();
  return (
    <div className="glass rounded-xl p-3 w-[360px] text-xs max-h-[60vh] overflow-auto">
      <div className="flex items-center justify-between mb-2">
        <div className="uppercase tracking-widest text-zinc-400">conversation</div>
        <div className="text-[10px] text-zinc-500 tabular-nums">{entries.length}</div>
      </div>
      {recent.length === 0 && (
        <div className="text-zinc-600 italic text-[11px]">no questions yet</div>
      )}
      <ul className="space-y-2">
        {recent.map((e) => (
          <li key={e.id} className="border-l-2 border-white/5 pl-2">
            <div className="text-zinc-400 text-[11px] truncate">
              <span className="text-zinc-500">you · </span>
              <span className="text-zinc-200">"{e.prompt}"</span>
            </div>
            <Outcome e={e} />
          </li>
        ))}
      </ul>
    </div>
  );
}

/** Filter wave-field surface words: drop anything > 20 chars or
 *  containing digits / special chars. The walker can pick up
 *  artifacts from the underlying corpus (footnote markers, citation
 *  fragments, OCR noise). Keep letters, internal apostrophes, and
 *  internal hyphens only; allow trailing simple punctuation. */
function cleanSurface(text: string | null): string {
  if (!text) return "";
  const allowed = /^[A-Za-z][A-Za-z'-]{0,19}[.,!?;:]?$/;
  return text
    .split(/\s+/)
    .filter((w) => w.length > 0 && w.length <= 20 && allowed.test(w))
    .join(" ");
}

function Outcome({ e }: { e: ConversationEntry }) {
  if (e.decisionType === "chosen") {
    const label = e.generator === "wave_field" ? "wave" : "mind";
    const labelColor = e.generator === "wave_field"
      ? "text-sky-300"
      : "text-emerald-400";
    // Wave-field outputs go through cleanSurface to strip corpus
    // artifacts (numbers, OCR noise, long fragments). Native-head
    // outputs are already templated text and pass through unchanged.
    const display = e.generator === "wave_field"
      ? cleanSurface(e.surface)
      : (e.surface ?? "");
    return (
      <div className="text-[11px] mt-0.5">
        <span className={labelColor}>{label} · </span>
        <span className="text-zinc-100">"{display}"</span>
        {e.gap !== null && (
          <span className="text-zinc-600 ml-1">· gap {e.gap.toFixed(3)}</span>
        )}
      </div>
    );
  }
  if (e.decisionType === "revision") {
    return (
      <div className="text-[11px] mt-0.5">
        <span className="text-amber-400">mind · revised </span>
        {e.gap !== null && (
          <span className="text-zinc-500">(gap {e.gap.toFixed(3)} above tolerance)</span>
        )}
      </div>
    );
  }
  if (e.decisionType === "suppression") {
    return (
      <div className="text-[11px] mt-0.5">
        <span className="text-rose-400">mind · suppressed </span>
        {e.gap !== null && (
          <span className="text-zinc-500">(gap {e.gap.toFixed(3)} too large)</span>
        )}
      </div>
    );
  }
  return (
    <div className="text-[11px] mt-0.5 text-zinc-600 italic">
      mind · silent
    </div>
  );
}

/** Translate a v1.1 wave-field /ingest_runtime response into a log entry. */
let _waveEntrySeq = 1_000_000;
export function entryFromWaveResponse(
  prompt: string, res: RuntimeResponse,
): ConversationEntry {
  const id = _waveEntrySeq++;
  if (res.status !== "ok" || !res.response) {
    return {
      id, t: Date.now() / 1000, prompt,
      decisionType: null, surface: null, reason: "thinking",
      gap: null, generator: "wave_field",
    };
  }
  return {
    id,
    t: Date.now() / 1000,
    prompt,
    decisionType: "chosen",
    surface: res.response,
    reason: null,
    gap: typeof res.gap === "number" ? res.gap : null,
    generator: "wave_field",
  };
}

/** Translate a CycleEvent (when triggered by a user ingest) into a log entry. */
export function entryFromCycle(prompt: string, ev: CycleEvent): ConversationEntry {
  const expr = ev.expression;
  if (!expr) {
    return {
      id: ev.stimulus_id,
      t: ev.now,
      prompt,
      decisionType: null,
      surface: null,
      reason: null,
      gap: null,
    };
  }
  if (expr.type === "chosen") {
    return {
      id: ev.stimulus_id,
      t: ev.now,
      prompt,
      decisionType: "chosen",
      surface: ev.emitted_surface ?? expr.surface,
      reason: null,
      gap: expr.expression_gap,
    };
  }
  return {
    id: ev.stimulus_id,
    t: ev.now,
    prompt,
    decisionType: expr.type,
    surface: null,
    reason: expr.reason,
    gap: expr.best_gap,
  };
}
