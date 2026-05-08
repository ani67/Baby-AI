import { useState } from "react";
import { idle, save, sleep } from "../lib/api";

export function Controls() {
  const [busy, setBusy] = useState<string | null>(null);
  const [lastSleep, setLastSleep] = useState<{ replays: number; abstractions: number } | null>(null);
  const [autoIdle, setAutoIdle] = useState(true);

  // Note: auto-idle effect lives in App.tsx (it has access to the timer logic
  // and the latest activity timestamp). This component just owns the toggle.

  async function run<T>(label: string, op: () => Promise<T>): Promise<T | null> {
    setBusy(label);
    try { return await op(); }
    catch (e) { console.error(label, "failed:", e); return null; }
    finally { setBusy(null); }
  }

  async function doIdle()  { await run("idle",  () => idle(2)); }
  async function doSleep() {
    const r = await run("sleep", () => sleep(2.0));
    if (r) setLastSleep({ replays: r.replays_fired, abstractions: r.abstractions_formed });
  }
  async function doSave() {
    const r = await run("save", save);
    if (r) console.log(`saved ${r.size_bytes} bytes to ${r.path}`);
  }

  return (
    <div className="glass rounded-xl p-3 w-[200px] text-xs">
      <div className="uppercase tracking-widest text-zinc-400 mb-2">controls</div>
      <div className="grid gap-1.5">
        <Btn label="idle"  busy={busy === "idle"}  onClick={doIdle} />
        <Btn label="sleep" busy={busy === "sleep"} onClick={doSleep} accent="violet" />
        <Btn label="save"  busy={busy === "save"}  onClick={doSave}  accent="zinc" />
      </div>
      <label className="mt-2 flex items-center gap-2 text-[11px] text-zinc-400 cursor-pointer select-none">
        <input
          type="checkbox"
          checked={autoIdle}
          onChange={(e) => {
            setAutoIdle(e.target.checked);
            // Communicate via a custom event the App listens for.
            window.dispatchEvent(new CustomEvent("mind:auto-idle", { detail: e.target.checked }));
          }}
        />
        <span>auto-idle every 10s</span>
      </label>
      {lastSleep && (
        <div className="mt-2 text-[10px] text-zinc-500">
          last sleep: {lastSleep.replays} replays · {lastSleep.abstractions} abstractions
        </div>
      )}
    </div>
  );
}

function Btn({
  label, busy, onClick, accent = "white",
}: { label: string; busy: boolean; onClick: () => void; accent?: string }) {
  const base = "px-3 py-1.5 rounded text-xs uppercase tracking-wider transition-colors disabled:opacity-50 disabled:cursor-not-allowed";
  const colors =
    accent === "violet" ? "bg-violet-500/15 hover:bg-violet-500/30 border border-violet-500/30 text-violet-100"
    : accent === "zinc" ? "bg-zinc-500/15 hover:bg-zinc-500/30 border border-zinc-500/30 text-zinc-100"
                        : "bg-white/8 hover:bg-white/16 border border-white/10 text-zinc-100";
  return (
    <button className={`${base} ${colors}`} disabled={busy} onClick={onClick}>
      {busy ? `${label}…` : label}
    </button>
  );
}
