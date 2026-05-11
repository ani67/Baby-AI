import type { RuntimeStatus } from "../lib/api";
import type { Stats, StateSnapshot } from "../lib/types";

type Props = {
  stats: Stats;
  full: StateSnapshot | null;
  runtimeStatus?: RuntimeStatus | null;
};

export function StatsPanel({ stats, full, runtimeStatus }: Props) {
  return (
    <div className="glass rounded-xl px-4 py-3 w-[260px] text-xs">
      <div className="flex items-center justify-between mb-2">
        <div className="uppercase tracking-widest text-zinc-400">the mind</div>
        <div className="text-[10px] text-zinc-500">
          {stats.consolidation_active ? "sleeping" : "awake"}
        </div>
      </div>
      <Row k="cycle" v={stats.cycle_count} />
      <Row k="concepts" v={stats.node_count} />
      <Row k="edges" v={stats.edge_count} />
      <Row k="pins" v={stats.pin_count} />
      <Row k="replay buffer" v={stats.replay_buffer_size} />
      {full && (
        <>
          <Row k="surprises" v={`${full.surprise_count} / ${full.observation_count}`} />
          <Row k="agents" v={full.agent_count} />
          <Row k="self cid" v={`#${full.self_concept_id}`} />
          <div className="mt-3 pt-2 border-t border-white/5">
            <div className="text-[10px] uppercase tracking-widest text-zinc-500 mb-1">
              top active
            </div>
            <ul className="space-y-0.5 max-h-32 overflow-auto">
              {full.top_active_concepts.slice(0, 6).map((c) => (
                <li key={c.id} className="flex items-center justify-between gap-2 text-[11px]">
                  <span className="truncate text-zinc-300">{c.name || `(unnamed #${c.id})`}</span>
                  <span className="tabular-nums text-zinc-500 shrink-0">{c.activation_count}</span>
                </li>
              ))}
            </ul>
          </div>
        </>
      )}
      {runtimeStatus && runtimeStatus.available && (
        <div className="mt-3 pt-2 border-t border-white/5">
          <div className="flex items-center justify-between mb-1">
            <div className="text-[10px] uppercase tracking-widest text-sky-300/80">
              wave field
            </div>
            <div className="text-[10px] text-zinc-500 tabular-nums">
              {runtimeStatus.total_outputs ?? 0} / {runtimeStatus.total_inputs ?? 0}
            </div>
          </div>
          <div className="py-0.5">
            <div className="flex justify-between mb-0.5">
              <span className="text-zinc-500">wave energy</span>
              <span className="tabular-nums text-zinc-400 text-[10px]">
                {(runtimeStatus.wave_energy ?? 0).toFixed(3)}
              </span>
            </div>
            <WaveBar energy={runtimeStatus.wave_energy ?? 0} />
          </div>
          <Row k="peak activation" v={(runtimeStatus.peak_activation ?? 0).toFixed(3)} />
          <Row k="wave steps" v={(runtimeStatus.step_count ?? 0).toLocaleString()} />
          <Row k="active concepts" v={runtimeStatus.active_concepts ?? 0} />
        </div>
      )}
    </div>
  );
}

function WaveBar({ energy }: { energy: number }) {
  // energy is unbounded above ~0; visually clamp at 0.5 (well-saturated wave).
  const pct = Math.min(100, energy * 200);
  const hue = Math.max(0, 200 - energy * 100);
  return (
    <div
      style={{
        width: "100%",
        height: 4,
        background: "rgba(255,255,255,0.08)",
        borderRadius: 2,
        overflow: "hidden",
      }}
    >
      <div
        style={{
          width: `${pct}%`,
          height: "100%",
          background: `hsl(${hue}, 80%, 60%)`,
          transition: "width 0.2s ease, background 0.2s ease",
        }}
      />
    </div>
  );
}

function Row({ k, v }: { k: string; v: string | number }) {
  return (
    <div className="flex justify-between py-0.5">
      <span className="text-zinc-500">{k}</span>
      <span className="tabular-nums text-zinc-200">{v}</span>
    </div>
  );
}
