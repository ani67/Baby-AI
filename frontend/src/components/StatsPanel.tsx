import type { Stats, StateSnapshot } from "../lib/types";

type Props = {
  stats: Stats;
  full: StateSnapshot | null;
};

export function StatsPanel({ stats, full }: Props) {
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
