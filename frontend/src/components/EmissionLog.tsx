import type { CycleEvent } from "../lib/types";

type Props = {
  emissions: { id: number; t: number; surface: string }[];
};

export function EmissionLog({ emissions }: Props) {
  if (!emissions.length) return null;
  const recent = emissions.slice(-6).reverse();
  return (
    <div className="glass rounded-xl p-3 w-[300px] text-xs max-h-[200px] overflow-auto">
      <div className="uppercase tracking-widest text-zinc-400 mb-2">recent expressions</div>
      <ul className="space-y-1">
        {recent.map((e) => (
          <li key={e.id} className="text-zinc-200">
            <span className="text-emerald-400">▸ </span>
            "{e.surface}"
          </li>
        ))}
      </ul>
    </div>
  );
}

/** Build the emissions list incrementally from cycle events. */
export function selectEmissions(events: CycleEvent[]) {
  const out: { id: number; t: number; surface: string }[] = [];
  for (const ev of events) {
    if (ev.emitted_surface) {
      out.push({ id: ev.stimulus_id, t: ev.now, surface: ev.emitted_surface });
    }
  }
  return out;
}
