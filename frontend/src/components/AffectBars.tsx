type Props = {
  composite: number[];
  arousal: number;
  consolidationActive: boolean;
};

/**
 * 12 vertical bars, one per affect dimension. Zero is a thin centerline;
 * positive values rise red-warm, negative values fall blue-cool.
 * Smooth 600ms easing on every height change (CSS transition in styles.css).
 */
export function AffectBars({ composite, arousal, consolidationActive }: Props) {
  const N = composite.length || 12;
  // Normalize so the visual scale stays meaningful across regimes:
  // pin to max(0.4, abs.max) so a steady-state mind doesn't render flat-zero.
  const ceiling = Math.max(0.4, ...composite.map((v) => Math.abs(v)));

  return (
    <div className="glass rounded-xl p-3 w-[260px]">
      <div className="flex items-center justify-between mb-2">
        <div className="text-xs uppercase tracking-widest text-zinc-400">affect</div>
        <div className="text-[10px] text-zinc-500 tabular-nums">arousal {arousal.toFixed(2)}</div>
      </div>
      <div className="flex gap-1 items-end h-24 relative">
        <div
          className="absolute left-0 right-0 top-1/2 h-px bg-white/10 z-0"
          style={{ transform: "translateY(-0.5px)" }}
        />
        {composite.map((v, i) => {
          const norm = v / ceiling; // [-1, 1]
          const heightPct = Math.min(1, Math.abs(norm)) * 50;
          const positive = norm >= 0;
          const color = positive
            ? `hsl(${20 + 5 * Math.sin(i)}, 80%, ${50 + 20 * Math.abs(norm)}%)`
            : `hsl(${210 - 5 * Math.sin(i)}, 80%, ${50 + 20 * Math.abs(norm)}%)`;
          return (
            <div key={i} className="relative flex-1 h-full">
              <div
                className="bar-fill absolute left-0 right-0 rounded-sm"
                style={{
                  ...(positive
                    ? { bottom: "50%", height: `${heightPct}%` }
                    : { top: "50%", height: `${heightPct}%` }),
                  background: color,
                  boxShadow: `0 0 6px ${color}66`,
                  opacity: 0.92,
                }}
              />
            </div>
          );
        })}
      </div>
      <div className="mt-2 flex items-center justify-between text-[10px] text-zinc-500">
        <span>dim 0</span>
        <span className="tabular-nums">N = {N}</span>
        <span>dim {N - 1}</span>
      </div>
      {consolidationActive && (
        <div className="mt-2 text-[10px] text-violet-300 pulse-soft">consolidation active · 1.5×</div>
      )}
    </div>
  );
}
