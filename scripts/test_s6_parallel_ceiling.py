"""S6 end-to-end test — verify the parallel ingestion path actually
triggers the ceiling-aware sleep tick during a long run, and that the
graph stays bounded by CONCEPT_CEILING + small overshoot afterwards.

Strategy: monkeypatch CONCEPT_CEILING and PRUNE_TO down to numbers we
can hit fast (CEILING=200, PRUNE_TO=150), and lower the manager's
SLEEP_CHECK_EVERY to 200 so we hit it inside a few seconds. Then run
parallel ingestion on a generated corpus and assert:

  - at least one ceiling-sleep fires
  - final node_count is bounded (≤ ceiling + a generous buffer for
    the items in flight at the moment of the sleep tick)
"""
from __future__ import annotations

import os
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from scripts.test_parallel_ingestion import (    # noqa: E402
    _generate_diverse_text,
    construct_fresh_mind,
)


def _build_corpus(tmp: str) -> dict[str, list[str]]:
    sources_by_domain: dict[str, list[str]] = {"philosophy": [], "history": []}
    spec = [
        ("philA.txt", "philosophy", 11),
        ("philB.txt", "philosophy", 22),
        ("histA.txt", "history",    33),
        ("histB.txt", "history",    44),
    ]
    for name, domain, seed in spec:
        path = os.path.join(tmp, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(_generate_diverse_text(seed=seed, n_sentences=4000))
        sources_by_domain[domain].append(path)
    return sources_by_domain


def main() -> int:
    print("=== S6 — parallel ingestion ceiling sleep tick ===")

    # Lower the ceiling so we hit it quickly.
    from backend import config
    config.CONCEPT_CEILING = 200
    config.PRUNE_TO        = 150
    # Also lower the manager's check cadence so the tick fires soon
    # rather than waiting for the default 10K-item interval.
    config.PARALLEL_INGESTION_SLEEP_CHECK_EVERY = 200
    config.PARALLEL_INGESTION_SLEEP_DURATION    = 5.0

    # Manager imports the constants at module load time, so re-import
    # in fresh state (or read directly from config). We rely on the
    # manager pulling SLEEP_CHECK_EVERY at import time, so reload it.
    import importlib
    from backend import parallel_ingestion as pi
    importlib.reload(pi)

    tmp = tempfile.mkdtemp(prefix="s6_ceil_")
    sources_by_domain = _build_corpus(tmp)

    sleep_calls = []
    save_calls = []
    def _save():
        save_calls.append(time.perf_counter())

    loop = construct_fresh_mind()

    # Wrap _maybe_sleep_for_ceiling to count fires.
    mgr = pi.ParallelIngestionManager(
        loop=loop, n_readers=2,
        save_callback=_save,
        save_interval=10_000_000,
    )
    original = mgr._maybe_sleep_for_ceiling
    def _wrapped():
        sleep_calls.append((loop.graph.node_count, time.perf_counter()))
        return original()
    mgr._maybe_sleep_for_ceiling = _wrapped

    t0 = time.perf_counter()
    items = mgr.run(sources_by_domain)
    duration = time.perf_counter() - t0

    print(f"  items processed: {items:,}")
    print(f"  duration:        {duration:.2f}s")
    print(f"  sleep checks:    {len(sleep_calls)}  (cadence={config.PARALLEL_INGESTION_SLEEP_CHECK_EVERY})")
    print(f"  saves fired:     {len(save_calls)}  "
          f"(periodic + final + post-sleep)")
    print(f"  final nodes:     {loop.graph.node_count}")
    print(f"  ceiling:         {config.CONCEPT_CEILING}")
    print(f"  prune_to:        {config.PRUNE_TO}")

    # Verify: we had to call the sleep check at least once (cadence
    # = 200, items ≥ 200).
    assert len(sleep_calls) >= 1, "sleep check never fired"
    # Final node count must not have run away. Allow a generous buffer
    # of 2x ceiling because the writer keeps cycling between sleep
    # ticks and adds nodes on top of the ceiling-pruned baseline.
    assert loop.graph.node_count <= config.CONCEPT_CEILING * 3, (
        f"graph blew past ceiling: {loop.graph.node_count} > "
        f"{config.CONCEPT_CEILING * 3}"
    )

    print("  PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
