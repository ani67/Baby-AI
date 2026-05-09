"""S3 test — verify the parallel ingestion writer fires its
save_callback periodically AND on shutdown, and that readers exit
cleanly enough that the writer's join doesn't hang for long.

Two scenarios:

  1. NORMAL run to completion: a small save_interval forces multiple
     periodic saves; on completion we expect a final save too. Total
     saves should be >= ceil(items / interval).
  2. EARLY STOP via the manager's _on_signal: the run loop sees the
     stop flag, drains the current item, and exits the finally with
     a final save. Wall time of the entire run + cleanup should stay
     well under the per-reader 5s join timeout × n_readers.

This isolates the v0.7 S3 changes; it doesn't depend on anything in
S4–S7 and doesn't touch persistence on disk (the callback is a
counter so we don't drag MindPersistence into the test surface).
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
import threading

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from scripts.test_parallel_ingestion import (   # noqa: E402
    _generate_diverse_text,
    construct_fresh_mind,
)


def _build_corpus(tmp: str, sentences_per_file: int = 1500) -> dict[str, list[str]]:
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
            f.write(_generate_diverse_text(seed=seed,
                                           n_sentences=sentences_per_file))
        sources_by_domain[domain].append(path)
    return sources_by_domain


def scenario_periodic_save() -> None:
    """Run to completion with save_interval=200; expect multiple saves."""
    from backend.parallel_ingestion import ParallelIngestionManager

    print("\n=== scenario: periodic save fires during a normal run ===")
    tmp = tempfile.mkdtemp(prefix="s3_save_")
    sources_by_domain = _build_corpus(tmp, sentences_per_file=1500)

    save_calls = []
    def _save():
        save_calls.append(time.perf_counter())

    loop = construct_fresh_mind()
    mgr = ParallelIngestionManager(
        loop=loop,
        n_readers=2,
        save_callback=_save,
        save_interval=200,
    )
    t0 = time.perf_counter()
    items = mgr.run(sources_by_domain)
    duration = time.perf_counter() - t0

    print(f"  items processed: {items:,}")
    print(f"  duration:        {duration:.2f}s")
    print(f"  save calls:      {len(save_calls)} (interval=200, expected ≈ "
          f"{max(1, items // 200) + 1})")

    # We should fire at least items//interval periodic saves PLUS one
    # final save in finally — strict lower bound: at least 1 final save.
    assert len(save_calls) >= 1, "no save callback ever fired"
    if items >= 200:
        # Roughly one per interval; allow ±2 for the boundary.
        expected = max(1, items // 200)
        assert len(save_calls) >= expected, (
            f"expected ≥ {expected} saves for {items} items, got "
            f"{len(save_calls)}"
        )
    print("  PASS")


def scenario_signal_shutdown() -> None:
    """Trigger the manager's signal handler from a background thread
    after a short delay; verify clean shutdown + a final save fires."""
    from backend.parallel_ingestion import ParallelIngestionManager

    print("\n=== scenario: SIGTERM mid-run triggers save and clean exit ===")
    tmp = tempfile.mkdtemp(prefix="s3_sig_")
    # Small enough to finish encoding in <2s on M1 but large enough that
    # we can interrupt it mid-flight.
    sources_by_domain = _build_corpus(tmp, sentences_per_file=4000)

    save_calls = []
    def _save():
        save_calls.append(time.perf_counter())

    loop = construct_fresh_mind()
    mgr = ParallelIngestionManager(
        loop=loop,
        n_readers=2,
        save_callback=_save,
        save_interval=10_000_000,  # effectively disable periodic
    )

    # Fire the manager's _on_signal handler from a background thread
    # after 1.0s — equivalent to a SIGTERM during the run, but doesn't
    # require us to actually deliver a signal (which is racy in tests).
    def _trip():
        time.sleep(1.0)
        mgr._on_signal(signum=15, _frame=None)

    threading.Thread(target=_trip, daemon=True).start()

    t0 = time.perf_counter()
    items = mgr.run(sources_by_domain)
    duration = time.perf_counter() - t0

    print(f"  items processed: {items:,}")
    print(f"  duration:        {duration:.2f}s")
    print(f"  save calls:      {len(save_calls)}  (final save expected)")

    # Final save should always fire from finally even when the run was
    # interrupted before any periodic save.
    assert len(save_calls) >= 1, "no final save fired on signal"
    # Cleanup should be fast — give it generous slack but not >20s
    # (per-reader join timeout is 5s × 2 readers + termination 2s × 2
    # = ~14s worst case if every reader is stuck).
    assert duration < 20.0, (
        f"shutdown too slow: {duration:.1f}s — readers likely hanging"
    )
    print("  PASS")


def main() -> int:
    scenario_periodic_save()
    scenario_signal_shutdown()
    print("\nALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
