"""Profile loop.cycle on a populated mind to locate the per-item cost.
Usage: python3 scripts/profile_cycle.py [N=200]"""
from __future__ import annotations

import cProfile
import os
import pstats
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

os.environ.setdefault("OMP_NUM_THREADS", "1")

from backend.input import encode_text                    # noqa: E402
from backend.mind_paths import MindPaths                  # noqa: E402
from backend.persistence import MindPersistence           # noqa: E402


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    paths = MindPaths("first")
    print(f"loading {paths.db}")
    loop = MindPersistence.load(paths.db)
    print(f"  nodes={loop.graph.node_count}  edges={loop.graph.edge_count}")

    # Build a small synthetic stream of varied texts so we hit the
    # surprise + write paths (not just dedup).
    sentences = [
        "the philosopher considers the law during the festival.",
        "an architect questions the temple before dawn.",
        "the historian examines the assembly across the centuries.",
        "an apprentice rejects the inheritance under the new regime.",
        "the citizen praises the alliance after the war.",
        "a poet condemns the verdict in private.",
        "the soldier discovers the treaty by the sea.",
        "a child trusts the offering during the campaign.",
        "the slave doubts the constitution in winter.",
        "a foreigner abandons the tribute near the agora.",
    ] * (n // 10 + 1)
    sentences = sentences[:n]

    # Pre-encode (we want to profile cycle, not encoding)
    reps = [encode_text(s) for s in sentences]
    loop.predict_engine.set_ingestion_mode(True)

    # Wall-time the bare cycle path first — what we really care about.
    t0 = time.perf_counter()
    for s, rep in zip(sentences, reps):
        now = time.time()
        ingest = loop.input_pipeline.ingest_text(s, now=now, representation=rep)
        loop.cycle(ingest, now=now + 1e-3, force_respond=False, skip_simulation=True)
    elapsed = time.perf_counter() - t0
    print(f"\n[wall] {n} cycles in {elapsed*1000:.1f}ms = "
          f"{elapsed*1000/n:.2f}ms/cycle  ({n/elapsed:,.0f}/s)")
    print(f"  graph after: nodes={loop.graph.node_count} edges={loop.graph.edge_count}")

    # Now profile a fresh batch (some new sentences to keep surprise active)
    sentences2 = [
        f"the {role} {verb} the {obj} {qual}."
        for role in ("scholar", "merchant", "elder", "judge")
        for verb in ("interrogates", "remembers", "fears", "celebrates")
        for obj in ("verdict", "alliance", "harvest", "harbor")
        for qual in ("at dawn", "at noon", "at dusk", "at night")
    ][:n]
    reps2 = [encode_text(s) for s in sentences2]

    pr = cProfile.Profile()
    pr.enable()
    for s, rep in zip(sentences2, reps2):
        now = time.time()
        ingest = loop.input_pipeline.ingest_text(s, now=now, representation=rep)
        loop.cycle(ingest, now=now + 1e-3, force_respond=False, skip_simulation=True)
    pr.disable()
    print(f"\n=== top 25 cumulative time (cycle) ===")
    pstats.Stats(pr).sort_stats("cumulative").print_stats(25)
    print(f"\n=== top 15 internal time (cycle) ===")
    pstats.Stats(pr).sort_stats("tottime").print_stats(15)
    return 0


if __name__ == "__main__":
    sys.exit(main())
