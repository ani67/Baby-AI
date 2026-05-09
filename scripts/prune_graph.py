"""One-time pruning script for oversized graphs.

Run after a curriculum completes if node_count >> CONCEPT_CEILING.
Loads the mind, runs k-means abstraction first (so cluster members
become protected by their IS_A edges before pruning evaluates them),
then loops prune_to_ceiling until node_count <= PRUNE_TO. Saves.

Usage:
    MIND_NAME=first python3 scripts/prune_graph.py
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.config import CONCEPT_CEILING, PRUNE_TO   # noqa: E402
from backend.mind_paths import MindPaths               # noqa: E402
from backend.persistence import MindPersistence        # noqa: E402


def main() -> int:
    mind_name = os.environ.get("MIND_NAME", "first")
    paths = MindPaths(mind_name)

    print(f"loading mind '{mind_name}' from {paths.db}")
    loop = MindPersistence.load(paths.db)
    g = loop.graph
    # Curriculum-side workload — build FAISS index for the spread/auto-link
    # paths that find_or_match touches during abstraction formation.
    g._rebuild_faiss_index()
    print(f"  before: {g.node_count:,} nodes / {g.edge_count:,} edges / "
          f"{g.pin_count} pins")
    print(f"  ceiling: {CONCEPT_CEILING:,}   prune_to: {PRUNE_TO:,}")
    print()

    # 1. Run k-means abstraction first so cluster centroids land before
    # the prune scores everything. Members of clusters become protected
    # by the inbound IS_A edges _form_abstractions lays.
    print("step 1 — running k-means abstraction …")
    t0 = time.perf_counter()
    n_abs = loop._form_abstractions(now=time.time())
    dt_abs = time.perf_counter() - t0
    print(f"  abstractions formed: {n_abs:,}  ({dt_abs:.1f}s)")
    print(f"  graph after:         {g.node_count:,} nodes / "
          f"{g.edge_count:,} edges")
    print()

    # 2. Loop prune_to_ceiling until under the ceiling. One call should
    # be enough — but if the resulting state is somehow still over the
    # ceiling (immune-set growth, etc.), iterate.
    print("step 2 — pruning weakest concepts …")
    rounds = 0
    while g.node_count > CONCEPT_CEILING:
        rounds += 1
        t0 = time.perf_counter()
        pre = g.node_count
        n_dropped = g.prune_to_ceiling(now=time.time())
        dt = time.perf_counter() - t0
        print(f"  round {rounds}: dropped {n_dropped:,}  "
              f"({pre:,} → {g.node_count:,})  ({dt:.1f}s)")
        if n_dropped == 0:
            # Nothing more we can drop (everything else is protected).
            print("  (no further candidates — all remaining are pinned or "
                  "members of abstractions)")
            break
    print()
    print(f"  after:  {g.node_count:,} nodes / {g.edge_count:,} edges / "
          f"{g.pin_count} pins")

    # 3. Save.
    print()
    print(f"saving to {paths.db} …")
    MindPersistence(paths.db).save(loop, now=time.time())
    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
