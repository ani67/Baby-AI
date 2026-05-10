"""One-shot score-based prune on mind 'first'.

The v0.7 pruner protected anything incident on an IS_A edge
(both abstraction parents AND members). On the v0.8 mature graph
that set saturated to ~96% — collapsing the candidate pool. Fix
in graph.py: protect_is_a_members defaults to False, parents
remain protected (always), members are scored along with everyone
else.

This script:
  1. loads mind 'first'
  2. rebuilds the MPS index
  3. forces a low ceiling/target (60K)
  4. calls prune_to_ceiling with protect_is_a_members=False
  5. saves
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend import config                              # noqa: E402
from backend import graph as graph_mod                  # noqa: E402
from backend.mind_paths import MindPaths                # noqa: E402
from backend.persistence import MindPersistence         # noqa: E402


TARGET_CEILING = 60_000
TARGET_PRUNE_TO = 55_000


def main() -> int:
    paths = MindPaths(mind_name="first")
    print(f"[score-prune] loading {paths.db}")
    loop = MindPersistence.load(paths.db)
    g = loop.graph

    nodes_before = g.node_count
    edges_before = g.edge_count
    print(f"[score-prune] before: {nodes_before:,} nodes / {edges_before:,} edges")

    print("[score-prune] rebuilding cosine index (one-time)")
    t_idx = time.perf_counter()
    g._rebuild_index()
    print(f"  index ntotal={g._index.ntotal:,} in {time.perf_counter() - t_idx:.2f}s")

    # The early-return guard reads CONCEPT_CEILING from the graph
    # module's namespace (`from .config import CONCEPT_CEILING, PRUNE_TO`
    # at import time). Monkey-patch both there for this script so the
    # score-based prune actually fires.
    print(f"[score-prune] forcing ceiling={TARGET_CEILING:,} prune_to={TARGET_PRUNE_TO:,}")
    graph_mod.CONCEPT_CEILING = TARGET_CEILING
    graph_mod.PRUNE_TO = TARGET_PRUNE_TO

    # Sanity-check the graph module's CONCEPT_CEILING is what we think.
    print(f"  graph_mod.CONCEPT_CEILING={graph_mod.CONCEPT_CEILING:,} "
          f"graph_mod.PRUNE_TO={graph_mod.PRUNE_TO:,}")

    print("[score-prune] running prune_to_ceiling(protect_is_a_members=False) …")
    t0 = time.perf_counter()
    n_removed = g.prune_to_ceiling(time.time(), protect_is_a_members=False)
    dt = time.perf_counter() - t0
    print(f"  removed {n_removed:,} concepts in {dt:.2f}s")

    nodes_after = g.node_count
    edges_after = g.edge_count
    print(f"[score-prune] after:  {nodes_after:,} nodes / {edges_after:,} edges")
    print(f"[score-prune] delta:  -{nodes_before - nodes_after:,} nodes "
          f"/ -{edges_before - edges_after:,} edges")

    print("[score-prune] saving …")
    t1 = time.perf_counter()
    MindPersistence(paths.db).save(loop, time.time())
    print(f"  saved in {time.perf_counter() - t1:.2f}s — OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
