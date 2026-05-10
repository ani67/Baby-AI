"""One-shot quality prune on mind 'first'.

Loads the mind, runs ConceptGraph.prune_weak_concepts, saves.
Single-process script — the lifespan-style stale-snapshot guard
that protects the API doesn't apply here, so we save unconditionally.
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.mind_paths import MindPaths              # noqa: E402
from backend.persistence import MindPersistence       # noqa: E402


def main() -> int:
    paths = MindPaths(mind_name="first")
    print(f"[quality-prune] loading {paths.db}")
    loop = MindPersistence.load(paths.db)
    g = loop.graph

    nodes_before = g.node_count
    edges_before = g.edge_count
    print(f"[quality-prune] before: {nodes_before:,} nodes / {edges_before:,} edges")

    # The MPS index is unbuilt after MindPersistence.load() (the loader
    # leaves it for the caller to populate). prune_weak_concepts calls
    # self._index.remove_many() which is safe on an empty index, but we
    # rebuild here so the saved-back state has a populated index for the
    # next consumer of mind.db. Mirrors what scripts/prune_graph.py does.
    print("[quality-prune] rebuilding cosine index (one-time)")
    t_idx = time.perf_counter()
    g._rebuild_index()
    print(f"  index ntotal={g._index.ntotal:,} in {time.perf_counter() - t_idx:.2f}s")

    print("[quality-prune] running prune_weak_concepts …")
    t0 = time.perf_counter()
    n_removed = g.prune_weak_concepts(time.time())
    dt = time.perf_counter() - t0
    print(f"  removed {n_removed:,} weak concepts in {dt:.2f}s")

    nodes_after = g.node_count
    edges_after = g.edge_count
    print(f"[quality-prune] after:  {nodes_after:,} nodes / {edges_after:,} edges")
    print(f"[quality-prune] delta:  -{nodes_before - nodes_after:,} nodes "
          f"/ -{edges_before - edges_after:,} edges")

    print("[quality-prune] saving …")
    t1 = time.perf_counter()
    MindPersistence(paths.db).save(loop, time.time())
    print(f"  saved in {time.perf_counter() - t1:.2f}s — OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
