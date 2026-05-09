"""One-time densification: bring a graph built under AUTO_LINK_K=1
up to AUTO_LINK_K=3 density.

The v0.7b production run wrote each fresh concept with similar_to
edges to its single nearest neighbor. The result was a ~3.7
edges/node graph that the spread function could barely traverse:
active sets stuck at 1-3 concepts even with 73K nodes available.

This script walks every concept, finds its top-3 nearest neighbors
above AUTO_LINK_MIN_COSINE, and adds similar_to edges that don't
already exist. add_edge is idempotent on (source, target, type), so
re-running this is safe.

Usage:
    MIND_NAME=first python3 scripts/densify_edges.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

os.environ.setdefault("OMP_NUM_THREADS", "1")

from backend.config import AUTO_LINK_K, AUTO_LINK_MIN_COSINE   # noqa: E402
from backend.graph import EdgeType                              # noqa: E402
from backend.mind_paths import MindPaths                        # noqa: E402
from backend.persistence import MindPersistence                 # noqa: E402


def main() -> int:
    mind_name = os.environ.get("MIND_NAME", "first")
    paths = MindPaths(mind_name)
    print(f"loading mind '{mind_name}' from {paths.db}")
    loop = MindPersistence.load(paths.db)
    g = loop.graph
    # Persistence does NOT auto-build the cosine index after load (the
    # caller is supposed to). The densification path's whole point is
    # to call search_k against the index, so build it here.
    print("  building cosine index over loaded nodes …")
    t_idx = time.perf_counter()
    g._rebuild_index()
    print(f"    index ntotal={g._index.ntotal:,}  "
          f"({time.perf_counter() - t_idx:.1f}s)")
    now = time.time()

    nodes_n = g.node_count
    edges_before = g.edge_count
    mean_before = edges_before / max(nodes_n, 1)
    print(f"  before: {nodes_n:,} nodes / {edges_before:,} edges / "
          f"{mean_before:.1f} mean edges per node")
    print(f"  AUTO_LINK_K={AUTO_LINK_K}  AUTO_LINK_MIN_COSINE={AUTO_LINK_MIN_COSINE}")
    print()

    # search_k(k=K+1) so we can drop self from the top result.
    K = AUTO_LINK_K
    SEARCH_K = K + 1

    edges_added = 0
    skipped_existing = 0
    skipped_low_sim = 0
    skipped_self = 0

    items = list(g.nodes.items())
    t0 = time.perf_counter()

    for i, (cid, node) in enumerate(items):
        if i and i % 5000 == 0:
            elapsed = time.perf_counter() - t0
            rate = i / max(elapsed, 1e-9)
            eta = (len(items) - i) / max(rate, 1e-9)
            print(f"  [{i:>6,}/{len(items):,}]  "
                  f"+{edges_added:,} edges  ({rate:.0f}/s, "
                  f"ETA {eta:.0f}s)")

        emb = node.embedding
        norm = float(np.linalg.norm(emb))
        if norm < 1e-9:
            continue
        q = (emb / norm).astype(np.float32, copy=False)

        sims, neighbor_ids = g._index.search_k(q, k=SEARCH_K)
        for sim, nid in zip(sims, neighbor_ids):
            if int(nid) == int(cid):
                skipped_self += 1
                continue
            if sim < AUTO_LINK_MIN_COSINE:
                skipped_low_sim += 1
                break  # sims are sorted descending; rest are lower
            key = (int(cid), int(nid), EdgeType.SIMILAR_TO)
            if key in g._edges:
                skipped_existing += 1
                continue
            g.add_edge(
                source_id=int(cid),
                target_id=int(nid),
                type=EdgeType.SIMILAR_TO,
                weight=float(sim),
                now=now,
            )
            edges_added += 1

    elapsed = time.perf_counter() - t0
    edges_after = g.edge_count
    mean_after = edges_after / max(nodes_n, 1)

    print()
    print(f"  done in {elapsed:.1f}s")
    print(f"  edges added:       {edges_added:,}")
    print(f"  already existed:   {skipped_existing:,}")
    print(f"  below min-cosine:  {skipped_low_sim:,}")
    print(f"  self-skips:        {skipped_self:,}")
    print()
    print(f"  after:  {nodes_n:,} nodes / {edges_after:,} edges / "
          f"{mean_after:.1f} mean edges per node")
    print(f"  delta:  +{edges_after - edges_before:,} edges  "
          f"({mean_after - mean_before:+.1f} mean/node)")

    print()
    print("saving …")
    MindPersistence(paths.db).save(loop, now=time.time())
    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
