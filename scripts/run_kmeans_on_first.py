"""Run the new k-means _form_abstractions on data/first/mind.db
(read-only-ish — we load, cluster, and write abstraction parents back).

Reports:
  - eligible-node count (excluding existing members + parents)
  - chosen k
  - abstractions_formed
  - for each new abstraction, the cosine-nearest NAMED concept to its
    centroid (the qualitative interpretation of "what did the mind
    decide was abstract enough to represent multiple others?")
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.graph import EdgeType                 # noqa: E402
from backend.persistence import MindPersistence    # noqa: E402


def main() -> int:
    db = "data/first/mind.db"
    print(f"[real] loading mind from {db}")
    loop = MindPersistence.load(db)
    g = loop.graph
    print(f"  graph: {g.node_count} nodes / {g.edge_count} edges")

    # How many already-abstracted? Should be 0 on a mind that never
    # successfully ran the BFS abstraction path.
    n_is_a = sum(1 for e in g._edges.values() if e.type is EdgeType.IS_A)
    print(f"  pre-existing is_a edges: {n_is_a}")
    print()

    # Eligible count
    already_member = {e.source_id for e in g._edges.values() if e.type is EdgeType.IS_A}
    is_abs         = {e.target_id for e in g._edges.values() if e.type is EdgeType.IS_A}
    eligible = [cid for cid in g.nodes if cid not in (already_member | is_abs)]
    n = len(eligible)
    k = max(8, int(np.ceil(np.sqrt(n))))
    k = min(k, n // 3)
    print(f"  eligible concepts: {n}")
    print(f"  k (chosen):        {k}")
    print()

    pre_nodes = g.node_count
    pre_edges = g.edge_count

    print("[real] running _form_abstractions …")
    t0 = time.perf_counter()
    n_formed = loop._form_abstractions(now=time.time())
    dt = time.perf_counter() - t0
    print(f"  abstractions_formed: {n_formed}")
    print(f"  duration:            {dt:.2f}s")
    print(f"  nodes: {pre_nodes:,} -> {g.node_count:,}")
    print(f"  edges: {pre_edges:,} -> {g.edge_count:,}")
    print()

    # For each new abstraction parent, find the nearest NAMED real concept
    # (excluding other abstraction parents and self) and print it.
    # Each parent has many incoming is_a edges (one per member) — dedup.
    new_parents = sorted(set(
        e.target_id for e in g._edges.values()
        if e.type is EdgeType.IS_A and e.target_id > pre_nodes
    ))
    abstraction_ids = set(new_parents)

    print(f"[real] each new abstraction → cosine-nearest named non-abstraction concept:")
    print(f"  (these are the mind's emerging categories — what does it think these clusters MEAN?)")
    print()
    for parent in new_parents[:60]:
        emb = g.nodes[parent].embedding
        # Find nearest NAMED concept that isn't itself an abstraction
        # parent and isn't this same parent.
        best = None
        best_sim = -1.0
        rep_norm = float(np.linalg.norm(emb))
        if rep_norm < 1e-9:
            continue
        unit = emb / rep_norm
        for other_cid, other in g.nodes.items():
            if other_cid == parent: continue
            if other_cid in abstraction_ids: continue
            if not other.name or other.name.startswith("abstraction:"): continue
            on = float(np.linalg.norm(other.embedding))
            if on < 1e-9: continue
            sim = float(np.dot(unit, other.embedding / on))
            if sim > best_sim:
                best_sim = sim
                best = other_cid
        members_of_this = [
            e.source_id for e in g._edges.values()
            if e.type is EdgeType.IS_A and e.target_id == parent
        ]
        n_members = len(members_of_this)
        if best is not None:
            best_name = (g.nodes[best].name or "").replace("\n", " ")[:80]
            print(f"  #{parent:>5d}  members={n_members:>4d}  "
                  f"nearest_concept_sim={best_sim:.3f}  "
                  f"[{g.nodes[parent].name}]")
            print(f"           → near {best_name!r}")
        else:
            print(f"  #{parent:>5d}  members={n_members:>4d}  (no named neighbor)")
    print()
    if n_formed > 60:
        print(f"  (… and {n_formed - 60} more)")
    print(f"[real] DONE — wrote {n_formed} new abstraction parents to {db}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
