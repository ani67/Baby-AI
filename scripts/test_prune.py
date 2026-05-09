"""Test prune_to_ceiling protects pinned concepts + abstraction-incident
concepts, drops the lowest-salience rest. Synthetic 200-node graph with
10 pinned, 5 abstraction parents (each owning 4 members), ceiling=50.
Patches CONCEPT_CEILING and PRUNE_TO at runtime so the test doesn't
need to mutate config.py.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend import graph as graph_module           # noqa: E402
from backend.affect import AffectStack              # noqa: E402
from backend.config import D_REP                    # noqa: E402
from backend.graph import ConceptGraph, EdgeType    # noqa: E402


def main() -> int:
    # Patch the constants for this test only.
    orig_ceiling = graph_module.CONCEPT_CEILING
    orig_prune_to = graph_module.PRUNE_TO
    graph_module.CONCEPT_CEILING = 50
    graph_module.PRUNE_TO = 40
    try:
        return _run()
    finally:
        graph_module.CONCEPT_CEILING = orig_ceiling
        graph_module.PRUNE_TO = orig_prune_to


def _run() -> int:
    rng = np.random.default_rng(0xc0de)
    g = ConceptGraph()
    a = AffectStack(birth_seed=1, t_birth=time.time())
    composite = a.composite(time.time())

    print("[prune] building 200 concepts (varied surprise + activation)")

    def unit():
        v = rng.standard_normal(D_REP).astype(np.float32)
        v /= np.linalg.norm(v) + 1e-9
        return v

    cids: list[int] = []
    for i in range(200):
        cid, _ = g.write_on_surprise(
            representation=unit(),
            surprise=float((i % 9) + 1),  # 1..9, varied
            current_affect=composite,
            name_hint=f"n{i}",
            now=time.time() - (i * 30),  # varied recency
        )
        # Vary activation_count so scoring has range.
        g.nodes[cid].activation_count = int((i % 7) + 1)
        # Give half a stronger running affect.
        if i % 2 == 0:
            g.nodes[cid].affect_trace.running_state += 0.4
        cids.append(cid)

    # Pin 10 concepts.
    pinned_cids = cids[:10]
    for cid in pinned_cids:
        g.pin(cid, reason="test")

    # Build 5 abstraction parents (each with 4 members) using IS_A edges.
    abstraction_parents: list[int] = []
    abstraction_members: list[int] = []
    for k in range(5):
        parent_cid, _ = g.write_on_surprise(
            representation=unit(), surprise=0.0,
            current_affect=composite,
            name_hint=f"abstraction:test{k}",
            now=time.time(), force_write=True,
        )
        abstraction_parents.append(parent_cid)
        # Pick 4 unpinned members.
        members = cids[10 + 4*k : 10 + 4*k + 4]
        for m in members:
            g.add_edge(m, parent_cid, EdgeType.IS_A, weight=0.7, now=time.time())
            abstraction_members.append(m)

    print(f"  graph: {g.node_count} nodes / {g.edge_count} edges / {g.pin_count} pins")
    print(f"  pinned_cids:           {pinned_cids}")
    print(f"  abstraction_parents:   {abstraction_parents}")
    print(f"  abstraction_members:   {abstraction_members}")
    print()

    # Run prune.
    print(f"[prune] CONCEPT_CEILING={graph_module.CONCEPT_CEILING}  "
          f"PRUNE_TO={graph_module.PRUNE_TO}")
    print("[prune] calling prune_to_ceiling …")
    t0 = time.perf_counter()
    n_dropped = g.prune_to_ceiling(now=time.time())
    dt = time.perf_counter() - t0
    print(f"  dropped {n_dropped} concepts in {dt*1000:.1f}ms")
    print(f"  graph after: {g.node_count} nodes / {g.edge_count} edges")
    print()

    # Assertions.
    print("[prune] assertions:")
    assert g.node_count <= graph_module.CONCEPT_CEILING, \
        f"node_count {g.node_count} should be <= ceiling {graph_module.CONCEPT_CEILING}"
    print(f"  node_count {g.node_count} <= ceiling {graph_module.CONCEPT_CEILING}: OK")

    for cid in pinned_cids:
        assert cid in g.nodes, f"pinned cid {cid} was dropped"
    print(f"  all 10 pinned concepts survived: OK")

    for cid in abstraction_parents:
        assert cid in g.nodes, f"abstraction parent {cid} was dropped"
    print(f"  all 5 abstraction parents survived: OK")

    for cid in abstraction_members:
        assert cid in g.nodes, f"abstraction member {cid} was dropped"
    print(f"  all 20 abstraction members survived: OK")

    # The protected set is 10 + 5 + 20 = 35. Plus we want under ceiling=50.
    # So 50 - 35 = 15 unprotected concepts can survive.
    survivors = set(g.nodes.keys())
    protected = set(pinned_cids) | set(abstraction_parents) | set(abstraction_members)
    unprotected_survivors = survivors - protected
    print(f"  protected: {len(protected)}  "
          f"unprotected_survivors: {len(unprotected_survivors)}  "
          f"(ceiling-protected = {graph_module.CONCEPT_CEILING - len(protected)})")

    print()
    print("[prune] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
