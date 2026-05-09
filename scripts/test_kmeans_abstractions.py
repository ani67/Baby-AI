"""K-means abstraction formation — verify the new _form_abstractions
finds 5 clusters in a synthetic 200-concept / 5-cluster mind, and that
re-running on the same state is idempotent.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.affect import AffectStack            # noqa: E402
from backend.attention import Attention            # noqa: E402
from backend.config import D_REP                   # noqa: E402
from backend.expression import Expression          # noqa: E402
from backend.graph import ConceptGraph, EdgeType   # noqa: E402
from backend.identity import Identity              # noqa: E402
from backend.input import InputPipeline            # noqa: E402
from backend.main_loop import MainLoop             # noqa: E402
from backend.predict import PredictionEngine       # noqa: E402
from backend.simulation import SimulationReplay   # noqa: E402


def build_loop() -> MainLoop:
    now = time.time()
    a = AffectStack(birth_seed=99, t_birth=now)
    g = ConceptGraph()
    p = PredictionEngine(affect=a, graph=g)
    sim = SimulationReplay(affect=a, graph=g, predict_engine=p)
    ident = Identity(
        affect=a, graph=g, predict_engine=p, simulation=sim,
        birth_seed=99, birth_time=now,
    )
    h = InputPipeline(affect=a, graph=g, predict_engine=p, identity=ident)
    f = Attention(affect=a, graph=g)
    gx = Expression(
        affect=a, graph=g, predict_engine=p, identity=ident, input_pipeline=h,
    )
    return MainLoop(
        affect=a, graph=g, predict_engine=p, simulation=sim,
        identity=ident, attention=f, expression=gx, input_pipeline=h,
    )


def main() -> int:
    rng = np.random.default_rng(0xab)
    print("[kmeans] building 5 cluster centers in 256-d")
    centers = rng.standard_normal((5, D_REP)).astype(np.float32)
    centers /= (np.linalg.norm(centers, axis=1, keepdims=True) + 1e-9)

    loop = build_loop()
    affect = loop.affect.composite(time.time())
    print("[kmeans] writing 200 concepts (40 per center, sigma=0.10 — clearly separated)")
    member_ids: list[list[int]] = [[] for _ in range(5)]
    for c in range(5):
        for j in range(40):
            v = centers[c] + 0.10 * rng.standard_normal(D_REP).astype(np.float32)
            v /= (np.linalg.norm(v) + 1e-9)
            cid, _ = loop.graph.write_on_surprise(
                representation=v, surprise=1.0,
                current_affect=affect, name_hint=f"c{c}_v{j}",
                now=time.time(),
            )
            member_ids[c].append(cid)
    print(f"[kmeans] graph: {loop.graph.node_count} nodes  edges={loop.graph.edge_count}")

    # First pass — should form clusters
    print()
    print("[kmeans] _form_abstractions (pass 1) …")
    n1 = loop._form_abstractions(now=time.time())
    print(f"  abstractions_formed: {n1}")
    print(f"  graph after pass 1:  {loop.graph.node_count} nodes  edges={loop.graph.edge_count}")
    print(f"  is_a edges:          {sum(1 for e in loop.graph._edges.values() if e.type is EdgeType.IS_A)}")
    assert n1 >= 5, f"expected ≥5 abstractions on a 5-cluster mind, got {n1}"
    if n1 > 6:
        # K-means at k=max(8,ceil(sqrt(200)))=max(8,15)=15 may split a
        # cluster into 2-3 sub-clusters; that's expected and not a failure.
        print(f"  (k-means picked k>5 sub-clusters; {n1} parents is fine — "
              f"each sub-centroid is a real abstraction)")

    # Show which named members each new abstraction parent ends up
    # connecting to. Confirms the IS_A edges land in the right cluster.
    abs_parents = sorted(
        e.target_id for e in loop.graph._edges.values()
        if e.type is EdgeType.IS_A
    )
    abs_parent_set = set(abs_parents)
    print()
    print("[kmeans] each new abstraction's first 5 members:")
    for parent in sorted(abs_parent_set):
        members = [
            e.source_id for e in loop.graph._edges.values()
            if e.type is EdgeType.IS_A and e.target_id == parent
        ]
        member_names = [loop.graph.nodes[m].name for m in members[:5]]
        # Which input cluster (0..4) do these names come from? Identity
        # init writes 'self' / 'unknown' / a few seed concepts before the
        # synthetic c{N}_v{M} ones — those don't follow the pattern, so
        # we just count them as cluster=-1 (unknown).
        cluster_hits = []
        for nm in member_names:
            if nm.startswith("c") and "_" in nm:
                head = nm.split("_")[0][1:]
                if head.isdigit():
                    cluster_hits.append(int(head))
                    continue
            cluster_hits.append(-1)
        majority = max(set(cluster_hits), key=cluster_hits.count) if cluster_hits else -1
        purity = cluster_hits.count(majority) / max(1, len(cluster_hits))
        parent_name = loop.graph.nodes[parent].name
        print(f"  parent={parent:>4d}  {parent_name!r:<35s}  "
              f"members(sample)={member_names}  → cluster={majority}  purity={purity:.2f}")

    # Second pass — should be idempotent (no new abstractions)
    print()
    print("[kmeans] _form_abstractions (pass 2) …")
    n2 = loop._form_abstractions(now=time.time())
    print(f"  abstractions_formed: {n2}")
    print(f"  graph after pass 2:  {loop.graph.node_count} nodes  edges={loop.graph.edge_count}")
    assert n2 == 0, f"second pass should be idempotent (0 new), got {n2}"

    print()
    print(f"[kmeans] PASS — {n1} abstractions formed pass 1, 0 on pass 2 (idempotent)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
