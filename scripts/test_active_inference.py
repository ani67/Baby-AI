"""Active inference smoke test.

Builds a fresh in-memory MainLoop, seeds the graph with ~100 synthetic
concepts via the standard ingest path (so activation_count and edges
look realistic), then runs idle() repeatedly and verifies:

  - the IdleResult carries the new inference_events / new_connections
    fields and they sum > 0 over the run;
  - at least one cycle wrote a new concept (was_new_write surfaced
    through gap → IdleResult.new_connections);
  - the graph's edge count grew between baseline (just after seeding)
    and the final state — auto-link inside observe() lays SIMILAR_TO
    edges from each new bridge concept to its cosine neighbors.

Mind 'first' is owned by Subagent 2's curriculum run; this test never
touches mind.db. The MainLoop here is constructed in-memory only.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# OMP / MPS guard before backend imports — the test runs encoders briefly
# during ingest and the unguarded variants can deadlock under multiprocess
# pytest-style runners. We're not using mp here, but matching the rest of
# the test suite keeps behavior comparable.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from backend.affect import AffectStack             # noqa: E402
from backend.attention import Attention             # noqa: E402
from backend.config import D_REP                    # noqa: E402
from backend.expression import Expression           # noqa: E402
from backend.graph import ConceptGraph              # noqa: E402
from backend.identity import Identity               # noqa: E402
from backend.input import InputPipeline             # noqa: E402
from backend.main_loop import MainLoop              # noqa: E402
from backend.predict import PredictionEngine        # noqa: E402
from backend.simulation import SimulationReplay     # noqa: E402


N_SEED_CONCEPTS = 100
N_IDLE_CYCLES   = 10
BIRTH_SEED      = 9919


def construct_fresh_mind() -> MainLoop:
    """Same shape as scripts/test_parallel_ingestion.construct_fresh_mind
    — duplicated locally so this test is self-contained and immune to
    refactors over there."""
    now = time.time()
    a = AffectStack(birth_seed=BIRTH_SEED, t_birth=now)
    g = ConceptGraph()
    p = PredictionEngine(affect=a, graph=g)
    sim = SimulationReplay(affect=a, graph=g, predict_engine=p)
    ident = Identity(
        affect=a, graph=g, predict_engine=p, simulation=sim,
        birth_seed=BIRTH_SEED, birth_time=now,
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


def seed_graph(loop: MainLoop, n: int) -> None:
    """Write n synthetic concepts directly into the graph and bump their
    activation_count. We bypass the input pipeline's text path because
    there's no useful coverage in encoding random text — the encoder
    isn't what's under test, the active inference loop is. write_on_surprise
    + a manual activation bump is the smallest way to produce a
    realistic-looking graph: fresh ConceptNode objects with non-zero
    activation_count and embeddings that span rep space.

    Concepts are sampled from K=8 cluster centers so cosine neighbors
    actually exist within the AUTO_LINK_MIN_COSINE=0.25 threshold; a
    pure random-unit-vector population sits at cosine ≈ 0 between any
    two points and link_to_nearest_neighbors finds nothing to attach.
    """
    rng = np.random.default_rng(BIRTH_SEED + 1)
    now = time.time()
    affect = loop.affect.composite(now)

    # K clusters, each producing roughly n/K members. Cluster centers
    # are random unit vectors; members are center + small noise then
    # renormalized. In D_REP=256, a sigma of 0.05 keeps within-cluster
    # cosine in the 0.85+ range while across-cluster stays near zero —
    # so AUTO_LINK_MIN_COSINE=0.25 actually has neighbors to find.
    # (At sigma 0.4 the noise dominates the centroid in high-D and
    # everything looks random again.)
    n_clusters = 8
    centers = rng.standard_normal((n_clusters, D_REP)).astype(np.float32)
    centers /= np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-9)

    for i in range(n):
        cluster = i % n_clusters
        noise = 0.05 * rng.standard_normal(D_REP).astype(np.float32)
        v = centers[cluster] + noise
        v = v / max(float(np.linalg.norm(v)), 1e-9)
        cid, is_new = loop.graph.write_on_surprise(
            representation=v,
            surprise=1.0,
            current_affect=affect,
            name_hint=f"seed_{i}_c{cluster}",
            now=now + i * 1e-3,
        )
        # Auto-link so the graph has the same shape as a real run.
        if is_new:
            loop.graph.link_to_nearest_neighbors(cid, now=now + i * 1e-3)
        # Bump activation_count — in a real run, find_or_match strengthen
        # paths and F.spread visits would do this naturally. Without the
        # bump every node has activation_count == 1 (set by write_on_surprise),
        # which still satisfies MIN_ACTIVATION_COUNT, but we want a
        # non-uniform distribution so the weighted sampler has signal.
        bumps = int(rng.integers(0, 5))
        for _ in range(bumps):
            loop.graph.nodes[cid].activation_count += 1


def main() -> int:
    loop = construct_fresh_mind()

    # Seed.
    seed_graph(loop, N_SEED_CONCEPTS)
    baseline_nodes = loop.graph.node_count
    baseline_edges = loop.graph.edge_count
    print(
        f"baseline: nodes={baseline_nodes}  edges={baseline_edges}  "
        f"(after seeding {N_SEED_CONCEPTS} synthetic concepts)"
    )

    # Push last_observation_t into the past so idle()'s "no fresh
    # stimulus in last 5s" gate is satisfied. Birth time is from
    # construct_fresh_mind — we shove it back 60s so idle() runs.
    loop.last_observation_t = time.time() - 60.0

    total_events = 0
    total_writes = 0
    saw_replayed_field = False
    saw_new_field = False

    for cycle_idx in range(N_IDLE_CYCLES):
        result = loop.idle(now=time.time(), max_replays=2)
        # Field presence — the test would crash on the next line if
        # IdleResult didn't actually carry the new fields.
        events = int(result.inference_events)
        writes = int(result.new_connections)
        total_events += events
        total_writes += writes
        if hasattr(result, "replayed_count"):
            saw_replayed_field = True
        if hasattr(result, "new_connections"):
            saw_new_field = True
        print(
            f"  idle[{cycle_idx + 1:2d}] events={events}  "
            f"new_writes={writes}  replays={result.replayed_count}  "
            f"nodes={loop.graph.node_count}  edges={loop.graph.edge_count}"
        )

    final_nodes = loop.graph.node_count
    final_edges = loop.graph.edge_count
    print()
    print(f"summary: total_events={total_events}  "
          f"total_new_writes={total_writes}")
    print(
        f"         nodes {baseline_nodes} → {final_nodes} "
        f"(Δ {final_nodes - baseline_nodes})"
    )
    print(
        f"         edges {baseline_edges} → {final_edges} "
        f"(Δ {final_edges - baseline_edges})"
    )

    # ---- assertions ----
    assert saw_replayed_field, "IdleResult missing legacy `replayed_count` field"
    assert saw_new_field, "IdleResult missing new `new_connections` field"
    assert total_events > 0, (
        f"active inference produced 0 events across {N_IDLE_CYCLES} idle "
        f"cycles — pair sampling or eligibility filter is broken"
    )
    assert total_writes > 0, (
        f"active inference produced 0 new concepts across {N_IDLE_CYCLES} "
        f"cycles — bridge composition never crossed surprise threshold"
    )
    assert final_edges > baseline_edges, (
        f"graph edge count did not grow ({baseline_edges} → {final_edges}) "
        f"— auto-link via observe() should have laid SIMILAR_TO edges from "
        f"each new bridge concept"
    )

    print()
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
