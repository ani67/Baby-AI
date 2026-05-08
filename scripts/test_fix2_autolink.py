"""Fix 2 — verify auto-link emits exactly AUTO_LINK_K similar_to edges
per fresh concept (not 3). Also confirm symmetric pair-laying so the
edge count grows by 2*K per write."""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.affect import AffectStack            # noqa: E402
from backend.config import AUTO_LINK_K, D_REP     # noqa: E402
from backend.graph import ConceptGraph, EdgeType  # noqa: E402


def main() -> int:
    print(f"[fix2] AUTO_LINK_K = {AUTO_LINK_K}")
    rng = np.random.default_rng(0xbeef)
    g = ConceptGraph()
    affect = AffectStack(birth_seed=1, t_birth=time.time())
    composite = affect.composite(time.time())

    # Random unit vectors in 256-d are nearly orthogonal (cosine ~0.06)
    # so they'd all fall below min_cosine=0.25. We instead build clustered
    # vectors: 5 cluster centers, each new vector is a perturbation of a
    # randomly-chosen center. That gives realistic cosine neighborhoods
    # similar to what GloVe-encoded sentences produce.
    n_centers = 5
    centers = rng.standard_normal((n_centers, D_REP)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True) + 1e-9

    def sample_clustered() -> np.ndarray:
        c = centers[rng.integers(n_centers)]
        noise = 0.15 * rng.standard_normal(D_REP).astype(np.float32)
        v = c + noise
        return v / (np.linalg.norm(v) + 1e-9)

    seed_count = 5
    for i in range(seed_count):
        g.write_on_surprise(sample_clustered(), 1.0, composite, f"seed{i}", time.time())
    for cid in list(g.nodes.keys()):
        g.link_to_nearest_neighbors(cid, now=time.time())

    print(f"[fix2] after seeding {seed_count} clustered nodes  edges={g.edge_count}")

    n_writes = 100
    n_per_write_edges: list[int] = []

    for i in range(n_writes):
        v = sample_clustered()
        e_before = g.edge_count
        cid, is_new = g.write_on_surprise(v, 1.0, composite, f"v{i}", time.time())
        # find_or_match dedup may have hit (clustered vectors near R_MATCH=0.92).
        if not is_new:
            n_per_write_edges.append(0)
            continue
        new_edges = g.link_to_nearest_neighbors(cid, now=time.time())
        e_after = g.edge_count
        n_per_write_edges.append(e_after - e_before)

    # link_to_nearest_neighbors lays symmetric pairs: K pairs = 2*K edges.
    expected_per_write = 2 * AUTO_LINK_K

    # The per-call invariant: each write+link call adds either 0 edges
    # (no neighbor crossed min_cosine) or exactly 2*K edges (symmetric
    # pair laying for K neighbors). Out-edge count *over time* doesn't
    # check this because later concepts' link calls add reverse-direction
    # OUT edges to earlier concepts.
    distinct = sorted(set(n_per_write_edges))
    print(f"[fix2] per-write edge growth distinct values: {distinct}")
    print(f"[fix2] expected:                                "
          f"{{0, {expected_per_write}}}")
    valid = {0, expected_per_write}
    bad = [c for c in n_per_write_edges if c not in valid]
    n_zero = sum(1 for c in n_per_write_edges if c == 0)
    n_pair = sum(1 for c in n_per_write_edges if c == expected_per_write)
    n_dedup = n_writes - n_zero - n_pair
    print(f"[fix2] of {n_writes} writes:")
    print(f"          {n_pair:>3d}  added 2*K = {expected_per_write} edges (neighbor above min_cosine)")
    print(f"          {n_zero:>3d}  added 0 edges (no neighbor / dedup)")
    print(f"          {n_dedup:>3d}  added something else  ← FAIL if non-zero")
    assert not bad, f"per-write edge growth violated: {bad[:10]}"

    # OLD behavior would have added up to 2*3=6 per write. Our distinct
    # set must NOT include any value > 2 for K=1.
    assert max(distinct) <= expected_per_write, \
        f"per-write growth exceeds 2*K — auto-link stuck on K=3? distinct={distinct}"

    print()
    print(f"[fix2] PASS — auto-link emits AUTO_LINK_K={AUTO_LINK_K} edges per write")
    return 0


if __name__ == "__main__":
    sys.exit(main())
