"""Fix 1 — verify FAISS nearest is bit-identical to brute-force, then
benchmark the speedup at N=5,000 over 1,000 queries.

Builds a synthetic graph with 5K random unit-norm vectors, queries 1K
random unit-norm vectors. Compares FAISS results to a numpy brute-force
implementation over the same matrix. Asserts every (cid, sim) pair
matches within float32 tolerance.

Then re-runs Phase-1-style write_on_surprise / find_or_match smoke checks
to confirm no regression.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.affect import AffectStack            # noqa: E402
from backend.config import D_REP                   # noqa: E402
from backend.graph import ConceptGraph             # noqa: E402


def _unit(rng: np.random.Generator, n: int, d: int) -> np.ndarray:
    v = rng.standard_normal((n, d)).astype(np.float32)
    norms = np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
    return v / norms


def main() -> int:
    rng = np.random.default_rng(0xc0ffee)

    print("[fix1] building synthetic graph: 5,000 unit vectors")
    g = ConceptGraph()
    affect = AffectStack(birth_seed=1, t_birth=time.time())
    composite = affect.composite(time.time())
    embeddings = _unit(rng, 5000, D_REP)
    t0 = time.perf_counter()
    for i in range(5000):
        g.write_on_surprise(
            representation=embeddings[i],
            surprise=1.0,
            current_affect=composite,
            name_hint=f"v{i}",
            now=time.time(),
        )
    t_build = time.perf_counter() - t0
    print(f"[fix1] build done in {t_build:.2f}s  -> nodes={g.node_count}, "
          f"faiss.ntotal={g._faiss_index.ntotal}")

    # --- Correctness: brute force vs FAISS, 1K queries --
    print()
    print("[fix1] correctness: brute-force vs FAISS over 1,000 queries")
    queries = _unit(rng, 1000, D_REP)
    # Build the brute-force matrix (matches existing _rebuild_matrix).
    g._rebuild_matrix()
    M = g._matrix
    M_ids = g._matrix_ids
    assert M is not None
    assert len(M_ids) == 5000

    mismatches = 0
    sim_diffs = []
    for q in queries:
        # Brute force: argmax over M @ q.
        sims = M @ q
        bf_idx = int(np.argmax(sims))
        bf_cid = M_ids[bf_idx]
        bf_sim = float(sims[bf_idx])

        # FAISS via nearest()
        result = g.nearest(q)
        assert result is not None
        faiss_cid, faiss_sim = result
        if bf_cid != faiss_cid:
            mismatches += 1
        sim_diffs.append(abs(bf_sim - faiss_sim))

    max_diff = max(sim_diffs)
    print(f"[fix1] mismatches: {mismatches} / 1,000")
    print(f"[fix1] max |sim_diff|: {max_diff:.2e}")
    assert mismatches == 0, "FAISS results differ from brute force"
    assert max_diff < 1e-5, f"sim values diverge too much: {max_diff}"

    # --- Speed: brute vs FAISS over the same 1K queries --
    # This first variant queries against a STATIC graph: brute force gets
    # to keep its rebuilt matrix across all queries (best case for brute).
    print()
    print("[fix1] speed benchmark — STATIC graph, 1,000 queries")
    t0 = time.perf_counter()
    for q in queries:
        sims = M @ q
        _ = int(np.argmax(sims))
    t_bf = time.perf_counter() - t0

    t0 = time.perf_counter()
    for q in queries:
        _ = g.nearest(q)
    t_faiss = time.perf_counter() - t0

    bf_qps = 1000 / max(t_bf, 1e-9)
    faiss_qps = 1000 / max(t_faiss, 1e-9)
    print(f"  brute force : {t_bf:.3f}s  ({bf_qps:>8,.0f} q/s)")
    print(f"  FAISS       : {t_faiss:.3f}s  ({faiss_qps:>8,.0f} q/s)")
    print(f"  speedup     : {t_bf / max(t_faiss, 1e-9):.2f}x")

    # --- Speed: realistic write+query interleaving (the curriculum hot path)
    # Every write_on_surprise sets _matrix_dirty=True, and the subsequent
    # nearest() call triggers a full O(N*D) rebuild of the brute-force matrix.
    # FAISS keeps its index in sync incrementally so this overhead vanishes.
    print()
    print("[fix1] speed benchmark — INTERLEAVED write+query (curriculum hot path)")

    # Brute-force version: build graph from scratch, no FAISS used.
    # We simulate by importing graph and disabling its faiss path.
    g_bf = ConceptGraph()
    # The class always maintains FAISS now, but we disable it for the
    # brute-force timing by short-circuiting nearest() to use _cosine_to_all.
    def bf_nearest(rep):
        if not g_bf.nodes:
            return None
        sims = g_bf._cosine_to_all(rep)
        idx = int(np.argmax(sims))
        return g_bf._matrix_ids[idx], float(sims[idx])

    t0 = time.perf_counter()
    for i in range(2000):
        g_bf.write_on_surprise(embeddings[i], 1.0, composite, f"v{i}", time.time())
        bf_nearest(embeddings[i])
    t_bf_inter = time.perf_counter() - t0

    g_faiss = ConceptGraph()
    t0 = time.perf_counter()
    for i in range(2000):
        g_faiss.write_on_surprise(embeddings[i], 1.0, composite, f"v{i}", time.time())
        g_faiss.nearest(embeddings[i])
    t_faiss_inter = time.perf_counter() - t0

    bf_inter_qps = 2000 / max(t_bf_inter, 1e-9)
    faiss_inter_qps = 2000 / max(t_faiss_inter, 1e-9)
    print(f"  brute force (rebuild every write) : {t_bf_inter:.3f}s  "
          f"({bf_inter_qps:>8,.0f} cycles/s)")
    print(f"  FAISS (incremental add)            : {t_faiss_inter:.3f}s  "
          f"({faiss_inter_qps:>8,.0f} cycles/s)")
    print(f"  speedup                             : "
          f"{t_bf_inter / max(t_faiss_inter, 1e-9):.2f}x")

    # --- Regression: Phase-1 minimum (write + find_or_match) --
    print()
    print("[fix1] regression: write_on_surprise + find_or_match dedup")
    g2 = ConceptGraph()
    a, b = _unit(rng, 2, D_REP)
    cid_a, new_a = g2.write_on_surprise(a, 1.0, composite, "a", time.time())
    cid_b, new_b = g2.write_on_surprise(b, 1.0, composite, "b", time.time())
    assert new_a and new_b
    assert cid_a != cid_b
    # Re-write of `a` (within R_MATCH default 0.92) should dedupe.
    a_jitter = a + 0.0001 * rng.standard_normal(D_REP).astype(np.float32)
    a_jitter /= np.linalg.norm(a_jitter)
    cid_a2, new_a2 = g2.write_on_surprise(a_jitter, 1.0, composite, "a2", time.time())
    assert not new_a2
    assert cid_a2 == cid_a
    print(f"  dedup OK: re-written near-duplicate of cid={cid_a} returned same cid")

    # --- Regression: load → rebuild path --
    print()
    print("[fix1] regression: simulate persistence reload (rebuild_faiss_index)")
    g3 = ConceptGraph()
    for i in range(100):
        g3.write_on_surprise(embeddings[i], 1.0, composite, f"r{i}", time.time())
    assert g3._faiss_index.ntotal == 100
    g3._rebuild_faiss_index()
    assert g3._faiss_index.ntotal == 100
    # Verify nearest still works after rebuild
    res = g3.nearest(embeddings[42])
    assert res is not None
    cid, sim = res
    print(f"  post-rebuild nearest(emb[42]) -> cid={cid} sim={sim:.4f}  (expect sim=1.0)")
    assert sim > 0.99

    print()
    print("[fix1] PASS — correctness + speedup + regressions")
    return 0


if __name__ == "__main__":
    sys.exit(main())
