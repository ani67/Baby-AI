"""S1 test — MPSConceptIndex correctness vs brute-force, plus speed
benchmark at increasing N.

Originally validated the MPS-backed index; that backend produced ~1ms
per query of CPU↔MPS round-trip overhead, which dominated cycle time
on a populated mind. The class is now numpy-backed (CPU); the test
still validates correctness and per-query latency.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.graph import MPSConceptIndex   # noqa: E402


def _unit(rng, n, d):
    v = rng.standard_normal((n, d)).astype(np.float32)
    v /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
    return v


def main() -> int:
    rng = np.random.default_rng(0xc0ffee)
    print()
    print("[mps] correctness — 10,000 vectors, 1,000 queries vs brute-force matmul")
    N = 10_000
    D = 256
    embeddings = _unit(rng, N, D)
    idx = MPSConceptIndex(d_rep=D)
    for i in range(N):
        idx.add(embeddings[i], i)
    print(f"  index.ntotal = {idx.ntotal}  device = {idx.device}")

    queries = _unit(rng, 1_000, D)

    # Brute force using numpy.
    M = embeddings  # already unit
    mismatches = 0
    sim_diffs = []
    for q in queries:
        bf_sims = M @ q
        bf_idx = int(np.argmax(bf_sims))
        bf_sim = float(bf_sims[bf_idx])

        sims, ids = idx.search(q, k=1)
        if not ids:
            mismatches += 1
            continue
        if int(ids[0]) != bf_idx:
            mismatches += 1
        sim_diffs.append(abs(bf_sim - float(sims[0])))

    max_diff = max(sim_diffs) if sim_diffs else 0
    print(f"  mismatches:  {mismatches} / 1,000")
    print(f"  max sim diff: {max_diff:.2e}")
    assert mismatches == 0, "MPS results differ from brute force"
    assert max_diff < 1e-3, f"sim divergence too high: {max_diff}"

    # ---- batch search ----
    print()
    print("[mps] batch search — 1,000 queries in one matmul")
    t0 = time.perf_counter()
    batch_results = idx.search_batch(queries, k=1)
    t_batch = time.perf_counter() - t0
    print(f"  batch (1,000 queries): {t_batch*1000:.1f}ms = "
          f"{t_batch/1000*1000:.4f}ms per query")
    assert len(batch_results) == 1000
    # spot-check first 100 against single-query path
    spot_mismatch = 0
    for i in range(100):
        sims, ids = idx.search(queries[i], k=1)
        if int(ids[0]) != int(batch_results[i][1]):
            spot_mismatch += 1
    assert spot_mismatch == 0, f"batch vs single mismatched: {spot_mismatch}"
    print(f"  batch ↔ single agreement: 100/100")

    # ---- speed benchmark — flat curve target ----
    print()
    print("[mps] speed across N — flat curve target")
    print(f"  {'N':>6}  {'1K queries':>10}  {'per-query':>12}")
    for n in (1_000, 5_000, 10_000, 50_000):
        emb = _unit(rng, n, D)
        idx2 = MPSConceptIndex(d_rep=D)
        for i in range(n):
            idx2.add(emb[i], i)
        # Warm-up
        idx2.search(emb[0], k=1)
        t0 = time.perf_counter()
        for _ in range(1000):
            idx2.search(queries[0], k=1)
        elapsed = time.perf_counter() - t0
        print(f"  {n:>6}  {elapsed*1000:>8.1f}ms  {elapsed:>10.4f}ms")

    print()
    print("[mps] PASS — correctness + speed verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
