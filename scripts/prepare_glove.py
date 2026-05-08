"""Reduce GloVe 6B 300d → 256d via PCA and write a fast-loading binary.

Inputs : data/encoders/glove.6B.300d.txt   (1 GB; can be deleted after)
Outputs: data/encoders/glove_pca256_v1.npz  ({words, vectors})

We keep the top VOCAB_LIMIT most-frequent tokens (GloVe is pre-sorted by
frequency) — at 100K words × 256 dims × float32 the binary is ~102 MB,
small enough to load in <500 ms on M1 and broad enough to cover any
text the mind will see.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np


VOCAB_LIMIT = 100_000
SOURCE      = "data/encoders/glove.6B.300d.txt"
OUTPUT      = "data/encoders/glove_pca256_v1.npz"
TARGET_DIM  = 256


def load_glove(path: str, limit: int) -> tuple[list[str], np.ndarray]:
    words: list[str] = []
    rows: list[np.ndarray] = []
    t0 = time.perf_counter()
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            parts = line.rstrip().split(" ")
            words.append(parts[0])
            rows.append(np.asarray(parts[1:], dtype=np.float32))
            if (i + 1) % 25_000 == 0:
                print(f"  loaded {i + 1:>7,d} words in {time.perf_counter() - t0:.1f}s")
    M = np.stack(rows)
    print(f"  done: {M.shape} in {time.perf_counter() - t0:.1f}s")
    return words, M


def pca_reduce(X: np.ndarray, target_dim: int) -> np.ndarray:
    """Standard PCA via SVD on the mean-centered matrix.

    Returns reduced vectors of shape (n_samples, target_dim). The
    projection itself is `Vt[:target_dim].T`; we apply it inline here
    rather than persist it because the encoder only ever consumes the
    reduced vectors.
    """
    print(f"PCA: centering {X.shape}…")
    mean = X.mean(axis=0)
    Xc = X - mean
    print(f"PCA: SVD …")
    t0 = time.perf_counter()
    # full_matrices=False is essential — otherwise SVD allocates a
    # 100K × 100K U matrix and dies.
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    print(f"  SVD complete in {time.perf_counter() - t0:.1f}s; "
          f"top-{target_dim} singular values explain "
          f"{(S[:target_dim] ** 2).sum() / (S ** 2).sum() * 100:.1f}% of variance")
    components = Vt[:target_dim]               # (target_dim, 300)
    reduced = Xc @ components.T                # (n_samples, target_dim)
    return reduced.astype(np.float32)


def main() -> int:
    if not os.path.exists(SOURCE):
        print(f"missing {SOURCE}", file=sys.stderr)
        return 1
    if os.path.exists(OUTPUT):
        print(f"already exists: {OUTPUT}")
        return 0

    print(f"loading top {VOCAB_LIMIT:,} GloVe vectors from {SOURCE}…")
    words, X = load_glove(SOURCE, VOCAB_LIMIT)

    reduced = pca_reduce(X, TARGET_DIM)

    print(f"writing {OUTPUT} …")
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    np.savez_compressed(
        OUTPUT,
        words=np.asarray(words, dtype=object),
        vectors=reduced.astype(np.float32),
    )
    size_mb = os.path.getsize(OUTPUT) / 1024 / 1024
    print(f"done: {OUTPUT}  ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
