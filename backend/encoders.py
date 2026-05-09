"""Concrete text encoders used by H's EncoderRegistry.

Two are present:

  text:char-trigram-bow-v1  — the Phase 1 encoder. Hash-based, no
    pretrained weights, deterministic. Kept registered but INACTIVE
    so concepts written by it (encoder_id = "text:char-trigram-bow-v1")
    keep their referencing metadata; they do not auto-reproject.

  text:glove-meanpool-pca256-v1  — the Phase 5 semantic encoder.
    GloVe 6B 300d reduced to 256d via PCA (99.5% variance preserved
    on the top-100K vocabulary), token-level mean-pool, L2-normalize.
    Captures real semantic neighborhoods: cos(cat, feline) > 0.7;
    cos(cat, democracy) < 0.2.
"""
from __future__ import annotations

import hashlib
import os
import re
import threading

import numpy as np

from backend.config import D_REP


# ============================================================
# Phase 1: char-trigram BoW (kept for backwards compat / dual-encoder graphs)
# ============================================================

ENCODER_TEXT_CHAR_TRIGRAM = "text:char-trigram-bow-v1"


def encode_text_char_trigram(text: str, dim: int = D_REP) -> np.ndarray:
    """Original Phase 1 encoder — md5-hashed signed-bucket char trigrams."""
    s = text.lower().strip()
    if not s:
        return np.zeros(dim, dtype=np.float32)
    padded = f"  {s}  "
    vec = np.zeros(dim, dtype=np.float32)
    for i in range(len(padded) - 2):
        digest = hashlib.md5(padded[i:i + 3].encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if (digest[4] & 1) else -1.0
        vec[idx] += sign
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm >= 1e-12 else vec


# ============================================================
# Phase 5: GloVe + PCA
# ============================================================

ENCODER_TEXT_GLOVE = "text:glove-meanpool-pca256-v1"
GLOVE_BINARY_PATH  = "data/encoders/glove_pca256_v1.npz"

# Tokenizer: lowercase + strip non-word chars + split on whitespace.
# Matches GloVe's tokenization closely enough — GloVe vocab includes
# common punctuation as separate entries but for sentence-level encoding
# we just want the words.
_TOKEN_RE = re.compile(r"[a-z][a-z'-]*[a-z]|[a-z]")


class _GloveStore:
    """Lazily-loaded singleton holding the PCA-reduced GloVe matrix
    and a word → row index. Thread-safe load (encoder may be called
    concurrently in the FastAPI ws/handler path)."""
    _instance: "_GloveStore | None" = None
    _lock = threading.Lock()

    def __init__(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"GloVe binary not found at {path}. Run "
                f"`python3 scripts/prepare_glove.py` first."
            )
        data = np.load(path, allow_pickle=True)
        words = data["words"]
        vectors = data["vectors"].astype(np.float32, copy=False)
        if vectors.shape[1] != D_REP:
            raise RuntimeError(
                f"glove binary dim {vectors.shape[1]} != D_REP {D_REP}"
            )
        self.vectors = vectors                                # (V, D_REP)
        self.index: dict[str, int] = {
            str(w): i for i, w in enumerate(words)
        }
        self.path = path

    @classmethod
    def get(cls) -> "_GloveStore":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = _GloveStore(GLOVE_BINARY_PATH)
        return cls._instance


def encode_text_glove(text: str, dim: int = D_REP) -> np.ndarray:
    """Tokenize → GloVe lookup → mean-pool → L2-normalize → 256-d unit.

    Unknown tokens contribute nothing (their absence reduces the
    denominator by 1 each). Empty input or all-unknown input returns
    the zero vector — C's find_or_match refuses to bind it against
    anything, matching char-trigram behavior.
    """
    if dim != D_REP:
        raise ValueError(f"encode_text_glove needs dim=D_REP={D_REP}, got {dim}")

    s = text.lower().strip()
    if not s:
        return np.zeros(D_REP, dtype=np.float32)

    store = _GloveStore.get()
    tokens = _TOKEN_RE.findall(s)
    if not tokens:
        return np.zeros(D_REP, dtype=np.float32)

    accum = np.zeros(D_REP, dtype=np.float32)
    n_known = 0
    for tok in tokens:
        idx = store.index.get(tok)
        if idx is None:
            continue
        accum += store.vectors[idx]
        n_known += 1

    if n_known == 0:
        return np.zeros(D_REP, dtype=np.float32)

    pooled = accum / n_known
    norm = float(np.linalg.norm(pooled))
    return (pooled / norm).astype(np.float32) if norm >= 1e-12 else pooled


def glove_is_available() -> bool:
    return os.path.exists(GLOVE_BINARY_PATH)


def encode_text_glove_batch(texts: list[str], dim: int = D_REP) -> np.ndarray:
    """Batch wrapper around encode_text_glove. Same shape contract.

    A previous v0.7 attempt routed this through torch.embedding_bag on
    MPS; on M1 Pro at the realistic batch sizes our offline encoder
    uses (64–256 mixed-length items), the GPU upload + kernel-launch
    overhead more than swallowed the gather speedup. The CPU per-item
    path runs at ~90K items/s on this hardware, well above the
    multilevel-preprocessor and SQLite-write rates that bound
    encode_corpus.py — so encoding is not the bottleneck either way.
    Keeping the loop simple matches the rest of the encoder module.
    """
    if dim != D_REP:
        raise ValueError(f"encode_text_glove_batch needs dim=D_REP={D_REP}")
    if not texts:
        return np.empty((0, D_REP), dtype=np.float32)
    out = np.empty((len(texts), D_REP), dtype=np.float32)
    for i, t in enumerate(texts):
        out[i] = encode_text_glove(t, dim=dim)
    return out


def encode_text_char_trigram_batch(texts: list[str], dim: int = D_REP) -> np.ndarray:
    """Char-trigram batch wrapper. Same shape contract."""
    if not texts:
        return np.empty((0, dim), dtype=np.float32)
    out = np.empty((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        out[i] = encode_text_char_trigram(t, dim=dim)
    return out
