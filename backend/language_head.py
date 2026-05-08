"""The language head — a small conditional GRU.

Architecture
------------
A 2-layer GRU language model. At generation time it is conditioned on
a concatenation of the current affect composite (12 dims) and the
top-5 active concept embeddings (5 × 256 = 1280 dims), totaling 1292d.
The conditioning vector is projected to the GRU's initial hidden state
across both layers.

The head is the mind's mouth, not the mind itself. It learns fluent
English from the corpus the mind was surprised by; the mind chooses
WHEN to speak (D's action selection) and WHAT to speak ABOUT (the
active set built by F.processing_loop). The language head only
decides HOW to phrase it.

Inference: temperature 0.7, max 20 tokens, multinomial sampling.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Special tokens (lock these IDs — vocab files reference them).
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]

# Architecture constants — locked so train + load match exactly.
EMBED_DIM       = 256
HIDDEN_DIM      = 256
NUM_LAYERS      = 2
COND_DIM        = 12 + 5 * 256          # affect[12] + top-5 concept embeddings
TOP_K_CONCEPTS  = 5
GENERATION_MAX  = 20
GENERATION_TEMP = 0.7


@dataclass
class Vocab:
    """Token ↔ id dictionary plus the reverse list for fast id→token."""
    token_to_id: dict[str, int]
    id_to_token: list[str]

    @property
    def size(self) -> int:
        return len(self.id_to_token)

    def encode(self, tokens: list[str]) -> list[int]:
        unk = UNK_ID
        return [self.token_to_id.get(t, unk) for t in tokens]

    def decode(self, ids: list[int]) -> list[str]:
        return [self.id_to_token[i] if 0 <= i < self.size else "<unk>" for i in ids]

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"id_to_token": self.id_to_token}, f)

    @classmethod
    def load(cls, path: str) -> "Vocab":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        ids = data["id_to_token"]
        return cls(token_to_id={t: i for i, t in enumerate(ids)}, id_to_token=ids)


class LanguageHead(nn.Module):
    """Conditional GRU language model. Tiny — ~5M params at vocab=8K."""

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, EMBED_DIM, padding_idx=PAD_ID)
        # Project the 1292-d conditioning vector into NUM_LAYERS × HIDDEN_DIM
        # initial-hidden tensor for the GRU stack.
        self.cond_proj = nn.Linear(COND_DIM, NUM_LAYERS * HIDDEN_DIM)
        self.gru = nn.GRU(EMBED_DIM, HIDDEN_DIM, num_layers=NUM_LAYERS, batch_first=True)
        self.out = nn.Linear(HIDDEN_DIM, vocab_size)

    def init_hidden(self, cond: torch.Tensor) -> torch.Tensor:
        """cond: (B, COND_DIM) → h0: (NUM_LAYERS, B, HIDDEN_DIM)."""
        B = cond.shape[0]
        h = self.cond_proj(cond).view(B, NUM_LAYERS, HIDDEN_DIM)
        # GRU expects (num_layers, batch, hidden).
        return torch.tanh(h.permute(1, 0, 2).contiguous())

    def forward(self, tokens: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """tokens: (B, T)  cond: (B, COND_DIM)  →  logits: (B, T, V)."""
        h0 = self.init_hidden(cond)
        emb = self.embed(tokens)
        out, _ = self.gru(emb, h0)
        return self.out(out)

    @torch.no_grad()
    def generate(
        self,
        cond: torch.Tensor,
        max_len: int = GENERATION_MAX,
        temperature: float = GENERATION_TEMP,
    ) -> list[int]:
        """Greedy-multinomial decode from <bos> until <eos> or max_len.

        cond: shape (1, COND_DIM) — single example only (no batched gen).
        Returns the list of generated ids (excludes the <bos> seed and
        the terminal <eos>).
        """
        self.eval()
        device = next(self.parameters()).device
        h = self.init_hidden(cond.to(device))
        x = torch.tensor([[BOS_ID]], device=device, dtype=torch.long)
        produced: list[int] = []
        for _ in range(max_len):
            emb = self.embed(x)
            out, h = self.gru(emb, h)
            logits = self.out(out[:, -1, :]) / max(temperature, 1e-6)
            probs = torch.softmax(logits, dim=-1)
            tok = int(torch.multinomial(probs[0], num_samples=1).item())
            if tok in (EOS_ID, PAD_ID):
                break
            if tok == UNK_ID:
                # Skip <unk> — re-sample from the top of the distribution.
                top_ids = torch.topk(probs[0], k=10).indices.tolist()
                top_ids = [t for t in top_ids if t not in (PAD_ID, UNK_ID, BOS_ID, EOS_ID)]
                if not top_ids:
                    break
                tok = top_ids[0]
            produced.append(tok)
            x = torch.tensor([[tok]], device=device, dtype=torch.long)
        return produced


def build_conditioning_vector(
    affect_composite: np.ndarray,
    concept_embeddings: list[np.ndarray],
) -> np.ndarray:
    """Pack (affect, top-5 concept embeddings) into a fixed COND_DIM vector.

    Pads with zeros if fewer than TOP_K_CONCEPTS embeddings are supplied.
    Returns float32[COND_DIM].
    """
    if affect_composite.shape != (12,):
        raise ValueError(f"affect must be (12,), got {affect_composite.shape}")
    out = np.zeros(COND_DIM, dtype=np.float32)
    out[:12] = affect_composite.astype(np.float32, copy=False)
    for i, emb in enumerate(concept_embeddings[:TOP_K_CONCEPTS]):
        if emb.shape != (256,):
            raise ValueError(f"concept embedding must be (256,), got {emb.shape}")
        out[12 + i * 256: 12 + (i + 1) * 256] = emb.astype(np.float32, copy=False)
    return out


# ============================================================
# Tokenization (matches scripts/build_vocab.py)
# ============================================================

import re

_TOKEN_RE = re.compile(r"[a-z][a-z'-]*[a-z]|[a-z]")


def tokenize(text: str) -> list[str]:
    """Lowercase + simple word-boundary tokenization. Identical to the
    encoder's tokenization so vocab built from book text aligns with
    what the LM was trained on.
    """
    return _TOKEN_RE.findall(text.lower())


def render_tokens(ids: list[int], vocab: Vocab) -> str:
    return " ".join(vocab.decode(ids))


# ============================================================
# Disk paths + load/save helpers
# ============================================================

DEFAULT_WEIGHTS_PATH = "data/language_head.pt"
DEFAULT_VOCAB_PATH   = "data/language_head_vocab.json"


def save_checkpoint(model: LanguageHead, path: str = DEFAULT_WEIGHTS_PATH) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"vocab_size": model.vocab_size, "state_dict": model.state_dict()}, path)


def load_checkpoint(path: str = DEFAULT_WEIGHTS_PATH, device: str = "cpu") -> LanguageHead:
    obj = torch.load(path, map_location=device, weights_only=True)
    model = LanguageHead(vocab_size=int(obj["vocab_size"]))
    model.load_state_dict(obj["state_dict"])
    model.to(device)
    model.eval()
    return model


def is_available() -> bool:
    return os.path.exists(DEFAULT_WEIGHTS_PATH) and os.path.exists(DEFAULT_VOCAB_PATH)
