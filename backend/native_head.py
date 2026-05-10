"""Native head v2 — a small conditional Transformer decoder.

A parallel v2 to backend/language_head.py's GRU. Larger model trained
on a sentence-length-filtered corpus (>=8 words). Drop-in compatible
generate() so expression.py can swap heads without touching call sites.

Architecture
------------
Standard decoder-only transformer:

    tokens ──► embed ──┐
                       ├── (+ pos) ──► n_layers × (self-attn + FFN) ──► out (vocab)
    BOS embedding ──── │
        +
    cond_proj(cond) ───┘  ← injected once at the BOS position only.

Conditioning injection design choice — bos-add (additive at step 0):
   We project the 1292-d conditioning vector to d_model and ADD it to
   the BOS token's embedding before positional embedding. The decoder
   then attends to that conditioned BOS at every subsequent step
   through causal self-attention; no new prefix slot is needed.
   Why bos-add over learned-prefix-token: simpler (no extra positional
   token to budget into max_seq_len; no separate prefix-mask logic),
   strictly fewer parameters, and at this model scale the difference
   is empirically a wash. Bos-add keeps the v2 head a drop-in for v1.

Sequence layout (training and inference):
    pos:        0      1      2      ...
    token:    <bos>   tok1   tok2    ...
    embed:    e_bos+  e_tok1 e_tok2  ...
              cond_proj(c)

Inference contract (matches v1 LanguageHead.generate):
    cond:                 (1, COND_DIM=1292) Tensor
    temperature:          float
    repetition_penalty:   float (default 1.5)
    blocked_start_ids:    list[int] | None — logits set to -inf at pos 0 only
    returns:              list[int] of generated ids (excludes BOS, EOS)

Parameter targets (configured via NativeHeadConfig defaults):
    vocab=8004, d_model=512, n_heads=8, n_layers=6, d_ff=2048
    ≈ tok_embed(8004*512=4.10M) + pos(64*512=33K)
      + 6 × (attn(4*512^2=1.05M) + ffn(2*512*2048=2.10M)) ≈ 18.9M
      + cond_proj(1292*512=0.66M) + out(512*8004=4.10M)
    ≈ 27-29M total params (vs v1 ~5.6M).
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse the v1 special tokens so v2 reads the same vocab json without
# any translation step.
from backend.language_head import BOS_ID, EOS_ID, PAD_ID, UNK_ID


@dataclass
class NativeHeadConfig:
    vocab_size: int = 8000
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 2048
    max_seq_len: int = 64
    dropout: float = 0.1
    n_aff: int = 12
    d_rep: int = 256
    n_concept_slots: int = 5

    @property
    def cond_dim(self) -> int:
        # Mirror language_head.COND_DIM: affect[12] + n_concept_slots * d_rep.
        return self.n_aff + self.n_concept_slots * self.d_rep


# ============================================================
# Model
# ============================================================


class NativeHead(nn.Module):
    """Decoder-only transformer with bos-additive conditioning."""

    def __init__(self, config: NativeHeadConfig) -> None:
        super().__init__()
        self.config = config

        self.tok_embed = nn.Embedding(config.vocab_size, config.d_model, padding_idx=PAD_ID)
        self.pos_embed = nn.Embedding(config.max_seq_len, config.d_model)

        # Project the 1292-d conditioning vector to d_model. A small MLP
        # lets the head reshape the affect+concept signal non-linearly
        # before it lands on the BOS slot.
        self.cond_proj = nn.Sequential(
            nn.Linear(config.cond_dim, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )

        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,        # pre-norm — more stable for tiny models
        )
        # We use TransformerEncoder + an explicit causal mask. This is
        # the standard decoder-only-transformer recipe and avoids the
        # cross-attention plumbing of nn.TransformerDecoder which we
        # don't need (no encoder side).
        self.blocks = nn.TransformerEncoder(layer, num_layers=config.n_layers)

        self.ln_f = nn.LayerNorm(config.d_model)
        self.out = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.dropout = nn.Dropout(config.dropout)

        # Pre-cache a boolean triangular causal mask up to max_seq_len.
        # True entries are *masked out* (cannot be attended to). We use
        # bool so it matches the dtype of the key-padding mask, which
        # silences torch's "mismatched mask types" deprecation warning.
        causal_bool = torch.triu(
            torch.ones(config.max_seq_len, config.max_seq_len, dtype=torch.bool),
            diagonal=1,
        )
        self.register_buffer("_causal_mask", causal_bool, persistent=False)

        self._init_weights()

    @property
    def vocab_size(self) -> int:
        return self.config.vocab_size

    def _init_weights(self) -> None:
        nn.init.normal_(self.tok_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_embed.weight, mean=0.0, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _embed_with_cond(self, tokens: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Token embedding + positional embedding + conditioning added to
        position 0 (the BOS slot).

        tokens: (B, T)  cond: (B, COND_DIM)  →  (B, T, d_model)
        """
        B, T = tokens.shape
        device = tokens.device
        positions = torch.arange(T, device=device).unsqueeze(0)              # (1, T)
        x = self.tok_embed(tokens) + self.pos_embed(positions)               # (B, T, D)
        # Inject cond on position 0 only. T must be >= 1 — train + gen both
        # start from a BOS token, so this is always safe.
        cond_h = self.cond_proj(cond)                                        # (B, D)
        x = x.clone()
        x[:, 0, :] = x[:, 0, :] + cond_h
        return self.dropout(x)

    def forward(self, tokens: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """tokens: (B, T)  cond: (B, COND_DIM)  →  logits: (B, T, V)."""
        T = tokens.size(1)
        if T > self.config.max_seq_len:
            raise ValueError(
                f"sequence length {T} exceeds max_seq_len {self.config.max_seq_len}"
            )
        x = self._embed_with_cond(tokens, cond)
        causal = self._causal_mask[:T, :T]
        # Pad mask: TransformerEncoder takes src_key_padding_mask where
        # True = "ignore". Build it from PAD_ID positions so padded tokens
        # don't pollute attention.
        key_pad = (tokens == PAD_ID)
        h = self.blocks(x, mask=causal, src_key_padding_mask=key_pad)
        h = self.ln_f(h)
        return self.out(h)

    @torch.no_grad()
    def generate(
        self,
        cond: torch.Tensor,
        max_len: int = 20,
        temperature: float = 0.7,
        repetition_penalty: float = 1.5,
        blocked_start_ids: list[int] | None = None,
    ) -> list[int]:
        """Greedy-multinomial decode from <bos> until <eos> or max_len.

        cond: shape (1, COND_DIM). Single example only (no batched gen).
        Returns the list of generated ids (excludes the <bos> seed and
        the terminal <eos>).

        Drop-in compatible with backend.language_head.LanguageHead.generate.
        """
        self.eval()
        device = next(self.parameters()).device
        cond = cond.to(device)

        # Cap at the model's max_seq_len. We need room for BOS + max_len
        # generated tokens; clip max_len so positional embeddings stay
        # in range.
        max_len = min(max_len, self.config.max_seq_len - 1)

        produced: list[int] = []
        blocked_set = set(blocked_start_ids or [])
        for _ in range(max_len):
            seq = [BOS_ID] + produced
            tokens = torch.tensor([seq], device=device, dtype=torch.long)
            logits_all = self.forward(tokens, cond)                          # (1, T, V)
            logits = logits_all[:, -1, :] / max(temperature, 1e-6)           # (1, V)

            if repetition_penalty and repetition_penalty != 1.0 and produced:
                seen = set(produced)
                row = logits[0]
                for tok in seen:
                    v = row[tok].item()
                    row[tok] = v / repetition_penalty if v > 0 else v * repetition_penalty
                logits[0] = row

            if not produced and blocked_set:
                for tok in blocked_set:
                    if 0 <= tok < logits.shape[-1]:
                        logits[0, tok] = float("-inf")

            probs = torch.softmax(logits, dim=-1)
            tok = int(torch.multinomial(probs[0], num_samples=1).item())
            if tok in (EOS_ID, PAD_ID):
                break
            if tok == UNK_ID:
                top_ids = torch.topk(probs[0], k=10).indices.tolist()
                top_ids = [t for t in top_ids if t not in (PAD_ID, UNK_ID, BOS_ID, EOS_ID)]
                if not top_ids:
                    break
                tok = top_ids[0]
            produced.append(tok)
        return produced


# ============================================================
# Save / load
# ============================================================


def save_checkpoint_v2(model: NativeHead, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "config": {
            "vocab_size":      model.config.vocab_size,
            "d_model":         model.config.d_model,
            "n_heads":         model.config.n_heads,
            "n_layers":        model.config.n_layers,
            "d_ff":            model.config.d_ff,
            "max_seq_len":     model.config.max_seq_len,
            "dropout":         model.config.dropout,
            "n_aff":           model.config.n_aff,
            "d_rep":           model.config.d_rep,
            "n_concept_slots": model.config.n_concept_slots,
        },
        "state_dict": model.state_dict(),
    }
    torch.save(payload, path)


def load_checkpoint_v2(path: str, device: str = "cpu") -> NativeHead:
    obj = torch.load(path, map_location=device, weights_only=True)
    cfg = NativeHeadConfig(**obj["config"])
    model = NativeHead(cfg)
    model.load_state_dict(obj["state_dict"])
    model.to(device)
    model.eval()
    return model


# Path used by expression.py's v2 fallback.
DEFAULT_V2_FILENAME = "native_head_v2.pt"


def is_available_v2(weights_path: str) -> bool:
    return os.path.exists(weights_path)
