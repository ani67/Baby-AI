"""A2 smoke test: ConditionedDecoder with 5 prefix tokens.

Constructs a fresh ConditionedDecoder, runs one forward+backward pass
on a dummy batch, samples 5 tokens. If this passes, the 20-epoch
training run will not blow up on architecture mistakes.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from backend.language_head import (   # noqa: E402
    COND_DIM,
    ConditionedDecoder,
    GPT2_HIDDEN_DIM,
    N_PREFIX_TOKENS,
)


def main() -> int:
    print(f"[smoke] N_PREFIX_TOKENS = {N_PREFIX_TOKENS}")
    print(f"[smoke] COND_DIM = {COND_DIM}, GPT2_HIDDEN_DIM = {GPT2_HIDDEN_DIM}")

    print("[smoke] constructing ConditionedDecoder …")
    cd = ConditionedDecoder()

    expected_out = N_PREFIX_TOKENS * GPT2_HIDDEN_DIM
    actual_out = cd.condition_proj.out_features
    print(f"[smoke] condition_proj.out_features = {actual_out}  (expected {expected_out})")
    assert actual_out == expected_out, "condition_proj output dim mismatch"

    # Trainable param count
    n_train = sum(p.numel() for p in cd.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in cd.parameters())
    print(f"[smoke] trainable: {n_train/1e6:.2f}M  /  total: {n_total/1e6:.2f}M params")

    # Dummy forward pass
    cond = torch.randn(2, COND_DIM)
    prefix = cd._build_prefix(cond)
    print(f"[smoke] _build_prefix(cond[2,{COND_DIM}]) -> shape {tuple(prefix.shape)}  "
          f"(expected (2, {N_PREFIX_TOKENS}, {GPT2_HIDDEN_DIM}))")
    assert prefix.shape == (2, N_PREFIX_TOKENS, GPT2_HIDDEN_DIM)

    # Forward+backward
    token_ids = torch.tensor([
        [50256, 1212, 318, 257, 1332, 13, 50256, 0],
        [50256, 1212, 318, 1194, 13, 50256, 0, 0],
    ], dtype=torch.long)
    attn = torch.tensor([
        [1, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 0, 0],
    ], dtype=torch.long)
    loss = cd.forward_with_prefix(cond, token_ids, attn)
    print(f"[smoke] forward_with_prefix loss = {loss.item():.4f}")
    loss.backward()
    print(f"[smoke] backward pass succeeded")

    # Inference
    cd.eval()
    with torch.no_grad():
        text = cd.generate(cond[:1], max_tokens=8, temperature=0.7)
    print(f"[smoke] generate(8 tokens) -> {text!r}")

    print("[smoke] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
