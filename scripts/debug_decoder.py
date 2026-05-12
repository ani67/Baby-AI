"""Probe the decoder at initialization. The 28x LM loss inflation
(observed 271 vs expected log(16384)=9.7) means either the logits are
mis-scaled or the loss reduction is wrong. This script answers both.
"""
import sys
sys.path.insert(0, '.')

import torch
from backend.unified_mind import UnifiedMind

mind = UnifiedMind.load_or_create('first')
# attribute names: UnifiedMind stores components as mind.mt,
# mind.enc, mind.dec, mind.affect_module, mind.memory_bank,
# mind.tokenizer (per Agent-8's integration report).
mind.mt.eval()
mind.enc.eval()
mind.dec.eval()

device = mind.bank.device
vocab_size = mind.tokenizer.vocab_size if hasattr(mind.tokenizer, 'vocab_size') else 16384

with torch.no_grad():
    enc = mind.enc(text="justice and natural selection")
    # encoder returns 'mem_active' (top-K active slots) per the v2.0 spec.
    # Memory transformer operates on the full memory_bank.slots, then we
    # subset to active for downstream uses. Mirror what training does.
    memory_full = mind.mt(mind.bank.slots)
    active_idx = enc['active_slot_indices']
    mem_active = memory_full[active_idx]
    print(f"=== Memory state ===")
    print(f"  memory_full: shape={tuple(memory_full.shape)} dtype={memory_full.dtype}")
    print(f"    mean={memory_full.float().mean().item():+.4f}  std={memory_full.float().std().item():.4f}")
    print(f"  mem_active: shape={tuple(mem_active.shape)} dtype={mem_active.dtype}")
    print(f"    mean={mem_active.float().mean().item():+.4f}  std={mem_active.float().std().item():.4f}")

    input_ids = torch.randint(0, vocab_size, (1, 16), device=device)
    logits, _ = mind.dec(input_ids, mem_active)

    print(f"\n=== Logits at init ===")
    print(f"  shape: {tuple(logits.shape)}  dtype: {logits.dtype}")
    print(f"  mean:  {logits.mean().item():+.4f}")
    print(f"  std:   {logits.std().item():.4f}")
    print(f"  max:   {logits.max().item():+.4f}")
    print(f"  min:   {logits.min().item():+.4f}")
    print(f"  expected at proper init: mean≈0, std≈1-5")

    targets = torch.randint(0, vocab_size, (1, 16), device=device)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, vocab_size),
        targets.reshape(-1).long(),
    )
    print(f"\n=== LM loss at init ===")
    print(f"  observed:           {loss.item():.4f}")
    expected = torch.log(torch.tensor(float(vocab_size))).item()
    print(f"  expected (log vocab): {expected:.4f}")
    print(f"  ratio:              {loss.item() / expected:.2f}x")

    # also check if `ignore_index=0` (PAD) materially shifts things
    loss_pad = torch.nn.functional.cross_entropy(
        logits.reshape(-1, vocab_size),
        targets.reshape(-1).long(),
        ignore_index=0,
    )
    print(f"  with ignore_index=0: {loss_pad.item():.4f}")

    # sum reduction sanity check — verifies our 'mean' interpretation
    loss_sum = torch.nn.functional.cross_entropy(
        logits.reshape(-1, vocab_size),
        targets.reshape(-1).long(),
        reduction='sum',
    )
    print(f"  reduction='sum':    {loss_sum.item():.4f}  (= mean × N_tokens = {loss.item() * 16:.4f})")

    # The training loop's actual call — replicate exactly
    print(f"\n=== Replicating training call ===")
    # In compute_losses: tgt = batch['target_ids']  shape (1, T) per
    # example_to_batch in train_unified.py. Then:
    #   bos = torch.full((tgt.shape[0], 1), 1, ...)
    #   inp = torch.cat([bos, tgt[:, :-1]], dim=1)
    #   logits, _ = self.dec(inp, mem_active_xfm)
    #   losses['lm'] = F.cross_entropy(
    #       logits.reshape(-1, cfg.vocab_size),
    #       tgt.reshape(-1).long(),
    #       ignore_index=0)
    tgt = torch.randint(1, vocab_size, (1, 64), device=device)  # avoid PAD
    tgt[0, 30:] = 0  # simulate padding
    bos = torch.full((1, 1), 1, device=device, dtype=torch.long)
    inp = torch.cat([bos, tgt[:, :-1]], dim=1)
    logits2, _ = mind.dec(inp, mem_active)
    loss2 = torch.nn.functional.cross_entropy(
        logits2.reshape(-1, vocab_size),
        tgt.reshape(-1).long(),
        ignore_index=0,
    )
    print(f"  loss (1,64 with pad after pos 30): {loss2.item():.4f}")
    print(f"  vs observed in training step 1: 271.7")
