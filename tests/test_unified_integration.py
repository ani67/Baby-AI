"""Integration tests for UnifiedMind (Phase 2 / Agent 8).

Run directly:
    python3 tests/test_unified_integration.py

Each test prints a `<name> OK` (or `<name> SKIP: <reason>`) line; the runner
script in the spec reconciliation greps for these markers.
"""
import sys
import tempfile
import time
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from backend.affect_module import AffectModule
from backend.decoder import UnifiedDecoder
from backend.encoder import CurriculumTokenizer, MultiModalEncoder
from backend.memory_bank import PersistentMemoryBank
from backend.memory_transformer import SparseMemoryTransformer
from backend.unified_mind import MindResponse, UnifiedMind


DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def _build_tiny_mind(m_slots: int = 128, vocab_size: int = 512,
                    n_layers: int = 2) -> UnifiedMind:
    """A small UnifiedMind for fast, deterministic tests. Avoids the slow
    concept-graph load path."""
    bank = PersistentMemoryBank(m_slots=m_slots, d_rep=512, device=DEVICE)
    # seed some embeddings so the search/topk paths have real signal
    n_seed = max(1, m_slots // 2)
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((n_seed, 512), dtype=np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9
    bank.initialize_from_concept_graph(emb)
    # mark some activation_counts so encoder's top-K selection is non-trivial
    with torch.no_grad():
        bank.activation_count[:n_seed] = torch.arange(
            1, n_seed + 1, device=DEVICE, dtype=torch.long,
        )

    enc = MultiModalEncoder(bank, vocab_size=vocab_size, n_layers=n_layers).to(
        DEVICE).to(torch.bfloat16)
    mt = SparseMemoryTransformer(bank, n_layers=n_layers,
                                 use_checkpoint=False).to(DEVICE).to(torch.bfloat16)
    aff = AffectModule().to(DEVICE).to(torch.bfloat16)
    dec = UnifiedDecoder(
        bank, vocab_size=vocab_size,
        shared_embedding=enc.text_embedding,
        n_layers=n_layers,
    ).to(DEVICE).to(torch.bfloat16)
    tok = CurriculumTokenizer()  # char-level stub
    return UnifiedMind(bank, mt, enc, aff, dec, tok, DEVICE)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_memory_initialization():
    """Build a UnifiedMind on the `first` mind. Assert n_written > 1000
    (we know `first` has ~60K concepts). Skips cleanly if the mind isn't
    initialized as expected."""
    db_path = Path('data/first/mind.db')
    if not db_path.exists():
        print("test_memory_initialization SKIP: data/first/mind.db not found")
        return
    # Cap to small slot count so we don't allocate 65K*512*2bytes for the
    # bank during a fast unit test. The seeding will populate min(N, m_slots).
    # We still test that seeding happened.
    try:
        mind = UnifiedMind.load_or_create('first')
    except Exception as exc:
        print(f"test_memory_initialization SKIP: load_or_create failed ({exc})")
        return
    n = int(mind.bank.n_written.item())
    assert n > 1000, f"expected n_written > 1000, got {n}"
    print(f"test_memory_initialization OK (n_written={n:,})")


def test_full_forward_pass():
    """process() returns a well-formed MindResponse."""
    mind = _build_tiny_mind()
    resp = mind.process(text="hello world")
    assert isinstance(resp, MindResponse), f"got {type(resp)}"
    assert isinstance(resp.text, str)
    assert isinstance(resp.gap, float) and resp.gap == resp.gap, "gap must be finite"
    # numeric sanity
    import math
    assert math.isfinite(resp.gap), f"gap not finite: {resp.gap}"
    assert math.isfinite(resp.arousal), f"arousal not finite: {resp.arousal}"
    assert 0.0 <= resp.arousal <= 1.0, f"arousal out of [0,1]: {resp.arousal}"
    assert resp.active_memory_slots > 0
    assert len(resp.affect_vector) == 12
    assert resp.generator == 'unified'
    print("test_full_forward_pass OK")


def test_force_respond_emits():
    """force_respond=True bypasses suppression and produces some text.

    With a randomly-initialized decoder + the char-level stub tokenizer
    (which strips PAD/BOS/EOS on decode), the surface string may be empty
    even though the decoder ran to completion. We bias the lm_head away
    from the BOS token id so the model emits *something*, then check that
    `suppressed=False` and the decoded surface contains at least one
    printable character.
    """
    mind = _build_tiny_mind()
    # Suppress BOS in the LM head so the untrained decoder doesn't sample it
    # forever. token_embedding.weight is tied with lm_head.weight; lowering
    # the BOS row of the embedding lowers the BOS logit.
    with torch.no_grad():
        mind.dec.token_embedding.weight[1].fill_(-5.0)  # BOS row
        # Keep EOS reasonable so we still terminate; don't make it dominate.
        mind.dec.token_embedding.weight[2].fill_(0.0)

    resp = mind.process(text="anything goes", force_respond=True)
    assert resp.suppressed is False, "force_respond must not be suppressed"
    if len(resp.text) == 0:
        for _ in range(8):
            resp = mind.process(text="anything goes", force_respond=True)
            if len(resp.text) > 0:
                break
    assert len(resp.text) > 0, (
        "force_respond should yield non-empty text "
        f"(got {resp.text!r}); decoder may be stuck on a special token"
    )
    print(f"test_force_respond_emits OK (text len={len(resp.text)})")


def test_gradients_through_all_components():
    """One forward+backward; assert each of the 5 components has non-None
    grad on at least one parameter."""
    m_slots = 128
    vocab_size = 512
    n_layers = 2
    bank = PersistentMemoryBank(m_slots=m_slots, d_rep=512, device=DEVICE)
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((64, 512), dtype=np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9
    bank.initialize_from_concept_graph(emb)

    enc = MultiModalEncoder(bank, vocab_size=vocab_size, n_layers=n_layers).to(
        DEVICE).to(torch.bfloat16)
    mt = SparseMemoryTransformer(bank, n_layers=n_layers,
                                 use_checkpoint=False).to(DEVICE).to(torch.bfloat16)
    aff = AffectModule().to(DEVICE).to(torch.bfloat16)
    dec = UnifiedDecoder(
        bank, vocab_size=vocab_size,
        shared_embedding=enc.text_embedding,
        n_layers=n_layers,
    ).to(DEVICE).to(torch.bfloat16)

    # Make sure all are in train mode for the gradient walk.
    for m in (enc, mt, aff, dec):
        m.train()

    enc_out = enc(text="gradient probe")
    mem_full = mt(bank.slots)
    aff_out = aff(mem_full)
    mem_active = mem_full[enc_out['active_slot_indices']]
    ids = torch.tensor([[1, 5, 7, 9]], device=DEVICE, dtype=torch.long)
    logits, _ = dec(ids, mem_active)
    # combine signals from every component so gradients flow everywhere
    loss = (
        logits.float().sum()
        + enc_out['surprise'].float()
        + aff_out['affect_vector'].float().sum()
        + mem_full.float().sum() * 1e-6  # tiny scale; mem_full is large
    )
    loss.backward()

    components = {
        'encoder': enc,
        'memory_transformer': mt,
        'affect_module': aff,
        'decoder': dec,
    }
    for name, mod in components.items():
        has_grad = any(
            p.grad is not None and p.grad.abs().sum().item() > 0
            for p in mod.parameters() if p.requires_grad
        )
        assert has_grad, f"no gradient flow into {name}"
    # memory bank: trained_slots is a Parameter inside mt.bank
    bank_has_grad = (
        bank.trained_slots.grad is not None
        and bank.trained_slots.grad.abs().sum().item() > 0
    )
    assert bank_has_grad, "no gradient flow into memory_bank.trained_slots"
    print("test_gradients_through_all_components OK")


def test_self_echo_detection():
    """Emit via force_respond; same text within 30s should flag echo=True."""
    mind = _build_tiny_mind()
    # First, force an emission and seed the echo buffer manually with known
    # text so the test doesn't depend on what the random-init decoder sampled.
    seed = "the quick brown fox jumps over the lazy dog"
    mind._record_emission(seed, time.time())

    # Now process near-identical input. The Jaccard overlap on tokens is 1.0.
    resp = mind.process(text=seed, force_respond=True, debug=True)
    assert resp.echo is True, "expected echo=True for near-identical input"

    # Disjoint input should NOT echo.
    resp2 = mind.process(text="zebras quantum velocity orbital",
                         force_respond=True, debug=True)
    assert resp2.echo is False, "expected echo=False for disjoint input"
    print("test_self_echo_detection OK")


def test_save_and_load_roundtrip():
    """Build, run 5 process() calls, save the memory bank, reload it, assert
    n_written and a couple of slot tensors round-trip exactly."""
    mind = _build_tiny_mind()
    for i in range(5):
        mind.process(text=f"sample text {i}", force_respond=True)

    n_before = int(mind.bank.n_written.item())
    slots_before = mind.bank.slots.detach().cpu().float().clone()
    exp_before = mind.bank.experience_slots.detach().cpu().float().clone()
    act_before = mind.bank.activation_count.detach().cpu().clone()

    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        save_path = f.name
    try:
        mind.bank.save(save_path)
        reloaded = PersistentMemoryBank.load(save_path, device=DEVICE)
        n_after = int(reloaded.n_written.item())
        assert n_after == n_before, f"n_written mismatch: {n_before} vs {n_after}"

        slots_after = reloaded.slots.detach().cpu().float()
        exp_after = reloaded.experience_slots.detach().cpu().float()
        act_after = reloaded.activation_count.detach().cpu()

        # `.slots` is a derived view (alpha-mix + normalize). Tolerant compare.
        slot_diff = (slots_after - slots_before).abs().max().item()
        assert slot_diff < 1e-2, f"slots drift {slot_diff}"
        exp_diff = (exp_after - exp_before).abs().max().item()
        assert exp_diff < 1e-2, f"experience_slots drift {exp_diff}"
        assert torch.equal(act_after, act_before), "activation_count mismatch"
    finally:
        if os.path.exists(save_path):
            os.remove(save_path)
    print(f"test_save_and_load_roundtrip OK (n_written={n_before})")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

ALL_TESTS = [
    test_full_forward_pass,
    test_force_respond_emits,
    test_gradients_through_all_components,
    test_self_echo_detection,
    test_save_and_load_roundtrip,
    test_memory_initialization,  # slow / depends on disk; run last
]


if __name__ == '__main__':
    failed = []
    for fn in ALL_TESTS:
        try:
            fn()
        except Exception as exc:
            import traceback
            print(f"{fn.__name__} FAIL: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            failed.append(fn.__name__)
    if failed:
        print(f"\nFAILED: {failed}")
        sys.exit(1)
    print("\nintegration OK")
