import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import math

import torch

from backend.affect_module import AffectModule
from backend.decoder import UnifiedDecoder
from backend.encoder import MultiModalEncoder
from backend.memory_bank import PersistentMemoryBank
from backend.memory_transformer import SparseMemoryTransformer
from backend.training import TrainingConfig, UnifiedMindTrainer


def test_train_step():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    V = 1024
    bank = PersistentMemoryBank(m_slots=512, d_rep=512, device=device)
    mt = SparseMemoryTransformer(bank, n_layers=2, use_checkpoint=False)
    enc = MultiModalEncoder(bank, vocab_size=V, n_layers=2)
    aff = AffectModule()
    dec = UnifiedDecoder(
        bank, vocab_size=V, shared_embedding=enc.text_embedding, n_layers=2,
    )

    cfg = TrainingConfig(
        batch_size=1,
        grad_accum=2,
        warmup_steps=0,
        gradnorm_every=1_000_000,  # disable for the unit test
        empty_cache_every=1_000_000,
        vocab_size=V,
    )
    trainer = UnifiedMindTrainer(cfg, bank, mt, enc, aff, dec, device)

    batch = {
        'text': 'hello world test',
        'target_ids': torch.tensor([[1, 2, 3, 4, 5, 0, 0, 0]], device=device),
        'is_surprising': torch.tensor([1.0], device=device),
    }
    out = trainer.train_step(batch)

    expected_keys = {'total_loss', 'mask', 'lm', 'align', 'affect', 'surp', 'weights'}
    missing = expected_keys - set(out.keys())
    assert not missing, f"missing keys: {missing}"

    total = out['total_loss']
    assert isinstance(total, float)
    assert math.isfinite(total), f"non-finite total_loss: {total}"
    for k in ('mask', 'lm', 'align', 'affect', 'surp'):
        v = out[k]
        assert math.isfinite(v), f"non-finite loss for {k}: {v}"
    assert len(out['weights']) == 5

    print("training OK")


if __name__ == '__main__':
    test_train_step()
