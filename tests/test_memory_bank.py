import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np, torch
from backend.memory_bank import PersistentMemoryBank
from backend.unified_config import MAX_DRIFT_RATE
def test_bank():
    bank = PersistentMemoryBank(m_slots=2048, d_rep=512)
    emb = np.random.randn(1500, 512).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    assert bank.initialize_from_concept_graph(emb) == 1500
    rep = torch.randn(512, device=bank.device)
    slot, dr = bank.soft_write(rep, torch.tensor(0.8))
    assert slot.shape == (1,) and 0 < float(dr[0]) <= MAX_DRIFT_RATE
    sims, idx = bank.search(rep.unsqueeze(0), k=10)
    assert sims.shape == (1, 10) and idx.shape == (1, 10)
    # search chunks: verify large-Q path
    queries = torch.randn(3000, 512, device=bank.device)
    sims, idx = bank.search(queries, k=5)
    assert sims.shape == (3000, 5)
    # combined slots are normalized
    s = bank.slots
    norms = s.norm(dim=-1).float()
    assert (norms - 1).abs().max() < 0.05
    # gradients flow through trained_slots via .slots
    loss = bank.slots.sum()
    loss.backward()
    assert bank.trained_slots.grad is not None
    print("memory_bank OK")
if __name__ == '__main__': test_bank()
