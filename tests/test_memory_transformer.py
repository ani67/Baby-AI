import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np, torch
from backend.memory_bank import PersistentMemoryBank
from backend.memory_transformer import SparseMemoryTransformer
def test_xfmr():
    bank = PersistentMemoryBank(m_slots=1024, d_rep=512)
    emb = np.random.randn(800, 512).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    bank.initialize_from_concept_graph(emb)
    t = SparseMemoryTransformer(bank, n_layers=2,
                                use_checkpoint=False).to(bank.device).to(torch.bfloat16)
    out = t()
    assert out.shape == (1024, 512)
    bias = torch.zeros(8, device=bank.device, dtype=torch.bfloat16)
    out2 = t(affect_bias=bias + 2.0)
    diff = (out2 - out).abs().mean()
    assert diff > 0
    out.sum().backward()
    print("memory_transformer OK")
if __name__ == '__main__': test_xfmr()
