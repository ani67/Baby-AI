import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np, torch
from backend.memory_bank import PersistentMemoryBank
from backend.encoder import MultiModalEncoder
def test_enc():
    bank = PersistentMemoryBank(m_slots=512, d_rep=512)
    emb = np.random.randn(300, 512).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    bank.initialize_from_concept_graph(emb)
    enc = MultiModalEncoder(bank, vocab_size=1024, n_layers=2).to(bank.device).to(torch.bfloat16)
    r = enc(text="justice and natural selection")
    assert r['updated_inputs'].dim() == 2
    assert r['input_rep'].shape == (512,)
    assert len(r['pc_errors']) == 2
    r['surprise'].backward()
    print("encoder OK")
if __name__ == '__main__': test_enc()
