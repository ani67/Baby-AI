import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np, torch
from backend.memory_bank import PersistentMemoryBank
from backend.decoder import UnifiedDecoder
def test_dec():
    bank = PersistentMemoryBank(m_slots=256, d_rep=512)
    emb = np.random.randn(200, 512).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    bank.initialize_from_concept_graph(emb)
    V = 1024
    d = UnifiedDecoder(bank, vocab_size=V, n_layers=2).to(bank.device).to(torch.bfloat16)
    assert d.lm_head.weight is d.token_embedding.weight   # weight tying
    mem_active = bank.slots[:128]                          # (K, D)
    ids = torch.randint(0, V, (2, 16), device=bank.device)
    logits, _ = d(ids, mem_active)
    assert logits.shape == (2, 16, V)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, V),
        torch.randint(0, V, (2 * 16,), device=bank.device))
    loss.backward()
    print("decoder OK")
if __name__ == '__main__': test_dec()
