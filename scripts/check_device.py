"""Diagnose where v2.0 components actually live + benchmark MPS matmul."""
import sys, time
sys.path.insert(0, '.')

import torch

print(f"=== Environment ===")
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built:     {torch.backends.mps.is_built()}")
print(f"torch version: {torch.__version__}")

print(f"\n=== Component placement ===")
from backend.memory_bank import PersistentMemoryBank
bank = PersistentMemoryBank(m_slots=8192)
print(f"bank.device:            {bank.device}")
print(f"bank.trained_slots.dev: {bank.trained_slots.device}")
print(f"bank.trained_slots.dt:  {bank.trained_slots.dtype}")
print(f"bank.experience_slots:  {bank.experience_slots.device} / {bank.experience_slots.dtype}")
print(f"bank.alpha_logit:       {bank.alpha_logit.device} / {bank.alpha_logit.dtype}")
print(f"bank.activation_count:  {bank.activation_count.device}")
print(f"bank.slots prop dev:    {bank.slots.device} / {bank.slots.dtype}")

from backend.memory_transformer import SparseMemoryTransformer
mt = SparseMemoryTransformer(bank)
first = next(mt.parameters())
print(f"mt first param:         {first.device} / {first.dtype}")
mt_cast = mt.to(bank.device).to(torch.bfloat16)
first_cast = next(mt_cast.parameters())
print(f"mt after to(bf16):      {first_cast.device} / {first_cast.dtype}")

from backend.encoder import MultiModalEncoder
enc = MultiModalEncoder(bank, vocab_size=1024, n_layers=2).to(bank.device).to(torch.bfloat16)
enc_p = next(enc.parameters())
print(f"encoder:                {enc_p.device} / {enc_p.dtype}")

from backend.affect_module import AffectModule
aff = AffectModule().to(bank.device).to(torch.bfloat16)
aff_p = next(aff.parameters())
print(f"affect_module:          {aff_p.device} / {aff_p.dtype}")

print(f"\n=== MPS matmul benchmark ===")
device = torch.device('mps')
a = torch.randn(8192, 512, device=device, dtype=torch.bfloat16)
b = torch.randn(512, device=device, dtype=torch.bfloat16)

# warmup
for _ in range(5):
    _ = a @ b
torch.mps.synchronize()

t0 = time.time()
for _ in range(100):
    sims = a @ b
torch.mps.synchronize()
elapsed = time.time() - t0
print(f"MPS matmul (8192,512)@(512,) ×100: {elapsed:.3f}s ({elapsed/100*1000:.2f}ms each)")
print(f"  expected if MPS: <1ms each")
print(f"  if CPU fallback: ~10ms+ each")

print(f"\n=== Search benchmark ===")
import numpy as np
emb = np.random.randn(4000, 512).astype(np.float32)
emb /= np.linalg.norm(emb, axis=1, keepdims=True)
bank.initialize_from_concept_graph(emb)

queries = torch.randn(1024, 512, device=bank.device, dtype=torch.bfloat16)
# warmup
_ = bank.search(queries, k=64)
torch.mps.synchronize()
t0 = time.time()
for _ in range(10):
    _, _ = bank.search(queries, k=64)
torch.mps.synchronize()
elapsed = time.time() - t0
print(f"bank.search(1024×512, k=64) ×10: {elapsed:.3f}s ({elapsed/10*1000:.1f}ms each)")

print(f"\n=== Memory transformer single layer benchmark ===")
mt_test = SparseMemoryTransformer(bank, n_layers=1, use_checkpoint=False).to(bank.device).to(torch.bfloat16)
x = bank.slots.detach()
torch.mps.synchronize()
# warmup
_ = mt_test(x)
torch.mps.synchronize()
t0 = time.time()
for _ in range(3):
    out = mt_test(x)
torch.mps.synchronize()
elapsed = time.time() - t0
print(f"mt_test 1-layer (8192 slots) ×3: {elapsed:.3f}s ({elapsed/3*1000:.0f}ms each)")
