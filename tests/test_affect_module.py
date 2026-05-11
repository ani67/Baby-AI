import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch, time
from backend.affect_module import AffectModule
def test_affect():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    m = AffectModule().to(device).to(torch.bfloat16)
    mem = torch.randn(256, 512, device=device, dtype=torch.bfloat16)
    r = m(mem)
    assert r['affect_vector'].shape == (12,)
    assert r['attention_bias'].shape == (8,)
    assert 0 <= float(r['arousal']) <= 1
    r['affect_vector'].sum().backward()
    m.update_timescales(r['affect_vector'].detach(), time.time())
    assert m.composite().shape == (12,)
    print("affect_module OK")
if __name__ == '__main__': test_affect()
