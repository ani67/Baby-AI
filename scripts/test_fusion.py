import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.fusion import MultimodalFusion

fusion = MultimodalFusion()
print(f"[fusion] device={fusion.device}")

text_rep = np.random.randn(512).astype(np.float32)
text_rep /= np.linalg.norm(text_rep) + 1e-9

fused = fusion.fuse(text_rep)
print(f"  output shape: {fused.shape}")
print(f"  output norm:  {np.linalg.norm(fused):.4f}")
assert fused.shape == (512,)
assert abs(np.linalg.norm(fused) - 1.0) < 0.01
print("test_fusion: PASS")
