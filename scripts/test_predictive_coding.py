import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.predictive_coding import PredictiveCodingHierarchy

pc = PredictiveCodingHierarchy()
print(f"[pc] device={pc.device}  levels={len(pc.levels)}")

rep = np.random.randn(512).astype(np.float32)
rep /= np.linalg.norm(rep) + 1e-9

errors = []
for i in range(50):
    result = pc.process(rep, learn=True)
    errors.append(result['surprise'])

print(f"  surprise at step 0:  {errors[0]:.4f}")
print(f"  surprise at step 25: {errors[25]:.4f}")
print(f"  surprise at step 49: {errors[49]:.4f}")
assert errors[49] < errors[0], "surprise should decrease with learning"
print("test_predictive_coding: PASS")
