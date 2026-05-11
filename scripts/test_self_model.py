import sys, os, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.self_model import SelfModel

sm = SelfModel()

surface = "justice is the foundation of society"
encoding = np.random.randn(512).astype(np.float32)
encoding /= np.linalg.norm(encoding) + 1e-9
now = time.time()
sm.register_output(surface, encoding, now)

assert sm.is_self_echo("justice is the foundation", now + 5), "echo within window"
assert not sm.is_self_echo("justice is the foundation", now + 35), "no echo after window"
assert not sm.is_self_echo("the weather is beautiful", now + 5), "no false echo"
print("[self_model] self-echo detection: PASS")

affect = np.random.randn(12).astype(np.float32)
centroid = np.random.randn(512).astype(np.float32)
centroid /= np.linalg.norm(centroid) + 1e-9
losses = []
for i in range(100):
    sm.update(affect, centroid, now + i)
    if sm.self_prediction_loss > 0:
        losses.append(sm.self_prediction_loss)
if losses:
    print(f"[self_model] self-prediction loss {losses[0]:.4f} -> {losses[-1]:.4f}")
print("test_self_model: PASS")
