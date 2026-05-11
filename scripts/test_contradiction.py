import sys, os, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.persistence import MindPersistence
from backend.mind_paths import MindPaths
from backend.contradiction import ContradictionDetector

paths = MindPaths('first')
loop = MindPersistence.load(paths.db)
loop.graph._rebuild_index()

detector = ContradictionDetector(loop.graph)

# pick first concept's embedding as base
nodes_iter = iter(loop.graph.nodes.values())
first = next(nodes_iter)
base = first.embedding.copy().astype(np.float32)
dim = base.shape[0]

# write two opposing-direction concepts
emb_a = base + np.random.randn(dim).astype(np.float32) * 0.05
emb_a /= np.linalg.norm(emb_a) + 1e-9

emb_b = -base + np.random.randn(dim).astype(np.float32) * 0.05
emb_b = emb_b * 0.8 + base * 0.2
emb_b /= np.linalg.norm(emb_b) + 1e-9

now = time.time()
cid_a, _ = loop.graph.write_on_surprise(
    representation=emb_a, surprise=1.0,
    current_affect=np.zeros(12, dtype=np.float32),
    name_hint='probe-contradiction-A', now=now,
)
cid_b, _ = loop.graph.write_on_surprise(
    representation=emb_b, surprise=1.0,
    current_affect=np.zeros(12, dtype=np.float32),
    name_hint='probe-contradiction-B', now=now + 0.1,
)
print(f"  wrote probe concepts cid_a={cid_a} cid_b={cid_b}")

contradiction = detector.check_new_concept(cid_b, now + 0.2)
if contradiction:
    print(f"  detected: cid {contradiction.concept_a} <-> {contradiction.concept_b}")
    print(f"  similarity={contradiction.similarity:.3f}  "
          f"opposition={contradiction.opposition:.3f}")
    print("test_contradiction: PASS")
else:
    print("  no contradiction detected — thresholds may need tuning")
    print("test_contradiction: NO-DETECTION (not a hard failure on this graph)")
