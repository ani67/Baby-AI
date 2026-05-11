"""Smoke test for WaveField. Verifies it loads, injects, settles."""
import sys, os, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.persistence import MindPersistence
from backend.mind_paths import MindPaths
from backend.wave_field import WaveField

paths = MindPaths('first')
print(f"[test_wave_field] loading {paths.db}")
loop = MindPersistence.load(paths.db)
loop.graph._rebuild_index()
print(f"  graph: {loop.graph.node_count} nodes / {loop.graph.edge_count} edges")

wf = WaveField(loop.graph)
print(f"  WaveField: N={wf.N} device={wf.device}")

# inject a few well-known concepts (whatever the graph has — first 3 nodes)
inject_ids = list(loop.graph.nodes.keys())[:3]
wf.inject(inject_ids, strength=1.0)

t0 = time.time()
steps = wf.run_until_settled()
elapsed = time.time() - t0
print(f"  settled in {steps} steps, {elapsed*1000:.0f}ms ({steps/max(elapsed,1e-6):.0f}/s)")

top = wf.get_top_concepts(10)
print(f"  top 10 concepts:")
for cid, act in top:
    name = loop.graph.nodes[cid].name[:50] if cid in loop.graph.nodes else '?'
    print(f"    [{act:.3f}] cid={cid}  {name!r}")

print("test_wave_field: PASS")
