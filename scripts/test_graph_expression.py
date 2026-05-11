import sys, os, numpy as np, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.persistence import MindPersistence
from backend.mind_paths import MindPaths
from backend.wave_field import WaveField  # may not exist yet if SA1 hasn't landed
from backend.expression_graph import GraphTraversalExpression

paths = MindPaths('first')
loop = MindPersistence.load(paths.db)
loop.graph._rebuild_index()

wf = WaveField(loop.graph)
expr = GraphTraversalExpression(loop.graph, wf)
print(f"[graph_expression] word nodes: {len(expr.word_nodes)}")

# seed first 3 concepts
seed = list(loop.graph.nodes.keys())[:3]
wf.inject(seed, strength=1.0)
wf.run_until_settled()

result = expr.generate(max_words=15)
print(f"  generated: '{result}'  ({len(result.split())} words)")
assert len(result.split()) > 0, "should generate at least one word"
print("test_graph_expression: PASS")
