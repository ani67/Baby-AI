# SPEC: Viz Emitter
*Component 6 of 9 — Streams the living graph to the frontend*

---

## What it is

Reads the Baby Model's graph state after every step and streams
a structured diff to any connected frontend WebSocket clients.

The frontend uses these diffs to animate the latent space visualization —
nodes pulsing when active, edges drawing in when formed, clusters
splitting when BUD fires, the whole graph slowly acquiring structure.

One-way data flow. The Viz Emitter never receives instructions.
It only pushes.

---

## Location in the project

```
project/
  backend/
    viz/
      emitter.py        ← VizEmitter class
      projector.py      ← UMAP dimensionality reduction (graph → 3D positions)
      diff.py           ← computes graph diffs between steps
```

---

## What it emits

Two types of WebSocket messages: **step updates** and **snapshots**.

### Step update (every step)

Lightweight. Contains only what changed plus the current activation state.
Sent after every learning loop step.

```json
{
  "type": "step",
  "step": 1042,
  "stage": 2,
  "graph_diff": {
    "nodes_added": [],
    "nodes_removed": [],
    "edges_added": [
      {"from": "c_04", "to": "c_07", "strength": 0.10}
    ],
    "edges_removed": [],
    "edges_updated": [
      {"from": "c_02", "to": "c_05", "strength": 0.73}
    ],
    "clusters_added": [],
    "clusters_removed": [],
    "clusters_updated": [
      {"id": "c_04", "type": "integration", "density": 0.71}
    ]
  },
  "activations": {
    "c_00": 0.82,
    "c_02": 0.61,
    "c_04": 0.44
  },
  "growth_events": [
    {"type": "CONNECT", "from": "c_04", "to": "c_07"}
  ],
  "dialogue": {
    "question": "What do dogs and cats have in common?",
    "answer": "They are both mammals kept as pets.",
    "curiosity_score": 0.74,
    "is_positive": true
  },
  "positions_stale": false
}
```

`positions_stale: true` means node positions haven't been recomputed
since the last projection run. The frontend should not animate
position changes when stale — hold nodes in place.

### Snapshot (every N steps or on demand)

Full graph state. Sent when the frontend first connects,
and every `snapshot_interval` steps (default 50).
Also sent on demand when frontend calls `GET /snapshot`.

```json
{
  "type": "snapshot",
  "step": 1000,
  "stage": 2,
  "nodes": [
    {
      "id": "n_042",
      "cluster": "c_04",
      "activation_mean": 0.61,
      "activation_variance": 0.12,
      "age": 204,
      "pos": [0.42, 0.18, 0.61],
      "alive": true
    }
  ],
  "clusters": [
    {
      "id": "c_04",
      "type": "integration",
      "density": 0.78,
      "node_count": 18,
      "layer_index": 1.0,
      "label": null,
      "dormant": false,
      "age": 412
    }
  ],
  "edges": [
    {
      "from": "c_04",
      "to": "c_07",
      "strength": 0.88,
      "age": 204,
      "direction": "bidirectional"
    }
  ],
  "model_stats": {
    "total_nodes": 312,
    "total_clusters": 14,
    "total_edges": 48,
    "dormant_clusters": 2,
    "layer_count": 4,
    "step": 1000,
    "stage": 2
  }
}
```

---

## Interface

```python
class VizEmitter:
    def __init__(
        self,
        snapshot_interval: int = 50,
        projection_interval: int = 200
    ):
        self._clients: set[WebSocket] = set()
        self._last_graph_json: dict = {}
        self._projector = Projector()
        self._snapshot_interval = snapshot_interval
        self._projection_interval = projection_interval
        self._step = 0

    # Called by FastAPI WebSocket handler
    async def connect(self, ws: WebSocket) -> None
    async def disconnect(self, ws: WebSocket) -> None

    # Called by Learning Loop after each step
    async def emit_step(
        self,
        step: int,
        stage: int,
        graph: Graph,
        activations: dict,          # cluster_id → float
        last_question: str,
        last_answer: str,
        curiosity_score: float,
        is_positive: bool,
        growth_events: list[dict]
    ) -> None

    # Called by API handler on GET /snapshot
    async def emit_snapshot_to(self, ws: WebSocket) -> None

    # Called by API handler for initial page load (REST)
    def get_current_snapshot(self) -> dict
```

---

## emit_step implementation

```python
async def emit_step(self, step, stage, graph, activations,
                    last_question, last_answer,
                    curiosity_score, is_positive,
                    growth_events) -> None:

    self._step = step

    # Compute diff against last known graph state
    current_json = graph.to_json()
    diff = compute_diff(self._last_graph_json, current_json)
    self._last_graph_json = current_json

    # Check if positions need recomputing
    positions_stale = (step % self._projection_interval != 0)
    if not positions_stale:
        # Run UMAP projection — updates pos on all nodes
        await self._projector.reproject(graph)

    # Build step message
    message = {
        "type": "step",
        "step": step,
        "stage": stage,
        "graph_diff": diff,
        "activations": activations,
        "growth_events": growth_events,
        "dialogue": {
            "question": last_question,
            "answer": last_answer,
            "curiosity_score": curiosity_score,
            "is_positive": is_positive
        },
        "positions_stale": positions_stale
    }

    await self._broadcast(message)

    # Full snapshot on interval
    if step % self._snapshot_interval == 0:
        snapshot = self._build_snapshot(step, stage, graph)
        await self._broadcast(snapshot)
```

---

## connect / disconnect

```python
async def connect(self, ws: WebSocket) -> None:
    """
    Called when a new frontend client connects.
    Sends the current full snapshot immediately so the
    frontend can render the current state without waiting
    for the next step update.
    """
    await ws.accept()
    self._clients.add(ws)

    # Send current snapshot immediately
    if self._last_graph_json:
        snapshot = self._build_snapshot_from_json(self._last_graph_json)
        await ws.send_json(snapshot)

async def disconnect(self, ws: WebSocket) -> None:
    self._clients.discard(ws)

async def _broadcast(self, message: dict) -> None:
    """
    Sends to all connected clients.
    Removes clients that have disconnected without notice
    (catches send errors silently).
    """
    dead = set()
    for ws in self._clients:
        try:
            await ws.send_json(message)
        except Exception:
            dead.add(ws)
    self._clients -= dead
```

---

## Projector (UMAP to 3D)

```python
class Projector:
    """
    Reduces node weight vectors from 512 dimensions to 3D positions
    for the frontend visualizer.

    Uses UMAP — preserves local structure (nearby nodes in 512D
    stay nearby in 3D) better than PCA or t-SNE for this use case.

    Runs every projection_interval steps (default: 200).
    Not every step — UMAP is O(N log N) and takes 0.5-3s
    depending on node count.

    Between projections: nodes hold their last computed positions.
    The frontend interpolates smoothly between old and new positions
    when a new projection arrives.
    """

    def __init__(self):
        self._umap = None        # lazy init — import umap on first use
        self._last_positions: dict[str, list[float]] = {}

    async def reproject(self, graph: Graph) -> None:
        """
        Runs UMAP on all living node weight vectors.
        Updates pos on each node in place.
        Runs in a thread pool executor to avoid blocking the event loop.
        """
        living_nodes = [
            n for c in graph.clusters if not c.dormant
            for n in c.nodes if n.alive
        ]
        if len(living_nodes) < 4:
            # UMAP needs at least 4 points
            for node in living_nodes:
                node.pos = [0.0, 0.0, 0.0]
            return

        # Collect weight matrix
        weight_matrix = torch.stack(
            [n.weights for n in living_nodes]
        ).numpy()    # UMAP needs numpy

        # Run in thread pool — UMAP is CPU-bound and blocks
        loop = asyncio.get_event_loop()
        positions = await loop.run_in_executor(
            None,
            self._run_umap,
            weight_matrix
        )

        # Write positions back to nodes
        for node, pos in zip(living_nodes, positions):
            node.pos = pos.tolist()
            self._last_positions[node.id] = node.pos

    def _run_umap(self, weight_matrix: np.ndarray) -> np.ndarray:
        """Synchronous UMAP call — runs in thread pool."""
        import umap
        if self._umap is None or weight_matrix.shape[0] != self._last_n:
            # Reinitialize UMAP if node count changed significantly
            n_neighbors = min(15, weight_matrix.shape[0] - 1)
            self._umap = umap.UMAP(
                n_components=3,
                n_neighbors=n_neighbors,
                min_dist=0.1,
                metric="cosine",    # matches our L2-normalized space
                random_state=42,
                n_jobs=1            # single thread inside the executor
            )
            self._last_n = weight_matrix.shape[0]

        positions = self._umap.fit_transform(weight_matrix)

        # Normalize positions to [-1, 1] cube for frontend
        for i in range(3):
            col = positions[:, i]
            rng = col.max() - col.min()
            if rng > 0:
                positions[:, i] = 2 * (col - col.min()) / rng - 1

        return positions
```

**UMAP install:**
```bash
pip install umap-learn
```
Note: `umap-learn` (the correct package) vs `umap` (wrong, unrelated).

**Why cosine metric:**
All node weight vectors are L2-normalized — they live on the unit hypersphere.
Cosine distance on the hypersphere is the natural distance metric.
Euclidean distance on L2-normalized vectors is equivalent to cosine distance
but cosine is more numerically stable in UMAP's graph construction.

---

## diff.py

```python
def compute_diff(old_json: dict, new_json: dict) -> dict:
    """
    Computes the structural difference between two graph states.
    Returns only what changed — not the full graph.

    Compares by id: anything with a new id is added,
    anything missing an old id is removed,
    anything with the same id but changed values is updated.
    """
    if not old_json:
        # No previous state — everything is "added"
        # But don't send full graph as diff — that's what snapshot is for
        return {
            "nodes_added": [],
            "nodes_removed": [],
            "edges_added": new_json.get("edges", []),
            "edges_removed": [],
            "edges_updated": [],
            "clusters_added": new_json.get("clusters", []),
            "clusters_removed": [],
            "clusters_updated": []
        }

    old_clusters = {c["id"]: c for c in old_json.get("clusters", [])}
    new_clusters = {c["id"]: c for c in new_json.get("clusters", [])}
    old_edges = {(e["from"], e["to"]): e for e in old_json.get("edges", [])}
    new_edges = {(e["from"], e["to"]): e for e in new_json.get("edges", [])}
    old_nodes = {n["id"]: n for n in old_json.get("nodes", [])}
    new_nodes = {n["id"]: n for n in new_json.get("nodes", [])}

    return {
        "nodes_added": [
            new_nodes[k] for k in new_nodes if k not in old_nodes
        ],
        "nodes_removed": [
            old_nodes[k] for k in old_nodes if k not in new_nodes
        ],
        "edges_added": [
            new_edges[k] for k in new_edges if k not in old_edges
        ],
        "edges_removed": [
            old_edges[k] for k in old_edges if k not in new_edges
        ],
        "edges_updated": [
            new_edges[k] for k in new_edges
            if k in old_edges
            and abs(new_edges[k]["strength"] - old_edges[k]["strength"]) > 0.02
        ],
        "clusters_added": [
            new_clusters[k] for k in new_clusters if k not in old_clusters
        ],
        "clusters_removed": [
            old_clusters[k] for k in old_clusters if k not in new_clusters
        ],
        "clusters_updated": [
            new_clusters[k] for k in new_clusters
            if k in old_clusters
            and new_clusters[k] != old_clusters[k]
        ]
    }
```

The edge update threshold of `0.02` avoids spamming the frontend
with tiny strength changes every step. Only meaningful strength
changes (edge strengthening or weakening noticeably) are sent.

---

## What the frontend does with each message type

```
ON "snapshot":
  Replace entire local graph state
  Re-render all nodes, edges, clusters from scratch
  Reset animation state
  (This is the expensive render — happens rarely)

ON "step" with positions_stale=false:
  Apply diff (add/remove nodes, edges, clusters)
  Smoothly animate nodes to their new positions
  (Tween over 500ms toward new pos)
  Flash activated clusters (pulse animation ~300ms)
  Animate growth events:
    CONNECT: draw edge gradually (0% → 100% opacity over 800ms)
    PRUNE:   fade edge out (100% → 0% opacity over 600ms)
    BUD:     split animation (parent shrinks, two children expand)
    INSERT:  new cluster fades in between two existing ones
  Update dialogue feed with new Q&A

ON "step" with positions_stale=true:
  Apply diff (structural changes only)
  DO NOT animate position changes
  Flash activated clusters
  Animate growth events
  Update dialogue feed
```

---

## Edge update threshold for frontend

Not every edge strength change is worth animating.
The diff already filters changes < 0.02.
The frontend should also apply a visual threshold:

```
strength 0.0 - 0.2:   nearly invisible line (opacity 0.1)
strength 0.2 - 0.5:   faint line (opacity 0.3)
strength 0.5 - 0.8:   visible line (opacity 0.6)
strength 0.8 - 1.0:   strong line (opacity 1.0, slightly thicker)
```

This means newly formed edges (strength 0.1) are barely visible
and must earn their visual weight by surviving and strengthening.
Old stable connections are prominent. Exactly what you want.

---

## Tests

```python
# test_viz_emitter.py

async def test_connect_sends_snapshot():
    emitter = VizEmitter()
    # Seed with a known graph state
    emitter._last_graph_json = minimal_graph_json()

    ws = MockWebSocket()
    await emitter.connect(ws)
    assert len(ws.sent_messages) == 1
    assert ws.sent_messages[0]["type"] == "snapshot"

async def test_no_snapshot_on_connect_if_empty():
    emitter = VizEmitter()
    # No graph state yet
    ws = MockWebSocket()
    await emitter.connect(ws)
    assert len(ws.sent_messages) == 0   # nothing to send

async def test_emit_step_broadcasts_to_all_clients():
    emitter = VizEmitter()
    ws1, ws2 = MockWebSocket(), MockWebSocket()
    await emitter.connect(ws1)
    await emitter.connect(ws2)
    ws1.sent_messages.clear()
    ws2.sent_messages.clear()

    graph = minimal_graph()
    await emitter.emit_step(
        step=1, stage=0, graph=graph,
        activations={"c_00": 0.5},
        last_question="test?", last_answer="test.",
        curiosity_score=0.7, is_positive=True,
        growth_events=[]
    )
    assert len(ws1.sent_messages) >= 1
    assert len(ws2.sent_messages) >= 1

async def test_dead_client_removed_on_broadcast():
    emitter = VizEmitter()
    good = MockWebSocket()
    dead = FailingWebSocket()   # raises on send
    await emitter.connect(good)
    await emitter.connect(dead)

    graph = minimal_graph()
    await emitter.emit_step(
        step=1, stage=0, graph=graph,
        activations={}, last_question="q", last_answer="a",
        curiosity_score=0.5, is_positive=True, growth_events=[]
    )
    assert dead not in emitter._clients
    assert good in emitter._clients

async def test_snapshot_sent_on_interval():
    emitter = VizEmitter(snapshot_interval=5)
    graph = minimal_graph()
    ws = MockWebSocket()
    await emitter.connect(ws)
    ws.sent_messages.clear()

    for i in range(1, 11):
        await emitter.emit_step(
            step=i, stage=0, graph=graph,
            activations={}, last_question="q", last_answer="a",
            curiosity_score=0.5, is_positive=True, growth_events=[]
        )

    snapshot_messages = [
        m for m in ws.sent_messages if m["type"] == "snapshot"
    ]
    assert len(snapshot_messages) == 2   # steps 5 and 10

def test_diff_detects_new_edge():
    old = {"nodes": [], "clusters": [], "edges": []}
    new = {
        "nodes": [],
        "clusters": [],
        "edges": [{"from": "c_00", "to": "c_01", "strength": 0.1}]
    }
    diff = compute_diff(old, new)
    assert len(diff["edges_added"]) == 1
    assert len(diff["edges_removed"]) == 0

def test_diff_detects_removed_cluster():
    old = {"nodes": [], "edges": [],
           "clusters": [{"id": "c_00", "type": "integration"}]}
    new = {"nodes": [], "edges": [], "clusters": []}
    diff = compute_diff(old, new)
    assert len(diff["clusters_removed"]) == 1

def test_diff_ignores_small_strength_change():
    edge = {"from": "c_00", "to": "c_01", "strength": 0.50}
    old = {"nodes": [], "clusters": [], "edges": [edge]}
    new_edge = {"from": "c_00", "to": "c_01", "strength": 0.51}
    new = {"nodes": [], "clusters": [], "edges": [new_edge]}
    diff = compute_diff(old, new)
    assert len(diff["edges_updated"]) == 0   # change < 0.02, filtered

async def test_projector_runs_without_error():
    projector = Projector()
    graph = minimal_graph_with_nodes(n=10)
    await projector.reproject(graph)
    for cluster in graph.clusters:
        for node in cluster.nodes:
            assert node.pos is not None
            assert len(node.pos) == 3

async def test_projector_handles_fewer_than_4_nodes():
    projector = Projector()
    graph = minimal_graph_with_nodes(n=2)
    # Should not raise — should just set pos to [0,0,0]
    await projector.reproject(graph)
```

---

## Hard parts

**UMAP reinitializes when node count changes.**
UMAP builds a k-nearest-neighbor graph during `fit_transform`.
When nodes are added (BUD, INSERT, EXTEND), the shape of the
input changes and the old UMAP model is invalid.
The spec reinitializes UMAP whenever `n != last_n`.
This is correct but means the positions will jump discontinuously
when the graph structure changes significantly.
The frontend should detect a full snapshot message (which arrives
when UMAP reruns) and do a hard re-render rather than a smooth
position tween.

**UMAP is slow as the graph grows.**
At 100 nodes: ~0.1s. At 500 nodes: ~1s. At 2000 nodes: ~5-10s.
All UMAP runs happen in a thread pool executor to avoid blocking
the asyncio event loop — but they do consume CPU.
On M1, UMAP is single-threaded (n_jobs=1 inside the executor).
Multi-threaded UMAP inside an executor causes thread-safety issues.
If UMAP becomes a bottleneck at large node counts, reduce
`projection_interval` (run less often) before parallelizing.

**WebSocket client management under rapid emit.**
At max speed the loop emits ~3 messages/second.
If a WebSocket client is slow (mobile browser on wifi),
the send buffer can back up. FastAPI's WebSocket implementation
handles this gracefully — slow clients get slow delivery,
fast clients get real-time delivery, and a crashed client
is caught by the exception handler in `_broadcast`.
No special handling needed beyond the dead-client cleanup
already in `_broadcast`.

**Memory leak from `_last_graph_json`.**
This dict holds the full serialized graph state from the last step.
At 2000 nodes, this is ~500KB. It's replaced every step,
so it doesn't grow — but the old dict is held in memory until
Python's GC collects it. On M1 with unified memory this is fine.
Don't serialize per-node activation history into this dict —
keep it to weights and structural metadata only.

**Projection positions in snapshot vs step messages.**
Step messages contain `positions_stale` but not actual positions.
Snapshot messages contain actual `pos` values on every node.
The frontend must track "last known position" per node locally
and use that when receiving a step message (no position update)
vs replace all positions when receiving a snapshot.
This split reduces message size significantly —
step messages are ~1-5KB, snapshots are ~50-500KB.

---

## M1-specific notes

UMAP uses numpy and scikit-learn internally — both run on CPU.
The thread pool executor keeps UMAP off the asyncio event loop thread.
No MPS or MLX in this component.

`umap-learn` on M1 requires:
```bash
pip install umap-learn
# If llvm-related errors appear during install:
brew install llvm
```

The `run_in_executor(None, ...)` uses Python's default thread pool
(ThreadPoolExecutor with `min(32, os.cpu_count() + 4)` workers).
On M1 with 8-10 cores this is plenty — UMAP only needs one thread.
