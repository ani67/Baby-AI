# SPEC: Baby Model
*Component 4 of 9 — The growing neural architecture*

---

## What it is

The core thing being built. A neural network that starts near-empty and grows
its own structure through experience.

Unlike every standard model:
- Architecture is not fixed at initialization
- Parameter count is not predetermined
- Computation per inference scales with what's being processed, not total size
- The graph topology IS the knowledge — not a container for it

It is organized into four layers of implementation, each building on the last:

```
4a. Node          primitive unit of computation
4b. Cluster       functional group of nodes
4c. Growth Ops    the six structural change operations
4d. BabyModel     the assembled system
```

---

## Location in the project

```
project/
  backend/
    model/
      node.py           ← Node primitive
      cluster.py        ← Cluster (group of nodes)
      growth.py         ← Growth operations (BUD, CONNECT, etc.)
      forward_forward.py ← Local learning rule
      baby_model.py     ← Assembled model
      graph.py          ← Graph state (nodes, edges, cluster registry)
      serializer.py     ← Save/load to .pt + JSON
```

---

---

# 4a. Node

---

## What a node is

The atomic unit. A node holds:
- A weight vector (what it has learned to respond to)
- A bias scalar
- An activation history (short rolling window)
- A plasticity scalar (how fast it currently learns)
- An age counter (how many steps it has been alive)

A node is NOT a neuron in the biological sense — it is closer to a
small feature detector. It fires when its weight vector aligns with
the incoming signal.

```python
@dataclass
class Node:
    id: str                          # "n_042"
    cluster_id: str                  # which cluster owns this node
    weights: torch.Tensor            # shape (512,) — input dim
    bias: torch.Tensor               # shape (1,)
    plasticity: float                # 0.0 to 1.0
    age: int                         # steps since creation
    activation_history: deque        # last 64 activation values (floats)
    alive: bool                      # False = dormant, excluded from forward pass
```

## Node forward pass

```python
def activate(self, x: torch.Tensor) -> float:
    """
    x: input vector, shape (512,)
    Returns: scalar activation value in [-1, 1]
    
    Simple dot product + bias + tanh.
    No matrix multiply — each node is one dot product.
    """
    raw = torch.dot(self.weights, x) + self.bias
    activation = torch.tanh(raw).item()
    self.activation_history.append(activation)
    self.age += 1
    return activation
```

## Node local update (Forward-Forward)

```python
def ff_update(
    self,
    activation: float,
    is_positive: bool,
    learning_rate: float
) -> None:
    """
    Forward-Forward local update rule.
    
    Positive data (confirmed correct by teacher):
        push weights toward current input — increase activation
    Negative data (incorrect, or low-confidence prediction):
        push weights away from current input — decrease activation
    
    No backward pass. No gradient from other nodes.
    Each node updates from its own activation alone.
    
    is_positive: True if the teacher confirmed this activation was appropriate
                 False if this activation should be suppressed
    
    The update is scaled by:
        plasticity × learning_rate × |activation|
    
    Nodes that barely fired get barely updated.
    Nodes that fired strongly get strongly updated.
    """
    sign = 1.0 if is_positive else -1.0
    magnitude = self.plasticity * learning_rate * abs(activation)
    # gradient of tanh(w·x) with respect to w is x * (1 - tanh²(w·x))
    grad = self.weights.grad_fn  # not used — we compute manually
    update = sign * magnitude * self._last_input * (1 - activation**2)
    self.weights += update
    self.weights = F.normalize(self.weights, dim=0)  # keep on unit sphere
```

## Node statistics

```python
@property
def mean_activation(self) -> float:
    if not self.activation_history:
        return 0.0
    return sum(self.activation_history) / len(self.activation_history)

@property
def activation_variance(self) -> float:
    if len(self.activation_history) < 2:
        return 0.0
    hist = list(self.activation_history)
    mean = sum(hist) / len(hist)
    return sum((x - mean)**2 for x in hist) / len(hist)

@property
def is_responsive(self) -> bool:
    """True if this node is doing meaningful work."""
    return self.mean_activation > 0.1 and self.activation_variance > 0.01
```

---

---

# 4b. Cluster

---

## What a cluster is

A functional group of nodes that share an activation pattern.
The cluster is the unit the rest of the system talks about —
the graph is a graph of clusters, not individual nodes.

```python
class Cluster:
    id: str                          # "c_04"
    nodes: list[Node]                # owned nodes
    layer_index: int                 # which depth level (0 = earliest)
    cluster_type: str                # "integration"|"transformation"|
                                     # "arbitration"|"routing"
    internal_edges: dict             # node_id → list[node_id] (within cluster)
    interface_nodes: list[str]       # node ids that face outward
    plasticity: float                # cluster-level plasticity (avg of nodes)
    age: int                         # steps since creation
    dormant: bool                    # True = excluded from all computation
```

## Cluster types (emerge, not designed)

```
INTEGRATION     high internal density, low external connections
                early layers, sensory / perceptual
                nodes cross-check each other to extract stable signal

TRANSFORMATION  medium internal density, both input and output connections
                middle layers, feature conversion
                translates one representation to another

ARBITRATION     low internal density, many inputs, few outputs
                late layers, decision making
                nodes should NOT agree — diversity is the point

ROUTING         near-zero internal density, high fan-in AND fan-out
                anywhere — acts as a switchboard
                no computation, just signal direction
```

The cluster's type is not assigned — it's inferred from its connectivity pattern:

```python
@property
def cluster_type(self) -> str:
    internal_density = self._compute_internal_density()
    external_ratio = len(self.interface_nodes) / max(len(self.nodes), 1)

    if internal_density > 0.6:
        return "integration"
    elif internal_density > 0.3 and external_ratio > 0.3:
        return "transformation"
    elif internal_density < 0.2 and external_ratio > 0.5:
        return "routing"
    else:
        return "arbitration"
```

## Cluster forward pass

```python
def forward(
    self,
    x: torch.Tensor,            # input vector (512,)
    incoming_edge_signals: dict  # {from_cluster_id: tensor (512,)}
) -> torch.Tensor:
    """
    1. Combine input x with any signals arriving via edges from other clusters
    2. Activate all living nodes against the combined input
    3. Collect activations into output vector
    4. Return output vector (512,) — projected back to embedding dim
    
    The output vector is a weighted sum of node weight vectors,
    weighted by their activation values.
    This is the cluster's "vote" about what it sees.
    """
    combined = x.clone()
    for signal in incoming_edge_signals.values():
        combined = combined + 0.3 * signal   # edge signals are additive, weighted

    node_activations = []
    for node in self.nodes:
        if node.alive:
            act = node.activate(combined)
            node_activations.append((node, act))

    if not node_activations:
        return torch.zeros(512)

    # Output = weighted sum of node weight vectors
    output = torch.zeros(512)
    total_weight = sum(abs(act) for _, act in node_activations)
    if total_weight > 0:
        for node, act in node_activations:
            output += (act / total_weight) * node.weights

    return F.normalize(output, dim=0)
```

## Cluster local update

```python
def ff_update(
    self,
    x: torch.Tensor,
    is_positive: bool,
    learning_rate: float
) -> None:
    """
    Calls ff_update on each living node in the cluster.
    Nodes that fired strongly update more than nodes that barely fired.
    """
    for node in self.nodes:
        if node.alive:
            activation = node.activation_history[-1] if node.activation_history else 0.0
            node.ff_update(activation, is_positive, learning_rate)
```

## Cluster statistics (used by growth detection)

```python
@property
def internal_density(self) -> float:
    """
    Ratio of actual internal edges to possible internal edges.
    Possible = N*(N-1)/2 for N nodes.
    """

@property
def activation_bimodality(self) -> float:
    """
    Hartigan's dip statistic on recent node activations.
    High bimodality = cluster is being pulled in two directions
    = candidate for BUD operation.
    Values > 0.05 are considered bimodal.
    """

@property
def output_coherence(self) -> float:
    """
    Cosine similarity between the cluster's outputs over recent steps.
    High coherence = stable, consistent representation.
    Low coherence = noisy or overloaded — may need to BUD.
    """

@property
def residual_structure(self) -> float:
    """
    PCA on the difference between this cluster's input and output.
    High value = structured residual = missing processing stage.
    Triggers INSERT layer operation if consistently high.
    """
```

---

---

# 4c. Growth Operations

---

## Overview

Six structural operations that change the model's architecture.
They are triggered automatically by monitoring cluster statistics.
Each is logged to the State Store as a `graph_event`.

```
BUD       one cluster splits into two
CONNECT   new edge forms between two clusters
INSERT    new cluster inserts between two existing ones
EXTEND    new cluster appended at the top of the graph
PRUNE     edge removed due to disuse
DORMANT   cluster suspended due to low activation
```

None of these operations require a backward pass.
They happen between forward passes, during the "growth check" phase
that runs every N steps (default N=100).

---

## BUD

```
Trigger:    cluster.activation_bimodality > 0.05
            AND cluster.output_coherence < 0.4
            AND cluster.age > 200  (don't split young clusters)

What it does:
  1. Cluster all nodes in the parent by their weight vectors (k-means, k=2)
  2. Create child_a with cluster_1 nodes, child_b with cluster_2 nodes
  3. Transfer internal edges: edges between nodes now within same child stay
     edges between nodes now in different children become inter-cluster edges
  4. Any external edges pointing to parent now point to both children
  5. Add an edge between child_a and child_b (they were one — they remember)
  6. Remove parent cluster from graph
  7. Log BUD event to State Store

Invariant: no information is lost — BUD is a split, not a deletion.
           All weights are preserved. All external connections are preserved.
```

```python
def bud(cluster: Cluster, graph: Graph) -> tuple[Cluster, Cluster]:
    """
    Splits cluster into two children.
    Returns (child_a, child_b).
    Modifies graph in place.
    """
    # 1. K-means split on node weight vectors
    weight_matrix = torch.stack([n.weights for n in cluster.nodes])
    labels = kmeans_2(weight_matrix)   # simple 2-center k-means, 10 iterations

    nodes_a = [n for n, l in zip(cluster.nodes, labels) if l == 0]
    nodes_b = [n for n, l in zip(cluster.nodes, labels) if l == 1]

    if len(nodes_a) == 0 or len(nodes_b) == 0:
        return None  # degenerate split — all nodes in one group, abort

    # 2. Create children
    child_a = Cluster(
        id=f"{cluster.id}a",
        nodes=nodes_a,
        layer_index=cluster.layer_index,
        plasticity=cluster.plasticity
    )
    child_b = Cluster(
        id=f"{cluster.id}b",
        nodes=nodes_b,
        layer_index=cluster.layer_index,
        plasticity=cluster.plasticity
    )

    # 3-6. Update graph
    graph.replace_cluster(cluster, [child_a, child_b])
    graph.add_edge(child_a.id, child_b.id, strength=0.5)  # sibling edge

    return child_a, child_b
```

---

## CONNECT

```
Trigger:    two clusters have co-activation correlation > 0.75
            over the last 200 steps
            AND no edge already exists between them

What it does:
  1. Create an Edge object between cluster_a and cluster_b
  2. Edge starts with strength = 0.1 (weak — must earn its place)
  3. Add to graph edge registry
  4. Log CONNECT event

Edge strength grows via Hebbian update each step they co-activate.
Edge strength decays each step they do not co-activate.
```

```python
@dataclass
class Edge:
    from_id: str
    to_id: str
    strength: float = 0.1          # 0.0 to 1.0
    age: int = 0
    direction: str = "bidirectional"  # or "forward"
    steps_since_activation: int = 0

    def hebbian_update(
        self,
        from_activation: float,
        to_activation: float,
        decay: float = 0.001
    ) -> None:
        """
        Δstrength = η × from_act × to_act - decay
        Strength bounded to [0.0, 1.0].
        If strength falls below 0.02 for 300+ steps: mark for PRUNE.
        """
        delta = 0.01 * from_activation * to_activation - decay
        self.strength = max(0.0, min(1.0, self.strength + delta))
        if from_activation > 0.1 and to_activation > 0.1:
            self.steps_since_activation = 0
        else:
            self.steps_since_activation += 1
        self.age += 1
```

---

## INSERT

```
Trigger:    residual between two adjacent clusters is structured
            cluster_a.residual_structure > 0.5
            AND this has been true for 100+ consecutive steps
            AND the residual PCA top components explain > 40% of variance

What it does:
  1. Create a new cluster with layer_index between the two
  2. Initialize its node weights from the principal components
     of the structured residual (it starts knowing what it needs to learn)
  3. Rewire: cluster_a → new_cluster → cluster_b
  4. Set new cluster plasticity high (0.9) — it has a lot to learn
  5. Log INSERT event
```

```python
def insert_layer(
    cluster_a: Cluster,
    cluster_b: Cluster,
    residual_samples: torch.Tensor,   # recent residuals, shape (N, 512)
    graph: Graph
) -> Cluster:
    """
    Creates new cluster between cluster_a and cluster_b.
    Initializes node weights from PCA of residual_samples.
    Returns the new cluster.
    """
    # PCA on residuals to find the structure
    U, S, V = torch.pca_lowrank(residual_samples, q=8)  # top 8 components
    initial_weights = V.T   # shape (8, 512) — one weight vector per component

    new_nodes = []
    for i, w in enumerate(initial_weights):
        node = Node(
            id=f"n_{graph.next_node_id()}",
            cluster_id="",   # set when cluster is created
            weights=F.normalize(w, dim=0),
            bias=torch.zeros(1),
            plasticity=0.9   # high — learn fast
        )
        new_nodes.append(node)

    new_cluster = Cluster(
        id=f"c_{graph.next_cluster_id()}",
        nodes=new_nodes,
        layer_index=(cluster_a.layer_index + cluster_b.layer_index) / 2,
        plasticity=0.9
    )

    graph.insert_cluster_between(cluster_a, new_cluster, cluster_b)
    return new_cluster
```

---

## EXTEND

```
Trigger:    the topmost cluster's outputs are collapsing —
            distinct input concepts are producing nearly identical outputs
            top_cluster.output_coherence < 0.2 (everything looks the same)
            AND this has been true for 50+ consecutive steps
            AND the system is in Stage 2 or higher

What it does:
  1. Create new cluster with layer_index = max_layer + 1
  2. Initialize nodes randomly (no residual to guide — it's a blank slate)
  3. Connect it to the current top cluster
  4. Set plasticity high (0.85)
  5. Log EXTEND event

This is new abstraction capacity forming above what already exists.
It will learn to disentangle what the layer below conflated.
```

---

## PRUNE

```
Trigger:    edge.strength < 0.02
            AND edge.steps_since_activation > 300

What it does:
  1. Remove edge from graph edge registry
  2. Remove from both clusters' interface node lists
  3. Log PRUNE event

Pruning is permanent but not destructive.
The clusters that were connected still exist — just no longer communicate.
If they start co-activating again, CONNECT will re-form an edge.
```

---

## DORMANT

```
Trigger:    cluster.mean_activation < 0.05
            over the last 500 steps
            AND cluster.age > 500

What it does:
  1. Set cluster.dormant = True
  2. Remove from forward pass traversal
  3. Keep weights in memory (not deleted)
  4. Log DORMANT event

Reactivation trigger:
  incoming signal whose nearest neighbor in embedding space
  is one of this cluster's node weight vectors
  (similarity > 0.7)
  → set dormant = False, resume normal operation

This is the model's version of long-term potentiation —
skills that go unused are quieted but not forgotten.
```

---

---

# 4d. BabyModel (assembled)

---

## Overview

The assembled system. Holds the Graph, runs the forward pass,
triggers growth operations, manages plasticity schedules,
exposes the interface that the Learning Loop calls.

```python
class BabyModel:
    def __init__(
        self,
        input_dim: int = 512,         # must match encoder output
        initial_clusters: int = 4,    # start small
        nodes_per_cluster: int = 8,   # small clusters initially
        initial_plasticity: float = 1.0,
        growth_check_interval: int = 100,   # check for growth every N steps
        snapshot_interval: int = 50         # snapshot to State Store every N steps
    ):
        self.graph = Graph()
        self.step = 0
        self.stage = 0
        self._init_clusters(initial_clusters, nodes_per_cluster)
        self._growth_monitor = GrowthMonitor(self.graph)
        self._plasticity_schedule = PlasticitySchedule()
```

## Initial state

```
4 clusters, 8 nodes each = 32 total nodes
No edges between clusters
Layer indices: 0, 0, 1, 1   (two "early" clusters, two "late")
All nodes initialized with random weight vectors, L2-normalized
All plasticity = 1.0 (maximally plastic — learn fast)
No edges (the rhizome has no connections yet — it will grow them)
```

This looks almost nothing like the final model will look.
That's intentional — starting with structure would bias the growth.

---

## Forward pass

```python
def forward(
    self,
    x: torch.Tensor,             # (512,) input vector from Encoder
    return_activations: bool = False
) -> tuple[torch.Tensor, dict]:
    """
    Routes x through the active subgraph.

    Algorithm:
    1. Find entry clusters (layer_index == 0, not dormant)
    2. Activate entry clusters with x
    3. For each cluster, gather signals from incoming edges
    4. Activate next-layer clusters with combined signals
    5. Continue until no more downstream clusters
    6. Collect output from the highest layer_index clusters
    7. Return output vector (512,) + activation dict

    The path taken depends on which edges exist and which clusters
    are active — NOT a fixed sequence through all clusters.
    Clusters with no path from any active cluster are not visited.

    Computational cost = O(active_clusters × nodes_per_cluster)
                        NOT O(total_clusters)
    """
    activations = {}
    outputs = {}
    visited = set()

    # Topological traversal
    queue = self.graph.entry_clusters()   # layer_index == 0
    while queue:
        cluster = queue.pop(0)
        if cluster.id in visited or cluster.dormant:
            continue
        visited.add(cluster.id)

        incoming = {
            edge.from_id: outputs[edge.from_id]
            for edge in self.graph.incoming_edges(cluster.id)
            if edge.from_id in outputs
        }
        output = cluster.forward(x, incoming)
        outputs[cluster.id] = output
        activations[cluster.id] = cluster.mean_activation

        for edge in self.graph.outgoing_edges(cluster.id):
            neighbor = self.graph.get_cluster(edge.to_id)
            if neighbor and not neighbor.dormant:
                queue.append(neighbor)

    # Final output = mean of highest-layer cluster outputs
    top_layer = max(
        (c for c in self.graph.clusters if not c.dormant),
        key=lambda c: c.layer_index,
        default=None
    )
    if top_layer is None:
        return torch.zeros(512), activations

    final_output = outputs.get(top_layer.id, torch.zeros(512))

    if return_activations:
        return final_output, activations
    return final_output, {}
```

---

## Update pass (Forward-Forward)

```python
def update(
    self,
    x: torch.Tensor,
    is_positive: bool,
    learning_rate: float | None = None
) -> dict:
    """
    Runs a Forward-Forward update on all visited clusters.

    Called after forward() with the same input.
    is_positive: True if teacher confirmed correct, False otherwise.

    The Learning Loop determines is_positive:
      - If the model's prediction matched the teacher's answer: True
      - If it didn't: False
      - Early stages: always True (no prediction yet, just absorbing)

    Returns dict of per-cluster weight change magnitudes (for delta_summary).
    """
    if learning_rate is None:
        learning_rate = self._plasticity_schedule.current_rate(self.step)

    changes = {}
    for cluster in self.graph.clusters:
        if not cluster.dormant and cluster.id in self._last_visited:
            before = self._cluster_weight_snapshot(cluster)
            cluster.ff_update(x, is_positive, learning_rate)
            after = self._cluster_weight_snapshot(cluster)
            changes[cluster.id] = torch.dist(before, after).item()

    # Hebbian update on all edges
    for edge in self.graph.edges:
        from_act = self._last_activations.get(edge.from_id, 0.0)
        to_act = self._last_activations.get(edge.to_id, 0.0)
        edge.hebbian_update(from_act, to_act)

    self.step += 1
    return changes
```

---

## Growth check

```python
def growth_check(self, store: StateStore) -> list[dict]:
    """
    Runs every growth_check_interval steps.
    Checks all growth triggers.
    Executes any triggered operations.
    Logs each operation to store.
    Returns list of events that fired.
    """
    if self.step % self.growth_check_interval != 0:
        return []

    events = []
    monitor = self._growth_monitor

    # Check BUD
    for cluster in self.graph.clusters:
        if monitor.should_bud(cluster):
            child_a, child_b = bud(cluster, self.graph)
            if child_a:
                event = store.log_graph_event(
                    step=self.step, event_type="BUD",
                    cluster_a=cluster.id, cluster_b=None,
                    metadata={
                        "parent": cluster.id,
                        "child_a": child_a.id,
                        "child_b": child_b.id,
                        "reason": "bimodal_activation",
                        "node_count_before": len(cluster.nodes)
                    }
                )
                events.append(event)

    # Check CONNECT
    for pair, corr in monitor.get_coactivation_candidates():
        if corr > 0.75 and not self.graph.edge_exists(*pair):
            self.graph.add_edge(*pair, strength=0.1)
            events.append(store.log_graph_event(
                step=self.step, event_type="CONNECT",
                cluster_a=pair[0], cluster_b=pair[1],
                metadata={"correlation": corr,
                          "reason": "coactivation_threshold"}
            ))

    # Check PRUNE
    for edge in list(self.graph.edges):
        if monitor.should_prune(edge):
            self.graph.remove_edge(edge)
            events.append(store.log_graph_event(
                step=self.step, event_type="PRUNE",
                cluster_a=edge.from_id, cluster_b=edge.to_id,
                metadata={"reason": "disuse",
                          "steps_unused": edge.steps_since_activation,
                          "final_strength": edge.strength}
            ))

    # Check INSERT
    for cluster_a, cluster_b in self.graph.adjacent_pairs():
        residuals = monitor.get_residuals(cluster_a.id, cluster_b.id)
        if residuals is not None and monitor.should_insert(residuals):
            new_cluster = insert_layer(cluster_a, cluster_b, residuals, self.graph)
            events.append(store.log_graph_event(
                step=self.step, event_type="INSERT",
                cluster_a=cluster_a.id, cluster_b=cluster_b.id,
                metadata={"new_cluster": new_cluster.id,
                          "reason": "structured_residual"}
            ))

    # Check EXTEND
    if monitor.should_extend(self.stage):
        new_cluster = extend_top(self.graph)
        events.append(store.log_graph_event(
            step=self.step, event_type="EXTEND",
            cluster_a=None, cluster_b=None,
            metadata={"new_cluster": new_cluster.id,
                      "reason": "top_layer_collapse"}
        ))

    # Check DORMANT
    for cluster in self.graph.clusters:
        if monitor.should_dormant(cluster):
            cluster.dormant = True
            events.append(store.log_graph_event(
                step=self.step, event_type="DORMANT",
                cluster_a=cluster.id, cluster_b=None,
                metadata={"reason": "low_activation",
                          "steps_inactive": cluster.steps_since_activation}
            ))

    return events
```

---

## Plasticity Schedule

```python
class PlasticitySchedule:
    """
    Controls the global learning rate over time.

    Early stages: high plasticity — the model is a blank slate,
                  everything should update readily
    Later stages: lower plasticity — learned concepts should be stable,
                  new things can still be learned but existing knowledge
                  should not be easily overwritten

    Per-cluster plasticity is separate — young clusters are more plastic
    regardless of the global schedule.
    """
    def current_rate(self, step: int) -> float:
        # Exponential decay from 0.01 to 0.001 over 10,000 steps
        base = 0.01
        floor = 0.001
        decay = 0.0003
        return max(floor, base * math.exp(-decay * step))

    def cluster_rate(self, cluster: Cluster, global_rate: float) -> float:
        # Young clusters learn faster regardless of global rate
        age_factor = min(1.0, cluster.age / 500)
        return global_rate * (2.0 - age_factor)   # 2x rate when new, 1x when mature
```

---

## Serialization

```python
class ModelSerializer:
    """
    Saves and loads the full model state.
    Two files per checkpoint:
      step_{N}.pt    — PyTorch state dict (all node weights and biases)
      step_{N}.json  — Graph structure (cluster membership, edges, metadata)
    """

    def save(self, model: BabyModel, path_prefix: str) -> None:
        # 1. Collect all node weights into a state dict
        state_dict = {}
        for cluster in model.graph.clusters:
            for node in cluster.nodes:
                state_dict[f"{node.id}.weights"] = node.weights
                state_dict[f"{node.id}.bias"] = node.bias
        torch.save(state_dict, f"{path_prefix}.pt")

        # 2. Save graph structure as JSON
        graph_json = self._graph_to_json(model.graph)
        with open(f"{path_prefix}.json", "w") as f:
            json.dump(graph_json, f)

    def load(self, model: BabyModel, path_prefix: str) -> None:
        # 1. Load graph structure first (determines which nodes exist)
        with open(f"{path_prefix}.json") as f:
            graph_json = json.load(f)
        self._json_to_graph(graph_json, model.graph)

        # 2. Load weights into the nodes that now exist
        state_dict = torch.load(f"{path_prefix}.pt", map_location="cpu")
        for cluster in model.graph.clusters:
            for node in cluster.nodes:
                if f"{node.id}.weights" in state_dict:
                    node.weights = state_dict[f"{node.id}.weights"]
                    node.bias = state_dict[f"{node.id}.bias"]
```

---

## GrowthMonitor

```python
class GrowthMonitor:
    """
    Tracks statistics needed by growth operations.
    Runs in the background — updated every step,
    queried every growth_check_interval steps.
    """
    def __init__(self, graph: Graph):
        self._coactivation = {}     # (id_a, id_b) → deque of corr values
        self._residuals = {}        # (id_a, id_b) → deque of residual vectors
        self._activation_history = {}   # cluster_id → deque

    def record_step(
        self,
        activations: dict,          # cluster_id → float
        outputs: dict               # cluster_id → tensor
    ) -> None:
        """Called every step by BabyModel after forward()."""
        # Update co-activation matrix
        active = [k for k, v in activations.items() if abs(v) > 0.1]
        for i, a in enumerate(active):
            for b in active[i+1:]:
                key = tuple(sorted([a, b]))
                if key not in self._coactivation:
                    self._coactivation[key] = deque(maxlen=200)
                corr = activations[a] * activations[b]
                self._coactivation[key].append(corr)

        # Record per-cluster activation history
        for cid, act in activations.items():
            if cid not in self._activation_history:
                self._activation_history[cid] = deque(maxlen=500)
            self._activation_history[cid].append(act)

    def should_bud(self, cluster: Cluster) -> bool:
        return (
            cluster.activation_bimodality > 0.05
            and cluster.output_coherence < 0.4
            and cluster.age > 200
            and not cluster.dormant
        )

    def should_prune(self, edge: Edge) -> bool:
        return (
            edge.strength < 0.02
            and edge.steps_since_activation > 300
        )

    def get_coactivation_candidates(self) -> list[tuple]:
        """Returns (pair, mean_correlation) for pairs above 0.75."""
        candidates = []
        for pair, history in self._coactivation.items():
            if len(history) > 50:
                mean_corr = sum(history) / len(history)
                if mean_corr > 0.75:
                    candidates.append((pair, mean_corr))
        return candidates

    def should_insert(self, residuals: torch.Tensor) -> bool:
        if len(residuals) < 100:
            return False
        U, S, V = torch.pca_lowrank(residuals, q=4)
        explained = (S[:2]**2).sum() / (S**2).sum()
        return explained.item() > 0.4

    def should_extend(self, stage: int) -> bool:
        if stage < 2:
            return False
        top_clusters = self._graph.top_layer_clusters()
        if not top_clusters:
            return False
        return all(c.output_coherence < 0.2 for c in top_clusters)

    def should_dormant(self, cluster: Cluster) -> bool:
        history = self._activation_history.get(cluster.id, deque())
        if len(history) < 500:
            return False
        mean = sum(history) / len(history)
        return mean < 0.05 and cluster.age > 500
```

---

## Graph

```python
class Graph:
    """
    Registry for all clusters and edges.
    The topology of the model.
    """
    def __init__(self):
        self.clusters: list[Cluster] = []
        self.edges: list[Edge] = []
        self._cluster_index: dict[str, Cluster] = {}
        self._node_counter: int = 0
        self._cluster_counter: int = 0

    def add_cluster(self, cluster: Cluster) -> None: ...
    def remove_cluster(self, cluster_id: str) -> None: ...
    def get_cluster(self, cluster_id: str) -> Cluster | None: ...
    def add_edge(self, from_id: str, to_id: str, strength: float) -> None: ...
    def remove_edge(self, edge: Edge) -> None: ...
    def edge_exists(self, from_id: str, to_id: str) -> bool: ...
    def incoming_edges(self, cluster_id: str) -> list[Edge]: ...
    def outgoing_edges(self, cluster_id: str) -> list[Edge]: ...
    def entry_clusters(self) -> list[Cluster]: ...   # layer_index == 0
    def top_layer_clusters(self) -> list[Cluster]: ...
    def adjacent_pairs(self) -> list[tuple]: ...    # pairs with edges between them
    def replace_cluster(
        self,
        old: Cluster,
        new_clusters: list[Cluster]
    ) -> None: ...
    def insert_cluster_between(
        self,
        before: Cluster,
        new: Cluster,
        after: Cluster
    ) -> None: ...
    def next_node_id(self) -> str: ...
    def next_cluster_id(self) -> str: ...

    def to_json(self) -> dict:
        """Full serializable representation of the current graph."""

    def summary(self) -> dict:
        return {
            "cluster_count": len(self.clusters),
            "node_count": sum(len(c.nodes) for c in self.clusters),
            "edge_count": len(self.edges),
            "dormant_count": sum(1 for c in self.clusters if c.dormant),
            "layer_count": len(set(c.layer_index for c in self.clusters
                                   if not c.dormant))
        }
```

---

## Tests

```python
# test_baby_model.py

def test_initial_state():
    model = BabyModel(initial_clusters=4, nodes_per_cluster=4)
    summary = model.graph.summary()
    assert summary["cluster_count"] == 4
    assert summary["node_count"] == 16
    assert summary["edge_count"] == 0

def test_forward_returns_512_dim():
    model = BabyModel()
    x = F.normalize(torch.randn(512), dim=0)
    output, _ = model.forward(x)
    assert output.shape == (512,)

def test_forward_output_is_normalized():
    model = BabyModel()
    x = F.normalize(torch.randn(512), dim=0)
    output, _ = model.forward(x)
    assert abs(torch.norm(output).item() - 1.0) < 1e-4

def test_update_changes_weights():
    model = BabyModel()
    x = F.normalize(torch.randn(512), dim=0)
    before = model._cluster_weight_snapshot(model.graph.clusters[0])
    model.forward(x)
    model.update(x, is_positive=True)
    after = model._cluster_weight_snapshot(model.graph.clusters[0])
    assert not torch.allclose(before, after)

def test_connect_triggered_by_coactivation():
    model = BabyModel(initial_clusters=4, growth_check_interval=10)
    x = F.normalize(torch.randn(512), dim=0)
    # Force high correlation by running same input 200 times
    for _ in range(200):
        model.forward(x)
        model.update(x, is_positive=True)
    store_mock = MockStateStore()
    events = model.growth_check(store_mock)
    connect_events = [e for e in events if e["event_type"] == "CONNECT"]
    assert len(connect_events) > 0

def test_bud_splits_cluster():
    # Create a cluster with bimodal activation
    model = BabyModel(initial_clusters=1, nodes_per_cluster=16)
    cluster = model.graph.clusters[0]
    cluster.age = 300

    # Force bimodal by setting half weights to opposite directions
    half = len(cluster.nodes) // 2
    for i, node in enumerate(cluster.nodes):
        if i < half:
            node.weights = F.normalize(torch.ones(512), dim=0)
        else:
            node.weights = F.normalize(-torch.ones(512), dim=0)

    child_a, child_b = bud(cluster, model.graph)
    assert child_a is not None
    assert child_b is not None
    assert len(model.graph.clusters) == 1   # parent removed, children added
    assert model.graph.edge_exists(child_a.id, child_b.id)

def test_serialization_round_trip():
    model = BabyModel()
    x = F.normalize(torch.randn(512), dim=0)
    for _ in range(50):
        model.forward(x)
        model.update(x, is_positive=True)

    serializer = ModelSerializer()
    serializer.save(model, "/tmp/test_checkpoint")

    model2 = BabyModel()
    serializer.load(model2, "/tmp/test_checkpoint")

    out1, _ = model.forward(x)
    out2, _ = model2.forward(x)
    assert torch.allclose(out1, out2, atol=1e-5)

def test_dormant_cluster_excluded():
    model = BabyModel(initial_clusters=4)
    cluster = model.graph.clusters[-1]
    cluster.dormant = True

    x = F.normalize(torch.randn(512), dim=0)
    _, activations = model.forward(x, return_activations=True)
    assert cluster.id not in activations
```

---

## Hard parts

**Bimodality detection is noisy early on.**
Hartigan's dip test on small samples (< 50 activations) is unreliable.
The `age > 200` guard on BUD is important — don't let it trigger
on a cluster that hasn't been seen enough to have a real distribution.
In the first 200 steps almost no growth operations should fire.

**K-means in BUD can degenerate.**
If all nodes have nearly identical weights (hasn't specialized yet),
k-means will put all nodes in one cluster and none in the other.
The `if len(nodes_a) == 0 or len(nodes_b) == 0: return None` guard
is essential — a degenerate split should abort cleanly, not produce
a zero-node cluster.

**Topological forward pass ordering.**
The traversal queue must process clusters in layer order to avoid
a cluster receiving an edge signal from a cluster that hasn't
been computed yet this step. Sort the queue by `layer_index` before
processing. Mixed layer_index values (from INSERT, which creates
fractional indices) require float comparison — use `round(x, 2)`
for bucketing if exact equality is needed.

**Residual tracking for INSERT is memory-intensive.**
Storing 100 recent residual vectors of shape (512,) per adjacent pair
is 100 × 512 × 4 bytes = 200KB per pair. With 20 adjacent pairs
that's 4MB of residual buffer. Bounded deques with maxlen=100 keep
this from growing unbounded. But don't track every adjacent pair —
only pairs that have been stable neighbors for > 50 steps.

**Forward-Forward positive/negative signal at early stages.**
At Stage 0, the model has no predictions — it's just absorbing.
Treating all early updates as `is_positive=True` is correct
(the teacher's answers are ground truth, just absorb them).
At Stage 1-2, the model can make predictions before seeing the answer.
Compare the model's output vector to the encoded answer vector.
If cosine similarity > 0.5: the model was close → is_positive=True.
If cosine similarity < 0.5: it was wrong → is_positive=False.
This threshold is a hyperparameter that may need tuning.

---

## M1-specific notes

**MPS backend for PyTorch.**
All tensors in the Baby Model should be on MPS device when available:
```python
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
```
Node weight vectors (512-dim) and all operations on them benefit from MPS.
The growth operations (BUD's k-means, INSERT's PCA) run on CPU —
they're called infrequently and the data is small enough that
the MPS transfer overhead isn't worth it.

**Memory budget.**
Starting: 32 nodes × 512 floats × 4 bytes = 65KB. Trivial.
After 10,000 steps with moderate growth: estimate 500-2000 nodes.
2000 × 512 × 4 bytes = 4MB. Still trivial.
The model weights will never be the memory constraint on M1.
The CLIP encoder (~330MB) will always dwarf the Baby Model.

**float32 throughout.**
MPS has inconsistent float16 support in some PyTorch versions.
Use float32 for all node weights and activations.
If memory ever becomes a concern (it won't), revisit.
