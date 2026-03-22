"""
Growth operations: BUD, CONNECT, INSERT, EXTEND, PRUNE, DORMANT.
"""

from collections import deque

import torch
import torch.nn.functional as F

from .node import Node
from .cluster import Cluster
from .graph import Graph, Edge


def kmeans_2(weight_matrix: torch.Tensor, iters: int = 10) -> list[int]:
    """Simple 2-center k-means on rows of weight_matrix. Returns list of 0/1 labels."""
    n = weight_matrix.shape[0]
    if n < 2:
        return [0] * n

    # Initialize with two most distant points
    idx_a = 0
    dists = torch.cdist(weight_matrix[idx_a:idx_a + 1], weight_matrix).squeeze()
    idx_b = dists.argmax().item()

    center_a = weight_matrix[idx_a].clone()
    center_b = weight_matrix[idx_b].clone()

    labels = [0] * n
    for _ in range(iters):
        # Assign
        for i in range(n):
            da = torch.dist(weight_matrix[i], center_a).item()
            db = torch.dist(weight_matrix[i], center_b).item()
            labels[i] = 0 if da <= db else 1

        # Update centers
        a_points = [weight_matrix[i] for i in range(n) if labels[i] == 0]
        b_points = [weight_matrix[i] for i in range(n) if labels[i] == 1]

        if a_points:
            center_a = torch.stack(a_points).mean(dim=0)
        if b_points:
            center_b = torch.stack(b_points).mean(dim=0)

    return labels


def bud(cluster: Cluster, graph: Graph) -> tuple[Cluster, Cluster] | None:
    """
    Splits cluster into two children.
    Returns (child_a, child_b) or None if degenerate.
    Modifies graph in place.
    """
    weight_matrix = torch.stack([n.weights for n in cluster.nodes])
    labels = kmeans_2(weight_matrix)

    nodes_a = [n for n, l in zip(cluster.nodes, labels) if l == 0]
    nodes_b = [n for n, l in zip(cluster.nodes, labels) if l == 1]

    if len(nodes_a) == 0 or len(nodes_b) == 0:
        return None

    # Pad each child to at least 4 nodes — after deep BUD chains
    # clusters can get thin (2 nodes). Fresh nodes seeded near the
    # child's centroid keep the cluster viable for further learning.
    MIN_NODES = 4
    for node_list in [nodes_a, nodes_b]:
        centroid = torch.stack([n.weights for n in node_list]).mean(dim=0)
        while len(node_list) < MIN_NODES:
            fresh = Node(
                id=graph.next_node_id(),
                cluster_id="",
                weights=F.normalize(centroid + torch.randn_like(centroid) * 0.1, dim=0),
                bias=torch.zeros(1),
                plasticity=cluster.plasticity,
            )
            node_list.append(fresh)

    child_a = Cluster(
        id=f"{cluster.id}a",
        nodes=nodes_a,
        layer_index=cluster.layer_index,
        plasticity=cluster.plasticity,
    )
    child_b = Cluster(
        id=f"{cluster.id}b",
        nodes=nodes_b,
        layer_index=cluster.layer_index,
        plasticity=cluster.plasticity,
    )

    graph.replace_cluster(cluster, [child_a, child_b])
    graph.add_edge(child_a.id, child_b.id, strength=0.5)

    return child_a, child_b


def insert_layer(
    cluster_a: Cluster,
    cluster_b: Cluster,
    residual_samples: torch.Tensor,
    graph: Graph,
) -> Cluster:
    """
    Creates new cluster between cluster_a and cluster_b.
    Initializes node weights from PCA of residual_samples.
    """
    U, S, V = torch.pca_lowrank(residual_samples, q=8)
    initial_weights = V.T  # shape (8, 512)

    new_nodes = []
    for i, w in enumerate(initial_weights):
        node = Node(
            id=graph.next_node_id(),
            cluster_id="",
            weights=F.normalize(w, dim=0),
            bias=torch.zeros(1),
            plasticity=0.9,
        )
        new_nodes.append(node)

    new_cluster = Cluster(
        id=graph.next_cluster_id(),
        nodes=new_nodes,
        layer_index=(cluster_a.layer_index + cluster_b.layer_index) / 2,
        plasticity=0.9,
    )

    graph.insert_cluster_between(cluster_a, new_cluster, cluster_b)
    return new_cluster


def extend_top(graph: Graph, nodes_per_cluster: int = 8) -> Cluster:
    """
    Appends a new cluster at the top of the graph.
    """
    max_layer = max((c.layer_index for c in graph.clusters if not c.dormant), default=0)

    new_nodes = []
    for _ in range(nodes_per_cluster):
        node = Node(
            id=graph.next_node_id(),
            cluster_id="",
            weights=F.normalize(torch.randn(512), dim=0),
            bias=torch.zeros(1),
            plasticity=0.85,
        )
        new_nodes.append(node)

    new_cluster = Cluster(
        id=graph.next_cluster_id(),
        nodes=new_nodes,
        layer_index=max_layer + 1,
        plasticity=0.85,
    )

    # Connect to current top clusters
    for c in graph.top_layer_clusters():
        graph.add_edge(c.id, new_cluster.id, strength=0.3)

    graph.add_cluster(new_cluster, source="extend")
    for node in new_cluster.nodes:
        node.cluster_id = new_cluster.id

    return new_cluster


class GrowthMonitor:
    """
    Tracks statistics needed by growth operations.
    Updated every step, queried every growth_check_interval steps.
    """

    def __init__(self, graph: Graph):
        self._graph = graph
        self._coactivation: dict[tuple, deque] = {}
        self._residuals: dict[tuple, deque] = {}
        self._activation_history: dict[str, deque] = {}
        self._bud_cooldown: dict[str, int] = {}  # cluster_id → step when last budded
        self.bud_cooldown_steps: int = 500

    def record_step(
        self,
        activations: dict,
        outputs: dict,
    ) -> None:
        """Called every step by BabyModel after forward()."""
        active = set(k for k, v in activations.items() if abs(v) > 0.01)
        all_ids = list(activations.keys())
        for i, a in enumerate(all_ids):
            for b in all_ids[i + 1:]:
                key = tuple(sorted([a, b]))
                if key not in self._coactivation:
                    self._coactivation[key] = deque(maxlen=200)
                # Record 1.0 if both active, 0.0 otherwise
                both_active = 1.0 if (a in active and b in active) else 0.0
                self._coactivation[key].append(both_active)

        for cid, act in activations.items():
            if cid not in self._activation_history:
                self._activation_history[cid] = deque(maxlen=500)
            self._activation_history[cid].append(act)

        # Record residuals between adjacent cluster pairs for INSERT
        for edge in self._graph.edges:
            if edge.from_id in outputs and edge.to_id in outputs:
                key = tuple(sorted([edge.from_id, edge.to_id]))
                if key not in self._residuals:
                    self._residuals[key] = deque(maxlen=200)
                residual = outputs[edge.to_id] - outputs[edge.from_id]
                self._residuals[key].append(residual.detach())

    def should_bud(self, cluster: Cluster) -> bool:
        # With L2-normalized weights, output_coherence stays near 1.0.
        # Use bimodality alone as the primary trigger — it measures whether
        # the cluster is receiving two distinct types of input.
        return (
            cluster.activation_bimodality > 0.05
            and cluster.age > 200
            and len(cluster.nodes) >= 2
            and not cluster.dormant
            and cluster.id not in self._bud_cooldown
        )

    def mark_budded(self, cluster_id: str, step: int) -> None:
        self._bud_cooldown[cluster_id] = step

    def clear_expired_cooldowns(self, current_step: int) -> None:
        cooldown = self.bud_cooldown_steps
        self._bud_cooldown = {
            cid: s for cid, s in self._bud_cooldown.items()
            if current_step - s < cooldown
        }

    def should_prune(self, edge: Edge) -> bool:
        return edge.strength < 0.01 and edge.steps_since_activation > 150

    def get_coactivation_candidates(self) -> list[tuple]:
        candidates = []
        for pair, history in self._coactivation.items():
            if len(history) > 20:
                mean_corr = sum(history) / len(history)
                if mean_corr > 0.3:
                    candidates.append((pair, mean_corr))
        return candidates

    def get_residuals(self, id_a: str, id_b: str) -> torch.Tensor | None:
        key = tuple(sorted([id_a, id_b]))
        if key not in self._residuals or len(self._residuals[key]) < 100:
            return None
        return torch.stack(list(self._residuals[key]))

    def should_insert(self, residuals: torch.Tensor) -> bool:
        if len(residuals) < 100:
            return False
        U, S, V = torch.pca_lowrank(residuals, q=4)
        explained = (S[:2] ** 2).sum() / (S ** 2).sum()
        return explained.item() > 0.4

    def should_extend(self, stage: int) -> bool:
        # Allow EXTEND after sufficient structure has formed (replaces stage gate)
        if len(self._graph.clusters) < 30:
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
