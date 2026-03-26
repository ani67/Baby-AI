"""
BabyModel — the assembled growing neural architecture.
"""

import math
import random

import torch
import torch.nn.functional as F

from .node import Node
from .cluster import Cluster
from .graph import Graph, Edge
from .growth import GrowthMonitor, bud, insert_layer, extend_top
from .forward_forward import PlasticitySchedule
from .weight_store import WeightStore


class BabyModel:
    def __init__(
        self,
        input_dim: int = 512,
        initial_clusters: int = 4,
        nodes_per_cluster: int = 8,
        initial_plasticity: float = 1.0,
        growth_check_interval: int = 50,
        snapshot_interval: int = 50,
    ):
        self.input_dim = input_dim
        self.graph = Graph()
        self.step = 0
        self.stage = 0
        self.growth_warning_threshold = 256  # soft warning only, no blocking
        self.inhibition_radius = 0.92
        self.suppression_factor = 0.5
        self.resonance_threshold = 0.05
        self.resonance_min_pass = 12
        self.growth_check_interval = growth_check_interval
        self.snapshot_interval = snapshot_interval
        self._growth_monitor = GrowthMonitor(self.graph)
        self._plasticity_schedule = PlasticitySchedule()
        self._last_visited: set[str] = set()
        self._last_activations: dict[str, float] = {}
        self._last_outputs: dict[str, torch.Tensor] = {}
        self._last_dampened: dict[str, int] = {}
        self._last_bud_step: int = -20  # global cooldown: no growth check for 20 steps after BUD
        self._restore_step: int = -200  # post-restore prune cooldown

        # Memory buffer: decaying echo of recent cluster activations
        self._activation_buffer = torch.zeros(input_dim)
        self.buffer_decay = 0.9
        self.buffer_weight = 0.15
        self.buffer_top_k = 5

        # C.1: Per-cluster staged learning signal
        self.per_cluster_signal = True
        self.per_cluster_global_steps = 5000
        self.per_cluster_blend_steps = 10000

        # C.2: Gates disabled — halving edge signals hurt spatial score
        self.gate_activation_step = float('inf')

        # FF Signal Enrichment Experiments (all OFF by default)
        self.exp_per_cluster_sign = False  # superseded by C.1
        self.exp_error_direction = False
        self.exp_contrastive_pairs = False
        self.exp_multi_target = False

        # Persistent identity matrix for vectorized resonance.
        # Only rows for changed clusters are recomputed each step (diff approach).
        self._identity_matrix: torch.Tensor | None = None  # (N, 512)
        self._identity_ids: list[str] = []                  # cluster_id per row
        self._identity_id_to_row: dict[str, int] = {}       # cluster_id → row index
        self._identity_dirty: set[str] = set()              # clusters needing row refresh

        self._init_clusters(initial_clusters, nodes_per_cluster, initial_plasticity)

        # GPU weight store — centralizes all node weights and edge gates
        self._weight_store = WeightStore(dim=input_dim)
        self._attach_store()

    def _attach_store(self) -> None:
        """Register all nodes and edge gates with the centralized GPU store."""
        store = self._weight_store
        self.graph._weight_store = store
        for cluster in self.graph.clusters:
            cluster._store = store
            for node in cluster.nodes:
                idx = store.alloc_node(node.id, node.weights, node.bias)
                node._store_idx = idx
        for edge in self.graph.edges:
            if edge.gate is not None:
                store.alloc_gate(edge.from_id, edge.to_id, edge.gate)
        # Move activation buffer to store device
        self._activation_buffer = self._activation_buffer.to(store.device)
        print(f"[store] attached: {len(store._node_map)} nodes, {len(store._edge_map)} gates on {store.device}", flush=True)

    def _sync_to_store(self, cluster_ids: set[str] | None = None) -> None:
        """Push changed node weights back to GPU store after FF updates."""
        store = self._weight_store
        targets = cluster_ids or self._identity_dirty
        for cid in targets:
            cluster = self.graph.get_cluster(cid)
            if cluster is None:
                continue
            for node in cluster.nodes:
                if node.alive and node._store_idx >= 0:
                    store.weights[node._store_idx] = node.weights.to(store.device)
                    store.biases[node._store_idx] = node.bias.squeeze().to(store.device)

    def restore_from_checkpoint(self, checkpoint: dict) -> None:
        """
        Rebuild the model from a saved checkpoint.
        checkpoint has: state_dict, graph_json, step, stage
        """
        state_dict = checkpoint["state_dict"]
        graph_json = checkpoint["graph_json"]

        # Clear the fresh graph
        self.graph = Graph()
        self._growth_monitor = GrowthMonitor(self.graph)
        self.step = checkpoint["step"]
        self.stage = checkpoint["stage"]

        # Rebuild clusters and nodes from graph_json
        clusters_json = graph_json.get("clusters", [])
        nodes_json = graph_json.get("nodes", [])
        edges_json = graph_json.get("edges", [])

        # Group nodes by cluster
        nodes_by_cluster: dict[str, list] = {}
        for nj in nodes_json:
            cid = nj.get("cluster") or nj.get("cluster_id", "")
            nodes_by_cluster.setdefault(cid, []).append(nj)

        # Rebuild clusters
        for cj in clusters_json:
            cid = cj["id"]
            node_defs = nodes_by_cluster.get(cid, [])

            nodes = []
            for nj in node_defs:
                nid = nj["id"]
                # Restore weights from state_dict, fall back to random
                weights = state_dict.get(f"{nid}.weights",
                    F.normalize(torch.randn(self.input_dim), dim=0))
                bias = state_dict.get(f"{nid}.bias", torch.zeros(1))

                node = Node(
                    id=nid,
                    cluster_id=cid,
                    weights=weights,
                    bias=bias,
                    plasticity=nj.get("plasticity", 1.0),
                    age=nj.get("age_steps", nj.get("age", 0)),
                    alive=nj.get("alive", True),
                )
                nodes.append(node)

            cluster = Cluster(
                id=cid,
                nodes=nodes,
                layer_index=cj.get("layer_index", 0),
                plasticity=cj.get("plasticity", 1.0),
                age=cj.get("age", 0),
                dormant=cj.get("dormant", False),
            )
            self.graph.add_cluster(cluster, source="restore")

        # Rebuild edges
        for ej in edges_json:
            edge = Edge(
                from_id=ej["from"] if "from" in ej else ej.get("from_id", ""),
                to_id=ej["to"] if "to" in ej else ej.get("to_id", ""),
                strength=ej.get("strength", 0.1),
                age=ej.get("age_steps", ej.get("age", 0)),
                direction=ej.get("direction", "bidirectional"),
                steps_since_activation=ej.get("steps_since_activation", 0),
            )
            self.graph.edges.append(edge)

        # Rebuild adjacency index after bulk edge loading
        self.graph.rebuild_edge_index()

        # Fix ID counters so new nodes/clusters don't collide
        max_node = 0
        for c in self.graph.clusters:
            for n in c.nodes:
                try:
                    num = int(n.id.split("_")[1])
                    max_node = max(max_node, num + 1)
                except (IndexError, ValueError):
                    pass
        self.graph._node_counter = max_node

        max_cluster = 0
        for c in self.graph.clusters:
            try:
                num = int(c.id.split("_")[1])
                max_cluster = max(max_cluster, num + 1)
            except (IndexError, ValueError):
                pass
        self.graph._cluster_counter = max_cluster

        nc = len(self.graph.clusters)
        nn = sum(len(c.nodes) for c in self.graph.clusters)
        ne = len(self.graph.edges)
        self._restore_step = self.step

        # Restore memory buffer if present
        if "_activation_buffer" in state_dict:
            self._activation_buffer = state_dict["_activation_buffer"]
        else:
            self._activation_buffer = torch.zeros(self.input_dim)

        # Re-attach GPU weight store
        self._weight_store = WeightStore(dim=self.input_dim)
        self._attach_store()
        print(f"[restore] loaded step={self.step} clusters={nc} edges={ne}", flush=True)

    def cleanup_excess_clusters(self) -> None:
        """No-op — hard cap removed. Inhibition controls activation instead."""
        active = [c for c in self.graph.clusters if not c.dormant]
        total = len(self.graph.clusters)
        print(f"[cleanup] {len(active)} active clusters (total={total}), no cap — inhibition active", flush=True)

    def reconnect_orphaned_clusters(self) -> None:
        """Add edges to active clusters that have no incoming connections."""
        import random
        active = [c for c in self.graph.clusters if not c.dormant]
        incoming_targets = set()
        for e in self.graph.edges:
            incoming_targets.add(e.to_id)
            if e.direction == "bidirectional":
                incoming_targets.add(e.from_id)
        entry_ids = {c.id for c in active if c.layer_index == 0}
        by_layer: dict[int, list] = {}
        for c in active:
            by_layer.setdefault(c.layer_index, []).append(c)
        reconnected = 0
        for c in active:
            if c.id in entry_ids:
                continue
            if c.id not in incoming_targets:
                prev_layer = c.layer_index - 1
                candidates = [p for p in by_layer.get(prev_layer, []) if p.id != c.id]
                if not candidates:
                    candidates = [p for p in by_layer.get(c.layer_index, []) if p.id != c.id]
                if candidates:
                    src = random.choice(candidates)
                    self.graph.add_edge(src.id, c.id, strength=0.1)
                    incoming_targets.add(c.id)
                    reconnected += 1
                    print(f"[cleanup] reconnect {src.id} → {c.id} (layer {src.layer_index}→{c.layer_index})", flush=True)
        if reconnected:
            print(f"[cleanup] reconnected {reconnected} orphaned clusters", flush=True)
        else:
            print(f"[cleanup] all {len(active)} active clusters are reachable", flush=True)

    def _rebuild_identity_matrix(self) -> None:
        """Full rebuild of identity matrix. Called on first use and after growth events."""
        active = [c for c in self.graph.clusters if not c.dormant]
        if not active:
            self._identity_matrix = None
            self._identity_ids = []
            self._identity_id_to_row = {}
            return
        self._identity_ids = [c.id for c in active]
        self._identity_id_to_row = {cid: i for i, cid in enumerate(self._identity_ids)}
        self._identity_matrix = torch.stack([c.identity for c in active])
        self._identity_dirty.clear()

    def _refresh_identity_matrix(self) -> None:
        """Diff-based update: only recompute rows for clusters whose weights changed."""
        if self._identity_matrix is None:
            self._rebuild_identity_matrix()
            return

        # Check if cluster count changed (growth/dormant events)
        active_count = sum(1 for c in self.graph.clusters if not c.dormant)
        if active_count != len(self._identity_ids):
            self._rebuild_identity_matrix()
            return

        # Patch only dirty rows (~20 per step instead of ~600)
        if self._identity_dirty:
            for cid in self._identity_dirty:
                row = self._identity_id_to_row.get(cid)
                if row is not None:
                    cluster = self.graph.get_cluster(cid)
                    if cluster and not cluster.dormant:
                        self._identity_matrix[row] = cluster.identity
            self._identity_dirty.clear()

    def _compute_resonance(self, input_vec: torch.Tensor) -> dict[str, float]:
        """
        Pre-screen clusters by cosine similarity to input using z-score filtering.
        Uses persistent identity matrix — only changed rows are recomputed (diff approach).
        Returns {cluster_id: resonance_score} for top-20 clusters.
        """
        self._refresh_identity_matrix()
        if self._identity_matrix is None:
            return {}

        input_norm = F.normalize(input_vec.to(self._identity_matrix.device), dim=0)

        # Single matrix-vector multiply: (N, 512) @ (512,) → (N,)
        sims = (self._identity_matrix @ input_norm).tolist()
        ids = self._identity_ids

        # Z-score filtering: pass clusters above mean + 1.0 * std
        n = len(sims)
        mean_s = sum(sims) / n
        std_s = (sum((v - mean_s) ** 2 for v in sims) / n) ** 0.5
        z_threshold = mean_s + 1.0 * std_s

        passed = {cid: s for cid, s in zip(ids, sims) if s > z_threshold}

        # Guarantee minimum clusters pass
        min_pass = self.resonance_min_pass
        if len(passed) < min_pass:
            top_indices = sorted(range(n), key=lambda i: sims[i], reverse=True)[:min_pass]
            for i in top_indices:
                passed[ids[i]] = sims[i]

        # Hard cap at 20
        MAX_ACTIVE = 20
        if len(passed) > MAX_ACTIVE:
            top_k = sorted(passed.items(), key=lambda x: x[1], reverse=True)[:MAX_ACTIVE]
            passed = dict(top_k)

        if self.step % 20 == 0:
            print(f"[resonance] step={self.step} screened={n} passed={len(passed)} z_threshold={z_threshold:.2f}", flush=True)

        return passed

    def _gates_active(self) -> bool:
        """Gates are noise before clusters differentiate. Skip until gate_activation_step."""
        return self.step >= self.gate_activation_step

    def _per_cluster_blend(self) -> float:
        """Returns blend ratio: 0.0 = all global signal, 1.0 = all per-cluster signal."""
        if self.step < self.per_cluster_global_steps:
            return 0.0
        if self.step < self.per_cluster_global_steps + self.per_cluster_blend_steps:
            elapsed = self.step - self.per_cluster_global_steps
            return elapsed / self.per_cluster_blend_steps
        return 1.0

    def _apply_buffer(self, x: torch.Tensor) -> torch.Tensor:
        """Mix activation buffer into input. Returns effective input for resonance + forward."""
        x = x.to(self._weight_store.device)
        if self._activation_buffer.norm().item() < 1e-6:
            return x
        return F.normalize(x + self.buffer_weight * self._activation_buffer, dim=0)

    def _update_activation_buffer(self) -> None:
        """Decay buffer and add top-K cluster identity vectors weighted by activation."""
        self._activation_buffer *= self.buffer_decay
        if not self._last_activations:
            return
        top_k = sorted(
            self._last_activations.items(), key=lambda kv: kv[1], reverse=True
        )[:self.buffer_top_k]
        for cid, act in top_k:
            cluster = self.graph.get_cluster(cid)
            if cluster is not None:
                self._activation_buffer += act * cluster.identity
        if self.step % 100 == 0:
            buf_norm = self._activation_buffer.norm().item()
            print(f"[buffer] step={self.step} norm={buf_norm:.3f} top_k={[f'{c}:{a:.3f}' for c,a in top_k[:3]]}", flush=True)

    def _apply_inhibition(self, activations: dict) -> dict:
        """
        Lateral inhibition: strongly activated clusters suppress similar neighbors.
        Sorted by activation strength descending — winners suppress losers.
        Only considers clusters with activation > 0.01.
        """
        # Filter to clusters worth considering
        active_ids = [cid for cid, act in activations.items() if act > 0.01]
        if len(active_ids) < 2:
            return activations

        # Sort by activation strength (strongest first)
        active_ids.sort(key=lambda cid: activations[cid], reverse=True)

        # Cache identity vectors
        identities = {}
        for cid in active_ids:
            cluster = self.graph.get_cluster(cid)
            if cluster:
                identities[cid] = cluster.identity

        processed = set()
        suppressed_clusters = set()

        for cid in active_ids:
            if cid in processed or cid not in identities:
                continue
            processed.add(cid)

            id_a = identities[cid]
            for other_cid in active_ids:
                if other_cid in processed or other_cid not in identities:
                    continue
                id_b = identities[other_cid]
                sim = torch.dot(id_a, id_b).item()
                if sim > self.inhibition_radius:
                    activations[other_cid] *= self.suppression_factor
                    suppressed_clusters.add(other_cid)

        if suppressed_clusters and self.step % 20 == 0:
            print(f"[inhibition] step={self.step} suppressed {len(suppressed_clusters)} clusters (radius={self.inhibition_radius})", flush=True)

        return activations

    def _init_clusters(
        self, num_clusters: int, nodes_per: int, plasticity: float
    ) -> None:
        """Create initial clusters: first half at layer 0, second half at layer 1."""
        for i in range(num_clusters):
            layer = 0 if i < num_clusters // 2 else 1
            # Handle odd numbers: at least one cluster at layer 0
            if num_clusters == 1:
                layer = 0
            nodes = []
            for _ in range(nodes_per):
                node = Node(
                    id=self.graph.next_node_id(),
                    cluster_id="",
                    weights=F.normalize(torch.randn(self.input_dim), dim=0),
                    bias=torch.zeros(1),
                    plasticity=plasticity,
                )
                nodes.append(node)

            cluster = Cluster(
                id=self.graph.next_cluster_id(),
                nodes=nodes,
                layer_index=layer,
                plasticity=plasticity,
            )
            for node in cluster.nodes:
                node.cluster_id = cluster.id
            self.graph.add_cluster(cluster, source="init")

        # Connect layer 0 clusters to layer 1 clusters
        layer0 = [c for c in self.graph.clusters if c.layer_index == 0]
        layer1 = [c for c in self.graph.clusters if c.layer_index == 1]
        for c0 in layer0:
            for c1 in layer1:
                self.graph.add_edge(c0.id, c1.id, strength=0.2)

    def _forward_with_resonance(self, x: torch.Tensor, resonant_ids: dict) -> tuple[torch.Tensor, dict]:
        """Forward pass with pre-computed resonance IDs (used by update_batch)."""
        return self.forward(x, return_activations=False, _precomputed_resonance=resonant_ids)

    def _compute_traversal_order(self, resonant_ids: dict) -> list[tuple[str, list[tuple[str, Edge]]]]:
        """Pre-compute BFS traversal order + edge refs. Reusable across batch samples."""
        visited = set()
        order = []  # [(cluster_id, [(src_id, edge_ref), ...])]

        queue = [c for c in self.graph.clusters if not c.dormant and c.id in resonant_ids]
        while queue:
            queue.sort(key=lambda c: c.layer_index)
            cluster = queue.pop(0)
            if cluster.id in visited or cluster.dormant or cluster.id not in resonant_ids:
                continue
            visited.add(cluster.id)

            # Pre-resolve incoming edges (carry edge ref for gated strength)
            edges_info = []
            for edge in self.graph.incoming_edges(cluster.id):
                src = edge.from_id if edge.from_id != cluster.id else edge.to_id
                if src in visited:
                    edges_info.append((src, edge))
            order.append((cluster.id, edges_info))

            for edge in self.graph.outgoing_edges(cluster.id):
                neighbor = self.graph.get_cluster(edge.to_id)
                if neighbor and not neighbor.dormant and neighbor.id in resonant_ids and neighbor.id not in visited:
                    queue.append(neighbor)

        return order

    def _forward_fast(self, x: torch.Tensor, traversal: list[tuple[str, list[tuple[str, Edge]]]]) -> tuple[torch.Tensor, dict]:
        """Fast forward using pre-computed traversal order. Skips BFS rebuild."""
        effective_x = self._apply_buffer(x)
        activations = {}
        outputs = {}
        use_gates = self._gates_active()

        for cid, edges_info in traversal:
            cluster = self.graph.get_cluster(cid)
            if not cluster:
                continue

            # Vectorized gate computation: one matmul per cluster instead of per-edge loops
            ready_edges = [(src_id, edge) for src_id, edge in edges_info if src_id in outputs]
            incoming = {}
            if ready_edges and use_gates:
                gates = [e.gate for _, e in ready_edges if e.gate is not None]
                if gates:
                    gate_matrix = torch.stack(gates)  # (K, 512)
                    gate_scores = torch.sigmoid(gate_matrix @ effective_x).tolist()  # (K,)
                    gi = 0
                    for src_id, edge in ready_edges:
                        if edge.gate is not None:
                            incoming[src_id] = (outputs[src_id], edge.strength * gate_scores[gi])
                            gi += 1
                        else:
                            incoming[src_id] = (outputs[src_id], edge.strength)
                else:
                    incoming = {src_id: (outputs[src_id], edge.strength) for src_id, edge in ready_edges}
            else:
                incoming = {src_id: (outputs[src_id], edge.strength) for src_id, edge in ready_edges}

            output = cluster.forward(effective_x, incoming)
            outputs[cid] = output
            node_acts = [n.activation_history[-1] for n in cluster.nodes if n.alive and n.activation_history]
            activations[cid] = sum(abs(a) for a in node_acts) / len(node_acts) if node_acts else 0.0

        # Run inhibition + record
        self._growth_monitor.record_step(activations, outputs)
        activations = self._apply_inhibition(activations)
        self._last_visited = {cid for cid, _ in traversal}
        self._last_activations = activations
        self._last_outputs = outputs

        # Output from highest layer
        if not traversal:
            return torch.zeros(self.input_dim), {}
        visited_clusters = [self.graph.get_cluster(cid) for cid, _ in traversal]
        visited_clusters = [c for c in visited_clusters if c]
        if visited_clusters:
            max_layer = max(c.layer_index for c in visited_clusters)
            top = [c for c in visited_clusters if c.layer_index == max_layer]
            top_outputs = [outputs[c.id] for c in top if c.id in outputs]
            if top_outputs:
                result = F.normalize(torch.stack(top_outputs).mean(dim=0), dim=0)
            else:
                result = torch.zeros(self.input_dim)
        else:
            result = torch.zeros(self.input_dim)

        return result.cpu(), activations

    def forward(
        self,
        x: torch.Tensor,
        return_activations: bool = False,
        _precomputed_resonance: dict | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Routes x through the active subgraph.
        Returns (output_vector (512,), activations dict).
        """
        activations = {}
        outputs = {}
        visited = set()

        # Log effective LR on first step
        if self.step == 0 or (self.step % 500 == 0):
            lr = self._plasticity_schedule.current_rate(self.step, self.stage)
            print(f"[lr] stage={self.stage} step={self.step} effective_lr={lr:.6f}", flush=True)

        # Apply memory buffer: bias input toward recently active cluster directions
        effective_x = self._apply_buffer(x)

        # Resonance pre-screening — only clusters relevant to this input participate
        resonant_ids = _precomputed_resonance if _precomputed_resonance is not None else self._compute_resonance(effective_x)

        # Seed BFS from ALL resonant clusters, sorted by layer.
        # Old approach only seeded from layer-0 entry clusters, which broke
        # when rapid BUD growth pushed all resonant clusters to deeper layers.
        # Resonance screening already filters to top-20 — all are valid entry points.
        queue = [c for c in self.graph.clusters
                 if not c.dormant and c.id in resonant_ids]

        while queue:
            # Sort by layer_index to ensure proper ordering
            queue.sort(key=lambda c: c.layer_index)
            cluster = queue.pop(0)
            if cluster.id in visited or cluster.dormant or cluster.id not in resonant_ids:
                continue
            visited.add(cluster.id)

            incoming = {}
            inc_edges = self.graph.incoming_edges(cluster.id)
            ready = []
            for edge in inc_edges:
                src = edge.from_id if edge.from_id != cluster.id else edge.to_id
                if src in outputs:
                    ready.append((src, edge))
            if ready and self._gates_active():
                gates = [e.gate for _, e in ready if e.gate is not None]
                if gates:
                    gate_matrix = torch.stack(gates)
                    gate_scores = torch.sigmoid(gate_matrix @ effective_x).tolist()
                    gi = 0
                    for src, edge in ready:
                        if edge.gate is not None:
                            incoming[src] = (outputs[src], edge.strength * gate_scores[gi])
                            gi += 1
                        else:
                            incoming[src] = (outputs[src], edge.strength)
                else:
                    for src, edge in ready:
                        incoming[src] = (outputs[src], edge.strength)
            else:
                for src, edge in ready:
                    incoming[src] = (outputs[src], edge.strength)

            output = cluster.forward(effective_x, incoming)
            outputs[cluster.id] = output
            # Use instantaneous activation (last step), not rolling average
            node_acts = [
                n.activation_history[-1]
                for n in cluster.nodes
                if n.alive and n.activation_history
            ]
            instant_act = (
                sum(abs(a) for a in node_acts) / len(node_acts)
                if node_acts else 0.0
            )
            activations[cluster.id] = instant_act

            for edge in self.graph.outgoing_edges(cluster.id):
                neighbor = self.graph.get_cluster(edge.to_id)
                if neighbor and not neighbor.dormant and neighbor.id in resonant_ids and neighbor.id not in visited:
                    queue.append(neighbor)

        # Run MPS texture-based forward pass alongside the standard forward
        mps_scores = self.graph.mps_forward(x)

        # Record coactivation pairs BEFORE inhibition — growth monitor needs raw firing patterns
        if self.step % 20 == 0:
            active_ids = [k for k, v in activations.items() if abs(v) > 0.01]
            print(f"[forward] step={self.step} visited={len(visited)} activations={len(activations)} active(>0.01)={len(active_ids)} vals={[f'{k}:{v:.3f}' for k,v in list(activations.items())[:4]]}", flush=True)
        self._growth_monitor.record_step(activations, outputs)

        # Save pre-inhibition top-4 cluster outputs as fallback for zero-vector protection
        pre_inhibition_top4 = None
        if activations:
            top4_ids = sorted(activations, key=activations.get, reverse=True)[:4]
            top4_vecs = [outputs[cid] for cid in top4_ids if cid in outputs]
            if top4_vecs:
                pre_inhibition_top4 = F.normalize(torch.stack(top4_vecs).mean(dim=0), dim=0)

        # THEN apply lateral inhibition — only affects signal/learning, not growth tracking
        activations = self._apply_inhibition(activations)

        self._last_visited = visited
        self._last_activations = activations
        self._last_outputs = outputs

        # Final output = from highest-layer visited clusters
        if not visited:
            result = torch.zeros(self.input_dim)
        else:
            visited_clusters = [
                self.graph.get_cluster(cid) for cid in visited
                if self.graph.get_cluster(cid) is not None
            ]
            if not visited_clusters:
                result = torch.zeros(self.input_dim)
            else:
                max_layer = max(c.layer_index for c in visited_clusters)
                top = [c for c in visited_clusters if c.layer_index == max_layer]
                if top:
                    top_outputs = [outputs[c.id] for c in top if c.id in outputs]
                    if top_outputs:
                        result = torch.stack(top_outputs).mean(dim=0)
                        result = F.normalize(result, dim=0)
                    else:
                        result = torch.zeros(self.input_dim)
                else:
                    result = torch.zeros(self.input_dim)

        # Zero-vector protection: never return a near-zero output
        if result.norm().item() < 0.001:
            print(f"[forward] WARNING step={self.step} near-zero output, using fallback", flush=True)
            if pre_inhibition_top4 is not None:
                result = pre_inhibition_top4
            elif resonant_ids:
                best_cid = max(resonant_ids, key=resonant_ids.get)
                best_cluster = self.graph.get_cluster(best_cid)
                if best_cluster is not None:
                    result = best_cluster.identity.clone()
                else:
                    result = F.normalize(torch.randn(self.input_dim), dim=0)
            else:
                result = F.normalize(torch.randn(self.input_dim), dim=0)

        result = result.cpu() if result.device.type != 'cpu' else result
        if return_activations:
            return result, activations
        return result, {}

    def update(
        self,
        x: torch.Tensor,
        is_positive: bool,
        learning_rate: float | None = None,
    ) -> dict:
        """
        Runs a Forward-Forward update on all visited clusters.
        Returns dict of per-cluster weight change magnitudes.
        """
        if learning_rate is None:
            learning_rate = self._plasticity_schedule.current_rate(self.step, self.stage)

        changes = {}
        for cid in self._last_visited:
            cluster = self.graph.get_cluster(cid)
            if cluster and not cluster.dormant:
                before = self._cluster_weight_snapshot(cluster)
                cluster.ff_update(x, is_positive, learning_rate)
                after = self._cluster_weight_snapshot(cluster)
                changes[cluster.id] = torch.dist(before, after).item()
                self._identity_dirty.add(cid)

        # Hebbian update — only edges where at least one endpoint fired
        for cid in self._last_activations:
            for edge in self.graph._edges_from.get(cid, []):
                to_act = self._last_activations.get(edge.to_id, 0.0)
                edge.hebbian_update(self._last_activations[cid], to_act)
            for edge in self.graph._edges_to.get(cid, []):
                from_act = self._last_activations.get(edge.from_id, 0.0)
                edge.hebbian_update(from_act, self._last_activations[cid])

        # Maintain quadtree: refresh textures, split/collapse tiles
        if self.step % 10 == 0:
            self.graph.maintain_quadtree()

        # Update memory buffer: decay + record top-K cluster echoes
        self._update_activation_buffer()

        self.step += 1
        return changes

    def forward_batch(
        self,
        vectors: list[torch.Tensor],
    ) -> list[tuple[torch.Tensor, dict]]:
        """
        Run forward pass for each vector in the batch.
        Returns list of (output, activations) tuples.
        Does NOT increment step or apply updates — caller handles that.
        """
        results = []
        for x in vectors:
            output, activations = self.forward(x, return_activations=True)
            results.append((output, activations))
        return results

    def update_batch(
        self,
        samples: list[tuple],
    ) -> dict:
        """
        Accumulate FF updates across a batch of sample tuples.
        Supports: (vec, is_positive) or (vec, is_positive, teacher_vec) for enriched signal.
        Increments step by len(samples). Returns per-cluster weight change magnitudes.
        """
        if not samples:
            return {}

        batch_size = len(samples)
        base_lr = self._plasticity_schedule.current_rate(self.step, self.stage)
        # Scale learning rate by 1/batch_size so total weight movement
        # matches a single step — just in a better-informed direction.
        batch_lr = base_lr / batch_size

        # Compute resonance + traversal order ONCE for the batch.
        # All 32 samples reuse the same BFS path (same clusters, same edges).
        first_x = samples[0][0]
        shared_resonant = self._compute_resonance(self._apply_buffer(first_x))
        traversal = self._compute_traversal_order(shared_resonant)
        visited_ids = {cid for cid, _ in traversal}

        # Snapshot only visited clusters (~20), not all 600+
        snapshots_before: dict[str, torch.Tensor] = {}
        for cid in visited_ids:
            cluster = self.graph.get_cluster(cid)
            if cluster and not cluster.dormant:
                snapshots_before[cid] = self._cluster_weight_snapshot(cluster)

        # Fast forward + FF update for each sample (reuses traversal)
        all_activations: list[dict[str, float]] = []

        # Exp 3: contrastive pairs — pair up samples, winner gets +, loser gets -
        if self.exp_contrastive_pairs and len(samples) >= 2:
            paired_samples = []
            for i in range(0, len(samples) - 1, 2):
                s_a, s_b = samples[i], samples[i + 1]
                vec_a, vec_b = s_a[0], s_b[0]
                teacher_a = s_a[2] if len(s_a) > 2 else vec_a
                teacher_b = s_b[2] if len(s_b) > 2 else vec_b
                # Run forward on both, compare outputs to their teachers
                out_a, _ = self.forward(vec_a)
                out_b, _ = self.forward(vec_b)
                sim_a = torch.dot(out_a, F.normalize(teacher_a, dim=0)).item()
                sim_b = torch.dot(out_b, F.normalize(teacher_b, dim=0)).item()
                # Better one gets positive, worse gets negative
                if sim_a >= sim_b:
                    paired_samples.append((vec_a, True, teacher_a))
                    paired_samples.append((vec_b, False, teacher_b))
                else:
                    paired_samples.append((vec_a, False, teacher_a))
                    paired_samples.append((vec_b, True, teacher_b))
            samples = paired_samples

        for sample in samples:
            x = sample[0]
            is_positive = sample[1]
            teacher_vec = sample[2] if len(sample) > 2 else None
            patches = sample[3] if len(sample) > 3 else None  # C.3: (49, 512) or None

            # C.3: precompute per-cluster best patch if patches available
            cluster_patch_input: dict[str, torch.Tensor] = {}
            if patches is not None and self._identity_matrix is not None:
                # (N, 512) @ (512, 49) → (N, 49) — each cluster identity vs each patch
                patch_scores = self._identity_matrix @ patches.T
                # Each cluster picks its best-matching patch
                best_patch_indices = patch_scores.argmax(dim=1)  # (N,)
                for i, cid in enumerate(self._identity_ids):
                    cluster_patch_input[cid] = patches[best_patch_indices[i].item()]

            self._forward_fast(x, traversal)
            blend = self._per_cluster_blend() if self.per_cluster_signal else 0.0
            for cid in self._last_visited:
                cluster = self.graph.get_cluster(cid)
                if cluster and not cluster.dormant:
                    cluster_positive = is_positive

                    # C.1: Staged per-cluster signal
                    if blend > 0.0 and teacher_vec is not None and cid in self._last_outputs:
                        out_cpu = self._last_outputs[cid].cpu()
                        cluster_sim = torch.dot(
                            out_cpu, F.normalize(teacher_vec, dim=0)
                        ).item()
                        cluster_is_positive = cluster_sim > 0.0
                        if random.random() < blend:
                            cluster_positive = cluster_is_positive

                    # C.3: Use patch-specific input if available
                    update_vec = cluster_patch_input.get(cid, x)

                    # Local target learning: use rich 512-d teacher signal
                    if teacher_vec is not None:
                        cluster.local_target_update(
                            F.normalize(teacher_vec, dim=0), batch_lr
                        )
                    else:
                        cluster.ff_update(update_vec, cluster_positive, batch_lr)

                    self._identity_dirty.add(cid)

            all_activations.append(dict(self._last_activations))

        # Sync updated weights back to GPU store
        self._sync_to_store()

        # Hebbian update: average activations across batch
        avg_activations: dict[str, float] = {}
        for act_dict in all_activations:
            for cid, val in act_dict.items():
                avg_activations[cid] = avg_activations.get(cid, 0.0) + val
        n = len(all_activations)
        for cid in avg_activations:
            avg_activations[cid] /= n

        # Hebbian update — only edges where at least one endpoint fired
        for cid in avg_activations:
            for edge in self.graph._edges_from.get(cid, []):
                to_act = avg_activations.get(edge.to_id, 0.0)
                edge.hebbian_update(avg_activations[cid], to_act)
            for edge in self.graph._edges_to.get(cid, []):
                from_act = avg_activations.get(edge.from_id, 0.0)
                edge.hebbian_update(from_act, avg_activations[cid])

        # C.2: Batched gate learning — once per batch using last sample's signal
        if self._gates_active() and samples:
            last_x = samples[-1][0]
            last_teacher = samples[-1][2] if len(samples[-1]) > 2 else None
            if last_teacher is not None:
                teacher_norm = F.normalize(last_teacher, dim=0)
                for cid in self._last_visited:
                    if cid not in self._last_outputs:
                        continue
                    cid_positive = torch.dot(self._last_outputs[cid].cpu(), teacher_norm).item() > 0.0
                    edges = self.graph.incoming_edges(cid)
                    if not edges:
                        continue
                    gated = [e for e in edges if e.gate is not None]
                    if not gated:
                        continue
                    # Vectorized gate update
                    gate_stack = torch.stack([e.gate for e in gated])  # (K, 512)
                    sign = 1.0 if cid_positive else -1.0
                    gate_stack = gate_stack + sign * 0.001 * last_x.unsqueeze(0)
                    gate_stack = F.normalize(gate_stack, dim=1)
                    for i, e in enumerate(gated):
                        e.gate = gate_stack[i]

        # Compute weight changes
        changes: dict[str, float] = {}
        for cluster in self.graph.clusters:
            if cluster.id in snapshots_before:
                after = self._cluster_weight_snapshot(cluster)
                changes[cluster.id] = torch.dist(snapshots_before[cluster.id], after).item()

        # Maintain quadtree once per batch
        if self.step % 10 == 0:
            self.graph.maintain_quadtree()

        # Store last activations from final sample (for growth monitor)
        if all_activations:
            self._last_activations = all_activations[-1]

        # Update memory buffer once per batch (not per sample)
        self._update_activation_buffer()

        # Advance step by batch size
        self.step += len(samples)

        return changes

    def growth_check(self, store) -> list[dict]:
        """
        Checks all growth triggers. Executes any triggered operations.
        Logs each operation to store. Returns list of events that fired.
        """
        # Dynamic growth check interval: slows as model scales
        active_clusters = [c for c in self.graph.clusters if not c.dormant]
        check_every = 50 + (len(active_clusters) // 10)
        if self.step - getattr(self, '_last_growth_check_step', -check_every) < check_every:
            return []
        self._last_growth_check_step = self.step
        # BUD cooldown: skip BUD but still evaluate INSERT/EXTEND/PRUNE/DORMANT
        bud_on_cooldown = self.step - self._last_bud_step < 20
        total_clusters = len(self.graph.clusters)

        events = []
        monitor = self._growth_monitor
        monitor.clear_expired_cooldowns(self.step)

        # Debug: coactivation stats
        coact = monitor._coactivation
        num_pairs = len(coact)
        max_score = 0.0
        if coact:
            max_score = max(
                (sum(h) / len(h)) for h in coact.values() if len(h) > 0
            )
        print(f"[growth] step={self.step} active={len(active_clusters)} total={total_clusters} pairs={num_pairs} max_coact={max_score:.3f}", flush=True)

        if len(active_clusters) > self.growth_warning_threshold:
            print(f"[growth] WARNING: {len(active_clusters)} active clusters exceeds soft threshold {self.growth_warning_threshold}", flush=True)

        # Check BUD — rate scales with cluster count, skip if on cooldown.
        # Hard pause above 500 clusters to prevent performance collapse.
        bud_count = 0
        bud_skipped = 0
        if len(active_clusters) >= 500:
            max_buds_per_check = 0
        elif bud_on_cooldown:
            max_buds_per_check = 0
        else:
            max_buds_per_check = max(1, len(active_clusters) // 50)
        for cluster in list(self.graph.clusters):
            if monitor.should_bud(cluster):
                if bud_count >= max_buds_per_check:
                    bud_skipped += 1
                    continue
                result = bud(cluster, self.graph)
                if result is not None:
                    child_a, child_b = result
                    bud_count += 1
                    monitor.mark_budded(cluster.id, self.step)
                    self._last_bud_step = self.step
                    event = {
                        "event_type": "BUD",
                        "cluster_a": cluster.id,
                        "metadata": {
                            "parent": cluster.id,
                            "child_a": child_a.id,
                            "child_b": child_b.id,
                            "reason": "bimodal_activation",
                            "node_count_before": len(cluster.nodes),
                        },
                    }
                    store.log_graph_event(
                        step=self.step, event_type="BUD",
                        cluster_a=cluster.id, cluster_b=None,
                        metadata=event["metadata"],
                    )
                    events.append(event)
        if bud_count > 0:
            print(f"[growth] step={self.step} budded {bud_count} clusters (rate limited, {bud_skipped} eligible skipped)", flush=True)

        # Check CONNECT — skip if edge count already exceeds cap
        edge_cap = len(active_clusters) * 20
        if len(self.graph.edges) > edge_cap:
            print(f"[growth] edge cap: {len(self.graph.edges)} > {len(active_clusters)}*20, CONNECT skipped", flush=True)
        else:
            for pair, corr in monitor.get_coactivation_candidates():
                if not self.graph.edge_exists(pair[0], pair[1]):
                    self.graph.add_edge(pair[0], pair[1], strength=0.1)
                    event = {
                        "event_type": "CONNECT",
                        "cluster_a": pair[0],
                        "cluster_b": pair[1],
                        "metadata": {
                            "correlation": corr,
                            "reason": "coactivation_threshold",
                        },
                    }
                    store.log_graph_event(
                        step=self.step, event_type="CONNECT",
                        cluster_a=pair[0], cluster_b=pair[1],
                        metadata=event["metadata"],
                    )
                    events.append(event)

        # Check PRUNE — percentage-based: remove bottom 5% of edges each check.
        # Self-scaling: 98K edges → prune 4900. 500 edges → prune 25.
        # Protects minimum of 2 edges per cluster.
        min_edges = len(active_clusters) * 2
        pruned_count = 0
        prune_allowed = self.step - self._restore_step >= 200
        if prune_allowed and len(self.graph.edges) > min_edges:
            # Sort edges by strength ascending — weakest first
            sorted_edges = sorted(self.graph.edges, key=lambda e: e.strength)
            # Prune bottom 5% by strength — no additional condition.
            # If you're in the weakest 5%, you get cut.
            max_prune = max(1, len(sorted_edges) // 20)
            for edge in sorted_edges[:max_prune]:
                if len(self.graph.edges) <= min_edges:
                    break
                self.graph.remove_edge(edge)
                pruned_count += 1
        if pruned_count > 0:
            print(f"[prune] step={self.step} removed {pruned_count} edges, total={len(self.graph.edges)}", flush=True)

        # Check INSERT — paused above 500 clusters (same as BUD)
        growth_allowed = len(active_clusters) < 500
        insert_count = 0
        max_inserts_per_check = 2
        if growth_allowed:
            for cluster_a, cluster_b in self.graph.adjacent_pairs():
                if insert_count >= max_inserts_per_check:
                    break
                residuals = monitor.get_residuals(cluster_a.id, cluster_b.id)
                if residuals is not None and monitor.should_insert(residuals):
                    new_cluster = insert_layer(
                        cluster_a, cluster_b, residuals, self.graph
                    )
                    insert_count += 1
                    event = {
                        "event_type": "INSERT",
                        "cluster_a": cluster_a.id,
                        "cluster_b": cluster_b.id,
                        "metadata": {
                            "new_cluster": new_cluster.id,
                            "reason": "structured_residual",
                        },
                    }
                    store.log_graph_event(
                        step=self.step, event_type="INSERT",
                        cluster_a=cluster_a.id, cluster_b=cluster_b.id,
                        metadata=event["metadata"],
                    )
                    events.append(event)

        # Check EXTEND — allowed when top layer has diverse activation, paused above 500
        if growth_allowed and monitor.should_extend(0):
            new_cluster = extend_top(self.graph)
            print(f"[extend] new layer: {new_cluster.id} at layer={new_cluster.layer_index}", flush=True)
            event = {
                "event_type": "EXTEND",
                "metadata": {
                    "new_cluster": new_cluster.id,
                    "reason": "top_layer_collapse",
                },
            }
            store.log_graph_event(
                step=self.step, event_type="EXTEND",
                cluster_a=None, cluster_b=None,
                metadata=event["metadata"],
            )
            events.append(event)

        # Check DORMANT
        for cluster in self.graph.clusters:
            if monitor.should_dormant(cluster):
                cluster.dormant = True
                event = {
                    "event_type": "DORMANT",
                    "cluster_a": cluster.id,
                    "metadata": {
                        "reason": "low_activation",
                    },
                }
                store.log_graph_event(
                    step=self.step, event_type="DORMANT",
                    cluster_a=cluster.id, cluster_b=None,
                    metadata=event["metadata"],
                )
                events.append(event)

        # Register any new nodes/edges from growth with the GPU store
        if events and hasattr(self, '_weight_store'):
            ws = self._weight_store
            for cluster in self.graph.clusters:
                cluster._store = ws
                for node in cluster.nodes:
                    if node._store_idx < 0:
                        idx = ws.alloc_node(node.id, node.weights, node.bias)
                        node._store_idx = idx

        return events

    def save_topology(self, path: str) -> None:
        """Exp 5: Save graph topology (structure only, no weights) for reuse."""
        import json
        topo = {
            "clusters": [{"id": c.id, "layer": c.layer_index, "role": getattr(c, 'role', 'detector'),
                          "node_count": len(c.nodes)} for c in self.graph.clusters],
            "edges": [{"from": e.from_id, "to": e.to_id, "direction": e.direction}
                      for e in self.graph.edges],
        }
        with open(path, "w") as f:
            json.dump(topo, f, indent=2)
        print(f"[topology] saved {len(topo['clusters'])} clusters, {len(topo['edges'])} edges to {path}")

    def load_topology(self, path: str) -> None:
        """Exp 5: Load topology and rebuild graph with fresh random weights."""
        import json
        with open(path) as f:
            topo = json.load(f)
        self.graph = Graph()
        self._growth_monitor = GrowthMonitor(self.graph)
        for cj in topo["clusters"]:
            nodes = []
            for _ in range(cj.get("node_count", 8)):
                node = Node(
                    id=self.graph.next_node_id(), cluster_id="",
                    weights=F.normalize(torch.randn(self.input_dim), dim=0),
                    bias=torch.zeros(1),
                )
                nodes.append(node)
            cluster = Cluster(id=cj["id"], nodes=nodes, layer_index=cj["layer"])
            if hasattr(cluster, 'role'):
                cluster.role = cj.get("role", "detector")
            for n in cluster.nodes:
                n.cluster_id = cluster.id
            self.graph.add_cluster(cluster, source="topology")
        for ej in topo["edges"]:
            self.graph.add_edge(ej["from"], ej["to"])
        self.step = 0
        self._identity_dirty.add("__all__")
        print(f"[topology] loaded {len(topo['clusters'])} clusters, {len(topo['edges'])} edges with fresh weights")

    def _cluster_weight_snapshot(self, cluster: Cluster) -> torch.Tensor:
        """Flatten all node weights in a cluster into a single tensor, normalized by sqrt(n)."""
        if not cluster.nodes:
            return torch.zeros(self.input_dim)
        return torch.cat([n.weights.detach().clone() for n in cluster.nodes]) / math.sqrt(len(cluster.nodes))
