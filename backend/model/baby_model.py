"""
BabyModel — the assembled growing neural architecture.
"""

import math

import torch
import torch.nn.functional as F

from .node import Node
from .cluster import Cluster
from .graph import Graph, Edge
from .growth import GrowthMonitor, bud, insert_layer, extend_top
from .forward_forward import PlasticitySchedule


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

        # Curiosity buffer: recent inputs with low resonance (nothing matched well)
        self._low_resonance_inputs: list[torch.Tensor] = []  # capped at 32

        # Memory buffer: decaying echo of recent cluster activations
        self._activation_buffer = torch.zeros(input_dim)
        self.buffer_decay = 0.9
        self.buffer_weight = 0.15
        self.buffer_top_k = 5

        # Persistent node-weight matrix for multi-prototype resonance.
        # Each node's weight vector is a row — nodes ARE the prototypes.
        # resonance = max(dot(node.weights, input)) per cluster.
        self._proto_matrix: torch.Tensor | None = None      # (N*M, 512) M=nodes per cluster
        self._proto_cluster_ids: list[str] = []              # cluster_id per row (repeats per node)
        self._proto_dirty: set[str] = set()                  # clusters needing row refresh

        self._init_clusters(initial_clusters, nodes_per_cluster, initial_plasticity)

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

    def _rebuild_proto_matrix(self) -> None:
        """Full rebuild of node-weight matrix. Each node is a prototype row."""
        active = [c for c in self.graph.clusters if not c.dormant]
        if not active:
            self._proto_matrix = None
            self._proto_cluster_ids = []
            return
        rows = []
        ids = []
        for c in active:
            for node in c.nodes:
                if node.alive:
                    rows.append(node.weights)
                    ids.append(c.id)
        if not rows:
            self._proto_matrix = None
            self._proto_cluster_ids = []
            return
        self._proto_matrix = torch.stack(rows)
        self._proto_cluster_ids = ids
        self._proto_dirty.clear()

    def _refresh_proto_matrix(self) -> None:
        """Rebuild when dirty or structure changed. Nodes ARE the prototypes — no k-means needed."""
        if self._proto_matrix is None or self._proto_dirty:
            self._rebuild_proto_matrix()

    def _compute_resonance(self, input_vec: torch.Tensor) -> dict[str, float]:
        """
        Multi-prototype resonance: each node's weight vector is a prototype.
        resonance(cluster) = max(dot(node.weights, input)) over all nodes in cluster.
        Uses persistent matrix — rebuilt only when weights change.
        """
        self._refresh_proto_matrix()
        if self._proto_matrix is None:
            return {}

        input_norm = F.normalize(input_vec, dim=0)

        # Multi-head resonance: split 512-d into 4 subspaces of 128-d each.
        # A cluster can be strong on some heads (subject) and weak on others (action).
        # This enables compositional matching: "dog running" activates dog-head AND running-head.
        NUM_HEADS = 4
        head_dim = self.input_dim // NUM_HEADS
        # Reshape: (N*M, 512) → (N*M, 4, 128)
        proto_heads = self._proto_matrix.view(-1, NUM_HEADS, head_dim)
        input_heads = input_norm.view(NUM_HEADS, head_dim)
        # Per-head dot products: (N*M, 4, 128) × (4, 128) → (N*M, 4) → sum → (N*M,)
        head_sims = (proto_heads * input_heads.unsqueeze(0)).sum(dim=2)  # (N*M, 4)
        all_sims = head_sims.sum(dim=1).tolist()  # sum across heads

        # Max per cluster — each node is a prototype
        cluster_max: dict[str, float] = {}
        for sim, cid in zip(all_sims, self._proto_cluster_ids):
            if cid not in cluster_max or sim > cluster_max[cid]:
                cluster_max[cid] = sim

        # Z-score filtering on max-per-cluster scores
        ids = list(cluster_max.keys())
        sims = list(cluster_max.values())
        n = len(sims)
        if n == 0:
            return {}
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

        # Track low-resonance inputs for curiosity growth
        best_sim = max(sims) if sims else 0.0
        if best_sim < 0.15 and len(self._low_resonance_inputs) < 32:
            self._low_resonance_inputs.append(input_vec.detach().clone())

        if self.step % 20 == 0:
            print(f"[resonance] step={self.step} screened={n} passed={len(passed)} z_threshold={z_threshold:.2f} low_res_buf={len(self._low_resonance_inputs)}", flush=True)

        return passed

    def _apply_buffer(self, x: torch.Tensor) -> torch.Tensor:
        """Mix activation buffer into input. Returns effective input for resonance + forward."""
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

    def _compute_traversal_order(self, resonant_ids: dict) -> list[tuple[str, list[tuple[str, str, float, str]]]]:
        """Pre-compute BFS traversal order + edge map. Reusable across batch samples."""
        visited = set()
        order = []  # [(cluster_id, [(src_id, edge_from_id, strength, edge_type), ...])]

        queue = [c for c in self.graph.clusters if not c.dormant and c.id in resonant_ids]
        while queue:
            queue.sort(key=lambda c: c.layer_index)
            cluster = queue.pop(0)
            if cluster.id in visited or cluster.dormant or cluster.id not in resonant_ids:
                continue
            visited.add(cluster.id)

            # Pre-resolve incoming edges with type
            edges_info = []
            for edge in self.graph.incoming_edges(cluster.id):
                src = edge.from_id if edge.from_id != cluster.id else edge.to_id
                if src in visited:
                    edges_info.append((src, edge.from_id, edge.strength, getattr(edge, 'edge_type', 'excitatory')))
            order.append((cluster.id, edges_info))

            for edge in self.graph.outgoing_edges(cluster.id):
                neighbor = self.graph.get_cluster(edge.to_id)
                if neighbor and not neighbor.dormant and neighbor.id in resonant_ids and neighbor.id not in visited:
                    queue.append(neighbor)

        return order

    def _forward_fast(self, x: torch.Tensor, traversal: list, resonance_scores: dict | None = None) -> tuple[torch.Tensor, dict]:
        """Fast forward using pre-computed traversal order. Skips BFS rebuild.
        If resonance_scores provided, weights activations by resonance strength (soft resonance)."""
        effective_x = self._apply_buffer(x)
        activations = {}
        outputs = {}

        for cid, edges_info in traversal:
            cluster = self.graph.get_cluster(cid)
            if not cluster:
                continue

            incoming = {}
            for edge_info in edges_info:
                src_id, _, strength = edge_info[0], edge_info[1], edge_info[2]
                edge_type = edge_info[3] if len(edge_info) > 3 else "excitatory"
                if src_id in outputs:
                    incoming[src_id] = (outputs[src_id], strength, edge_type)

            output = cluster.forward(effective_x, incoming)
            outputs[cid] = output
            node_acts = [n.activation_history[-1] for n in cluster.nodes if n.alive and n.activation_history]
            raw_act = sum(abs(a) for a in node_acts) / len(node_acts) if node_acts else 0.0
            # Soft resonance: scale activation by resonance score (0-1)
            # Clusters with high resonance contribute more than borderline ones
            if resonance_scores and cid in resonance_scores:
                raw_act *= max(0.1, resonance_scores[cid])  # floor at 0.1 to not zero out
            activations[cid] = raw_act

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

        return result, activations

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
            for edge in self.graph.incoming_edges(cluster.id):
                src = edge.from_id
                if src == cluster.id:
                    src = edge.to_id
                if src in outputs:
                    incoming[src] = (outputs[src], edge.strength, getattr(edge, 'edge_type', 'excitatory'))

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
                self._proto_dirty.add(cid)

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
        samples: list[tuple[torch.Tensor, bool]] | list[tuple[torch.Tensor, bool, float]],
    ) -> dict:
        """
        Accumulate FF updates across a batch of (vector, is_positive[, signal_strength]) tuples.
        Increments step by len(samples).
        Returns aggregated per-cluster weight change magnitudes.
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
        for sample in samples:
            x, is_positive = sample[0], sample[1]
            signal_strength = sample[2] if len(sample) > 2 else 1.0
            self._forward_fast(x, traversal, resonance_scores=shared_resonant)
            for cid in self._last_visited:
                cluster = self.graph.get_cluster(cid)
                if cluster and not cluster.dormant:
                    cluster.ff_update(x, is_positive, batch_lr, signal_strength)
                    self._proto_dirty.add(cid)

            all_activations.append(dict(self._last_activations))

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
                    # If two clusters have very similar identities, make the edge
                    # inhibitory — they're competing for the same input space.
                    c_a = self.graph.get_cluster(pair[0])
                    c_b = self.graph.get_cluster(pair[1])
                    edge_type = "excitatory"
                    if c_a and c_b:
                        sim = torch.dot(c_a.identity, c_b.identity).item()
                        if sim > 0.85:
                            edge_type = "inhibitory"
                    self.graph.add_edge(pair[0], pair[1], strength=0.1)
                    # Set edge type on the just-added edge
                    self.graph.edges[-1].edge_type = edge_type
                    event = {
                        "event_type": "CONNECT",
                        "cluster_a": pair[0],
                        "cluster_b": pair[1],
                        "metadata": {
                            "correlation": corr,
                            "reason": "coactivation_threshold",
                            "edge_type": edge_type,
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

        # Check CURIOSITY — reactivate dormant clusters toward unrecognized inputs.
        # Instead of spawning new clusters (which would exceed the cap), repurpose
        # dormant ones by reseeding their weights toward low-resonance input directions.
        curiosity_count = 0
        dormant_clusters = [c for c in self.graph.clusters if c.dormant]
        if self._low_resonance_inputs and dormant_clusters:
            max_curiosity = min(2, len(dormant_clusters), len(self._low_resonance_inputs))
            for i in range(max_curiosity):
                target_vec = self._low_resonance_inputs.pop(0)
                cluster = dormant_clusters[i]
                # Reseed nodes toward the unrecognized input direction
                for node in cluster.nodes:
                    node.weights = F.normalize(target_vec + torch.randn_like(target_vec) * 0.1, dim=0)
                    node.bias = torch.zeros(1)
                    node.activation_history.clear()
                    node.alive = True
                cluster.dormant = False
                cluster._identity_cache = None
                curiosity_count += 1
                event = {
                    "event_type": "CURIOSITY",
                    "cluster_a": cluster.id,
                    "metadata": {"reason": "low_resonance_reactivation"},
                }
                store.log_graph_event(
                    step=self.step, event_type="CURIOSITY",
                    cluster_a=cluster.id, cluster_b=None,
                    metadata=event["metadata"],
                )
                events.append(event)
            if curiosity_count > 0:
                self._proto_dirty.update(c.id for c in dormant_clusters[:curiosity_count])
                print(f"[curiosity] step={self.step} reactivated {curiosity_count} dormant clusters toward novel inputs", flush=True)

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

        return events

    def _cluster_weight_snapshot(self, cluster: Cluster) -> torch.Tensor:
        """Flatten all node weights in a cluster into a single tensor, normalized by sqrt(n)."""
        if not cluster.nodes:
            return torch.zeros(self.input_dim)
        return torch.cat([n.weights.detach().clone() for n in cluster.nodes]) / math.sqrt(len(cluster.nodes))
