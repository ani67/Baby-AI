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

    def _compute_resonance(self, input_vec: torch.Tensor) -> dict[str, float]:
        """
        Pre-screen clusters by cosine similarity to input using z-score filtering.
        Returns {cluster_id: resonance_score} for clusters above mean + 1.0*std.
        Guarantees at least resonance_min_pass clusters pass (takes top N by score).
        """
        input_norm = F.normalize(input_vec, dim=0)
        scores = {}
        for c in self.graph.clusters:
            if c.dormant:
                continue
            sim = torch.dot(c.identity, input_norm).item()
            scores[c.id] = sim

        if not scores:
            return {}

        # Z-score filtering: pass clusters above mean + 1.0 * std
        vals = list(scores.values())
        mean_s = sum(vals) / len(vals)
        std_s = (sum((v - mean_s) ** 2 for v in vals) / len(vals)) ** 0.5
        z_threshold = mean_s + 1.0 * std_s

        passed = {cid: s for cid, s in scores.items() if s > z_threshold}

        # Guarantee minimum clusters pass
        min_pass = self.resonance_min_pass
        if len(passed) < min_pass and len(scores) >= min_pass:
            top_n = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:min_pass]
            for cid, s in top_n:
                passed[cid] = s
        elif len(passed) < min_pass:
            passed = scores

        if self.step % 20 == 0:
            print(f"[resonance] step={self.step} screened={len(scores)} passed={len(passed)} z_threshold={z_threshold:.2f}", flush=True)

        return passed

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

    def forward(
        self,
        x: torch.Tensor,
        return_activations: bool = False,
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

        # Resonance pre-screening — only clusters relevant to this input participate
        resonant_ids = self._compute_resonance(x)

        # Start with entry clusters that passed resonance, process in layer order
        queue = [c for c in self.graph.entry_clusters() if c.id in resonant_ids]
        # Also seed any resonant cluster that has no incoming edges (orphaned subgraphs)
        all_targets = set()
        for e in self.graph.edges:
            all_targets.add(e.to_id)
            if e.direction == "bidirectional":
                all_targets.add(e.from_id)
        for c in self.graph.clusters:
            if not c.dormant and c.id in resonant_ids and c.id not in all_targets and c.layer_index > 0:
                queue.append(c)

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
                    incoming[src] = (outputs[src], edge.strength)

            output = cluster.forward(x, incoming)
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

        # Maintain quadtree: refresh textures, split/collapse tiles
        if self.step % 10 == 0:
            self.graph.maintain_quadtree()

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
        samples: list[tuple[torch.Tensor, bool]],
    ) -> dict:
        """
        Accumulate FF updates across a batch of (vector, is_positive) pairs,
        then apply all accumulated gradients once.  Increments step by len(samples).
        Returns aggregated per-cluster weight change magnitudes.
        """
        if not samples:
            return {}

        batch_size = len(samples)
        base_lr = self._plasticity_schedule.current_rate(self.step, self.stage)
        # Scale learning rate by 1/batch_size so total weight movement
        # matches a single step — just in a better-informed direction.
        batch_lr = base_lr / batch_size

        # Snapshot weights before batch
        snapshots_before: dict[str, torch.Tensor] = {}
        for cluster in self.graph.clusters:
            if not cluster.dormant:
                snapshots_before[cluster.id] = self._cluster_weight_snapshot(cluster)

        # Accumulate FF updates: run forward + ff_update for each sample
        all_activations: list[dict[str, float]] = []
        for x, is_positive in samples:
            self.forward(x, return_activations=False)
            for cluster in self.graph.clusters:
                if not cluster.dormant and cluster.id in self._last_visited:
                    cluster.ff_update(x, is_positive, batch_lr)

            all_activations.append(dict(self._last_activations))

        # Hebbian update: average activations across batch
        avg_activations: dict[str, float] = {}
        for act_dict in all_activations:
            for cid, val in act_dict.items():
                avg_activations[cid] = avg_activations.get(cid, 0.0) + val
        n = len(all_activations)
        for cid in avg_activations:
            avg_activations[cid] /= n

        for edge in self.graph.edges:
            from_act = avg_activations.get(edge.from_id, 0.0)
            to_act = avg_activations.get(edge.to_id, 0.0)
            edge.hebbian_update(from_act, to_act)

        # Compute weight changes
        changes: dict[str, float] = {}
        for cluster in self.graph.clusters:
            if cluster.id in snapshots_before:
                after = self._cluster_weight_snapshot(cluster)
                changes[cluster.id] = torch.dist(snapshots_before[cluster.id], after).item()

        # Maintain quadtree once per batch
        if self.step % 10 == 0:
            self.graph.maintain_quadtree()

        # Advance step by batch size
        self.step += len(samples)

        # Store last activations from final sample (for growth monitor)
        if all_activations:
            self._last_activations = all_activations[-1]

        return changes

    def growth_check(self, store) -> list[dict]:
        """
        Checks all growth triggers. Executes any triggered operations.
        Logs each operation to store. Returns list of events that fired.
        """
        # Dynamic growth check interval: slows as model scales
        active_clusters = [c for c in self.graph.clusters if not c.dormant]
        check_every = 50 + (len(active_clusters) // 10)
        if self.step % check_every != 0:
            return []
        if self.step - self._last_bud_step < 20:
            return []
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

        # Check BUD — rate scales with cluster count, no stage-based cap
        bud_count = 0
        bud_skipped = 0
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

        # Check CONNECT
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

        # Check PRUNE — protect minimum edge density + 200-step post-restore cooldown
        min_edges = len(active_clusters) * 2
        pruned_count = 0
        prune_allowed = self.step - self._restore_step >= 200
        if prune_allowed:
            for edge in list(self.graph.edges):
                if len(self.graph.edges) <= min_edges:
                    break
                if monitor.should_prune(edge):
                    self.graph.remove_edge(edge)
                    pruned_count += 1
                    event = {
                        "event_type": "PRUNE",
                        "cluster_a": edge.from_id,
                        "cluster_b": edge.to_id,
                        "metadata": {
                            "reason": "disuse",
                            "steps_unused": edge.steps_since_activation,
                            "final_strength": edge.strength,
                        },
                    }
                    store.log_graph_event(
                        step=self.step, event_type="PRUNE",
                        cluster_a=edge.from_id, cluster_b=edge.to_id,
                        metadata=event["metadata"],
                    )
                    events.append(event)
        if pruned_count > 0:
            print(f"[prune] step={self.step} removed {pruned_count} edges (strength<0.05), total={len(self.graph.edges)}", flush=True)

        # Check INSERT — no stage-based cap, growth scales naturally
        growth_allowed = True
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

        # Check EXTEND — allowed when top layer collapses
        if growth_allowed and monitor.should_extend(0):
            new_cluster = extend_top(self.graph)
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

        return events

    def _cluster_weight_snapshot(self, cluster: Cluster) -> torch.Tensor:
        """Flatten all node weights in a cluster into a single tensor, normalized by sqrt(n)."""
        if not cluster.nodes:
            return torch.zeros(self.input_dim)
        return torch.cat([n.weights.detach().clone() for n in cluster.nodes]) / math.sqrt(len(cluster.nodes))
