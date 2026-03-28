"""
BabyModel V2 — thin wrapper around BrainState parallel engine.

Preserves the exact API that the orchestrator, viz emitter, and endpoints expect.
Internally delegates everything to BrainState (GPU-parallel, no Python loops).
"""

import torch
import torch.nn.functional as F

from .brain import BrainState


class GraphAdapter:
    """Thin adapter so orchestrator/viz can call graph.clusters, graph.summary(), etc."""

    def __init__(self, brain: BrainState):
        self._brain = brain

    @property
    def clusters(self):
        """Return list of cluster-like objects for API compatibility."""
        result = []
        for i in range(self._brain.n):
            cid = self._brain.cluster_ids[i]
            result.append(_ClusterView(self._brain, i, cid))
        return result

    @property
    def edges(self):
        """Return list of edge-like objects. Capped at 5000 for API speed at scale."""
        result = []
        limit = 5000
        for (i, j), s in self._brain._edge_strengths.items():
            if i < self._brain.n and j < self._brain.n:
                result.append(_EdgeView(
                    self._brain.cluster_ids[i],
                    self._brain.cluster_ids[j],
                    s,
                ))
                if len(result) >= limit:
                    break
        return result

    def summary(self) -> dict:
        return self._brain.summary()

    def to_json(self) -> dict:
        clusters = []
        nodes = []
        for i in range(self._brain.n):
            cid = self._brain.cluster_ids[i]
            dormant = self._brain.dormant[i].item()
            layer = self._brain.layer_indices[i].item()
            nid = f"n_{i:04d}"
            clusters.append({
                "id": cid,
                "dormant": dormant,
                "layer_index": layer,
                "plasticity": 1.0,
                "age": self._brain.ages[i].item(),
                "nodes": [{"id": nid, "plasticity": 1.0, "age": 0, "alive": not dormant}],
                "internal_edges": {},
                "interface_nodes": [],
            })
            nodes.append({
                "id": nid,
                "cluster_id": cid,
            })

        edges = []
        edge_limit = 10000  # cap serialization for speed at scale
        for (i, j), s in self._brain._edge_strengths.items():
            if i < self._brain.n and j < self._brain.n:
                edges.append({
                    "from": self._brain.cluster_ids[i],
                    "to": self._brain.cluster_ids[j],
                    "from_id": self._brain.cluster_ids[i],
                    "to_id": self._brain.cluster_ids[j],
                    "strength": s,
                    "age": 0,
                    "direction": "forward",
                    "steps_since_activation": 0,
                })
                if len(edges) >= edge_limit:
                    break

        return {
            "step": 0,  # filled by caller
            "stage": 0,
            "node_counter": self._brain.n,
            "cluster_counter": self._brain.n,
            "clusters": clusters,
            "edges": edges,
        }

    def get_cluster(self, cluster_id: str):
        idx = self._brain._id_to_idx.get(cluster_id)
        if idx is None:
            return None
        return _ClusterView(self._brain, idx, cluster_id)

    def incoming_edges(self, cluster_id: str):
        idx = self._brain._id_to_idx.get(cluster_id)
        if idx is None:
            return []
        result = []
        for (i, j), s in self._brain._edge_strengths.items():
            if j == idx and i < self._brain.n:
                result.append(_EdgeView(
                    self._brain.cluster_ids[i],
                    self._brain.cluster_ids[j], s,
                ))
        return result

    def outgoing_edges(self, cluster_id: str):
        idx = self._brain._id_to_idx.get(cluster_id)
        if idx is None:
            return []
        result = []
        for (i, j), s in self._brain._edge_strengths.items():
            if i == idx and j < self._brain.n:
                result.append(_EdgeView(
                    self._brain.cluster_ids[i],
                    self._brain.cluster_ids[j], s,
                ))
        return result


class _ClusterView:
    """Lightweight view into BrainState for a single cluster. Read-only."""

    def __init__(self, brain: BrainState, idx: int, cid: str):
        self._brain = brain
        self._idx = idx
        self.id = cid

    @property
    def dormant(self) -> bool:
        return self._brain.dormant[self._idx].item()

    @property
    def layer_index(self) -> float:
        return self._brain.layer_indices[self._idx].item()

    @property
    def identity(self) -> torch.Tensor:
        return F.normalize(self._brain.weights[self._idx], dim=0)

    @property
    def age(self) -> int:
        return self._brain.ages[self._idx].item()

    @property
    def nodes(self):
        """Single-node cluster for API compat."""
        return [_NodeView(self._brain, self._idx)]

    @property
    def mean_activation(self) -> float:
        return self._brain.fire_rates[self._idx].item()


class _NodeView:
    """Lightweight view for a single node (= the cluster's weight vector)."""

    def __init__(self, brain: BrainState, idx: int):
        self._brain = brain
        self._idx = idx
        self.id = f"n_{idx:04d}"
        self.alive = not brain.dormant[idx].item()
        self.plasticity = 1.0
        self.age = brain.ages[idx].item()
        self.activation_history = []
        self._store_idx = -1

    @property
    def weights(self) -> torch.Tensor:
        return self._brain.weights[self._idx]

    @property
    def bias(self) -> torch.Tensor:
        return torch.zeros(1)


class _EdgeView:
    """Lightweight edge view."""

    def __init__(self, from_id: str, to_id: str, strength: float):
        self.from_id = from_id
        self.to_id = to_id
        self.strength = strength
        self.age = 0
        self.direction = "forward"
        self.steps_since_activation = 0
        self.gate = None

    def hebbian_update(self, from_act: float, to_act: float):
        pass  # handled in BrainState._hebbian_update


class BabyModelV2:
    """
    V2 BabyModel — same API as V1, powered by BrainState parallel engine.
    Drop-in replacement: orchestrator, viz, endpoints all work unchanged.
    """

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
        self.step = 0
        self.stage = 0
        self.snapshot_interval = snapshot_interval
        self.growth_check_interval = growth_check_interval

        # V2 engine
        self.brain = BrainState(dim=input_dim, initial_size=initial_clusters)

        # Graph adapter for API compatibility
        self.graph = GraphAdapter(self.brain)

        # Buffer alias (orchestrator/dashboard access these directly)
        self._activation_buffer = self.brain.activation_buffer
        self.buffer_decay = self.brain.buffer_decay
        self.buffer_weight = self.brain.buffer_weight
        self.buffer_top_k = 5

        # Growth monitor stub (v2 growth is in BrainState)
        self._growth_monitor = _GrowthMonitorStub()
        self._last_bud_step = -20
        self._restore_step = -200

        # V1 compat attributes (health monitor reads/writes these via setattr)
        self.resonance_threshold = 0.10
        self.resonance_min_pass = 12
        self.inhibition_radius = 0.92
        self.suppression_factor = 0.5
        self.per_cluster_signal = True
        self.per_cluster_global_steps = 5000
        self.per_cluster_blend_steps = 10000
        self.gate_activation_step = float('inf')
        self.exp_per_cluster_sign = False
        self.exp_error_direction = False
        self.exp_contrastive_pairs = False
        self.exp_multi_target = False
        self.growth_warning_threshold = 256

    def forward(
        self,
        x: torch.Tensor,
        return_activations: bool = False,
        _precomputed_resonance: dict | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """Forward pass — delegates to BrainState parallel engine."""
        prediction, activations = self.brain.forward(x)
        self._activation_buffer = self.brain.activation_buffer
        if not return_activations:
            activations = {}
        return prediction, activations

    def update(self, x: torch.Tensor, is_positive: bool = True) -> dict:
        """Single-sample update (V1 compat for _step_single path)."""
        if is_positive:
            self.brain.update(x, x)  # Positive: teacher = input
        self.brain.adapt_thresholds()
        self.step += 1
        self._activation_buffer = self.brain.activation_buffer
        return {}

    def update_batch(
        self,
        samples: list[tuple],
    ) -> tuple[dict, list[dict]]:
        """
        Process a batch of samples. Each sample: (x, is_positive, teacher_vec?, patches?)
        Returns (changes_dict, all_activations_list).
        """
        all_activations = []

        for sample in samples:
            x = sample[0]
            teacher_vec = sample[2] if len(sample) > 2 else None

            # Forward
            prediction, activations = self.brain.forward(x)
            all_activations.append(activations)

            # Learn from correction
            if teacher_vec is not None:
                self.brain.update(x, teacher_vec)

            # Adapt thresholds
            self.brain.adapt_thresholds()

        # Advance step
        self.step += len(samples)

        # Sync buffer alias
        self._activation_buffer = self.brain.activation_buffer

        return {}, all_activations

    def growth_check(self, store) -> list[dict]:
        """Check growth triggers. Logs events to store."""
        events = self.brain.growth_check(self.step)
        for event in events:
            store.log_graph_event(
                step=self.step,
                event_type=event["event_type"],
                cluster_a=event.get("cluster_a"),
                cluster_b=event.get("cluster_b"),
                metadata=event.get("metadata", {}),
            )
        return events

    def restore_from_checkpoint(self, checkpoint: dict):
        """Restore from a checkpoint dict."""
        state_dict = checkpoint.get("state_dict", checkpoint.get("model_state_dict", {}))
        self.step = checkpoint.get("step", 0)
        self.stage = checkpoint.get("stage", 0)

        # If checkpoint has brain state (v2 format), restore directly
        if "brain_state" in state_dict:
            self.brain.load_state_dict(state_dict["brain_state"])
            self._activation_buffer = self.brain.activation_buffer
            return

        # Otherwise try v1 format (node weights + graph_json)
        # For now, just log and start fresh
        print("[v2] no v2 checkpoint found, starting fresh", flush=True)

    def _per_cluster_blend(self) -> float:
        """V1 compat — always 1.0 (per-cluster mode)."""
        return 1.0

    def cleanup_excess_clusters(self):
        pass

    def reconnect_orphaned_clusters(self):
        pass


class _GrowthMonitorStub:
    """Stub for API compatibility. V2 growth is in BrainState."""
    bud_cooldown_steps = 500
    def record_step(self, *a, **kw): pass
    def check(self, *a, **kw): pass
    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)
