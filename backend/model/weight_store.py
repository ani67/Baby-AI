"""
Centralized GPU-resident tensor store for all node weights, biases, and edge gates.

Instead of 4000+ scattered tensors on CPU, all data lives in pre-allocated
GPU tensors indexed by row. Eliminates torch.stack() overhead and enables
vectorized forward/update passes on MPS.
"""

import torch
import torch.nn.functional as F


class WeightStore:
    def __init__(self, dim: int = 512, max_nodes: int = 8192, max_edges: int = 32768):
        # MPS causes silent segfaults with large pre-allocated tensors + fancy indexing.
        # Stay on CPU until PyTorch MPS matures. The store still eliminates torch.stack().
        self.device = torch.device("cpu")
        self.dim = dim
        self._max_nodes = max_nodes
        self._max_edges = max_edges

        # Pre-allocated tensors on device
        self.weights = torch.zeros(max_nodes, dim, device=self.device)
        self.biases = torch.zeros(max_nodes, device=self.device)
        self.gates = torch.zeros(max_edges, dim, device=self.device)

        # Allocation tracking via free lists (LIFO for cache locality)
        self._node_free: list[int] = list(range(max_nodes - 1, -1, -1))
        self._edge_free: list[int] = list(range(max_edges - 1, -1, -1))

        # Mappings
        self._node_map: dict[str, int] = {}  # node_id → row
        self._edge_map: dict[tuple[str, str], int] = {}  # (from, to) → row

    # ── Node operations ────────────────────────────────────────────────

    def alloc_node(self, node_id: str, weights: torch.Tensor, bias: torch.Tensor) -> int:
        if node_id in self._node_map:
            idx = self._node_map[node_id]
            self.weights[idx] = weights.to(self.device)
            self.biases[idx] = bias.squeeze().to(self.device)
            return idx
        if not self._node_free:
            self._grow_nodes()
        idx = self._node_free.pop()
        self._node_map[node_id] = idx
        self.weights[idx] = weights.to(self.device)
        self.biases[idx] = bias.squeeze().to(self.device)
        return idx

    def free_node(self, node_id: str) -> None:
        idx = self._node_map.pop(node_id, None)
        if idx is not None:
            self.weights[idx].zero_()
            self.biases[idx].zero_()
            self._node_free.append(idx)

    def node_idx(self, node_id: str) -> int:
        return self._node_map[node_id]

    # ── Edge gate operations ───────────────────────────────────────────

    def alloc_gate(self, from_id: str, to_id: str, gate: torch.Tensor) -> int:
        key = (from_id, to_id)
        if key in self._edge_map:
            idx = self._edge_map[key]
            self.gates[idx] = gate.to(self.device)
            return idx
        if not self._edge_free:
            self._grow_edges()
        idx = self._edge_free.pop()
        self._edge_map[key] = idx
        self.gates[idx] = gate.to(self.device)
        return idx

    def free_gate(self, from_id: str, to_id: str) -> None:
        key = (from_id, to_id)
        idx = self._edge_map.pop(key, None)
        if idx is not None:
            self.gates[idx].zero_()
            self._edge_free.append(idx)

    def gate_idx(self, from_id: str, to_id: str) -> int | None:
        return self._edge_map.get((from_id, to_id))

    # ── Batch accessors ────────────────────────────────────────────────

    def get_cluster_weights(self, store_indices: list[int]) -> torch.Tensor:
        """(K, dim) weight matrix for a cluster's nodes. No torch.stack needed."""
        return self.weights[store_indices]

    def get_cluster_biases(self, store_indices: list[int]) -> torch.Tensor:
        """(K,) bias vector for a cluster's nodes."""
        return self.biases[store_indices]

    def scatter_identities(
        self, groups: dict[str, list[int]], cluster_order: list[str]
    ) -> torch.Tensor:
        """
        Compute identity (normalized mean of node weights) for all clusters
        in one GPU pass using scatter_add.

        groups: {cluster_id: [store_indices]}
        cluster_order: ordered list of cluster_ids for row ordering
        Returns: (N_clusters, dim) tensor on device
        """
        cid_to_row = {cid: i for i, cid in enumerate(cluster_order)}
        all_indices = []
        assignments = []
        for cid in cluster_order:
            idxs = groups.get(cid, [])
            all_indices.extend(idxs)
            assignments.extend([cid_to_row[cid]] * len(idxs))

        if not all_indices:
            return torch.zeros(len(cluster_order), self.dim, device=self.device)

        idx_t = torch.tensor(all_indices, dtype=torch.long, device=self.device)
        assign_t = torch.tensor(assignments, dtype=torch.long, device=self.device)

        W = self.weights[idx_t]  # (total_nodes, dim)
        n_clusters = len(cluster_order)

        # scatter_add + divide by count = mean
        sums = torch.zeros(n_clusters, self.dim, device=self.device)
        sums.scatter_add_(0, assign_t.unsqueeze(1).expand_as(W), W)
        counts = torch.zeros(n_clusters, device=self.device)
        counts.scatter_add_(0, assign_t, torch.ones(len(all_indices), device=self.device))
        counts = counts.clamp(min=1)
        means = sums / counts.unsqueeze(1)

        return F.normalize(means, dim=1)

    def cluster_variances(
        self, groups: dict[str, list[int]]
    ) -> dict[str, float]:
        """
        Per-cluster weight variance (BUD signal).
        High variance = nodes diverged = split candidate.
        One GPU scatter pass.
        """
        cluster_ids = list(groups.keys())
        if not cluster_ids:
            return {}

        cid_to_row = {cid: i for i, cid in enumerate(cluster_ids)}
        all_indices = []
        assignments = []
        for cid in cluster_ids:
            idxs = groups[cid]
            all_indices.extend(idxs)
            assignments.extend([cid_to_row[cid]] * len(idxs))

        if not all_indices:
            return {}

        idx_t = torch.tensor(all_indices, dtype=torch.long, device=self.device)
        assign_t = torch.tensor(assignments, dtype=torch.long, device=self.device)
        n = len(cluster_ids)

        W = self.weights[idx_t]  # (total, dim)

        # Compute centroids via scatter
        sums = torch.zeros(n, self.dim, device=self.device)
        sums.scatter_add_(0, assign_t.unsqueeze(1).expand_as(W), W)
        counts = torch.zeros(n, device=self.device)
        counts.scatter_add_(0, assign_t, torch.ones(len(all_indices), device=self.device))
        counts = counts.clamp(min=1)
        centroids = sums / counts.unsqueeze(1)

        # Variance: mean squared distance from centroid
        expanded = centroids[assign_t]  # (total, dim)
        sq_dist = ((W - expanded) ** 2).sum(dim=1)  # (total,)
        var_sums = torch.zeros(n, device=self.device)
        var_sums.scatter_add_(0, assign_t, sq_dist)
        variances = var_sums / counts

        var_list = variances.cpu().tolist()
        return {cid: var_list[i] for i, cid in enumerate(cluster_ids)}

    # ── Serialization ──────────────────────────────────────────────────

    def state_dict(self) -> dict:
        """Return serializable state. Tensors moved to CPU."""
        # Only save allocated rows to reduce file size
        node_ids = list(self._node_map.keys())
        node_indices = [self._node_map[nid] for nid in node_ids]
        edge_keys = list(self._edge_map.keys())
        edge_indices = [self._edge_map[k] for k in edge_keys]

        state = {
            "dim": self.dim,
            "node_ids": node_ids,
            "node_weights": self.weights[node_indices].cpu() if node_indices else torch.zeros(0, self.dim),
            "node_biases": self.biases[node_indices].cpu() if node_indices else torch.zeros(0),
            "edge_keys": edge_keys,
            "edge_gates": self.gates[edge_indices].cpu() if edge_indices else torch.zeros(0, self.dim),
        }
        return state

    def load_state_dict(self, state: dict) -> None:
        """Restore from saved state."""
        node_ids = state["node_ids"]
        node_weights = state["node_weights"]
        node_biases = state["node_biases"]
        for i, nid in enumerate(node_ids):
            self.alloc_node(nid, node_weights[i], node_biases[i:i+1])

        edge_keys = state["edge_keys"]
        edge_gates = state["edge_gates"]
        for i, key in enumerate(edge_keys):
            self.alloc_gate(key[0], key[1], edge_gates[i])

    def load_legacy(self, old_state: dict, nodes: list[tuple[str, str]], edges: list) -> None:
        """Import from old per-node state_dict format."""
        for node_id, _cluster_id in nodes:
            w_key = f"{node_id}.weights"
            b_key = f"{node_id}.bias"
            weights = old_state.get(w_key, torch.zeros(self.dim))
            bias = old_state.get(b_key, torch.zeros(1))
            self.alloc_node(node_id, weights, bias)

        for edge in edges:
            gate = F.normalize(torch.randn(self.dim), dim=0)
            self.alloc_gate(edge.from_id, edge.to_id, gate)

    # ── Internal ───────────────────────────────────────────────────────

    def _grow_nodes(self) -> None:
        """Double node capacity."""
        old_max = self._max_nodes
        new_max = old_max * 2
        new_weights = torch.zeros(new_max, self.dim, device=self.device)
        new_biases = torch.zeros(new_max, device=self.device)
        new_weights[:old_max] = self.weights
        new_biases[:old_max] = self.biases
        self.weights = new_weights
        self.biases = new_biases
        self._node_free.extend(range(new_max - 1, old_max - 1, -1))
        self._max_nodes = new_max

    def _grow_edges(self) -> None:
        """Double edge capacity."""
        old_max = self._max_edges
        new_max = old_max * 2
        new_gates = torch.zeros(new_max, self.dim, device=self.device)
        new_gates[:old_max] = self.gates
        self.gates = new_gates
        self._edge_free.extend(range(new_max - 1, old_max - 1, -1))
        self._max_edges = new_max
