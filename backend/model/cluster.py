from collections import deque
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from .node import Node


@dataclass
class Cluster:
    id: str                                         # "c_04"
    nodes: list[Node] = field(default_factory=list)
    layer_index: float = 0                          # which depth level (0 = earliest)
    internal_edges: dict = field(default_factory=dict)  # node_id → list[node_id]
    interface_nodes: list[str] = field(default_factory=list)
    plasticity: float = 1.0
    age: int = 0
    dormant: bool = False
    lens: torch.Tensor = field(default_factory=lambda: torch.zeros(512))
    _identity_cache: torch.Tensor | None = field(default=None, repr=False)
    _store: object | None = field(default=None, repr=False)  # WeightStore reference

    _output_history: deque = field(default_factory=lambda: deque(maxlen=64), repr=False)

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

    def _compute_internal_density(self) -> float:
        n = len(self.nodes)
        if n < 2:
            return 0.0
        possible = n * (n - 1) / 2
        actual = sum(len(targets) for targets in self.internal_edges.values()) / 2
        return actual / possible

    @property
    def identity(self) -> torch.Tensor:
        """Normalized mean of all node weight vectors — the cluster's semantic fingerprint.
        Cached until invalidated by ff_update."""
        if self._identity_cache is not None:
            return self._identity_cache
        living = [n for n in self.nodes if n.alive]
        if not living:
            return torch.zeros(512)
        if self._store is not None and living[0]._store_idx >= 0:
            indices = [n._store_idx for n in living]
            weights = self._store.weights[indices]
        else:
            weights = torch.stack([n.weights for n in living])
        self._identity_cache = F.normalize(weights.mean(dim=0), dim=0)
        return self._identity_cache

    @property
    def mean_activation(self) -> float:
        if not self.nodes:
            return 0.0
        acts = [n.mean_activation for n in self.nodes if n.alive]
        return sum(acts) / len(acts) if acts else 0.0

    @property
    def activation_bimodality(self) -> float:
        """
        Simplified bimodality detection.
        Measures the gap between the two halves of sorted activations.
        Higher value = more bimodal.
        """
        acts = []
        for node in self.nodes:
            if node.alive and node.activation_history:
                acts.extend(list(node.activation_history))
        if len(acts) < 10:
            return 0.0
        acts.sort()
        mid = len(acts) // 2
        lower_mean = sum(acts[:mid]) / mid
        upper_mean = sum(acts[mid:]) / (len(acts) - mid)
        overall_mean = sum(acts) / len(acts)
        overall_var = sum((x - overall_mean) ** 2 for x in acts) / len(acts)
        if overall_var < 1e-8:
            return 0.0
        gap = (upper_mean - lower_mean) ** 2 / (4 * overall_var)
        return min(gap, 1.0)

    @property
    def output_coherence(self) -> float:
        """Cosine similarity between cluster outputs over recent steps."""
        if len(self._output_history) < 2:
            return 1.0
        outputs = list(self._output_history)
        sims = []
        for i in range(len(outputs) - 1):
            sim = F.cosine_similarity(outputs[i].unsqueeze(0),
                                      outputs[i + 1].unsqueeze(0)).item()
            sims.append(sim)
        return sum(sims) / len(sims) if sims else 1.0

    @property
    def residual_structure(self) -> float:
        """Placeholder — computed externally via GrowthMonitor."""
        return 0.0

    def forward(
        self,
        x: torch.Tensor,
        incoming_edge_signals: dict,
    ) -> torch.Tensor:
        """
        Activate all living nodes via batched matmul, return weighted-sum output (512,).
        One matmul for all nodes instead of per-node dot products.
        """
        if self.lens.norm() > 1e-6:
            combined = F.normalize(x + self.lens, dim=0)
        else:
            combined = x.clone()
        for value in incoming_edge_signals.values():
            if isinstance(value, tuple):
                signal, strength = value
                combined = combined + strength * signal
            else:
                combined = combined + 0.3 * value

        living = [n for n in self.nodes if n.alive]
        if not living:
            return torch.zeros(512)

        # Batched activation: index into GPU store or stack from CPU
        if self._store is not None and living[0]._store_idx >= 0:
            indices = [n._store_idx for n in living]
            weight_matrix = self._store.weights[indices]  # (K, 512) — GPU index
            biases = self._store.biases[indices]           # (K,) — GPU index
            combined = combined.to(weight_matrix.device)
        else:
            weight_matrix = torch.stack([n.weights for n in living])  # (K, 512)
            biases = torch.stack([n.bias for n in living]).squeeze(-1)  # (K,)
        raw = weight_matrix @ combined + biases  # (K,) — one matmul
        acts = torch.tanh(raw)  # (K,)

        # Write back side effects: _last_input, activation_history, age
        acts_list = acts.tolist()
        combined_cpu = combined.detach().cpu()
        for i, node in enumerate(living):
            node._last_input = combined_cpu
            node.activation_history.append(acts_list[i])
            node.age += 1

        # Batched output computation: weighted sum of node weights by activation
        abs_acts = acts.abs()  # (K,)
        total_weight = abs_acts.sum()
        if total_weight > 0:
            coeffs = (acts / total_weight).unsqueeze(1)  # (K, 1)
            output = (coeffs * weight_matrix).sum(dim=0)  # (512,)
        else:
            output = torch.zeros(512)

        output = F.normalize(output, dim=0)
        self._output_history.append(output.detach().clone())
        self.age += 1
        return output

    def ff_update(
        self,
        x: torch.Tensor,
        is_positive: bool,
        learning_rate: float,
    ) -> None:
        """Calls ff_update on each living node."""
        for node in self.nodes:
            if node.alive:
                activation = node.activation_history[-1] if node.activation_history else 0.0
                node.ff_update(activation, is_positive, learning_rate)
        self._identity_cache = None  # weights changed, invalidate

    def local_target_update(
        self,
        teacher_vec: torch.Tensor,
        learning_rate: float,
    ) -> None:
        """
        Rich local learning: project teacher target into each node's space.
        Each node gets its own 512-d direction to learn toward.
        """
        for node in self.nodes:
            if node.alive:
                activation = node.activation_history[-1] if node.activation_history else 0.0
                if abs(activation) < 0.001:
                    continue
                # Each node's local target: blend of teacher direction
                # and its own specialization (so nodes don't all converge)
                node_sim = torch.dot(node.weights, teacher_vec).item()
                # Scale target by how relevant this node is to the teacher
                local_target = F.normalize(
                    teacher_vec + 0.5 * node.weights, dim=0
                )
                node.local_target_update(activation, local_target, learning_rate)
        self._identity_cache = None

    def update_lens(self, error: torch.Tensor, activation: float, lr: float):
        """Nudge lens toward reducing this cluster's error."""
        if abs(activation) < 0.01:
            return
        self.lens = self.lens + lr * activation * error

