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
    role: str = "detector"  # "detector" | "integrator" | "predictor"
    _identity_cache: torch.Tensor | None = field(default=None, repr=False)

    _output_history: deque = field(default_factory=lambda: deque(maxlen=64), repr=False)
    _error_history: deque = field(default_factory=lambda: deque(maxlen=64), repr=False)  # recent prediction errors (0-1)

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
        Activate all living nodes, return weighted-sum output (512,).
        Supports typed edges: excitatory add signal, inhibitory subtract.
        """
        combined = x.clone()
        for value in incoming_edge_signals.values():
            if isinstance(value, tuple):
                if len(value) == 3:
                    signal, strength, edge_type = value
                else:
                    signal, strength = value
                    edge_type = "excitatory"
                if edge_type == "inhibitory":
                    combined = combined - strength * signal
                else:
                    combined = combined + strength * signal
            else:
                combined = combined + 0.3 * value

        # Normalize combined signal before activation (prevents drift in deep graphs)
        combined = F.normalize(combined, dim=0)

        node_activations = []
        for node in self.nodes:
            if node.alive:
                act = node.activate(combined)
                node_activations.append((node, act))

        if not node_activations:
            return torch.zeros(512)

        output = torch.zeros(512)
        total_weight = sum(abs(act) for _, act in node_activations)
        if total_weight > 0:
            for node, act in node_activations:
                output += (act / total_weight) * node.weights

        # Residual connection: output = input + learned transformation
        # Enables gradient flow through deep graphs (proven in transformers)
        output = F.normalize(x + output, dim=0)
        self._output_history.append(output.detach().clone())
        self.age += 1
        return output

    def ff_update(
        self,
        x: torch.Tensor,
        is_positive: bool,
        learning_rate: float,
        signal_strength: float = 1.0,
    ) -> None:
        """Batched FF update — computes all node updates via tensor ops."""
        living = [n for n in self.nodes if n.alive]
        if not living:
            return

        sign = 1.0 if is_positive else -1.0

        # Batch: stack activations and inputs, compute all updates at once
        acts = torch.tensor([n.activation_history[-1] if n.activation_history else 0.0 for n in living])
        plasticities = torch.tensor([n.plasticity for n in living])
        magnitudes = plasticities * learning_rate * acts.abs() * signal_strength  # (N,)

        # All nodes share the same last_input (set during forward)
        last_input = living[0]._last_input
        if last_input is None:
            return

        # Update = sign * magnitude * input * (1 - act²), per node: (N, 512)
        updates = sign * magnitudes.unsqueeze(1) * last_input.unsqueeze(0) * (1 - acts.pow(2)).unsqueeze(1)

        # Apply momentum and update weights per node
        for i, node in enumerate(living):
            if node._momentum is None:
                node._momentum = torch.zeros_like(node.weights)
            node._momentum = 0.9 * node._momentum + 0.1 * updates[i]
            node.weights = F.normalize(node.weights + node._momentum, dim=0)

        self._identity_cache = None
