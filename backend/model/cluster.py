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
        """
        combined = x.clone()
        for value in incoming_edge_signals.values():
            if isinstance(value, tuple):
                signal, strength = value
                combined = combined + strength * signal
            else:
                combined = combined + 0.3 * value

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
