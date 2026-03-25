from collections import deque
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


@dataclass
class Node:
    id: str                                        # "n_042"
    cluster_id: str                                # which cluster owns this node
    weights: torch.Tensor                          # shape (512,) — input dim
    bias: torch.Tensor                             # shape (1,)
    plasticity: float = 1.0                        # 0.0 to 1.0
    age: int = 0                                   # steps since creation
    activation_history: deque = field(default_factory=lambda: deque(maxlen=64))
    alive: bool = True                             # False = dormant

    _last_input: torch.Tensor = field(default=None, repr=False)
    _momentum: torch.Tensor = field(default=None, repr=False)

    def activate(self, x: torch.Tensor) -> float:
        """
        x: input vector, shape (512,)
        Returns: scalar activation value in [-1, 1]
        """
        self._last_input = x.detach()
        raw = torch.dot(self.weights, x) + self.bias
        activation = torch.tanh(raw).item()
        self.activation_history.append(activation)
        self.age += 1
        return activation

    def ff_update(
        self,
        activation: float,
        is_positive: bool,
        learning_rate: float,
        signal_strength: float = 1.0,
    ) -> None:
        """
        Forward-Forward local update rule with continuous signal.
        signal_strength (0-1) scales the update magnitude — stronger
        similarity gets a stronger push, weak similarity gets a weak push.
        """
        if self._last_input is None:
            return
        sign = 1.0 if is_positive else -1.0
        magnitude = self.plasticity * learning_rate * abs(activation) * signal_strength
        update = sign * magnitude * self._last_input * (1 - activation ** 2)
        # Momentum: smooth weight updates for more stable convergence
        if self._momentum is None:
            self._momentum = torch.zeros_like(self.weights)
        self._momentum = 0.9 * self._momentum + 0.1 * update
        self.weights = self.weights + self._momentum
        self.weights = F.normalize(self.weights, dim=0)

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
        return sum((x - mean) ** 2 for x in hist) / len(hist)

    @property
    def is_responsive(self) -> bool:
        """True if this node is doing meaningful work."""
        return self.mean_activation > 0.1 and self.activation_variance > 0.01
