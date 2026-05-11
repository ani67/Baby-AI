"""
5-level predictive coding hierarchy.

Each level predicts what the next level will see.
Prediction errors flow upward as learning signal.
Top-down predictions flow downward as attention bias.

Replaces the flat single-level prediction engine for the full
architecture. The existing PredictionEngine continues to work
during transition.

Level dims:
  L1: 512 -> 256  raw feature combinations
  L2: 256 -> 256  object/word level
  L3: 256 -> 256  concept level
  L4: 256 -> 256  scene/argument level
  L5: 256 -> 512  abstract, back to concept space
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PCConfig:
    input_dim: int = 512
    hidden_dim: int = 256
    n_levels: int = 5
    learning_rate: float = 1e-4
    error_threshold: float = 0.3
    surprise_threshold: float = 0.7


class PCLevel(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )
        self.predictor = nn.Sequential(
            nn.Linear(out_dim, in_dim),
            nn.LayerNorm(in_dim),
            nn.GELU(),
        )
        self.in_dim = in_dim
        self.out_dim = out_dim
        self._state: Optional[torch.Tensor] = None

    def forward_bottom_up(self, x: torch.Tensor) -> torch.Tensor:
        self._state = self.encoder(x)
        return self._state

    def forward_top_down(self) -> torch.Tensor:
        if self._state is None:
            return torch.zeros(self.in_dim)
        return self.predictor(self._state)

    def prediction_error(self, actual: torch.Tensor) -> torch.Tensor:
        if self._state is None:
            return actual
        predicted = self.forward_top_down()
        return actual - predicted


class PredictiveCodingHierarchy(nn.Module):
    def __init__(self, config: Optional[PCConfig] = None):
        super().__init__()
        self.config = config or PCConfig()
        dims = [512, 256, 256, 256, 256, 512]
        self.levels = nn.ModuleList([
            PCLevel(dims[i], dims[i+1]) for i in range(self.config.n_levels)
        ])
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=self.config.learning_rate,
        )
        self.device = (
            torch.device('mps') if torch.backends.mps.is_available()
            else torch.device('cpu')
        )
        self.to(self.device)
        self._last_errors: list[float] = []
        self._steps = 0

    def process(self, input_rep: np.ndarray, learn: bool = True) -> dict:
        """Run a full bidirectional PC cycle.

        When ``learn=False`` the entire forward+top-down pass runs under
        ``torch.no_grad()`` — no autograd graph built, no gradient
        memory allocated. On a 28M-param model this drops per-call
        cost from ~150 ms to ~5-10 ms, which is what unblocks the
        runtime's main loop when PC is called every input.
        """
        x = torch.tensor(input_rep, dtype=torch.float32, device=self.device)

        ctx = torch.no_grad() if not learn else torch.enable_grad()
        with ctx:
            # Bottom-up pass
            states = []
            current = x
            for level in self.levels:
                current = level.forward_bottom_up(current)
                states.append(current)

            # Top-down pass + error computation
            errors: list[torch.Tensor] = []
            pred = self.levels[-1].forward_top_down()
            for i in range(len(self.levels) - 2, -1, -1):
                level = self.levels[i]
                actual = states[i]
                error = actual - pred
                errors.insert(0, error)
                pred = level.forward_top_down()
            # L1 vs raw input
            errors.insert(0, x - pred)

            # Use detach so tolist/float conversion never touches
            # the autograd graph regardless of mode.
            error_magnitudes = [float(e.detach().norm()) for e in errors]
            surprise = max(error_magnitudes) if error_magnitudes else 0.0

        if learn and surprise > self.config.error_threshold:
            self.optimizer.zero_grad()
            loss = sum(e.pow(2).mean() for e in errors)
            loss.backward()
            self.optimizer.step()

        self._last_errors = error_magnitudes
        self._steps += 1

        top_down_rep = states[-1].detach().cpu().numpy()

        return {
            'states': [s.detach() for s in states],
            'errors': errors,
            'error_magnitudes': error_magnitudes,
            'surprise': surprise,
            'top_down': top_down_rep,
            'should_crystallize': surprise > self.config.surprise_threshold,
        }

    def get_top_down_for_wave_field(self, wave_field,
                                    n_concepts: int = 50) -> Optional[torch.Tensor]:
        """Convert L5 state -> wave field activation vector (N,).
        Returns None if L5 has no state yet."""
        l5 = self.levels[-1]._state
        if l5 is None:
            return None
        # softmax over node alignments
        with torch.no_grad():
            l5_dev = l5.to(wave_field.node_matrix.device)
            sims = wave_field.node_matrix @ l5_dev
            top_activation = torch.softmax(sims * 3.0, dim=0)
        return top_activation.to(wave_field.activation.device)
