"""
Learned positional encoding for sequential processing.

Adds position-dependent signal to input vectors so the brain can distinguish
"dog at position 1" from "dog at position 5". Initialized with sinusoidal
pattern (gives structure to start), then learned via gradient-free training.

Used by:
  - NativeTextEncoder.encode_sequential() — per-word position
  - NativeVisionEncoder.encode_patches_grid() — spatial patch position
"""

import math

import torch
import torch.nn.functional as F


class PositionalEncoding:
    """Learned positional embedding table: (max_positions, dim).

    Sinusoidal initialization gives meaningful starting distances
    (nearby positions are similar, far positions are different).
    Embeddings are small (scaled to ~10% of input magnitude) so
    position tints the vector without overwhelming content.
    """

    def __init__(self, max_positions: int = 32, dim: int = 512):
        self.max_positions = max_positions
        self.dim = dim
        self.embeddings = self._sinusoidal_init(max_positions, dim)

    @staticmethod
    def _sinusoidal_init(max_pos: int, dim: int) -> torch.Tensor:
        """Standard sinusoidal positional encoding, scaled down to ~10%."""
        pe = torch.zeros(max_pos, dim)
        position = torch.arange(0, max_pos, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float) * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe * 0.1  # scale down: position is a hint, not the signal

    def encode(self, position: int, vector: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to a vector. Returns normalized result."""
        pos = min(position, self.max_positions - 1)
        pe = self.embeddings[pos]
        if pe.device != vector.device:
            pe = pe.to(vector.device)
        return F.normalize(vector + pe, dim=0)

    def encode_batch(self, vectors: list[torch.Tensor]) -> list[torch.Tensor]:
        """Add positional encoding to a sequence of vectors."""
        return [self.encode(i, v) for i, v in enumerate(vectors)]

    def state_dict(self) -> dict:
        return {"embeddings": self.embeddings.clone()}

    def load_state_dict(self, d: dict):
        if "embeddings" in d:
            self.embeddings = d["embeddings"]
