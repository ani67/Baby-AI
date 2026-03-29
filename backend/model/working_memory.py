"""
WorkingMemory — explicit scratchpad for multi-step reasoning.

Separate from the activation buffer (which is passive temporal context).
Working memory is an addressable K-slot store: write round-robin, read
by attention-weighted similarity to a query vector.

    +-------+-------+-------+-------+-------+-------+-------+-------+
    | slot0 | slot1 | slot2 | slot3 | slot4 | slot5 | slot6 | slot7 |
    +-------+-------+-------+-------+-------+-------+-------+-------+
         ^                                       ^
         |                                       |
       read (attention)                    write_head (round-robin)

All stored vectors are L2-normalized.
"""

import torch
import torch.nn.functional as F


class WorkingMemory:
    """
    A scratchpad of K slots, each holding a dim-dimensional vector.
    Write: round-robin into next slot.
    Read: attention-weighted readout (dot-product similarity to query).
    """

    def __init__(self, slots: int = 8, dim: int = 512, device: str = "cpu"):
        self.slots = slots
        self.dim = dim
        self.device = torch.device(device)
        self._buffer = torch.zeros(slots, dim, device=self.device)
        self._write_head = 0
        self._written = 0  # total writes (capped at slots for occupancy)

    def write(self, vector: torch.Tensor) -> None:
        """Store vector in next slot (round-robin). L2-normalized before storage."""
        vector = F.normalize(vector.detach().to(self.device), dim=0)
        self._buffer[self._write_head] = vector
        self._write_head = (self._write_head + 1) % self.slots
        self._written = min(self._written + 1, self.slots)

    def read(self, query: torch.Tensor) -> torch.Tensor:
        """Attention-weighted readout: softmax(slots @ query * 5.0) @ slots -> (dim,)."""
        if self._written == 0:
            return torch.zeros(self.dim, device=self.device)
        query = query.to(self.device)
        active = self._buffer[: self._active_count()]
        scores = active @ query * 5.0  # temperature-scaled dot product
        attn = F.softmax(scores, dim=0)  # (K,)
        result = attn @ active  # (dim,)
        return F.normalize(result, dim=0)

    def read_top_k(self, query: torch.Tensor, k: int = 3) -> torch.Tensor:
        """Read only from top-k most similar slots (sparser, more focused)."""
        if self._written == 0:
            return torch.zeros(self.dim, device=self.device)
        query = query.to(self.device)
        active = self._buffer[: self._active_count()]
        scores = active @ query  # (K,)
        actual_k = min(k, len(scores))
        top_vals, top_idx = scores.topk(actual_k)
        attn = F.softmax(top_vals * 5.0, dim=0)
        result = attn @ active[top_idx]
        return F.normalize(result, dim=0)

    def clear(self) -> None:
        """Zero all slots, reset write head."""
        self._buffer.zero_()
        self._write_head = 0
        self._written = 0

    def state_dict(self) -> dict:
        """For checkpointing."""
        return {
            "buffer": self._buffer.cpu().clone(),
            "write_head": self._write_head,
            "written": self._written,
        }

    def load_state_dict(self, d: dict) -> None:
        """Restore from checkpoint."""
        self._buffer = d["buffer"].to(self.device)
        self._write_head = d["write_head"]
        self._written = d["written"]
        # Ensure slot/dim match
        if self._buffer.shape != (self.slots, self.dim):
            print(
                f"[working_memory] shape mismatch: checkpoint {self._buffer.shape} "
                f"vs expected ({self.slots}, {self.dim}), resetting",
                flush=True,
            )
            self.clear()

    @property
    def occupancy(self) -> int:
        """How many slots have been written to (non-zero)."""
        return self._written

    def _active_count(self) -> int:
        """Number of valid slots to read from."""
        return self._written

    def to(self, device: torch.device) -> "WorkingMemory":
        """Move to a new device (for MPS migration)."""
        self.device = device
        self._buffer = self._buffer.to(device)
        return self
