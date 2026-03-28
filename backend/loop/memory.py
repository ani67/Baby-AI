"""
Episodic memory: store significant learning experiences and replay them.

Stores high-error and growth-event experiences as (input_vec, expected_vec)
pairs in SQLite. Replays them mixed into regular batches, prioritizing
categories where current performance is worst.
"""

import struct
import re
from collections import deque

import torch
import torch.nn.functional as F


_CATEGORY_RE = re.compile(
    r'\b(dog|cat|bird|car|bus|train|horse|sheep|cow|elephant|bear|zebra|giraffe|'
    r'person|man|woman|bicycle|motorcycle|airplane|boat|truck|skateboard|surfboard|'
    r'snowboard|tennis|baseball|pizza|cake|sandwich|broccoli|banana|orange|'
    r'chair|couch|bed|table|toilet|laptop|phone|clock|vase|cup|fork|knife|bowl|'
    r'bottle|book)\b', re.IGNORECASE,
)


def _tensor_to_blob(t: torch.Tensor) -> bytes:
    vals = t.detach().cpu().float().tolist()
    return struct.pack(f"{len(vals)}f", *vals)


def _blob_to_tensor(blob: bytes) -> torch.Tensor:
    n = len(blob) // 4
    vals = struct.unpack(f"{n}f", blob)
    return F.normalize(torch.tensor(vals, dtype=torch.float32), dim=0)


class EpisodicMemory:
    def __init__(self, store, capacity: int = 2000):
        self._store = store
        self._capacity = capacity
        self._error_history: deque = deque(maxlen=200)
        self._last_store_step: int = -10

    def maybe_store(self, item, error_magnitude: float, step: int,
                    growth_events: list) -> bool:
        """Store if top-25% error or growth event. Rate-limited to 1 per 10 steps."""
        self._error_history.append(error_magnitude)
        if step - self._last_store_step < 10:
            return False

        # Trigger: growth event OR top-25% error
        trigger = None
        if growth_events:
            trigger = "growth_event"
        elif len(self._error_history) >= 20:
            threshold = sorted(self._error_history)[len(self._error_history) * 3 // 4]
            if error_magnitude >= threshold:
                trigger = "high_error"

        if trigger is None:
            return False
        if item.expected_vector is None or item.input_vector is None:
            return False

        # Extract category
        category = None
        if item.description:
            match = _CATEGORY_RE.search(item.description)
            if match:
                category = match.group(1).lower()

        self._store.store_episodic_memory(
            step=step,
            category=category,
            input_vec=_tensor_to_blob(item.expected_vector),
            expected_vec=_tensor_to_blob(item.expected_vector),
            error_magnitude=error_magnitude,
            trigger=trigger,
        )
        self._last_store_step = step
        return True

    def sample_replay(self, n: int = 8,
                      category_weights: dict[str, float] | None = None) -> list[tuple]:
        """Return replay samples formatted for update_batch(): (vec, True, teacher_vec, None)."""
        if not category_weights:
            return []

        # Pick worst categories (highest weight = worst performance)
        sorted_cats = sorted(category_weights.items(), key=lambda x: -x[1])
        top_cats = [cat for cat, _ in sorted_cats[:n]]

        rows = self._store.sample_episodic_memories(
            categories=top_cats,
            current_step=0,
            limit=n,
        )
        if not rows:
            return []

        # Increment replay counts
        self._store.increment_replay_count(
            [r["id"] for r in rows], step=0,
        )

        # Convert to update_batch format
        samples = []
        for r in rows:
            vec = _blob_to_tensor(r["input_vec"])
            teacher_vec = _blob_to_tensor(r["expected_vec"])
            samples.append((vec, True, teacher_vec, None))
        return samples

    def evict(self) -> int:
        """Trim to capacity."""
        return self._store.evict_episodic_memories(self._capacity)

    def count(self) -> int:
        return self._store.get_episodic_memory_count()
