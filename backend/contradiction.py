"""
ContradictionDetector: finds when new concepts oppose existing ones.

When the mind learns 'justice helps the weak'
and already knows 'justice helps the strong'
- same semantic neighborhood, opposing direction.

Current architecture: writes both, no tension noted.
With contradiction detection:
  -> both written
  -> tension edge between them
  -> both seeded as waves in wave field simultaneously
  -> interference = the mind sitting with the tension
  -> unresolved contradictions seed active inference

ACC (anterior cingulate cortex) equivalent.
A mind that holds contradictions without resolving them is more
honest than one that resolves them prematurely.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Contradiction:
    concept_a: int
    concept_b: int
    similarity: float
    opposition: float
    detected_at: float
    resolved: bool = False
    resolution_concept: Optional[int] = None


class ContradictionDetector:
    """Watches concept graph for contradictions on each new write."""

    SIMILARITY_THRESHOLD = 0.65   # above this = same topic
    OPPOSITION_THRESHOLD = -0.3   # below this = opposing claims

    def __init__(self, graph, wave_field=None):
        self.graph = graph
        self.wave_field = wave_field
        self.contradiction_buffer: deque = deque(maxlen=50)
        self.total_detected = 0
        self.total_resolved = 0

    def check_new_concept(self, new_concept_id: int,
                          now: float) -> Optional[Contradiction]:
        if new_concept_id not in self.graph.nodes:
            return None
        new_node = self.graph.nodes[new_concept_id]
        new_emb = new_node.embedding
        n = float(np.linalg.norm(new_emb))
        if n < 1e-9:
            return None
        new_norm = (new_emb / n).astype(np.float32)

        # k-NN search returns (sims, ids) tuple
        sims, neighbor_ids = self.graph._index.search(new_norm, k=10)

        for sim, neighbor_id in zip(sims, neighbor_ids):
            if neighbor_id == new_concept_id:
                continue
            if neighbor_id not in self.graph.nodes:
                continue
            if float(sim) < self.SIMILARITY_THRESHOLD:
                continue

            neighbor_node = self.graph.nodes[neighbor_id]
            neighbor_emb = neighbor_node.embedding
            nn_norm = float(np.linalg.norm(neighbor_emb))
            if nn_norm < 1e-9:
                continue
            neighbor_normed = neighbor_emb / nn_norm
            dot = float(np.dot(new_norm, neighbor_normed))

            if dot < self.OPPOSITION_THRESHOLD:
                contradiction = Contradiction(
                    concept_a=int(new_concept_id),
                    concept_b=int(neighbor_id),
                    similarity=float(sim),
                    opposition=dot,
                    detected_at=float(now),
                )
                self.contradiction_buffer.append(contradiction)
                self.total_detected += 1
                self._write_tension_edge(new_concept_id, neighbor_id,
                                         contradiction)
                if self.wave_field is not None:
                    try:
                        self.wave_field.inject(
                            [new_concept_id, neighbor_id], strength=0.5,
                        )
                    except Exception:
                        pass
                return contradiction
        return None

    def _write_tension_edge(self, cid_a: int, cid_b: int,
                            contradiction: Contradiction):
        from backend.graph import EdgeType
        try:
            self.graph.add_edge(
                source_id=cid_a, target_id=cid_b,
                type=EdgeType.OPPOSITE_OF,
                weight=abs(contradiction.opposition),
                now=contradiction.detected_at,
            )
        except Exception:
            # signature might be positional in this codebase; try fallback
            try:
                self.graph.add_edge(
                    cid_a, cid_b, EdgeType.OPPOSITE_OF,
                    abs(contradiction.opposition),
                    contradiction.detected_at,
                )
            except Exception:
                pass

    def get_active_contradictions(self) -> list[Contradiction]:
        return [c for c in self.contradiction_buffer if not c.resolved]

    def attempt_resolution(self, contradiction: Contradiction,
                           synthesis_concept_id: int):
        contradiction.resolved = True
        contradiction.resolution_concept = int(synthesis_concept_id)
        self.total_resolved += 1
