"""Pull pre-encoded sentences from many sources in weighted rotation.

A sequential curriculum exhausts one source before moving to the next.
An interleaved curriculum draws from all active sources simultaneously,
weighted so smaller domains aren't drowned by larger ones. The mind
sees a mix of language structure, narrative consequence, philosophy,
history, etc. across every window of N sentences instead of waiting
through one domain at a time.

Each source is read from data/encoded_corpus.db. Encoding must happen
before the curriculum starts (run encode_corpus.py first).
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, Iterator

import numpy as np

from backend.preencoder import (
    DB_PATH as ENCODED_DB_PATH,
    fetch_encoded,
    is_source_encoded,
)


@dataclass
class SourceSpec:
    """One curriculum source. `weight` is multiplied into the source's
    pull probability; the JSON's `weight` field is exposed verbatim so
    operators can hand-tune what dominates."""
    name: str
    source_file: str
    domain: str
    weight: float = 1.0


@dataclass
class InterleavedSentence:
    sentence: str
    representation: np.ndarray
    domain: str
    source_name: str
    position_in_source: int


class InterleavedReader:
    """Yields (sentence, repr, domain, source_name) one at a time, drawing
    from many pre-encoded sources in weighted random rotation. Sources
    that exhaust drop out of the pool. Iteration ends when all sources
    are exhausted.

    Probability of pulling from source `s`:
        P(s) = (s.weight × size_factor(s)) / sum_over_active

    where size_factor = min(1.0, 5000 / len(s.sentences)). This keeps a
    100-sentence wiki page from being drowned by a 7000-sentence Plutarch
    while still letting natural-size differences influence the mix.
    """

    def __init__(
        self,
        sources: list[SourceSpec],
        encoded_db_path: str = ENCODED_DB_PATH,
        seed: int | None = None,
    ) -> None:
        self.encoded_db_path = encoded_db_path
        self._rng = random.Random(seed)

        # Materialize every source as a list — random access by index.
        # Memory cost: 256 floats × 4 bytes × ~20K sentences ≈ 20 MB.
        # Within the M1 budget; trades RAM for predictable random access.
        self._sources: list[dict] = []
        for spec in sources:
            if not is_source_encoded(spec.source_file, encoded_db_path):
                print(f"[interleaved] [skip] source not pre-encoded: {spec.source_file}")
                continue
            rows = list(fetch_encoded(spec.source_file, encoded_db_path))
            if not rows:
                print(f"[interleaved] [skip] empty source: {spec.source_file}")
                continue
            size_factor = min(1.0, 5000.0 / len(rows))
            self._sources.append({
                "spec": spec,
                "rows": rows,                  # list of (sentence, repr, position)
                "cursor": 0,
                "size_factor": size_factor,
                "active": True,
                "n_pulled": 0,
            })

    # ---- introspection ----

    @property
    def total_sentences(self) -> int:
        return sum(len(s["rows"]) for s in self._sources)

    @property
    def active_sources(self) -> int:
        return sum(1 for s in self._sources if s["active"])

    def per_source_pulls(self) -> dict[str, int]:
        return {s["spec"].name: s["n_pulled"] for s in self._sources}

    def per_domain_pulls(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for s in self._sources:
            d = s["spec"].domain
            out[d] = out.get(d, 0) + s["n_pulled"]
        return out

    # ---- iteration ----

    def __iter__(self) -> Iterator[InterleavedSentence]:
        return self

    def __next__(self) -> InterleavedSentence:
        active = [s for s in self._sources if s["active"]]
        if not active:
            raise StopIteration

        # Weighted choice across active sources.
        weights = [s["spec"].weight * s["size_factor"] for s in active]
        chosen = self._rng.choices(active, weights=weights, k=1)[0]

        idx = chosen["cursor"]
        sentence, rep, position = chosen["rows"][idx]
        chosen["cursor"] = idx + 1
        chosen["n_pulled"] += 1
        if chosen["cursor"] >= len(chosen["rows"]):
            chosen["active"] = False

        return InterleavedSentence(
            sentence=sentence,
            representation=rep,
            domain=chosen["spec"].domain,
            source_name=chosen["spec"].name,
            position_in_source=int(position),
        )
