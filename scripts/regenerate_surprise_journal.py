"""Regenerate surprised_sentences.jsonl from the existing concept graph.

Background: the parallel-ingestion path doesn't write to this file
(it goes through loop.cycle with skip_simulation=True, which never
reaches the journal-append branch). After a 1.4M-item parallel run
we have 70K+ new concepts in the graph but the journal is frozen at
6,135 entries from before the run.

This script reconstructs the missing entries from what actually
landed in the graph: every concept whose name looks like text gets
a surprise record built from its stored metadata.

Caveats:
- AffectTrace stores `running_state` (an EWMA) but no birth-time
  snapshot, so we use running_state as a proxy for the affect at
  the time the surprise fired. Records reconstructed this way are
  marked with `reconstructed: true`.
- `level` is unknown after the fact — we infer "word" / "phrase" /
  "sentence" / "paragraph" by length so the trainer can still bucket.
- We pad concept_embeddings with the top-5 graph-nearest neighbors
  (semantic context) — the same shape the original journal records
  carry.

Usage:
    MIND_NAME=first python3 scripts/regenerate_surprise_journal.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

os.environ.setdefault("OMP_NUM_THREADS", "1")

from backend.mind_paths import MindPaths           # noqa: E402
from backend.persistence import MindPersistence    # noqa: E402


def infer_level(text: str) -> str:
    n_words = len(text.split())
    if n_words <= 1:
        return "word"
    if n_words <= 4:
        return "phrase"
    if n_words <= 30:
        return "sentence"
    return "paragraph"


def main() -> int:
    mind_name = os.environ.get("MIND_NAME", "first")
    paths = MindPaths(mind_name)

    print(f"loading {mind_name} from {paths.db}")
    loop = MindPersistence.load(paths.db)
    g = loop.graph
    print(f"  graph: {g.node_count:,} nodes / {g.edge_count:,} edges")

    # Rebuild the cosine index (persistence leaves it empty) so we can
    # do top-k neighbor lookups for the concept_embeddings field.
    print("  rebuilding cosine index over loaded nodes …")
    t0 = time.perf_counter()
    g._rebuild_index()
    print(f"    ntotal={g._index.ntotal:,}  ({time.perf_counter() - t0:.1f}s)")

    journal_path = Path(paths.surprised_log)
    journal_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing journal sentences to dedup.
    existing: set[str] = set()
    if journal_path.exists():
        with open(journal_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    s = rec.get("sentence")
                    if isinstance(s, str):
                        existing.add(s)
                except Exception:
                    continue
    print(f"  existing journal entries: {len(existing):,}")

    # Walk concepts.
    new_records = 0
    skipped_short = 0
    skipped_dup = 0
    skipped_nonword = 0
    by_level = {"word": 0, "phrase": 0, "sentence": 0, "paragraph": 0}

    t0 = time.perf_counter()
    items = list(g.nodes.items())
    print(f"  walking {len(items):,} concepts …")

    with open(journal_path, "a", encoding="utf-8") as f:
        for i, (cid, node) in enumerate(items):
            if i and i % 5_000 == 0:
                rate = i / max(time.perf_counter() - t0, 1e-9)
                eta = (len(items) - i) / max(rate, 1e-9)
                print(
                    f"    [{i:>6,}/{len(items):,}]  +{new_records:,} records  "
                    f"({rate:.0f}/s, ETA {eta:.0f}s)"
                )

            name = (node.name or "").strip()
            if len(name) < 10:
                skipped_short += 1
                continue
            # only keep things that look like real text (must contain
            # at least one space — single tokens get filtered).
            if " " not in name:
                skipped_nonword += 1
                continue
            if name in existing:
                skipped_dup += 1
                continue

            # affect: running_state is an EWMA over composite affect at
            # each strengthen event. Closest available proxy to "felt
            # state when this surprise fired."
            affect_arr = node.affect_trace.running_state
            affect = [float(x) for x in affect_arr]

            # concept_embeddings: top-5 nearest neighbors (skip self).
            emb = node.embedding
            n_emb = float(np.linalg.norm(emb))
            concept_embeddings: list[list[float]] = []
            if n_emb >= 1e-9 and g._index.ntotal > 0:
                q = (emb / n_emb).astype(np.float32, copy=False)
                _sims, neighbor_ids = g._index.search_k(q, k=6)
                for nid in neighbor_ids:
                    if int(nid) == int(cid):
                        continue
                    nbr = g.nodes.get(int(nid))
                    if nbr is None:
                        continue
                    concept_embeddings.append(
                        [float(x) for x in nbr.embedding]
                    )
                    if len(concept_embeddings) >= 5:
                        break

            level = infer_level(name)
            by_level[level] = by_level.get(level, 0) + 1

            record = {
                "sentence":            name,
                "affect":              affect,
                "concept_embeddings":  concept_embeddings,
                "surprise_score":      float(node.surprise_at_birth),
                "concept_id":          int(cid),
                "level":               level,
                "t":                   float(node.created_at),
                "reconstructed":       True,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            existing.add(name)
            new_records += 1

    elapsed = time.perf_counter() - t0
    print(f"\n  walked {len(items):,} concepts in {elapsed:.1f}s")
    print(f"  records added:        {new_records:,}")
    print(f"    by inferred level:  word={by_level['word']:,}  "
          f"phrase={by_level['phrase']:,}  "
          f"sentence={by_level['sentence']:,}  "
          f"paragraph={by_level['paragraph']:,}")
    print(f"  skipped (too short):  {skipped_short:,}")
    print(f"  skipped (single word):{skipped_nonword:,}")
    print(f"  skipped (duplicate):  {skipped_dup:,}")
    print(f"\n  total journal size:   {len(existing):,}")
    print(f"  journal path:         {journal_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
