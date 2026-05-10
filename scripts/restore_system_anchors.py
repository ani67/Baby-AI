"""Restore missing system anchors (self/unknown concept) on a saved mind.

Triggered by the v0.9 crash:
    KeyError: 'source concept 2 not in graph'
where cid 2 (= spine.unknown_concept_id) had been pruned out of the
graph in a previous run. The pin call exists at boot (identity.py
line 256) but was somehow lost from graph._pins on this mind, so a
later score-based prune dropped the concept.

This script:
  1. loads the mind
  2. for each spine system anchor (self_concept_id, unknown_concept_id):
       - if missing from graph.nodes → recreate it via write_on_surprise
         using the same seeded embedding the v0.1 boot path uses, then
         update spine.{self,unknown}_concept_id to the new cid
       - if present but unpinned → pin it
  3. saves
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.config import D_REP                                # noqa: E402
from backend.identity import PinReason                          # noqa: E402
from backend.mind_paths import MindPaths                        # noqa: E402
from backend.persistence import MindPersistence                 # noqa: E402


def _seeded_unit_vector(rng: np.random.Generator, dim: int) -> np.ndarray:
    """Mirrors identity._seeded_unit_vector — same seed → same vector."""
    v = rng.normal(0.0, 1.0, size=dim).astype(np.float32)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


def main() -> int:
    paths = MindPaths(mind_name="first")
    print(f"[restore] loading {paths.db}")
    loop = MindPersistence.load(paths.db)
    g = loop.graph
    spine = loop.identity.spine
    now = time.time()

    print(f"[restore] before:")
    print(f"  self_concept_id    = {spine.self_concept_id}  "
          f"in_graph={spine.self_concept_id in g.nodes}  "
          f"pinned={g.is_pinned(spine.self_concept_id)}")
    print(f"  unknown_concept_id = {spine.unknown_concept_id}  "
          f"in_graph={spine.unknown_concept_id in g.nodes}  "
          f"pinned={g.is_pinned(spine.unknown_concept_id)}")

    # Replay the v0.1 boot embedding sequence — same draws in the same
    # order from the same seed so the embeddings match.
    rng = np.random.default_rng(spine.birth_seed)
    self_emb    = _seeded_unit_vector(rng, D_REP)
    unknown_emb = _seeded_unit_vector(rng, D_REP)
    composite_at_birth = loop.affect.composite(spine.birth_time)

    # ---- self ----
    if spine.self_concept_id not in g.nodes:
        new_self_cid, _ = g.write_on_surprise(
            representation=self_emb,
            surprise=0.0,
            current_affect=composite_at_birth,
            name_hint="self",
            now=spine.birth_time,
        )
        old = spine.self_concept_id
        spine.self_concept_id = new_self_cid
        print(f"[restore] recreated self anchor: cid {old} → {new_self_cid}")
    if not g.is_pinned(spine.self_concept_id):
        g.pin(spine.self_concept_id, reason=PinReason.SELF_REFERENT.value)
        print(f"[restore] pinned self_concept_id={spine.self_concept_id}")

    # ---- unknown ----
    if spine.unknown_concept_id not in g.nodes:
        new_unknown_cid, _ = g.write_on_surprise(
            representation=unknown_emb,
            surprise=0.0,
            current_affect=composite_at_birth,
            name_hint="unknown",
            now=spine.birth_time,
        )
        old = spine.unknown_concept_id
        spine.unknown_concept_id = new_unknown_cid
        # Re-key the OtherModel registry to the new cid.
        if old in spine.others:
            other = spine.others.pop(old)
            other.agent_concept_id = new_unknown_cid
            spine.others[new_unknown_cid] = other
        print(f"[restore] recreated unknown anchor: cid {old} → {new_unknown_cid}")
    if not g.is_pinned(spine.unknown_concept_id):
        g.pin(spine.unknown_concept_id, reason=PinReason.OTHER_AGENT.value)
        print(f"[restore] pinned unknown_concept_id={spine.unknown_concept_id}")

    print(f"[restore] after:")
    print(f"  self_concept_id    = {spine.self_concept_id}  "
          f"in_graph={spine.self_concept_id in g.nodes}  "
          f"pinned={g.is_pinned(spine.self_concept_id)}")
    print(f"  unknown_concept_id = {spine.unknown_concept_id}  "
          f"in_graph={spine.unknown_concept_id in g.nodes}  "
          f"pinned={g.is_pinned(spine.unknown_concept_id)}")

    print("[restore] saving …")
    t1 = time.perf_counter()
    MindPersistence(paths.db).save(loop, now)
    print(f"  saved in {time.perf_counter() - t1:.2f}s — OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
