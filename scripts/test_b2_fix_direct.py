"""B2 fix verification — direct call to generate_extended with budget=3.

Skips MainLoop's budget heuristic (which can pick 1 for calm inputs). The
goal of this test is to verify the FIX (evolved internal_repr + nearest-
match echo seeding) actually allows iter 2+ to land where the previous
build broke.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.expression import ExpressionIntent              # noqa: E402
from backend.mind_paths import MindPaths                     # noqa: E402
from backend.persistence import MindPersistence              # noqa: E402


PROMPTS = [
    "what is beautiful",
    "tell me everything you know about philosophy",
]


def main() -> int:
    paths = MindPaths("first")
    print(f"[direct] loading mind 'first' from {paths.db}")
    loop = MindPersistence.load(paths.db)
    loop.expression._lm_weights_path = paths.language_head
    loop.expression._lm_vocab_path   = paths.vocab
    if hasattr(loop.expression, "_cd_root"):
        loop.expression._cd_root = os.path.dirname(paths.language_head) or "."
    if hasattr(loop.expression, "_cd_weights_path"):
        from backend.language_head import CONDITIONED_DECODER_FILENAME
        loop.expression._cd_weights_path = os.path.join(
            loop.expression._cd_root or ".", CONDITIONED_DECODER_FILENAME,
        )
    loop.expression.reload_language_head()
    loop.expression.set_attention(loop.attention)

    cd_loaded = loop.expression._conditioned_decoder is not None
    print(f"[direct] CD loaded: {cd_loaded}")
    print(f"[direct] graph: {loop.graph.node_count} nodes, {loop.graph.edge_count} edges\n")

    h = loop.input_pipeline
    examiner = h.register_agent("direct", now=time.time())

    for prompt in PROMPTS:
        now = time.time()
        ing = h.ingest_text(prompt, agent_id=examiner, now=now)
        # Run F.attend(INPUT) + processing_loop to build a real active set.
        from backend.attention import AttentionPhase
        seeds: dict[int, float] = {}
        if ing.gap.concept_id is not None:
            seeds[ing.gap.concept_id] = 1.0
        else:
            nearest = loop.graph.nearest(ing.stimulus.representation)
            if nearest is not None:
                cid, sim = nearest
                if sim > 0.0:
                    seeds[int(cid)] = float(sim)
        spread, _ = loop.attention.attend(
            phase=AttentionPhase.INPUT,
            raw_seeds=seeds,
            now=now + 1e-3,
        )
        active = loop.attention.processing_loop(spread.active_set, now=now + 2e-3)
        if not active:
            print(f"[direct] empty active set for {prompt!r} — skipping")
            continue

        c = loop._active_set_centroid(active)
        centroid = c if c is not None else ing.stimulus.representation
        intent = ExpressionIntent(
            intent_id=f"direct/{prompt[:8]}",
            internal_repr=centroid.astype(np.float32),
            active_concepts=dict(active),
            now=now + 3e-3,
            audience_concept_id=None,
        )

        print("=" * 78)
        print(f"Q: {prompt!r}")
        print(f"   active set size: {len(active)}")
        print()
        sentences = loop.expression.generate_extended(
            intent, budget=3,
            parent_stimulus_id=ing.stimulus.stimulus_id,
        )
        print(f"   sentence count: {len(sentences)}")
        for i, s in enumerate(sentences, start=1):
            print(f"   [{i}] {s!r}")
        if not sentences:
            print(f"   (no sentences emitted)")
        print()

        time.sleep(2.0)

    return 0


if __name__ == "__main__":
    sys.exit(main())
