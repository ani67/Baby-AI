"""B2 fix verification — internal_repr now evolves with the merged
active set each iteration. Multi-sentence emission should land where
the previous fixed-internal_repr version stopped at iter 2."""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.mind_paths import MindPaths            # noqa: E402
from backend.persistence import MindPersistence     # noqa: E402


PROMPTS = [
    "what is beautiful",
    "tell me everything you know about philosophy",
]


def main() -> int:
    paths = MindPaths("first")
    print(f"[retest] loading mind 'first' from {paths.db}")
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
    cd_loaded = loop.expression._conditioned_decoder is not None
    print(f"[retest] CD loaded: {cd_loaded}")
    print(f"[retest] graph: {loop.graph.node_count} nodes, {loop.graph.edge_count} edges\n")

    h = loop.input_pipeline
    examiner = h.register_agent("retest", now=time.time())

    for prompt in PROMPTS:
        now = time.time()
        ing = h.ingest_text(prompt, agent_id=examiner, now=now)
        cyc = loop.cycle(ing, now=now + 1e-3, force_respond=True)
        budget = loop.compute_expression_budget(
            cyc.processed_active_set or cyc.input_active_set or {},
            now=now + 1e-3,
        )
        decision = type(cyc.expression_decision).__name__ if cyc.expression_decision else None
        proc_size = len(cyc.processed_active_set or {})
        arousal = float(loop.affect.current_arousal(time.time()))
        surface = cyc.emitted_surface or ""
        sentences = [s for s in surface.split(". ") if s.strip()]

        print("=" * 78)
        print(f"Q: {prompt!r}")
        print(f"   proc_size={proc_size}  arousal={arousal:.3f}  budget={budget}  decision={decision}")
        print(f"   sentence count: {len(sentences)}")
        print(f"   total chars:    {len(surface)}")
        print()
        for i, s in enumerate(sentences, start=1):
            print(f"   [{i}] {s!r}")
        if not sentences:
            print(f"   (no surface)  raw={surface!r}")
        print()

        # Decay before next probe.
        time.sleep(2.0)

    return 0


if __name__ == "__main__":
    sys.exit(main())
