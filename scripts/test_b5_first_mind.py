"""B5 supplemental — exercise budget>1 against the trained 'first' mind.

The animal mind in scripts/test_b5_animal_mind.py has only template-fallback
expression. To verify generate_extended actually produces multiple sentences
when the LM is rich enough, this script probes 'first' (has GPT-2 v1
loaded) with prompts whose post-processing active set is reliably large.

Reports per prompt:
  - action, budget, decision
  - response.split('. ') sentence count
  - full response
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.mind_paths import MindPaths           # noqa: E402
from backend.persistence import MindPersistence    # noqa: E402


PROMPTS = [
    "what are you",
    "what is beautiful",
    "do you dream",
    "tell me what surprised you",
]


def main() -> int:
    paths = MindPaths("first")
    print(f"[probe] loading mind 'first' from {paths.db}")
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
    print(f"[probe] CD loaded: {cd_loaded}")
    print(f"[probe] graph: {loop.graph.node_count} nodes, {loop.graph.edge_count} edges\n")

    h = loop.input_pipeline
    examiner = h.register_agent("probe", now=time.time())

    for prompt in PROMPTS:
        now = time.time()
        ing = h.ingest_text(prompt, agent_id=examiner, now=now)
        cyc = loop.cycle(ing, now=now + 1e-3, force_respond=True)
        action_kind = cyc.action.action.value if cyc.action else None
        budget = loop.compute_expression_budget(
            cyc.processed_active_set or cyc.input_active_set or {},
            now=now + 1e-3,
        )
        decision = type(cyc.expression_decision).__name__ if cyc.expression_decision else None
        proc_size = len(cyc.processed_active_set or {})
        arousal = float(loop.affect.current_arousal(time.time()))
        surface = cyc.emitted_surface or ""
        sentences = [s for s in surface.split(". ") if s.strip()]

        print(f"Q: {prompt!r}")
        print(f"   action={action_kind}  proc_size={proc_size}  arousal={arousal:.3f}  "
              f"budget={budget}  decision={decision}")
        print(f"   sentences={len(sentences)}  total_chars={len(surface)}")
        for i, s in enumerate(sentences, start=1):
            print(f"     [{i}] {s!r}")
        if not sentences:
            print(f"     (none — surface={surface!r})")
        print()

        # Sleep a moment so reaction decays toward fresh state for the next probe.
        time.sleep(2.0)

    return 0


if __name__ == "__main__":
    sys.exit(main())
