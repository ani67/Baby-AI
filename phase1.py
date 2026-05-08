"""Phase 1 vertical slice — the only test that matters for Phase 1.

Success condition (verbatim from the user):
    Feed it a sentence. The affect vector updates. A concept gets written.
    Feed it the same sentence again. It does not write again.
    Surprise fired once. Confirmation fired nothing.

Run: python3 phase1.py
"""
from __future__ import annotations

import sys

import numpy as np

from backend.affect import AffectStack
from backend.graph import ConceptGraph
from backend.input import InputPipeline
from backend.predict import PredictionEngine


def banner(label: str) -> None:
    print()
    print("=" * 64)
    print(label)
    print("=" * 64)


def fmt_vec(v: np.ndarray, prec: int = 3) -> str:
    return "[" + ", ".join(f"{x:+.{prec}f}" for x in v) + "]"


def main() -> int:
    sentence = "the dog is brown and runs across the field"
    t0 = 0.0
    t1 = 0.001    # 1 ms after the first ingest — reaction has barely decayed
    t2 = 0.002

    # ------------------------------------------------------------------
    # Construct the four components, wired exactly as Phase 1 requires.
    # ------------------------------------------------------------------
    affect = AffectStack(birth_seed=42, t_birth=t0)
    graph = ConceptGraph()
    predict_engine = PredictionEngine(affect=affect, graph=graph)
    pipeline = InputPipeline(
        affect=affect,
        graph=graph,
        predict_engine=predict_engine,
    )

    banner("Birth state")
    snap = affect.snapshot(t0)
    print(f"  composite norm   = {np.linalg.norm(snap['composite']):.4f}")
    print(f"  reaction norm    = {np.linalg.norm(snap['reaction']):.4f}")
    print(f"  character norm   = {np.linalg.norm(snap['character']):.4f}")
    print(f"  graph node count = {graph.node_count}")
    print(f'  sentence         = "{sentence}"')

    composite_at_birth = affect.composite(t0)

    # ------------------------------------------------------------------
    # FIRST FEED — novel sentence. Expect: surprise, write, affect shift.
    # ------------------------------------------------------------------
    banner("First feed (novel)")
    res1 = pipeline.ingest_text(sentence, now=t1)
    composite_after_1 = affect.composite(t1)
    snap_after_1 = affect.snapshot(t1)

    print(f"  representation norm        = {res1.representation_norm:.4f}")
    print(f"  prediction.is_degenerate   = {res1.prediction.is_degenerate}")
    print(f"  prediction.confidence      = {res1.prediction.confidence:.4f}")
    print(f"  ||predicted_mean||         = {np.linalg.norm(res1.prediction.mean):.4f}")
    print(f"  gap.magnitude              = {res1.gap.magnitude:.4f}")
    print(f"  gap.is_surprise            = {res1.gap.is_surprise}")
    print(f"  gap.surprise_score         = {res1.gap.surprise_score:.4f}")
    print(f"  gap.was_new_write          = {res1.gap.was_new_write}")
    print(f"  gap.concept_id             = {res1.gap.concept_id}")
    print()
    print(f"  graph node count after     = {graph.node_count}")
    print(f"  reaction norm after        = {np.linalg.norm(snap_after_1['reaction']):.4f}")
    print(f"  composite norm after       = {np.linalg.norm(composite_after_1):.4f}")
    print(f"  composite shift from birth = {np.linalg.norm(composite_after_1 - composite_at_birth):.4f}")
    if res1.gap.concept_id is not None:
        node = graph.nodes[res1.gap.concept_id]
        print(f"  written node #{node.concept_id}")
        print(f"     name              = {node.name!r}")
        print(f"     activation_count  = {node.activation_count}")
        print(f"     surprise_at_birth = {node.surprise_at_birth:.4f}")
        print(f"     ||embedding||     = {np.linalg.norm(node.embedding):.4f}")
        print(f"     birth_state[:6]   = {fmt_vec(node.affect_trace.birth_state[:6])}")

    composite_before_2 = affect.composite(t2)

    # ------------------------------------------------------------------
    # SECOND FEED — same sentence. Expect: no surprise, no write.
    # ------------------------------------------------------------------
    banner("Second feed (repeat — same sentence)")
    res2 = pipeline.ingest_text(sentence, now=t2)
    composite_after_2 = affect.composite(t2)
    snap_after_2 = affect.snapshot(t2)

    print(f"  representation norm        = {res2.representation_norm:.4f}")
    print(f"  prediction.is_degenerate   = {res2.prediction.is_degenerate}")
    print(f"  prediction.confidence      = {res2.prediction.confidence:.4f}")
    print(f"  ||predicted_mean||         = {np.linalg.norm(res2.prediction.mean):.4f}")
    print(f"  gap.magnitude              = {res2.gap.magnitude:.6f}")
    print(f"  gap.is_surprise            = {res2.gap.is_surprise}")
    print(f"  gap.surprise_score         = {res2.gap.surprise_score:.4f}")
    print(f"  gap.was_new_write          = {res2.gap.was_new_write}")
    print(f"  gap.concept_id             = {res2.gap.concept_id}")
    print()
    print(f"  graph node count after     = {graph.node_count}")
    print(f"  composite shift this step  = {np.linalg.norm(composite_after_2 - composite_before_2):.6f}")

    # ------------------------------------------------------------------
    # Verdict.
    # ------------------------------------------------------------------
    banner("Verdict")
    checks = [
        ("first feed produced surprise",
            res1.gap.is_surprise is True),
        ("first feed wrote a new concept",
            res1.gap.was_new_write is True and graph.node_count == 1),
        ("first feed shifted the composite affect vector",
            np.linalg.norm(composite_after_1 - composite_at_birth) > 1e-3),
        ("second feed did NOT produce surprise",
            res2.gap.is_surprise is False),
        ("second feed did NOT write a new concept",
            res2.gap.was_new_write is False and graph.node_count == 1),
        ("second feed produced near-zero gap",
            res2.gap.magnitude < 1e-4),
        ("graph contains exactly one concept",
            graph.node_count == 1),
        ("totals: 2 observations, 1 surprise",
            predict_engine.observation_count == 2 and predict_engine.surprise_count == 1),
    ]
    all_ok = True
    for label, ok in checks:
        mark = "OK  " if ok else "FAIL"
        print(f"  [{mark}] {label}")
        if not ok:
            all_ok = False

    print()
    print("PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
