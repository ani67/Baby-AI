"""B1 unit test — compute_expression_budget threshold table.

No model loading. Builds a minimal MainLoop with a stub affect that
returns a fixed arousal, then asserts the budget at each spec corner.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time

from backend.affect import AffectStack          # noqa: E402
from backend.attention import Attention          # noqa: E402
from backend.expression import Expression        # noqa: E402
from backend.graph import ConceptGraph           # noqa: E402
from backend.identity import Identity            # noqa: E402
from backend.input import InputPipeline          # noqa: E402
from backend.main_loop import MainLoop           # noqa: E402
from backend.predict import PredictionEngine     # noqa: E402
from backend.simulation import SimulationReplay  # noqa: E402


def _build_loop() -> MainLoop:
    now = time.time()
    a = AffectStack(birth_seed=1, t_birth=now)
    g = ConceptGraph()
    p = PredictionEngine(affect=a, graph=g)
    sim = SimulationReplay(affect=a, graph=g, predict_engine=p)
    ident = Identity(
        affect=a, graph=g, predict_engine=p, simulation=sim,
        birth_seed=1, birth_time=now,
    )
    h = InputPipeline(affect=a, graph=g, predict_engine=p, identity=ident)
    f = Attention(affect=a, graph=g)
    gx = Expression(
        affect=a, graph=g, predict_engine=p, identity=ident, input_pipeline=h,
    )
    return MainLoop(
        affect=a, graph=g, predict_engine=p, simulation=sim,
        identity=ident, attention=f, expression=gx, input_pipeline=h,
    )


def _set_arousal(loop: MainLoop, val: float) -> None:
    """Patch current_arousal on the live AffectStack to a constant."""
    loop.affect.current_arousal = lambda now=None, _v=val: float(_v)  # type: ignore[assignment]


def main() -> int:
    loop = _build_loop()
    now = time.time()

    # Build sized "active sets" (the dict's contents don't matter — only len).
    def s(n): return {i: 1.0 for i in range(n)}

    # The five corners of the spec table.
    cases = [
        # (arousal, set_size, expected_budget, label)
        (0.7, 35, 5, "paragraph: arousal>0.6 and set>30"),
        (0.7, 30, 3, "fall to 3 when set==30 (>30 required)"),
        (0.4, 16, 3, "multi-sentence: arousal>0.3 and set>15"),
        (0.4, 15, 2, "fall to 2 when set==15 (>15 required)"),
        (0.2,  9, 2, "two-sentence: arousal>0.1 and set>8"),
        (0.2,  8, 1, "fall to 1 when set==8 (>8 required)"),
        (0.05, 100, 1, "minimal: arousal too low"),
        (0.7,  3, 1, "minimal: set too small even at high arousal"),
        (0.7, 31, 5, "paragraph again at exact thresholds+1"),
    ]

    n_pass = 0
    for arousal, set_size, expected, label in cases:
        _set_arousal(loop, arousal)
        got = loop.compute_expression_budget(s(set_size), now=now)
        ok = (got == expected)
        n_pass += int(ok)
        print(f"  arousal={arousal:.2f}  |set|={set_size:>3d}  "
              f"got={got}  expected={expected}  "
              f"{'PASS' if ok else 'FAIL'}  ({label})")

    print()
    if n_pass == len(cases):
        print(f"[B1] all {n_pass}/{len(cases)} cases pass")
        return 0
    print(f"[B1] FAIL: {n_pass}/{len(cases)} pass")
    return 1


if __name__ == "__main__":
    sys.exit(main())
