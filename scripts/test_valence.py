"""Valence test — affect_in_rep_space + inject() valence modulation.

Phase 3 surprise-with-valence: incoming gaps carry a signed alignment
score with the current composite affect. Below the deadband |v| ≤
VALENCE_THRESHOLD, modulation is off (default behavior preserved).
Above it, sign decides direction:
    valence > 0 → approach: scale delta upward.
    valence < 0 → aversion: reverse delta and scale by |v|.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.affect import AffectStack, InjectionPoint   # noqa: E402
from backend.config import D_REP, N_AFF, VALENCE_THRESHOLD  # noqa: E402


def _fresh(birth_seed: int = 7) -> tuple[AffectStack, float]:
    now = time.time()
    return AffectStack(birth_seed=birth_seed, t_birth=now), now


def test_affect_in_rep_space_shape_and_norm() -> None:
    """The helper returns a (D_REP,) unit vector in rep space."""
    a, now = _fresh()
    out = a.affect_in_rep_space(now)
    assert out.shape == (D_REP,), f"shape: expected ({D_REP},), got {out.shape}"
    n = float(np.linalg.norm(out))
    # Unit-normalized (with the +1e-9 floor in the helper, exact 1.0 ± eps).
    assert abs(n - 1.0) < 1e-4, f"norm: expected ≈1.0, got {n}"


def test_inject_valence_zero_matches_default() -> None:
    """Default valence=0.0 keeps existing callers behavior-identical."""
    a1, now = _fresh(birth_seed=11)
    a2, _   = _fresh(birth_seed=11)

    strong_signal = np.zeros(N_AFF, dtype=np.float32)
    strong_signal[0] = 0.9
    strong_signal[1] = 0.5

    r1 = a1.inject(InjectionPoint.INPUT, strong_signal, 1.5, now)
    r2 = a2.inject(InjectionPoint.INPUT, strong_signal, 1.5, now, valence=0.0)
    assert np.allclose(r1, r2), (
        f"inject() with default valence != inject(valence=0.0); "
        f"diff_norm={float(np.linalg.norm(r1 - r2))}"
    )


def test_inject_deadband_unchanged() -> None:
    """Inside the deadband |valence| ≤ VALENCE_THRESHOLD, modulation is off."""
    a1, now = _fresh(birth_seed=19)
    a2, _   = _fresh(birth_seed=19)

    sig = np.zeros(N_AFF, dtype=np.float32)
    sig[2] = 0.7

    baseline = a1.inject(InjectionPoint.INPUT, sig, 1.0, now)
    # Just under the threshold — should still match baseline.
    in_band  = a2.inject(InjectionPoint.INPUT, sig, 1.0, now,
                         valence=VALENCE_THRESHOLD - 0.05)
    assert np.allclose(baseline, in_band), (
        f"deadband modulation leaked through; "
        f"diff_norm={float(np.linalg.norm(baseline - in_band))}"
    )


def test_inject_positive_valence_scales_delta_by_valence() -> None:
    """Above the deadband, positive valence multiplies delta by valence.
    Direction is preserved (cosine ≈ +1 with the default-valence delta),
    magnitude is scaled by `valence` itself."""
    a1, now = _fresh(birth_seed=23)
    a2, _   = _fresh(birth_seed=23)

    sig = np.zeros(N_AFF, dtype=np.float32)
    sig[0] = 0.9
    sig[3] = 0.4

    pre1 = a1.reaction.vector.copy()
    pre2 = a2.reaction.vector.copy()

    valence = 0.8                                        # > VALENCE_THRESHOLD = 0.3
    base   = a1.inject(InjectionPoint.INPUT, sig, 1.5, now)
    scaled = a2.inject(InjectionPoint.INPUT, sig, 1.5, now, valence=valence)

    base_delta   = base   - pre1
    scaled_delta = scaled - pre2

    bn = float(np.linalg.norm(base_delta))
    sn = float(np.linalg.norm(scaled_delta))
    assert bn > 1e-9 and sn > 1e-9, f"degenerate movement; bn={bn}, sn={sn}"

    # Same direction.
    cos = float(base_delta @ scaled_delta) / (bn * sn)
    assert cos > 0.99, (
        f"positive valence flipped direction; cosine={cos:.4f} (expected ≈+1)"
    )
    # Magnitude scaled by `valence`.
    ratio = sn / bn
    assert abs(ratio - valence) < 1e-3, (
        f"positive valence did not scale delta by valence; "
        f"|scaled|/|base|={ratio:.4f}, expected≈{valence}"
    )


def test_inject_negative_valence_reverses() -> None:
    """Negative valence above the deadband reverses delta — the new
    reaction lies on the OPPOSITE side of the prior reaction from
    where the default-valence inject would have moved it."""
    a1, now = _fresh(birth_seed=29)
    a2, _   = _fresh(birth_seed=29)

    sig = np.zeros(N_AFF, dtype=np.float32)
    sig[1] = 0.8
    sig[4] = 0.3

    pre1 = a1.reaction.vector.copy()
    pre2 = a2.reaction.vector.copy()

    base     = a1.inject(InjectionPoint.INPUT, sig, 1.5, now)
    aversive = a2.inject(InjectionPoint.INPUT, sig, 1.5, now, valence=-0.7)

    base_delta = base     - pre1
    avr_delta  = aversive - pre2

    # Reversed direction → negative cosine.
    bn = float(np.linalg.norm(base_delta))
    an = float(np.linalg.norm(avr_delta))
    assert bn > 1e-9 and an > 1e-9, (
        f"degenerate movement; bn={bn}, an={an}"
    )
    cos = float(base_delta @ avr_delta) / (bn * an)
    assert cos < -0.5, (
        f"negative valence did not reverse direction; cosine={cos:.4f} "
        f"(expected < -0.5)"
    )


def test_inject_shape_validation_still_works() -> None:
    """Valence kwarg doesn't break the existing shape guard."""
    a, now = _fresh()
    bad = np.zeros(N_AFF + 1, dtype=np.float32)
    raised = False
    try:
        a.inject(InjectionPoint.INPUT, bad, 0.5, now, valence=0.5)
    except ValueError:
        raised = True
    assert raised, "inject did not raise on wrong-shape gap_signal"


def main() -> int:
    test_affect_in_rep_space_shape_and_norm()
    test_inject_valence_zero_matches_default()
    test_inject_deadband_unchanged()
    test_inject_positive_valence_scales_delta_by_valence()
    test_inject_negative_valence_reverses()
    test_inject_shape_validation_still_works()
    print("All valence tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
