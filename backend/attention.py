"""Attention (Component F). Phase 3 minimal — the conductor.

F never produces an attention vector itself. It calls C.spread with the
right phase-mode mapping, samples affect once per attend() so the spread
all happens "under one feeling" deterministically, and returns the active
set that falls out. The active set IS the attention.

What exists:
  - AttentionPhase enum (INPUT / PROCESSING / OUTPUT) — distinct from
    A's InjectionPoint enum (same names, different domains; F's phase
    selects spread mode, A's injection point selects affect gain).
  - AttentionFrame: read-only record of one attend invocation, captures
    the affect snapshot used so observers know "this thought happened
    under this feeling."
  - Attention.attend(phase, raw_seeds, now) → SpreadResult.

What does NOT exist (deferred):
  - HabitOverlay (perceptual habits). Phase 3 later step.
  - PerceptualBaseline (the idle-attractor seed set).
  - processing_loop with multi-tick settle. Phase 3 later step.
  - tom_seed_provider, set_habit_temperature, request_habit_pin_refresh.
  - predict_prior wiring from B.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from backend.affect import AffectStack
from backend.graph import ConceptGraph, SpreadMode, SpreadResult


class AttentionPhase(Enum):
    INPUT      = "input"
    PROCESSING = "processing"
    OUTPUT     = "output"


_PHASE_TO_MODE: dict[AttentionPhase, SpreadMode] = {
    AttentionPhase.INPUT:      SpreadMode.PERCEIVE,
    AttentionPhase.PROCESSING: SpreadMode.PREDICT,
    AttentionPhase.OUTPUT:     SpreadMode.SIMULATE,
}


@dataclass
class AttentionFrame:
    """One invocation's read-only context. The affect fields are sampled
    exactly once at the start of `attend`; the spread runs entirely under
    these values. This preserves the spec's "this thought happened under
    this feeling" determinism — feelings can shift between attends, never
    mid-attend.
    """
    phase: AttentionPhase
    mode: SpreadMode
    seeds: dict[int, float]
    composite_affect: np.ndarray
    arousal: float
    now: float


class Attention:
    """Owns the call into C.spread. No persistent state in Phase 3 minimal."""

    def __init__(self, *, affect: AffectStack, graph: ConceptGraph) -> None:
        self.affect = affect
        self.graph = graph

    def attend(
        self,
        phase: AttentionPhase,
        raw_seeds: dict[int, float],
        now: float,
        *,
        max_steps: int = 3,
        budget: int = 256,
    ) -> tuple[SpreadResult, AttentionFrame]:
        """Build an AttentionFrame, call C.spread, return both.

        Returning the frame alongside the result lets callers (and tests)
        inspect the affect snapshot the spread ran under without making
        Attention hold per-call state.

        raw_seeds: dict[concept_id → activation]. The harness or H provides
        these; Phase 3 minimal does not augment them with habit seeds.
        """
        composite = self.affect.composite(now)
        arousal = self.affect.current_arousal(now)
        mode = _PHASE_TO_MODE[phase]

        # Phase 3 minimal: pass raw seeds straight through. HabitOverlay later.
        seeds = {cid: float(v) for cid, v in raw_seeds.items()}

        frame = AttentionFrame(
            phase=phase,
            mode=mode,
            seeds=seeds,
            composite_affect=composite,
            arousal=arousal,
            now=now,
        )

        result = self.graph.spread(
            seeds=seeds,
            affect=self.affect,
            composite_affect=composite,
            arousal=arousal,
            max_steps=max_steps,
            budget=budget,
            mode=mode,
        )
        return result, frame
