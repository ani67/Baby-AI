"""Affective Engine (Component A).

Five-layer timescale stack + the gap→affect projection W.

What exists:
  - AffectStack: the five timescale layers (reaction, working, mood, disposition,
    character), each holding a vector in N_AFF=12 dimensions and decaying at its
    own half-life. All decay is lazy — applied on read using `(now - t_layer)`.
  - W: a learned linear projection from D_REP-dim gap space to N_AFF-dim affect
    space. Initialized small-random with row-normalization. Updated live by an
    Oja-subspace Hebbian rule on every observation that passes through B.
  - project_gap(delta, magnitude): pure projection. Returns (gap_signal, mag).
  - project_affect_to_repr(): cached pinv(W). Used by G later for style biasing.
  - learn_W(raw_delta): in-place Hebbian update. Caller (B) drives it.
  - inject(injection_point, gap_signal, gap_magnitude, now): the only state
    mutator on the reaction layer; nudge_chain propagates upward.
  - composite(now): weighted sum across all five post-decay layers.
  - snapshot(now): read-only debug view; not a contract for any caller.

What does NOT exist (deferred):
  - simulate_inject (no D yet).
  - set_nudge_gain_multiplier (no D).
  - gate_attention (no F / no spread yet).
  - persistence (single-process for now).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from backend.config import (
    COMPOSITE_WEIGHTS,
    D_REP,
    HALF_LIFE_CHARACTER,
    HALF_LIFE_DISPOSITION,
    HALF_LIFE_MOOD,
    HALF_LIFE_REACTION,
    HALF_LIFE_WORKING,
    INJECTION_GAIN_INPUT,
    INJECTION_GAIN_OUTPUT,
    INJECTION_GAIN_PROCESSING,
    MAX_LAYER_NORM,
    N_AFF,
    NUDGE_GAINS,
    NUDGE_THRESHOLDS,
)


def _clamp_norm(v: np.ndarray, ceiling: float = MAX_LAYER_NORM) -> np.ndarray:
    """Cap an affect vector's L2 norm at `ceiling`, preserving direction.
    Used after every layer mutation so no layer can accumulate without bound
    under a high-arousal regime (e.g., dense book ingestion at 700+ cycles/sec
    where inter-cycle decay is effectively zero for slow-half-life layers)."""
    n = float(np.linalg.norm(v))
    if n <= ceiling:
        return v
    return (v * (ceiling / n)).astype(np.float32)


# Hebbian learning rate for W. Small enough that any single observation moves W
# only slightly; over many observations W converges toward the principal
# subspace of the raw-delta distribution.
W_LEARNING_RATE = 1e-3


class InjectionPoint(Enum):
    INPUT = "input"
    PROCESSING = "processing"
    OUTPUT = "output"


_INJECTION_GAIN = {
    InjectionPoint.INPUT:      INJECTION_GAIN_INPUT,
    InjectionPoint.PROCESSING: INJECTION_GAIN_PROCESSING,
    InjectionPoint.OUTPUT:     INJECTION_GAIN_OUTPUT,
}


@dataclass
class Layer:
    """One timescale of affect.

    `vector` is the value as of time `t`. Reads must apply forward decay using
    `(now - t)` and `half_life` — the layer never spontaneously updates itself.
    """
    vector: np.ndarray   # float32[N_AFF]
    t: float
    half_life: float


class AffectStack:
    """Five-layer affect state. All vectors live in N_AFF dims."""

    def __init__(self, *, birth_seed: int = 0, t_birth: float = 0.0) -> None:
        rng = np.random.default_rng(birth_seed)
        # Character: small jitter — the stable emotional fingerprint.
        character_init = rng.normal(0.0, 0.05, size=N_AFF).astype(np.float32)
        # Disposition starts at half of character (synthesis boot sequence).
        disposition_init = (0.5 * character_init).astype(np.float32)
        # Mood and working start at zero — the mind has no current mood at birth.
        zero = np.zeros(N_AFF, dtype=np.float32)
        # Reaction: light arousal jitter so the first input lands on a
        # non-degenerate state and can produce real surprise.
        reaction_init = rng.normal(0.0, 0.02, size=N_AFF).astype(np.float32)

        self.reaction    = Layer(reaction_init,    t_birth, HALF_LIFE_REACTION)
        self.working     = Layer(zero.copy(),      t_birth, HALF_LIFE_WORKING)
        self.mood        = Layer(zero.copy(),      t_birth, HALF_LIFE_MOOD)
        self.disposition = Layer(disposition_init, t_birth, HALF_LIFE_DISPOSITION)
        self.character   = Layer(character_init,   t_birth, HALF_LIFE_CHARACTER)

        # W: gap → affect projection, shape (N_AFF, D_REP). Init with small
        # Gaussian entries scaled so each row has expected norm ~1 — enough
        # signal to produce a non-degenerate gap_signal on the first input,
        # not so much that early outputs saturate the squashing tanh.
        scale = 1.0 / float(np.sqrt(D_REP))
        self.W = rng.normal(0.0, scale, size=(N_AFF, D_REP)).astype(np.float32)
        self._W_pinv_cache: np.ndarray | None = None
        self.W_update_count: int = 0

        # Nudge-gain multiplier — D installs 0.30 around each replay observation
        # so replays still feel real (reaction is full magnitude) but their
        # propagation upward into mood/disposition/character is muted.
        # Default 1.0 = no attenuation. Cleared back to 1.0 after each replay.
        self._nudge_gain_multiplier: float = 1.0

        # Consolidation envelope — sleep installs `_consolidation_multiplier`
        # (default 1.5) and sets the flag for the duration of sleep mode.
        # When active, it OVERRIDES the per-replay multiplier so a single
        # sleep envelope amplifies all upward nudges (including replay's)
        # without each replay's local set/clear leaking back to 1.0 between
        # events. Synthesis: this is what "REM-like consolidation" means
        # numerically — the sleeping mind shapes mood/disposition/character
        # 1.5× faster than the waking mind does.
        self._consolidation_active: bool = False
        self._consolidation_multiplier: float = 1.0

    @property
    def _ordered_layers(self) -> tuple[Layer, ...]:
        return (self.reaction, self.working, self.mood, self.disposition, self.character)

    # ---- reads ----

    def composite(self, now: float) -> np.ndarray:
        """Weighted sum across all five layers, post-decay. float32[N_AFF]."""
        out = np.zeros(N_AFF, dtype=np.float32)
        for w, layer in zip(COMPOSITE_WEIGHTS, self._ordered_layers):
            out += w * _decayed(layer, now)
        return out

    def snapshot(self, now: float) -> dict:
        """Debug-only view — never on a hot path. Returns post-decay copies."""
        return {
            "reaction":    _decayed(self.reaction, now).copy(),
            "working":     _decayed(self.working, now).copy(),
            "mood":        _decayed(self.mood, now).copy(),
            "disposition": _decayed(self.disposition, now).copy(),
            "character":   _decayed(self.character, now).copy(),
            "composite":   self.composite(now),
        }

    # ---- W: gap → affect projection ----

    def project_gap(
        self,
        delta: np.ndarray,
        magnitude: float | None = None,
    ) -> tuple[np.ndarray, float]:
        """Project a raw D_REP-dim gap into an N_AFF-dim affect signal.

        Pure function — does not mutate W. The returned magnitude is the
        rep-space norm of `delta`, not the projected norm; callers use it
        for surprise scoring and inject's tanh squashing, both of which
        want a stable rep-space scale rather than W's drifting norm.
        """
        if delta.shape != (D_REP,):
            raise ValueError(f"delta must be ({D_REP},), got {delta.shape}")
        if magnitude is None:
            magnitude = float(np.linalg.norm(delta))
        gap_signal = self.W @ delta.astype(np.float32)
        return gap_signal, float(magnitude)

    def project_affect_to_repr(self) -> np.ndarray:
        """Return pinv(W), shape (D_REP, N_AFF). Cached until W next changes.

        G uses this in the expression layer to bias the surface-rep generation
        toward the current affect state. Phase 2 just exposes it; the cache
        makes repeated calls cheap.
        """
        if self._W_pinv_cache is None:
            self._W_pinv_cache = np.linalg.pinv(self.W).astype(np.float32)
        return self._W_pinv_cache

    def learn_W(
        self,
        raw_delta: np.ndarray,
        learning_rate: float = W_LEARNING_RATE,
    ) -> None:
        """Apply Oja's subspace rule to W given one D_REP-dim observation.

        Update: ΔW = η · (y x^T − y y^T W), with x = unit(raw_delta) and
        y = W x. This is a stable Hebbian rule: rows of W converge toward
        an orthonormal basis spanning the principal subspace of whatever
        raw-delta distribution the system actually experiences. There is
        no separate training step — every observation that B sees produces
        one update.

        Skipped silently if the input has effectively zero magnitude — a
        zero direction has no useful information to bind into W.
        """
        mag = float(np.linalg.norm(raw_delta))
        if mag < 1e-9:
            return
        x = (raw_delta.astype(np.float32) / mag)             # D_REP, unit
        y = self.W @ x                                       # N_AFF
        delta_W = (np.outer(y, x) - np.outer(y, y) @ self.W) # N_AFF × D_REP
        self.W += learning_rate * delta_W
        self._W_pinv_cache = None
        self.W_update_count += 1

    # ---- replay attenuation hook (Phase 3 — D installs around each replay) ----

    def set_nudge_gain_multiplier(self, m: float) -> None:
        """D calls this before each replay observation to attenuate the chain
        propagation. Reaction layer is unaffected — replays still feel real —
        but their push into working/mood/disposition/character is scaled by m.

        When `_consolidation_active` is True (sleep mode), this value is
        ignored by the nudge chain — the consolidation multiplier wins.
        The set/clear pair still runs cleanly across the sleep boundary.
        """
        if m < 0.0:
            raise ValueError(f"nudge_gain_multiplier must be ≥ 0, got {m}")
        self._nudge_gain_multiplier = float(m)

    def clear_nudge_gain_multiplier(self) -> None:
        """D calls this after each replay observation. Restores 1.0 (no attenuation)."""
        self._nudge_gain_multiplier = 1.0

    # ---- consolidation envelope (Phase 4 — sleep wraps the system in this) ----

    def begin_consolidation(self, multiplier: float = 1.5) -> None:
        """Sleep mode entry. Sets the consolidation multiplier to override
        per-replay attenuation; while active, all upward nudges are scaled
        by this value regardless of `_nudge_gain_multiplier` games elsewhere.
        """
        if multiplier < 0.0:
            raise ValueError(f"consolidation multiplier must be ≥ 0, got {multiplier}")
        self._consolidation_active = True
        self._consolidation_multiplier = float(multiplier)

    def end_consolidation(self) -> None:
        """Sleep mode exit. Clears the override; nudges return to whatever
        `_nudge_gain_multiplier` is currently set to (1.0 in steady state)."""
        self._consolidation_active = False
        self._consolidation_multiplier = 1.0

    def _effective_multiplier(self) -> float:
        if self._consolidation_active:
            return self._consolidation_multiplier
        return self._nudge_gain_multiplier

    # ---- pure simulation (Phase 3 — used by D's simulate_chain rollouts) ----

    def simulate_inject(
        self,
        prior_affect: np.ndarray,
        gap_signal: np.ndarray,
        gap_magnitude: float,
        injection_point: InjectionPoint,
    ) -> np.ndarray:
        """Pure: compute the post-inject reaction WITHOUT mutating live state.

        D's simulate_chain rollouts use this to feel candidate paths forward
        without disturbing the system's actual affect. Same g + tanh(magnitude)
        + delta math as `inject`; nothing is timestamped, nothing propagates
        up the nudge chain (simulations don't shape mood).

        Returns the new reaction vector for that simulated step.
        """
        if gap_signal.shape != (N_AFF,):
            raise ValueError(f"gap_signal must be ({N_AFF},), got {gap_signal.shape}")
        if prior_affect.shape != (N_AFF,):
            raise ValueError(f"prior_affect must be ({N_AFF},), got {prior_affect.shape}")

        g = _INJECTION_GAIN[injection_point]
        squashed = float(np.tanh(gap_magnitude))
        delta = (g * squashed) * gap_signal.astype(np.float32)
        return prior_affect.astype(np.float32) + delta

    # ---- arousal + attention gating (Phase 3) ----

    def current_arousal(self, now: float) -> float:
        """Reaction-layer L2 norm post-decay. Drives the spread sparsity
        envelope (high arousal = narrow/intense, low = broad/diffuse) and
        the multiplicative-vs-additive blend in gate_attention.
        """
        return float(np.linalg.norm(_decayed(self.reaction, now)))

    def gate_attention(
        self,
        node_affect_trace,
        composite_affect: np.ndarray,
        arousal: float,
        semantic_score: float,
        predictive_score: float = 1.0,
    ) -> float:
        """Compute the affect gate for one spread propagation step.

        Three relevances combine into one gate:
          - affective: cosine alignment between current composite and the
            destination node's running_state (how much the system "feels"
            the concept right now).
          - semantic: caller-supplied. C uses edge weight in the simplest
            case; later G/D may pass cosine in rep space.
          - predictive: 1.0 baseline; F's predict_prior raises it for
            concepts B is currently expecting.

        Arousal interpolates between multiplicative (high arousal —
        selective, intense, narrow: all three factors must be present)
        and additive (low arousal — broad, diffuse, associative: even one
        strong factor lights the path) regimes.

        Phase 3 minimal: arousal acts as a linear blend in [0, 1]; non-
        linear schedule and per-band table land later.
        """
        running = node_affect_trace.running_state
        rn = float(np.linalg.norm(running))
        cn = float(np.linalg.norm(composite_affect))
        if rn < 1e-9 or cn < 1e-9:
            affect_align = 0.0
        else:
            cos = float((running @ composite_affect.astype(np.float32)) / (rn * cn))
            affect_align = max(0.0, cos)   # negative alignment → no contribution

        sem  = max(0.0, float(semantic_score))
        pred = max(0.0, float(predictive_score))

        mult_gate = affect_align * sem * pred
        add_gate  = (affect_align + sem + pred) / 3.0

        mix = min(1.0, max(0.0, arousal))
        return float(mix * mult_gate + (1.0 - mix) * add_gate)

    # ---- the only mutator on affect state ----

    def inject(
        self,
        injection_point: InjectionPoint,
        gap_signal: np.ndarray,
        gap_magnitude: float,
        now: float,
    ) -> np.ndarray:
        """B is the only legitimate caller (per synthesis).

        gap_signal is already in N_AFF dims — produced by `project_gap`.
        gap_magnitude is the rep-space norm used for tanh squashing.

        Returns the new reaction vector (post-decay + delta), for callers
        that want to log the per-event movement.
        """
        if gap_signal.shape != (N_AFF,):
            raise ValueError(f"gap_signal must be ({N_AFF},), got {gap_signal.shape}")

        g = _INJECTION_GAIN[injection_point]
        # Squash on magnitude so a single huge gap doesn't snap reaction
        # outside its plausible band.
        squashed = float(np.tanh(gap_magnitude))
        delta = (g * squashed) * gap_signal.astype(np.float32)

        decayed = _decayed(self.reaction, now)
        new_reaction = _clamp_norm(decayed + delta)
        self.reaction.vector = new_reaction
        self.reaction.t = now

        self._nudge_chain(now)
        return new_reaction.copy()

    # ---- internal ----

    def _nudge_chain(self, now: float) -> None:
        """Push affect upward: reaction → working → mood → disposition → character.

        Each step is threshold-gated. Below the lower layer's threshold magnitude,
        the upper layer is untouched. Above it, the upper layer absorbs a
        gain-scaled push along the lower layer's direction. Net effect: short,
        small reactions never reach character; sustained reactions integrate
        all the way up over their natural timescale.
        """
        layers = list(self._ordered_layers)
        # 1.0 normally; 0.30 during a single replay; 1.5 during sleep
        # consolidation (overrides replay's local set/clear).
        m = self._effective_multiplier()
        for i in range(4):
            lower = layers[i]
            upper = layers[i + 1]
            gain = NUDGE_GAINS[i] * m
            threshold = NUDGE_THRESHOLDS[i]

            lower_value = _decayed(lower, now)
            mag = float(np.linalg.norm(lower_value))
            if mag < threshold:
                continue

            # Saturate the push magnitude through tanh so a runaway lower
            # layer can't drive the upper layer linearly. Combined with the
            # layer-norm ceiling below, this bounds the entire chain under
            # any input regime.
            raw_push_mag = (mag - threshold) * gain
            push_mag = float(np.tanh(raw_push_mag))
            direction = lower_value / max(mag, 1e-12)
            push = direction * push_mag

            upper_value = _decayed(upper, now)
            upper.vector = _clamp_norm(upper_value + push)
            upper.t = now


def _decayed(layer: Layer, now: float) -> np.ndarray:
    """Apply exponential half-life decay forward from layer.t to now."""
    elapsed = max(0.0, now - layer.t)
    if layer.half_life <= 0.0:
        return layer.vector
    factor = 0.5 ** (elapsed / layer.half_life)
    return layer.vector * float(factor)
