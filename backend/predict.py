"""Prediction Engine (Component B).

Role: B is the orchestrator that ties A and C together. H feeds it an encoded
representation; B forms a prediction, computes the gap, scores surprise, and
routes side effects to A (always) and C (only on surprise).

What exists:
  - predict(query_seed, layer): nearest-neighbor lookup over the concept graph.
    Degenerate prediction (zero mean, confidence 0) when the graph is empty
    or nothing is close. Otherwise returns the nearest concept's embedding
    as the predicted mean, with confidence = clamped cosine similarity.
  - observe(prediction, actual, now, name_hint, replay_origin=False):
    computes delta and magnitude; scores surprise per-layer via cold-start
    absolute floor while count < COLD_START_N, then via Welford z-score
    once enough samples have accumulated. Calls A.project_gap, A.inject,
    A.learn_W on every observation. Calls C.write_on_surprise only when
    surprise fires. Honors replay_origin: when True, surprise is scored
    against current stats but the stats are NOT updated (synthesis T4).

Still deferred (no caller in Phase 2):
  - simulate_step, reconcile_simulation, replay_push (D's territory).
  - Pending-prediction lifecycle / TTL bookkeeping (predict and observe
    still collapse onto the same tick — no streaming/anticipation yet).
  - Per-dimension precision and weighted_delta. Confidence is a scalar.
  - Persistence of running stats across sessions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from backend.affect import AffectStack, InjectionPoint
from backend.config import (
    COLD_START_MAGNITUDE_FLOOR,
    COLD_START_N,
    D_REP,
    MIN_THRESHOLD,
    STDDEV_FLOOR,
)
from backend.graph import ConceptGraph

if TYPE_CHECKING:
    # Avoid circular import — D imports B's types, B only references D by attribute.
    from backend.simulation import SimulationReplay


_LAYER_TO_INJECTION_POINT = {
    "INPUT":      InjectionPoint.INPUT,
    "PROCESSING": InjectionPoint.PROCESSING,
    "OUTPUT":     InjectionPoint.OUTPUT,
}


@dataclass
class Prediction:
    mean: np.ndarray            # float32[D_REP]
    confidence: float           # scalar in [0, 1]; 0 = degenerate
    layer: str                  # "INPUT" / "PROCESSING" / "OUTPUT"
    tick: int
    is_degenerate: bool         # True iff mean is the zero placeholder


@dataclass
class PredictionGap:
    delta: np.ndarray           # float32[D_REP] = actual - predicted
    magnitude: float            # ||delta||
    surprise_score: float       # cold-start: synthetic MIN_THRESHOLD; post: z
    is_surprise: bool
    layer: str
    tick: int
    confidence_at_prediction: float
    replay_origin: bool                # synthesis T4 — set by D during replay
    # --- Phase 2 diagnostic fields (will move to D's replay log later) ---
    concept_id: int | None = None      # populated when a write/strengthen happened
    was_new_write: bool = False        # True iff a new ConceptNode was created


@dataclass
class SimulationFrame:
    """One step in a D-driven rollout. No state is registered with B —
    simulate_step does not produce a pending prediction. Frames are pure
    computation, so simulate_chain can run many candidate paths cheaply.
    """
    state: np.ndarray            # float32[D_REP] — predicted next-state
    affect: np.ndarray           # float32[N_AFF] — simulated reaction at this step
    expected_surprise: float     # 1 − cosine(state, nearest_in_graph), clipped ≥ 0
    chain_id: int
    depth: int
    action_kind: str             # decoupled from D's enum — string identifier


class WelfordStats:
    """Online mean and (sample) variance via Welford's algorithm.

    Updates are O(1) and numerically stable for long sequences. `count`,
    `mean`, and `M2` are the canonical Welford state; variance is computed
    on demand from M2 / (count - 1).
    """

    __slots__ = ("count", "mean", "M2")

    def __init__(self) -> None:
        self.count: int = 0
        self.mean: float = 0.0
        self.M2: float = 0.0

    def update(self, x: float) -> None:
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.M2 += delta * delta2

    @property
    def variance(self) -> float:
        if self.count < 2:
            return 0.0
        return self.M2 / (self.count - 1)

    @property
    def stddev(self) -> float:
        return float(self.variance ** 0.5)


class PredictionEngine:
    """Owns the predict/observe handshake and the routing into A and C."""

    def __init__(self, *, affect: AffectStack, graph: ConceptGraph) -> None:
        self.affect = affect
        self.graph = graph
        self._tick: int = 0
        self.observation_count: int = 0
        self.surprise_count: int = 0
        # Per-layer running stats. Phase 2 only exercises INPUT, but the dict
        # is initialized for all three layers so PROCESSING and OUTPUT can
        # join later without an init dance.
        self._stats: dict[str, WelfordStats] = {
            "INPUT":      WelfordStats(),
            "PROCESSING": WelfordStats(),
            "OUTPUT":     WelfordStats(),
        }
        # Optional D sink — set by D after construction (avoids circular init).
        # When attached, B auto-pushes every non-replay surprise to D for replay.
        self.replay_target: "SimulationReplay | None" = None

    # ---- predict ----

    def predict(self, query_seed: np.ndarray, layer: str = "INPUT") -> Prediction:
        """Nearest-neighbor prediction from the graph.

        Phase 2 still uses the encoded input itself as query_seed. With an
        empty graph, returns a zero-mean degenerate prediction. With an
        existing match, returns the nearest concept's embedding as the
        predicted mean and clamped cosine similarity as confidence.
        """
        if query_seed.shape != (D_REP,):
            raise ValueError(f"query_seed must be ({D_REP},), got {query_seed.shape}")

        self._tick += 1
        nearest = self.graph.nearest(query_seed)
        if nearest is None:
            return Prediction(
                mean=np.zeros(D_REP, dtype=np.float32),
                confidence=0.0,
                layer=layer,
                tick=self._tick,
                is_degenerate=True,
            )

        cid, sim = nearest
        predicted_embedding = self.graph.nodes[cid].embedding.astype(np.float32, copy=True)
        confidence = max(0.0, float(sim))
        return Prediction(
            mean=predicted_embedding,
            confidence=confidence,
            layer=layer,
            tick=self._tick,
            is_degenerate=False,
        )

    # ---- observe ----

    def observe(
        self,
        prediction: Prediction,
        actual: np.ndarray,
        now: float,
        name_hint: str = "",
        replay_origin: bool = False,
    ) -> PredictionGap:
        """Compute the gap, score surprise, route to A and (if surprise) C.

        replay_origin (synthesis T4): when True, this observation is being
        replayed by D. Surprise is still scored against the current stats —
        the system can re-experience and re-evaluate the memory — but the
        stats themselves are NOT updated. Otherwise replays would drag the
        threshold downward and inflate the surprise rate over time.
        """
        if actual.shape != (D_REP,):
            raise ValueError(f"actual must be ({D_REP},), got {actual.shape}")

        self.observation_count += 1

        delta = actual.astype(np.float32) - prediction.mean.astype(np.float32)
        magnitude = float(np.linalg.norm(delta))

        # Score against current stats first, before updating.
        surprise_score, is_surprise = self._score_surprise(magnitude, prediction.layer)

        # Update Welford stats unless this is a replay observation.
        if not replay_origin:
            self._stats[prediction.layer].update(magnitude)

        gap = PredictionGap(
            delta=delta,
            magnitude=magnitude,
            surprise_score=surprise_score,
            is_surprise=is_surprise,
            layer=prediction.layer,
            tick=prediction.tick,
            confidence_at_prediction=prediction.confidence,
            replay_origin=replay_origin,
        )

        # Always route to A. Zero gap → zero delta to reaction (handled by tanh).
        # A owns W: project, inject, then learn from this delta.
        gap_signal, gap_magnitude = self.affect.project_gap(delta, magnitude)
        injection_point = _LAYER_TO_INJECTION_POINT[prediction.layer]
        self.affect.inject(injection_point, gap_signal, gap_magnitude, now)
        self.affect.learn_W(delta)

        # Route to C only on surprise.
        if is_surprise:
            self.surprise_count += 1
            current_affect = self.affect.composite(now)
            cid, is_new = self.graph.write_on_surprise(
                representation=actual,
                surprise=surprise_score,
                current_affect=current_affect,
                name_hint=name_hint,
                now=now,
            )
            gap.concept_id = cid
            gap.was_new_write = is_new

            # Auto-push to D's replay buffer if attached. Skip when this
            # observation is itself a replay — pushing replays as new entries
            # would loop the buffer. Synthesis: replay_push happens on every
            # surprise crossing threshold from EmitGap.
            if self.replay_target is not None and not replay_origin:
                self.replay_target.on_surprise(
                    gap=gap,
                    actual_repr=actual.astype(np.float32, copy=True),
                    affect_snapshot=current_affect,
                    now=now,
                )

        return gap

    # ---- simulation (pure rollouts; no pending registration) ----

    def simulate_step(
        self,
        state: np.ndarray,
        affect_proxy: np.ndarray,
        action_kind: str,
        chain_id: int,
        depth: int,
    ) -> SimulationFrame:
        """Pure rollout step — does NOT register a pending prediction.

        Phase 3 minimal world model: predicted next-state is the nearest
        concept in the graph (or current state if action is "wait").
        Simulated affect is computed by A.simulate_inject on the gap between
        current and predicted states. Expected_surprise is `1 − sim` for the
        nearest neighbor — a proxy for how much the rollout is "stretching"
        the graph's known territory.

        action_kind is a string so B doesn't have to import D's ActionKind.
        D translates ActionKind → action_kind.value before calling.
        """
        if state.shape != (D_REP,):
            raise ValueError(f"state must be ({D_REP},), got {state.shape}")

        nearest = self.graph.nearest(state)
        if nearest is None:
            next_state = state.astype(np.float32, copy=True)
            expected_surprise = 0.0
        else:
            cid, sim = nearest
            if action_kind == "wait":
                # WAIT: do not advance state; the system holds its position.
                next_state = state.astype(np.float32, copy=True)
                expected_surprise = 0.0
            else:
                # Drift toward the nearest known concept.
                next_state = self.graph.nodes[cid].embedding.astype(np.float32, copy=True)
                expected_surprise = float(max(0.0, 1.0 - sim))

        # Simulated felt affect: project the simulated gap, simulate-inject.
        delta = next_state - state.astype(np.float32)
        gap_signal, mag = self.affect.project_gap(delta)
        next_affect = self.affect.simulate_inject(
            prior_affect=affect_proxy,
            gap_signal=gap_signal,
            gap_magnitude=mag,
            injection_point=InjectionPoint.PROCESSING,
        )

        return SimulationFrame(
            state=next_state,
            affect=next_affect,
            expected_surprise=expected_surprise,
            chain_id=chain_id,
            depth=depth,
            action_kind=action_kind,
        )

    # ---- diagnostic ----

    def layer_stats(self, layer: str) -> dict:
        """Read-only snapshot of one layer's running stats. For tests / debug."""
        s = self._stats[layer]
        return {
            "count":  s.count,
            "mean":   s.mean,
            "stddev": s.stddev,
            "threshold": s.mean + MIN_THRESHOLD * s.stddev if s.count >= COLD_START_N else None,
        }

    # ---- internals ----

    def _score_surprise(self, magnitude: float, layer: str) -> tuple[float, bool]:
        """Cold-start absolute floor while count < COLD_START_N, Welford z
        once stats have accumulated.

        Cold-start (count < 30): magnitude > COLD_START_MAGNITUDE_FLOOR
        registers a synthetic z = MIN_THRESHOLD crossing.

        Warm (count ≥ 30): z = (magnitude − mean) / max(stddev, eps);
        surprise iff magnitude > mean + MIN_THRESHOLD · stddev (equivalent
        to z > MIN_THRESHOLD). Returns the actual z as the score so callers
        get a calibrated number.
        """
        stats = self._stats[layer]
        if stats.count < COLD_START_N:
            if magnitude > COLD_START_MAGNITUDE_FLOOR:
                return MIN_THRESHOLD, True
            return 0.0, False

        std = max(stats.stddev, STDDEV_FLOOR)
        z = (magnitude - stats.mean) / std
        threshold_magnitude = stats.mean + MIN_THRESHOLD * stats.stddev
        return float(z), magnitude > threshold_magnitude
