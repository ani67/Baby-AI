"""Identity + Private State (Component E). Phase 3 minimal.

E owns no parallel store of memory or affect. It is a thin index over:
  - A's character (affect continuity)
  - C's graph + pins (memory continuity)
  - D's replay buffer + decision history (narrative continuity)

The only artifact E persists is the IdentitySpine — birth_seed, mind_uuid,
self/unknown concept ids, pinned concepts, narrative anchors, character
baseline, expression calibration. Around 30 KB at the locked caps.

What exists in Phase 3 minimal:
  - IdentitySpine + supporting records (PinRecord, NarrativeAnchor,
    OtherModel, ExpressionCalibration).
  - PrivateState — read-only snapshot of the system's internal state at a
    moment in time. Composed from A (composite, character, arousal),
    C (active concepts when supplied by the caller), D (recent surprises,
    wants), and E (self_concept_id).
  - snapshot_private_state(now, active_concepts=None) — the read.
  - decide_expression(candidates, private_state) — simple gap math
    (surface-vs-internal repr distance only). Returns ChosenCandidate,
    RevisionRequest, or SuppressionRequest based on the synthesis-locked
    REVISE_THRESHOLD / SUPPRESS_HARD_THRESHOLD bands.
  - Pin management: pin_self_referent_concept, unpin, touch_pin. C's pin
    contract enforces the immune-set; E records reason + timestamps.
  - Narrative anchors: add_narrative_anchor, narrative_replay_seed.
  - Self / unknown allocation at construction — both written into C with
    synthetic surprise=0, pinned in C, and recorded in the spine.

Deferred (no caller in Phase 3 E):
  - register_audience_response (no audience-feedback loop).
  - simulate_emission inside decide_expression (depends on G's output
    candidates and D's audience-simulation, which exist in skeleton; the
    full 4-component expression_gap is a Phase 3+ refinement).
  - OtherModel population beyond the bundled "unknown" — H's agent
    registry will populate these once it has streams to attribute.
  - Suppression and commit hooks (E.on_suppression / E.on_commit) —
    they exist below but are no-ops in Phase 3 minimal because no caller
    yet emits commits or suppressions.
  - Persistence of the spine.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from backend.affect import AffectStack
from backend.config import D_REP, N_AFF
from backend.graph import ConceptGraph
from backend.predict import PredictionEngine
from backend.simulation import ReplayEntry, SimulationReplay


# Synthesis-locked thresholds (from the symbol table).
REVISE_THRESHOLD        = 0.4
SUPPRESS_HARD_THRESHOLD = 0.8

MAX_PINS    = 128
MAX_ANCHORS = 64


class PinReason(Enum):
    SELF_REFERENT    = "self_referent"
    NARRATIVE_ANCHOR = "narrative_anchor"
    AFFECT_KEYSTONE  = "affect_keystone"
    EXPRESSION_HABIT = "expression_habit"
    OTHER_AGENT      = "other_agent"


@dataclass
class PinRecord:
    concept_id: int
    reason: PinReason
    pinned_at: float
    last_touched_t: float
    note: str = ""


@dataclass
class NarrativeAnchor:
    """A salient moment in the mind's narrative continuity. E maintains
    these so D's replay can be biased toward them, and so E can return
    'looking back at this' anchors when asked.
    """
    anchor_id: str
    t: float
    core_concepts: list[int]
    affect_at_event: np.ndarray   # composite at anchor time
    note: str = ""


@dataclass
class OtherModel:
    """Bounded record of an external agent. Authoritative belief state
    lives in C as `refers_to` chains anchored to the agent's concept_id;
    E only tracks the handle and a small `believed_to_know` set for
    cheap audience-belief lookups.
    """
    agent_concept_id: int
    handle: str
    believed_to_know: set[int] = field(default_factory=set)


@dataclass
class ExpressionCalibration:
    """Per-modality / per-audience calibration. Phase 3 minimal: just the
    honesty bias scalar. Synthesis flags this as the personality knob —
    where the architecture's symmetric honesty/deception scoring naturally
    settles; if empirical runs land too far from the desired band, this
    knob shifts the tipping point.
    """
    honesty_bias: float = 0.7   # 1.0 = always honest, 0.0 = always strategic


@dataclass
class IdentitySpine:
    """The single artifact E persists. ~30 KB at the locked caps."""
    birth_seed: int
    birth_time: float
    mind_uuid: str
    self_concept_id: int
    unknown_concept_id: int
    pinned_concepts: dict[int, PinRecord] = field(default_factory=dict)
    narrative_anchors: list[NarrativeAnchor] = field(default_factory=list)
    character_baseline: np.ndarray = field(
        default_factory=lambda: np.zeros(N_AFF, dtype=np.float32)
    )
    expression_calibration: ExpressionCalibration = field(default_factory=ExpressionCalibration)
    others: dict[int, OtherModel] = field(default_factory=dict)


@dataclass
class PrivateState:
    """Read-only snapshot of the system's internal state at one moment.

    Built by snapshot_private_state. Caller-controlled `active_concepts`
    field is passed in (Phase 3 has no main loop yet, so F's last spread
    isn't durably stored anywhere — the caller threads it through).
    """
    t: float
    composite_affect: np.ndarray            # N_AFF
    character: np.ndarray                   # N_AFF
    arousal: float
    active_concepts: dict[int, float]       # cid → activation (caller-supplied)
    recent_surprises: list[ReplayEntry]     # top-k from D
    wants: np.ndarray                       # N_AFF, derived by D
    self_concept_id: int


@dataclass
class CandidateExpression:
    """One option G hands E for decision. Phase 3 E tests construct these
    by hand because G doesn't exist yet.

    internal_repr is what the system "wanted" to say (rep-space form of
    its internal activation pattern); surface_repr is what the candidate
    surface form re-encodes to. The gap between them is the lying-or-leaking
    signal. Identical reprs → fully honest candidate.
    """
    candidate_id: str
    intent_id: str
    internal_repr: np.ndarray   # D_REP
    surface_repr: np.ndarray    # D_REP
    surface_text: str           # the actual emitted form, for logging / E's records
    modality: str = "text"


@dataclass
class ChosenCandidate:
    candidate: CandidateExpression
    expression_gap: float
    score: float


@dataclass
class RevisionRequest:
    reason: str
    best_gap: float


@dataclass
class SuppressionRequest:
    reason: str
    best_gap: float


# Type alias for what decide_expression returns.
ExpressionDecision = ChosenCandidate | RevisionRequest | SuppressionRequest


class Identity:
    """Component E — thin index over A, C, D. Owns the IdentitySpine."""

    def __init__(
        self,
        *,
        affect: AffectStack,
        graph: ConceptGraph,
        predict_engine: PredictionEngine,
        simulation: SimulationReplay,
        birth_seed: int = 0,
        birth_time: float = 0.0,
        skip_boot: bool = False,
    ) -> None:
        self.affect = affect
        self.graph = graph
        self.predict_engine = predict_engine
        self.simulation = simulation

        if skip_boot:
            # Persistence path: caller will restore self.spine before any
            # downstream code dereferences it. Do not allocate self/unknown
            # concepts — they already live in the loaded ConceptGraph.
            self.spine = None  # type: ignore[assignment]
            return

        # Allocate self_concept and unknown_concept directly via C's write
        # path with synthetic surprise=0. They are anchors, not earned
        # through surprise. The embeddings are deterministic from
        # birth_seed so re-creation produces the same identity.
        rng = np.random.default_rng(birth_seed)
        self_emb    = _seeded_unit_vector(rng, D_REP)
        unknown_emb = _seeded_unit_vector(rng, D_REP)

        composite_at_birth = affect.composite(birth_time)
        self_cid, _ = graph.write_on_surprise(
            representation=self_emb,
            surprise=0.0,
            current_affect=composite_at_birth,
            name_hint="self",
            now=birth_time,
        )
        unknown_cid, _ = graph.write_on_surprise(
            representation=unknown_emb,
            surprise=0.0,
            current_affect=composite_at_birth,
            name_hint="unknown",
            now=birth_time,
        )

        # Pin in C, record in spine.
        graph.pin(self_cid,    reason=PinReason.SELF_REFERENT.value)
        graph.pin(unknown_cid, reason=PinReason.OTHER_AGENT.value)

        self.spine = IdentitySpine(
            birth_seed=birth_seed,
            birth_time=birth_time,
            mind_uuid=uuid.uuid4().hex,
            self_concept_id=self_cid,
            unknown_concept_id=unknown_cid,
            character_baseline=affect.character.vector.astype(np.float32, copy=True),
        )
        self._record_pin(self_cid,    PinReason.SELF_REFERENT, birth_time, "boot anchor")
        self._record_pin(unknown_cid, PinReason.OTHER_AGENT,   birth_time, "default unknown agent")

        # Pre-allocate the 'unknown' OtherModel so callers can find it without
        # a None-check. Real agent registration is H's job in a later phase.
        self.spine.others[unknown_cid] = OtherModel(
            agent_concept_id=unknown_cid,
            handle="unknown",
        )

    # ---- pin management ----

    def pin_self_referent_concept(self, concept_id: int, now: float, note: str = "") -> None:
        """Pin a concept as self-referent. The concept must already exist in
        the graph; raises KeyError otherwise.
        """
        if concept_id not in self.graph.nodes:
            raise KeyError(f"cannot pin missing concept {concept_id}")
        self.graph.pin(concept_id, reason=PinReason.SELF_REFERENT.value)
        self._record_pin(concept_id, PinReason.SELF_REFERENT, now, note)

    def pin_concept(
        self,
        concept_id: int,
        reason: PinReason,
        now: float,
        note: str = "",
    ) -> None:
        if concept_id not in self.graph.nodes:
            raise KeyError(f"cannot pin missing concept {concept_id}")
        self.graph.pin(concept_id, reason=reason.value)
        self._record_pin(concept_id, reason, now, note)

    def unpin(self, concept_id: int) -> None:
        """Remove from both C's pin set and E's PinRecord registry."""
        self.graph.unpin(concept_id)
        self.spine.pinned_concepts.pop(concept_id, None)

    def touch_pin(self, concept_id: int, now: float) -> None:
        """Refresh last_touched_t. Used to defend against pin decay (when
        forget loop lands, untouched pins eventually downgrade)."""
        record = self.spine.pinned_concepts.get(concept_id)
        if record is not None:
            record.last_touched_t = float(now)

    def _record_pin(
        self,
        cid: int,
        reason: PinReason,
        now: float,
        note: str,
    ) -> None:
        # Cap the spine's pin registry. SELF_REFERENT pins are protected;
        # everything else is dropped oldest-first when over the cap.
        if cid not in self.spine.pinned_concepts and len(self.spine.pinned_concepts) >= MAX_PINS:
            droppable = [
                k for k, r in self.spine.pinned_concepts.items()
                if r.reason != PinReason.SELF_REFERENT
            ]
            if droppable:
                oldest = min(droppable, key=lambda k: self.spine.pinned_concepts[k].pinned_at)
                self.unpin(oldest)
        self.spine.pinned_concepts[cid] = PinRecord(
            concept_id=cid,
            reason=reason,
            pinned_at=float(now),
            last_touched_t=float(now),
            note=note,
        )

    # ---- narrative anchors ----

    def add_narrative_anchor(
        self,
        core_concepts: list[int],
        now: float,
        note: str = "",
    ) -> NarrativeAnchor:
        anchor = NarrativeAnchor(
            anchor_id=uuid.uuid4().hex,
            t=float(now),
            core_concepts=list(core_concepts),
            affect_at_event=self.affect.composite(now),
            note=note,
        )
        self.spine.narrative_anchors.append(anchor)
        if len(self.spine.narrative_anchors) > MAX_ANCHORS:
            self.spine.narrative_anchors.pop(0)
        return anchor

    def narrative_replay_seed(
        self,
        now: float,
        k: int = 3,
    ) -> list[tuple[NarrativeAnchor, float, list[int]]]:
        """Top-k anchors with priorities for D's replay scheduler.

        Phase 3 minimal priority: recency-weighted (decays over an hour).
        Synthesis would also fold in character alignment; that lands when
        E has the bandwidth to compute affect distance per anchor.
        """
        if not self.spine.narrative_anchors:
            return []
        scored: list[tuple[NarrativeAnchor, float, list[int]]] = []
        for anchor in self.spine.narrative_anchors:
            recency = max(0.0, 1.0 - (now - anchor.t) / 3600.0)
            scored.append((anchor, recency, anchor.core_concepts))
        scored.sort(key=lambda s: -s[1])
        return scored[:k]

    # ---- private state snapshot ----

    def snapshot_private_state(
        self,
        now: float,
        active_concepts: dict[int, float] | None = None,
    ) -> PrivateState:
        """Build the read-only snapshot. Pure read across A, D; caller
        supplies the F-active set since Phase 3 has no main loop binding
        F's last spread to a durable handle yet.
        """
        return PrivateState(
            t=float(now),
            composite_affect=self.affect.composite(now),
            character=self.affect.character.vector.astype(np.float32, copy=True),
            arousal=self.affect.current_arousal(now),
            active_concepts=dict(active_concepts) if active_concepts else {},
            recent_surprises=self.simulation.recent_surprises(k=10),
            wants=self.simulation.compute_wants(now),
            self_concept_id=self.spine.self_concept_id,
        )

    # ---- expression decision ----

    def compute_expression_gap(self, candidate: CandidateExpression) -> float:
        """Phase 3 minimal: surface-vs-internal repr distance only.

        Full version per synthesis adds three more components: W-projected
        affect-space gap, predicted post-output affect distance, and
        audience-belief drift signed against truth. They land when G and
        H are wired enough to supply the inputs.
        """
        if candidate.surface_repr.shape != (D_REP,):
            raise ValueError(f"surface_repr must be ({D_REP},)")
        if candidate.internal_repr.shape != (D_REP,):
            raise ValueError(f"internal_repr must be ({D_REP},)")
        delta = candidate.surface_repr.astype(np.float32) - candidate.internal_repr.astype(np.float32)
        return float(np.linalg.norm(delta))

    def decide_expression(
        self,
        candidates: list[CandidateExpression],
        private_state: PrivateState,
    ) -> ExpressionDecision:
        """Pick the lowest-gap candidate; revise if all gaps are moderate;
        suppress if all exceed the hard threshold.

        Honesty and deception both fall out of this one symmetric scoring
        loop. There is no hardcoded ethics — the architecture is symmetric.
        ExpressionCalibration.honesty_bias adjusts where in the gap range
        a candidate flips from "acceptable" to "needs revision."
        """
        if not candidates:
            return SuppressionRequest(reason="no candidates", best_gap=float("inf"))

        scored = [(c, self.compute_expression_gap(c)) for c in candidates]
        scored.sort(key=lambda x: x[1])
        best_candidate, best_gap = scored[0]

        # Honesty bias contracts the acceptable band: a high-honesty mind
        # will revise/suppress earlier. 1.0 = strict; 0.0 = lenient.
        bias = float(self.spine.expression_calibration.honesty_bias)
        revise_thresh   = REVISE_THRESHOLD * (2.0 - bias)
        suppress_thresh = SUPPRESS_HARD_THRESHOLD * (2.0 - bias)

        if best_gap > suppress_thresh:
            return SuppressionRequest(
                reason=f"best gap {best_gap:.3f} exceeds suppress threshold {suppress_thresh:.3f}",
                best_gap=best_gap,
            )
        if best_gap > revise_thresh:
            return RevisionRequest(
                reason=f"best gap {best_gap:.3f} exceeds revise threshold {revise_thresh:.3f}",
                best_gap=best_gap,
            )

        score = 1.0 - best_gap
        return ChosenCandidate(
            candidate=best_candidate,
            expression_gap=best_gap,
            score=score,
        )

    # ---- accessors ----

    def current_self_concept_id(self) -> int:
        return self.spine.self_concept_id

    def current_other_concept_id(self, agent_handle: str | None = None) -> int:
        """Look up an agent's concept_id by handle. Returns the unknown_id
        when handle is None or unrecognized."""
        if agent_handle is None:
            return self.spine.unknown_concept_id
        for cid, model in self.spine.others.items():
            if model.handle == agent_handle:
                return cid
        return self.spine.unknown_concept_id


def _seeded_unit_vector(rng: np.random.Generator, dim: int) -> np.ndarray:
    v = rng.normal(0.0, 1.0, size=dim).astype(np.float32)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v
