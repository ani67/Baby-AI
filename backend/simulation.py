"""Simulation + Replay (Component D). Phase 3 minimal.

D is two coupled mechanisms running on the same substrate:

  1. REPLAY — a priority buffer of past surprises. When the system is idle,
     D pulls a high-priority entry, sets A's nudge multiplier to 0.30, and
     re-fires the entry through B with `replay_origin=True`. Welford stats
     stay frozen (B honors the flag); the reaction layer feels it; mood and
     above are nudged at 30% strength so a single trauma can't drag the
     character vector. Each replay produces 0–1 graph mutations through
     C's normal find-or-match path (the embedding may match an existing
     node now where it didn't before, or vice versa).

  2. SIMULATION — chained `B.simulate_step` rollouts that use
     `A.simulate_inject` to feel candidate paths forward without touching
     live state. D proposes action candidates from the active set,
     simulates each forward, scores them against the system's wants
     (character + homeostatic delta), and picks the best.

What exists in Phase 3 minimal:
  - ActionKind, ActionDescriptor, ReplayEntry, CommittedDecisionRecord.
  - ReplayBuffer with priority-eviction and `on_surprise` auto-push from B.
  - replay_loop with attenuated-nudge envelope around each B.observe call.
  - simulate_chain over MAX_SIM_DEPTH_LOCAL=6 frames.
  - propose_candidates from the F-active set (top concepts → ATTEND/INTERROGATE,
    plus always WAIT).
  - score_chain with three terms: alignment-with-wants, confidence (inverse
    of expected_surprise), and action cost.
  - propose_and_choose orchestrator returning an ActionDescriptor.
  - WorldModelMetadata with the homeostatic_target_ema (locked to D per T17).

Deferred:
  - Counterfactual replay of the runner-up.
  - simulate_audience_response (needs G).
  - world_model.simulated_affect_if_visited (specific E need).
  - D.observe_outcome connecting committed decisions back to reality
    (needs the main loop). Methods exist as stubs of structure only when
    they are real callable interfaces — observe_outcome is left out
    entirely until there's something to observe.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from backend.affect import AffectStack, InjectionPoint
from backend.config import N_AFF
from backend.graph import ConceptGraph
from backend.predict import PredictionEngine, PredictionGap, SimulationFrame


# Synthesis-locked constants (subset relevant to Phase 3 D minimal).
REPLAY_CAPACITY                = 4096
REPLAY_NUDGE_GAIN_MULTIPLIER   = 0.30
MAX_SIM_DEPTH_LOCAL            = 6      # per-decision rollout depth (D-side cap)
MAX_SIM_CANDIDATES             = 6
PER_WINDOW_REPLAY_BUDGET       = 64
# Below this priority, an entry is evicted from the buffer after its next
# replay. Each replay decays priority by 0.7; surprise_score=1.5 starts hit
# the floor in ~14 replays. This is what makes "drain the buffer" terminate
# rather than infinite-loop on stale low-priority entries.
REPLAY_PRIORITY_FLOOR          = 0.01

# Action cost hints — synthesis-locked. EXPRESS is structurally costly so
# the mind doesn't initiate speech idly; humans get responses through the
# `force_respond` social-constraint path on MainLoop.cycle, not by biasing
# the architecture's own preferences.
ACTION_COST_HINTS: dict[str, float] = {
    "express":      0.40,
    "attend":       0.10,
    "interrogate":  0.00,
    "wait":         0.00,
    "suppress":     0.20,
}

# Score weights — Phase 3 minimal. Synthesis flags these as empirical.
SCORE_W_ALIGNMENT  = 0.6
SCORE_W_CONFIDENCE = 0.3
SCORE_W_COST       = 0.1


class ActionKind(Enum):
    EXPRESS     = "express"
    ATTEND      = "attend"
    INTERROGATE = "interrogate"
    WAIT        = "wait"
    SUPPRESS    = "suppress"


@dataclass
class ActionDescriptor:
    """The result of propose_and_choose. Returned to the main loop, which
    is responsible for actually carrying out the action (G handles EXPRESS,
    F handles ATTEND, etc.). Phase 3 minimal: action selection is real;
    main-loop dispatch is not yet wired.
    """
    action: ActionKind
    target_concept_id: int | None
    score: float
    chain_id: int
    decision_id: str


@dataclass
class CommittedDecisionRecord:
    """One row in the decision history. E reads these for narrative continuity."""
    decision_id: str
    chosen: ActionDescriptor
    runner_up: ActionDescriptor | None
    chain_summary: list[SimulationFrame]
    affect_at_decision: np.ndarray
    t: float


@dataclass
class ReplayEntry:
    """One past surprise, stored for re-firing during idle windows.

    Phase 3 fields are a strict subset of synthesis's locked structure;
    fields not yet read by anyone (tags bitfield, etc.) are omitted rather
    than stubbed.
    """
    entry_id: str
    original_tick: int
    original_t: float
    actual_repr: np.ndarray             # float32[D_REP]
    affect_snapshot: np.ndarray         # float32[N_AFF] composite at the time
    layer: str
    priority: float
    replay_count: int = 0
    last_replayed_t: float | None = None
    surprise_at_birth: float = 0.0      # diagnostic; the original surprise score


@dataclass
class WorldModelMetadata:
    """D-owned per-synthesis T17. The homeostatic_target_ema lives here
    rather than in A — preserves A's locked five-layer count while letting
    D track "what does this mind feel like, on average" for wants computation.
    """
    homeostatic_target_ema: np.ndarray   # float32[N_AFF]
    homeostatic_alpha: float = 0.001
    last_replay_run_t: float | None = None
    replays_this_window: int = 0
    window_started_t: float | None = None


class SimulationReplay:
    """Component D — owns the replay buffer and the simulation-then-choose pipeline."""

    def __init__(
        self,
        *,
        affect: AffectStack,
        graph: ConceptGraph,
        predict_engine: PredictionEngine,
    ) -> None:
        self.affect = affect
        self.graph = graph
        self.predict_engine = predict_engine

        # Replay buffer — flat dict keyed by entry_id for O(1) ops; priority
        # eviction sorts on demand. At 4096 entries the sort cost is trivial.
        self._buffer: dict[str, ReplayEntry] = {}

        # Decision history; ring-style cap so memory stays bounded.
        self._decisions: list[CommittedDecisionRecord] = []
        self._chain_id_seq: int = 0

        # World-model metadata — homeostatic target seeded at character.
        self.world_model = WorldModelMetadata(
            homeostatic_target_ema=affect.character.vector.astype(np.float32, copy=True),
        )

        # Wire the auto-push hook so B sends every non-replay surprise here.
        self.predict_engine.replay_target = self

    # ---- replay buffer ----

    @property
    def buffer_size(self) -> int:
        return len(self._buffer)

    def on_surprise(
        self,
        gap: PredictionGap,
        actual_repr: np.ndarray,
        affect_snapshot: np.ndarray,
        now: float,
    ) -> ReplayEntry:
        """Called by B when a non-replay observation crosses surprise threshold.

        Priority is the surprise score itself for Phase 3 minimal; later D
        will fold in recency, novelty, and affect magnitude.
        """
        entry = ReplayEntry(
            entry_id=uuid.uuid4().hex,
            original_tick=gap.tick,
            original_t=float(now),
            actual_repr=actual_repr.astype(np.float32, copy=True),
            affect_snapshot=affect_snapshot.astype(np.float32, copy=True),
            layer=gap.layer,
            priority=float(gap.surprise_score),
            surprise_at_birth=float(gap.surprise_score),
        )
        self._buffer[entry.entry_id] = entry
        if len(self._buffer) > REPLAY_CAPACITY:
            self._evict_lowest_priority()
        return entry

    def _evict_lowest_priority(self) -> None:
        if not self._buffer:
            return
        worst_id = min(self._buffer, key=lambda eid: self._buffer[eid].priority)
        del self._buffer[worst_id]

    def replay_loop(self, now: float, max_replays: int = 1) -> list[ReplayEntry]:
        """Replay up to `max_replays` of the highest-priority entries.

        For each entry: install A's nudge multiplier, re-call B.predict
        and B.observe with `replay_origin=True`, restore the multiplier.
        Welford stats stay frozen (B honors the flag). The reaction layer
        feels the replay; the upper layers receive 30% nudge.

        Returns the list of replayed entries (mutated: replay_count, etc.).
        """
        if not self._buffer:
            return []

        # Pick the top-priority entries.
        ordered = sorted(self._buffer.values(), key=lambda e: -e.priority)
        replayed: list[ReplayEntry] = []
        for entry in ordered[:max_replays]:
            self._replay_one(entry, now)
            replayed.append(entry)
        self.world_model.last_replay_run_t = float(now)
        self.world_model.replays_this_window += len(replayed)
        return replayed

    def _replay_one(self, entry: ReplayEntry, now: float) -> None:
        self.affect.set_nudge_gain_multiplier(REPLAY_NUDGE_GAIN_MULTIPLIER)
        try:
            prediction = self.predict_engine.predict(entry.actual_repr, layer=entry.layer)
            self.predict_engine.observe(
                prediction=prediction,
                actual=entry.actual_repr,
                now=now,
                name_hint=f"replay:{entry.entry_id[:8]}",
                replay_origin=True,
            )
        finally:
            self.affect.clear_nudge_gain_multiplier()

        entry.replay_count += 1
        entry.last_replayed_t = float(now)
        # Decay priority each replay so a single high-surprise event doesn't
        # monopolize the buffer's attention forever.
        entry.priority *= 0.7
        # Evict once the entry's priority falls below the floor — this is
        # what lets sleep's "drain the buffer" actually terminate.
        if entry.priority < REPLAY_PRIORITY_FLOOR:
            self._buffer.pop(entry.entry_id, None)

    # ---- simulation chains ----

    def simulate_chain(
        self,
        seed_state: np.ndarray,
        seed_affect: np.ndarray,
        action: ActionKind,
        depth: int = MAX_SIM_DEPTH_LOCAL,
    ) -> list[SimulationFrame]:
        """Roll forward `depth` simulation steps from a starting state under
        a single action. Returns the list of frames (length = depth).
        """
        self._chain_id_seq += 1
        chain_id = self._chain_id_seq

        frames: list[SimulationFrame] = []
        state = seed_state.astype(np.float32, copy=True)
        affect = seed_affect.astype(np.float32, copy=True)
        for d in range(depth):
            frame = self.predict_engine.simulate_step(
                state=state,
                affect_proxy=affect,
                action_kind=action.value,
                chain_id=chain_id,
                depth=d,
            )
            frames.append(frame)
            state = frame.state
            affect = frame.affect
        return frames

    # ---- action selection ----

    def propose_candidates(
        self,
        active_set: dict[int, float],
        current_state: np.ndarray,
    ) -> list[tuple[ActionKind, int | None, np.ndarray]]:
        """Generate up to MAX_SIM_CANDIDATES action candidates from the
        F-supplied active set. WAIT is always present so silence remains
        learnable. ATTEND fans out across the top active concepts. EXPRESS,
        SUPPRESS, INTERROGATE join when their owning components exist.
        """
        candidates: list[tuple[ActionKind, int | None, np.ndarray]] = []

        # Always include WAIT — silence is a real choice.
        candidates.append((ActionKind.WAIT, None, current_state.astype(np.float32, copy=True)))

        ranked = sorted(active_set.items(), key=lambda x: -x[1])

        # ATTEND to top active concepts.
        for cid, _ in ranked:
            if len(candidates) >= MAX_SIM_CANDIDATES:
                break
            if cid not in self.graph.nodes:
                continue
            seed = self.graph.nodes[cid].embedding.astype(np.float32, copy=True)
            candidates.append((ActionKind.ATTEND, cid, seed))

        # EXPRESS the top active concept when its activation is non-trivial.
        # Threshold 0.4 (was 0.7) so freshly-ingested concepts (which seed F
        # at activation=1.0 but spread can dilute) still propose expressing.
        if ranked:
            top_cid, top_activation = ranked[0]
            if top_activation > 0.4 and top_cid in self.graph.nodes:
                if len(candidates) < MAX_SIM_CANDIDATES:
                    seed = self.graph.nodes[top_cid].embedding.astype(np.float32, copy=True)
                    candidates.append((ActionKind.EXPRESS, top_cid, seed))

        return candidates

    def score_chain(
        self,
        frames: list[SimulationFrame],
        wants: np.ndarray,
        action: ActionKind,
    ) -> float:
        """Three-term scoring (Phase 3 minimal):
            alignment   — cosine(simulated final affect, wants)
            confidence  — 1 − mean(expected_surprise across chain)
            cost        — synthesis action-cost hint, subtracted

        wants: the affect direction the system would prefer to feel —
        Phase 3 minimal computes this as `character + (character − homeostatic)`,
        the standard "want what the mind tends to want, pulled toward the gap".
        """
        if not frames:
            return 0.0

        final_affect = frames[-1].affect
        align = _cosine(final_affect, wants)
        align_score = (align + 1.0) / 2.0   # map [-1, 1] → [0, 1]

        avg_surprise = float(np.mean([f.expected_surprise for f in frames]))
        confidence = max(0.0, 1.0 - avg_surprise)

        cost = ACTION_COST_HINTS.get(action.value, 0.0)

        return float(
            SCORE_W_ALIGNMENT  * align_score
            + SCORE_W_CONFIDENCE * confidence
            - SCORE_W_COST       * cost
        )

    def propose_and_choose(
        self,
        active_set: dict[int, float],
        current_state: np.ndarray,
        now: float,
    ) -> tuple[ActionDescriptor, CommittedDecisionRecord]:
        """The full pipeline: propose → simulate × N → score → choose.

        Returns the chosen ActionDescriptor and the CommittedDecisionRecord
        that holds the runner-up and the winning chain for E to inspect.
        """
        # Update the homeostatic EMA toward the current composite.
        composite = self.affect.composite(now)
        a = self.world_model.homeostatic_alpha
        self.world_model.homeostatic_target_ema += a * (composite - self.world_model.homeostatic_target_ema)

        # Compute wants. The mind wants to drift toward its character but
        # also wants to close the gap between where it is and where it
        # typically settles. Simple linear blend of the two pulls.
        wants = self.compute_wants(now)

        # Propose candidates.
        candidates = self.propose_candidates(active_set, current_state)

        # Simulate and score each.
        chains: list[tuple[ActionDescriptor, list[SimulationFrame]]] = []
        for action, target_id, seed in candidates:
            frames = self.simulate_chain(seed_state=seed, seed_affect=composite, action=action)
            score = self.score_chain(frames, wants=wants, action=action)
            descriptor = ActionDescriptor(
                action=action,
                target_concept_id=target_id,
                score=score,
                chain_id=frames[0].chain_id if frames else self._chain_id_seq,
                decision_id=uuid.uuid4().hex,
            )
            chains.append((descriptor, frames))

        # Choose the winner; keep the runner-up for E's introspection.
        chains.sort(key=lambda c: -c[0].score)
        winner_desc, winner_frames = chains[0]
        runner_up_desc = chains[1][0] if len(chains) > 1 else None

        record = CommittedDecisionRecord(
            decision_id=winner_desc.decision_id,
            chosen=winner_desc,
            runner_up=runner_up_desc,
            chain_summary=winner_frames,
            affect_at_decision=composite,
            t=float(now),
        )
        self._decisions.append(record)
        if len(self._decisions) > 256:
            self._decisions.pop(0)

        return winner_desc, record

    # ---- introspection ----

    def recent_surprises(self, k: int = 10) -> list[ReplayEntry]:
        """Top-k highest-priority replay entries — for E's narrative continuity."""
        return sorted(self._buffer.values(), key=lambda e: -e.priority)[:k]

    def committed_decision_history(self, k: int = 10) -> list[CommittedDecisionRecord]:
        return list(self._decisions[-k:])

    # ---- internals ----

    def compute_wants(self, now: float) -> np.ndarray:
        """Phase 3 minimal: wants = character + (character − homeostatic_ema).

        The first term pulls toward the stable signature; the second is the
        "boredom" / regression-to-mean term that makes the system seek what
        it's been missing relative to typical. Both come from existing state;
        no separate "wants" store is invented.
        """
        character = self.affect.character.vector.astype(np.float32)
        homeostatic = self.world_model.homeostatic_target_ema
        # Where the mind tends → where it typically is = the gap it would
        # rather close.
        wants = character + (character - homeostatic)
        n = float(np.linalg.norm(wants))
        if n < 1e-9:
            # Character is degenerate (very early in the mind's life). Fall
            # back to character alone, normalized; if even that is zero, all
            # actions become tied on alignment and only confidence/cost matter.
            return character
        return wants


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float((a @ b) / (na * nb))
