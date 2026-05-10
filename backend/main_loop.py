"""Main loop — the runtime that ties A, B, C, D, E, F, G, H together.

Synthesis boot sequence describes a 50 Hz tick:
    H.tick → for each new Stimulus → F.attend → F.processing_loop →
    D.propose_and_choose → (if EXPRESS) E.snapshot + G.request_expression
    → on commit: H.emit_output (delivers + INTERNAL_OUTPUT_ECHO via inject_internal)
    → if idle: D.replay_loop

Phase 3 minimal MainLoop is synchronous: callers provide a stimulus or a
"now" timestamp and one cycle runs to completion. No real-time scheduler,
no IngestQueue draining (H ingests synchronously and returns a result the
loop can act on directly). F.processing_loop is deferred — Phase 3
runs a single F.attend(INPUT) per cycle.

What exists:
  - MainLoop.cycle(ingest_result, now): runs F.attend(INPUT) seeded from
    the just-ingested stimulus's matched/written concept; calls
    D.propose_and_choose against the active set; if D picked EXPRESS,
    builds an ExpressionIntent from the active set + ingest representation,
    calls G.request_expression, and on ChosenCandidate commits the surface
    via H.emit_output (which delivers + injects the echo back through B).
  - MainLoop.idle(now): runs D.replay_loop with bounded budget when the
    system has nothing fresh.
  - CycleResult and IdleResult — read-only records the test harness uses
    to verify what happened in one tick.

Deferred:
  - F.processing_loop multi-tick settle (single F.attend for now).
  - Real time scheduling / 50 Hz dispatch.
  - Action carry-out for non-EXPRESS actions (ATTEND would re-prime F's
    next attend; INTERROGATE would synthesize a follow-up question;
    SUPPRESS would record into the replay buffer with a SUPPRESSED tag).
    Phase 3 loop honors D's choice but only EXPRESS has a downstream
    effect (the path that needs wiring to demonstrate end-to-end).
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from backend.active_inference import ActiveInference
from backend.affect import AffectStack
from backend.attention import Attention, AttentionFrame, AttentionPhase
from backend.expression import Expression, ExpressionIntent
from backend.config import R_MATCH
from backend.graph import ConceptGraph, EdgeType, SpreadResult
from backend.identity import (
    CandidateExpression,
    ChosenCandidate,
    ExpressionDecision,
    Identity,
    RevisionRequest,
    SuppressionRequest,
)
from backend.input import IngestResult, InputPipeline, Modality
from backend.predict import PredictionEngine
from backend.simulation import (
    ActionDescriptor,
    ActionKind,
    CommittedDecisionRecord,
    SimulationReplay,
)


@dataclass
class CycleResult:
    """Read-only record of one F→D→(G) cycle. The harness inspects this
    to verify the wiring did what it should.
    """
    stimulus_id: int
    attention_frame: AttentionFrame
    spread: SpreadResult
    action: ActionDescriptor
    decision_record: CommittedDecisionRecord
    expression_decision: ExpressionDecision | None
    emitted_surface: str | None
    echo_result: IngestResult | None
    # Phase 5: snapshots of the active set at the input phase and after
    # F.processing_loop, so callers can verify migration of activation
    # away from the seed and toward associated concepts.
    input_active_set: dict[int, float] | None = None
    processed_active_set: dict[int, float] | None = None


@dataclass
class IdleResult:
    replayed_count: int
    now: float
    # Phase 7+: counts emitted by ActiveInference.run_inference_cycle.
    # `inference_events` is the number of pairs probed (passed
    # eligibility + not-already-connected gating); `new_connections`
    # is the subset where observe() actually wrote a fresh concept
    # (was_new_write=True, which also auto-links it). Both default to
    # 0 so existing call sites that don't run active inference (and
    # the JSON shape of /idle when run_inference is disabled) stay
    # behavior-identical.
    inference_events: int = 0
    new_connections: int = 0


@dataclass
class SleepResult:
    """Outcome of one sleep envelope. duration_actual is wall-clock seconds
    measured by perf_counter; replays_fired counts entries replayed (each
    entry can be replayed multiple times); abstractions_formed counts new
    abstract parent nodes written during the abstraction-formation pass.
    """
    replays_fired: int
    abstractions_formed: int
    duration_actual: float


# Sleep tunables — Phase 4 minimal defaults.
SLEEP_REPLAYS_PER_BURST       = 8        # how many entries to replay per loop iteration
ABSTRACTION_MIN_MEMBERS       = 3        # cluster size threshold (per user spec)
ABSTRACTION_PARENT_WEIGHT     = 0.7      # is_a edge weight from member → parent
# ABSTRACTION_DENSITY_THRESHOLD removed — the BFS-on-similar_to algorithm
# it gated cannot form abstractions at AUTO_LINK_K=1 (single giant
# component, density 0.0008, 600× below 0.5). Replaced with vector-space
# clustering on embeddings via MiniBatchKMeans.


class MainLoop:
    """The runtime glue. Holds references to every component; owns no
    state beyond bookkeeping (last_observation_t, cycle counter)."""

    def __init__(
        self,
        *,
        affect: AffectStack,
        graph: ConceptGraph,
        predict_engine: PredictionEngine,
        simulation: SimulationReplay,
        identity: Identity,
        attention: Attention,
        expression: Expression,
        input_pipeline: InputPipeline,
    ) -> None:
        self.affect = affect
        self.graph = graph
        self.predict_engine = predict_engine
        self.simulation = simulation
        self.identity = identity
        self.attention = attention
        self.expression = expression
        self.input_pipeline = input_pipeline

        self.last_observation_t: float = identity.spine.birth_time
        self.cycle_count: int = 0

        # Phase 7 workstream B: bind Attention into Expression so that
        # Expression.generate_extended can spread from each sentence's
        # self-overhearing echo into the next sentence's intent. Done here
        # rather than in Expression's ctor so the existing Expression
        # constructor signature stays stable across persistence reload.
        self.expression.set_attention(self.attention)

        # Active inference: the mind generates its own questions during
        # idle by composing bridge representations between concepts that
        # are currently in awareness (high activation_count). Constructed
        # here rather than in the ctor signature so the existing wiring
        # in tests, persistence, and run_curriculum stays unchanged.
        self.active_inference = ActiveInference(
            graph=self.graph,
            predict_engine=self.predict_engine,
            affect=self.affect,
        )

    # ---- one cycle (run after each ingest) ----

    def cycle(
        self,
        ingest_result: IngestResult,
        now: float,
        *,
        force_respond: bool = False,
        skip_simulation: bool = False,
    ) -> CycleResult:
        """Run F → D → (G if EXPRESS) for one freshly-ingested stimulus.

        `force_respond` (Phase 4 social constraint): when True and D's
        choice is WAIT, the action is overridden to EXPRESS targeting the
        top active concept. The mind keeps its own preferences (D ran
        normally and chose WAIT — that's preserved in the decision_record)
        but a social pressure ("a human just spoke to you") flips the
        outward act. This does not change the system's internal state
        evaluation, only what it does next.

        `skip_simulation` (Phase 6 fast-ingestion): when True, skip the
        F.processing_loop, D.propose_and_choose, the EXPRESS path (G), and
        H.emit_output entirely. Only the perception path runs — the
        social-constraint write, auto-link, and one F.attend(INPUT) — so
        the input is absorbed into the graph and B's affect/Welford state
        but no action is selected and no surface is emitted. force_respond
        is ignored in this mode (no expression to force). Used by
        run_curriculum.py's book step where force_respond is False
        anyway and D's simulate_chain rollouts are wasted work.
        """
        self.cycle_count += 1
        self.last_observation_t = float(now)

        # 0. Social-constraint write. When a human directly addressed the
        #    mind (force_respond=True) and surprise didn't fire — typically
        #    because Welford has calibrated to the recent magnitude regime —
        #    write the input anyway. The mind records what humans tell it
        #    even when its own surprise threshold has habituated. find_or_match
        #    inside write_on_surprise still dedupes against existing
        #    near-duplicates, so this doesn't bloat the graph.
        if force_respond and ingest_result.gap.concept_id is None:
            cid, is_new = self.graph.write_on_surprise(
                representation=ingest_result.stimulus.representation,
                surprise=float(ingest_result.gap.magnitude),
                current_affect=self.affect.composite(now),
                name_hint=ingest_result.stimulus.name_hint,
                now=now,
            )
            ingest_result.gap.concept_id = cid
            ingest_result.gap.was_new_write = is_new

        # 0a. Auto-link freshly-written concepts to their cosine-nearest
        #     neighbors via similar_to. Without this, every fresh concept
        #     is structurally isolated and F.spread has nothing to bring in.
        if ingest_result.gap.was_new_write and ingest_result.gap.concept_id is not None:
            # top_k / min_cosine fall through to AUTO_LINK_K and
            # AUTO_LINK_MIN_COSINE in config (Phase 7 perf fix).
            self.graph.link_to_nearest_neighbors(
                ingest_result.gap.concept_id,
                now=now,
            )

        # 1. F.attend(INPUT). Seed the spread from the concept H just
        #    matched/wrote. If no concept was bound (gap below surprise
        #    threshold and no prior match), seed from the bare
        #    representation's nearest neighbor instead — the active set
        #    should reflect what the input made the system think of.
        seeds: dict[int, float] = {}
        if ingest_result.gap.concept_id is not None:
            seeds[ingest_result.gap.concept_id] = 1.0
        else:
            nearest = self.graph.nearest(ingest_result.stimulus.representation)
            if nearest is not None:
                cid, sim = nearest
                if sim > 0.0:
                    seeds[cid] = float(sim)

        spread, frame = self.attention.attend(
            phase=AttentionPhase.INPUT,
            raw_seeds=seeds,
            now=now,
        )
        input_active_set = dict(spread.active_set)

        # Fast-ingestion early exit. Phase 6: book ingestion runs at high
        # throughput; D.propose_and_choose's chained simulate_step rollouts
        # cost ~60% of cycle time and produce nothing observable when
        # force_respond is False. F.processing_loop and the EXPRESS path
        # are also wasted work for passive reading.
        if skip_simulation:
            stub_action = ActionDescriptor(
                action=ActionKind.WAIT,
                target_concept_id=None,
                score=0.0,
                chain_id=0,
                decision_id="skip_simulation",
            )
            stub_record = CommittedDecisionRecord(
                decision_id="skip_simulation",
                chosen=stub_action,
                runner_up=None,
                chain_summary=[],
                affect_at_decision=self.affect.composite(now),
                t=float(now),
            )
            return CycleResult(
                stimulus_id=ingest_result.stimulus.stimulus_id,
                attention_frame=frame,
                spread=spread,
                action=stub_action,
                decision_record=stub_record,
                expression_decision=None,
                emitted_surface=None,
                echo_result=None,
                input_active_set=input_active_set,
                processed_active_set=None,
            )

        # 1a. Processing loop (Phase 5). 4 internal PROCESSING-mode spread
        #     ticks; each decays the current actives by 0.85 and re-spreads,
        #     so input-concept dominance fades from 1.0 toward ~0.5 while
        #     associated concepts accumulate. Final active set is capped at
        #     0.5 max activation — the input is evidence, not mental state.
        processed_active_set = self.attention.processing_loop(
            input_active_set=input_active_set,
            now=now,
        )

        # 2. D.propose_and_choose. Active set drives candidate generation;
        #    current_state is the representation of the stimulus that
        #    triggered this cycle, so simulation rollouts start from "what
        #    just landed."
        action, decision_record = self.simulation.propose_and_choose(
            active_set=processed_active_set or input_active_set,
            current_state=ingest_result.stimulus.representation,
            now=now,
        )

        # 2a. Social-constraint override. When a human directly addressed
        # the mind (force_respond=True from /ingest with agent_handle), and
        # D's natural choice was WAIT, override to EXPRESS targeting the
        # strongest concept in the *processed* active set. The decision
        # record retains D's original choice — only the outward act flips.
        active_for_action = processed_active_set or input_active_set
        if force_respond and action.action is ActionKind.WAIT and active_for_action:
            top_cid = max(active_for_action, key=lambda c: active_for_action[c])
            action = ActionDescriptor(
                action=ActionKind.EXPRESS,
                target_concept_id=top_cid,
                score=action.score,            # carry D's own score for the record
                chain_id=action.chain_id,
                decision_id=action.decision_id,
            )

        # 3. EXPRESS path. Build an ExpressionIntent from F's active set
        #    and the ingest representation, ask G for a decision, and
        #    on ChosenCandidate commit via H.emit_output (which logs +
        #    re-injects the surface as INTERNAL_OUTPUT_ECHO).
        expression_decision: ExpressionDecision | None = None
        emitted_surface: str | None = None
        echo_result: IngestResult | None = None

        if action.action is ActionKind.EXPRESS:
            # internal_repr = weighted centroid of the *processed* active set
            # (post-processing-ticks). Now the centroid reflects the broader
            # mental state with the input concept down-weighted to ≤ 0.5,
            # giving G's candidates a real chance to land closer to a
            # neighbor's surface than to a verbatim echo.
            active_for_express = processed_active_set or input_active_set
            internal_repr = self._active_set_centroid(active_for_express)
            if internal_repr is None:
                internal_repr = ingest_result.stimulus.representation.astype(np.float32, copy=True)
            intent = ExpressionIntent(
                intent_id=action.decision_id,
                internal_repr=internal_repr,
                active_concepts=dict(active_for_express),
                now=now,
                audience_concept_id=action.target_concept_id,
            )

            # Phase 7 workstream B: length signal. budget=1 retains the
            # legacy single-sentence path (request_expression + emit_output)
            # so existing tests and force_respond=True minimal-arousal
            # cases stay byte-identical. budget>1 drops into the iterative
            # generate_extended which emits per-sentence echoes internally
            # and returns the list of surfaces; MainLoop joins and emits
            # the combined utterance once for the world delivery / UI.
            budget = self.compute_expression_budget(active_for_express, now)

            if budget <= 1:
                expression_decision = self.expression.request_expression(intent)
                if isinstance(expression_decision, ChosenCandidate):
                    emitted_surface = expression_decision.candidate.surface_text
                    echo_result = self.input_pipeline.emit_output(
                        modality=Modality.TEXT,
                        surface=emitted_surface,
                        now=now,
                        parent_stimulus_id=ingest_result.stimulus.stimulus_id,
                    )
            else:
                sentences = self.expression.generate_extended(
                    intent,
                    budget=budget,
                    parent_stimulus_id=ingest_result.stimulus.stimulus_id,
                )
                if sentences:
                    emitted_surface = " ".join(sentences)
                    # The world delivery: one combined utterance. Each
                    # sentence's per-step INTERNAL_OUTPUT_ECHO already
                    # fired inside generate_extended; this final emit
                    # delivers to emitted_log and produces one more echo
                    # of the joined surface (the system "remembers"
                    # what it just said as a unified statement).
                    echo_result = self.input_pipeline.emit_output(
                        modality=Modality.TEXT,
                        surface=emitted_surface,
                        now=now,
                        parent_stimulus_id=ingest_result.stimulus.stimulus_id,
                    )
                    # Synthesize a ChosenCandidate-shaped record for the
                    # CycleResult so downstream consumers (api.py, frontend
                    # ConversationLog) get a "Chosen"-type decision name
                    # for the joined utterance. The candidate's surface_repr
                    # equals internal_repr (joined response is self-honest:
                    # the system's internal centroid maps directly to what
                    # it said, with sentence-by-sentence gaps already
                    # absorbed into per-sentence echoes inside the loop).
                    expression_decision = ChosenCandidate(
                        candidate=CandidateExpression(
                            candidate_id=f"{intent.intent_id}/extended",
                            intent_id=intent.intent_id,
                            internal_repr=intent.internal_repr,
                            surface_repr=intent.internal_repr,
                            surface_text=emitted_surface,
                            modality="text",
                        ),
                        expression_gap=0.0,
                        score=float(len(sentences)),
                    )
                else:
                    # generate_extended returned []: every iteration
                    # bailed (Revision / Suppression / unknown). Fall
                    # through to a single-shot request so a budget>1
                    # turn never produces a worse outcome than budget=1.
                    # The cycle's expression_decision faithfully captures
                    # whichever path actually landed.
                    expression_decision = self.expression.request_expression(intent)
                    if isinstance(expression_decision, ChosenCandidate):
                        emitted_surface = expression_decision.candidate.surface_text
                        echo_result = self.input_pipeline.emit_output(
                            modality=Modality.TEXT,
                            surface=emitted_surface,
                            now=now,
                            parent_stimulus_id=ingest_result.stimulus.stimulus_id,
                        )

        return CycleResult(
            stimulus_id=ingest_result.stimulus.stimulus_id,
            attention_frame=frame,
            spread=spread,
            action=action,
            decision_record=decision_record,
            expression_decision=expression_decision,
            emitted_surface=emitted_surface,
            echo_result=echo_result,
            input_active_set=input_active_set,
            processed_active_set=processed_active_set,
        )

    # ---- expression budget (Phase 7 — workstream B) ----

    def compute_expression_budget(
        self,
        active_set: dict[int, float],
        now: float,
    ) -> int:
        """Decide how many sentences the mind has to say in this turn.

        Coupling: more aroused + a richer active set → more to say. The
        thresholds are spec-locked (Phase 7 workstream B step 1):

            arousal > 0.6  and  |active| > 30   →  5    paragraph
            arousal > 0.3  and  |active| > 15   →  3    multi-sentence
            arousal > 0.1  and  |active| >  8   →  2    two-sentence thought
            otherwise                            →  1    minimal

        Notes
        -----
        - The budget is consumed by Expression.generate_extended in the
          EXPRESS branch of `cycle`; budget=1 falls back to the existing
          single-sentence path so the runtime contract stays identical
          for the calm / sparse case.
        - This is a length signal, not an importance signal — it asks
          "how much is the mind currently holding," not "how badly does
          it want to talk."
        """
        arousal = float(self.affect.current_arousal(now))
        set_size = len(active_set)

        if arousal > 0.6 and set_size > 30:
            return 5
        if arousal > 0.3 and set_size > 15:
            return 3
        if arousal > 0.1 and set_size > 8:
            return 2
        return 1

    # ---- idle (run when nothing fresh has arrived) ----

    def idle(
        self,
        now: float,
        max_replays: int = 1,
        *,
        run_inference: bool = True,
    ) -> IdleResult:
        """Idle-window handling: D.replay_loop bounded + active inference.

        Synthesis idle conditions: no recent_observation_t in last 5 s
        AND arousal < 0.6 AND replays_this_window < PER_WINDOW_REPLAY_BUDGET.
        Phase 3 minimal MainLoop checks the time-since-last-observation
        and arousal; the per-window budget is honored by D.replay_loop's
        internal bookkeeping.

        run_inference (default True): also run ActiveInference.run_inference_cycle
        to give the mind a chance to compose its own bridge representations
        between currently-active concepts. Disabling this preserves the
        pre-active-inference IdleResult shape (counts default to 0). Replay
        is independent — it always fires regardless of run_inference.
        """
        if (now - self.last_observation_t) < 5.0:
            return IdleResult(replayed_count=0, now=float(now))
        if self.affect.current_arousal(now) >= 0.6:
            return IdleResult(replayed_count=0, now=float(now))

        replayed = self.simulation.replay_loop(now=now, max_replays=max_replays)

        inference_events = 0
        new_connections = 0
        if run_inference:
            inference_events, new_connections = (
                self.active_inference.run_inference_cycle(now=now)
            )

        return IdleResult(
            replayed_count=len(replayed),
            now=float(now),
            inference_events=inference_events,
            new_connections=new_connections,
        )

    # ---- sleep (Phase 4 — explicit consolidation pass) ----

    def sleep(self, now: float, duration_seconds: float) -> SleepResult:
        """Run a consolidation pass under the 1.5× nudge envelope.

        Drains the replay buffer (in PER_BURST chunks) until the buffer is
        empty or the wall-clock budget is exhausted. Then runs abstraction
        formation: connected components in the similar_to subgraph with
        ≥ ABSTRACTION_MIN_MEMBERS=3 nodes and mutual edge density ≥ 0.5
        promote to a new abstraction node, with `is_a` edges from each
        member to the parent. Concepts already abstracted (sources of an
        existing is_a edge) are excluded from new clusters; existing
        abstraction parents (targets of any is_a edge) likewise.

        The 1.5× envelope OVERRIDES the per-replay 0.30× attenuation —
        sleep is the time when replays shape character at full strength,
        not the time when each replay is muted.
        """
        if duration_seconds <= 0:
            raise ValueError(f"duration_seconds must be > 0, got {duration_seconds}")

        start = time.perf_counter()
        replays_fired = 0
        abstractions_formed = 0

        self.affect.begin_consolidation(multiplier=1.5)
        try:
            # Drain the replay buffer in bursts, honoring the wall-clock budget.
            replay_now = float(now)
            while self.simulation.buffer_size > 0:
                if (time.perf_counter() - start) >= duration_seconds:
                    break
                replayed = self.simulation.replay_loop(
                    now=replay_now,
                    max_replays=SLEEP_REPLAYS_PER_BURST,
                )
                if not replayed:
                    break
                replays_fired += len(replayed)
                # Advance replay clock fractionally so successive replays
                # don't all stamp the same time on running affect.
                replay_now += 0.001

            # Abstraction formation runs once per sleep, after consolidation
            # has had a chance to shape the running affect of each member.
            abstractions_formed = self._form_abstractions(replay_now)

            # Forgetting is curation. The default path is now quality-based:
            # drop concepts that have failed every weakness criterion
            # (activation_count==1, edge_count<=1, affect<0.1, surprise
            # below median). IS_A-incidence is no longer auto-protective —
            # weak members go regardless of taxonomy membership. Pinned
            # concepts remain immune.
            #
            # CONCEPT_CEILING is kept ONLY as an emergency brake: if the
            # quality prune fails to keep the graph under control, fall
            # back to size-based trim to PRUNE_TO.
            from backend.config import (                                # noqa: PLC0415
                CONCEPT_CEILING,
                QUALITY_PRUNE_ON_SLEEP,
            )
            if QUALITY_PRUNE_ON_SLEEP:
                pre = self.graph.node_count
                weak_dropped = self.graph.prune_weak_concepts(now=replay_now)
                print(
                    f"[quality-prune] removed {weak_dropped:,} weak concepts → "
                    f"{self.graph.node_count:,} remaining "
                    f"(was {pre:,})",
                    flush=True,
                )

            pruned_count = 0
            if self.graph.node_count > CONCEPT_CEILING:
                pre = self.graph.node_count
                pruned_count = self.graph.prune_to_ceiling(now=replay_now)
                print(
                    f"[prune-emergency] removed {pruned_count:,} concepts → "
                    f"{self.graph.node_count:,} remaining "
                    f"(was {pre:,})",
                    flush=True,
                )
        finally:
            self.affect.end_consolidation()

        duration_actual = time.perf_counter() - start
        return SleepResult(
            replays_fired=replays_fired,
            abstractions_formed=abstractions_formed,
            duration_actual=duration_actual,
        )

    # ---- active-set centroid (used to build EXPRESS intents) ----

    def _active_set_centroid(self, active_set: dict[int, float]) -> "np.ndarray | None":
        """Weighted L2-normalized centroid of the active concepts' embeddings.
        Returns None if the active set is empty or has no valid concepts.
        """
        if not active_set:
            return None
        weight_sum = 0.0
        accum = np.zeros(0, dtype=np.float32)
        for cid, w in active_set.items():
            node = self.graph.nodes.get(cid)
            if node is None:
                continue
            if accum.size == 0:
                accum = np.zeros_like(node.embedding)
            accum = accum + float(w) * node.embedding
            weight_sum += float(w)
        if weight_sum <= 1e-9 or accum.size == 0:
            return None
        accum = accum / weight_sum
        n = float(np.linalg.norm(accum))
        if n < 1e-9:
            return None
        return (accum / n).astype(np.float32)

    # ---- abstraction formation (called inside sleep) ----

    def _form_abstractions(self, now: float) -> int:
        """Vector-space clustering on concept embeddings (Phase 7 fix).

        BFS-on-similar_to was the prior algorithm; it cannot work at
        AUTO_LINK_K=1 because every new concept gets pulled into the
        existing giant connected component (density 2/n on a tree,
        which crosses 0.5 only at n ≤ 4). The diagnostic in
        scripts/diagnose_abstractions.py confirmed: at 6,474 nodes the
        similar_to subgraph collapsed to ONE component of 5,993 nodes
        with density 0.0008 — 600× below the 0.5 threshold.

        New approach: cluster the L2-normalized embeddings directly via
        MiniBatchKMeans at k = max(8, ceil(sqrt(N))). Each cluster of
        ≥ ABSTRACTION_MIN_MEMBERS becomes an abstraction parent if
        find_or_match (R_MATCH=0.92) doesn't already cover its centroid
        — that's the idempotency guarantee. Already-abstracted nodes
        and existing abstraction parents are excluded from
        re-clustering, same as the BFS version.

        Returns the number of new abstraction parents written.
        """
        try:
            from sklearn.cluster import MiniBatchKMeans
        except ImportError:
            # Graceful degradation if sklearn isn't installed — sleep
            # still runs replays + persistence saves.
            return 0

        # Exclude nodes already absorbed into an abstraction (have an
        # outbound is_a) and nodes that ARE abstractions (have an
        # inbound is_a). Same predicate as the BFS version.
        already_member: set[int] = {
            e.source_id for e in self.graph._edges.values() if e.type is EdgeType.IS_A
        }
        is_abstraction: set[int] = {
            e.target_id for e in self.graph._edges.values() if e.type is EdgeType.IS_A
        }
        excluded = already_member | is_abstraction

        eligible_ids = [cid for cid in self.graph.nodes if cid not in excluded]
        if len(eligible_ids) < ABSTRACTION_MIN_MEMBERS:
            return 0

        # Stack L2-normalized embeddings. The graph is supposed to keep
        # them unit-norm but we re-normalize defensively (older save
        # files may not have).
        rows = []
        for cid in eligible_ids:
            emb = self.graph.nodes[cid].embedding
            n = float(np.linalg.norm(emb))
            rows.append(emb / n if n >= 1e-9 else np.zeros_like(emb))
        X = np.stack(rows).astype(np.float32)

        n_concepts = len(eligible_ids)
        k = max(8, int(np.ceil(np.sqrt(n_concepts))))
        # Can't have more clusters than would viably hit MIN_MEMBERS each.
        k = min(k, n_concepts // ABSTRACTION_MIN_MEMBERS)
        if k < 1:
            return 0

        kmeans = MiniBatchKMeans(
            n_clusters=k,
            random_state=0,
            batch_size=256,
            n_init=3,
        )
        labels = kmeans.fit_predict(X)

        abstractions_formed = 0

        for cluster_idx in range(k):
            mask = labels == cluster_idx
            member_count = int(mask.sum())
            if member_count < ABSTRACTION_MIN_MEMBERS:
                continue

            # Centroid: mean of member embeddings, L2-normalized. We
            # recompute from the data rather than using
            # kmeans.cluster_centers_[i] since those are not unit-norm
            # (k-means centroids minimize squared distance, which
            # collapses lengths) — and find_or_match expects unit-norm
            # cosine comparisons.
            centroid = X[mask].mean(axis=0).astype(np.float32)
            cn = float(np.linalg.norm(centroid))
            if cn < 1e-9:
                continue
            centroid = centroid / cn

            # Idempotency: if any existing concept (member or otherwise)
            # already covers this region within R_MATCH, skip — the next
            # run of _form_abstractions would have done the same.
            match = self.graph.find_or_match(centroid, threshold=R_MATCH)
            if match is not None:
                continue

            members = [eligible_ids[i] for i, m in enumerate(mask) if m]
            mean_birth = np.mean(
                [self.graph.nodes[cid].affect_trace.birth_state for cid in members],
                axis=0,
            ).astype(np.float32)

            # force_write=True so the centroid doesn't dedup back to one
            # of its own members (centroids of tightly-clustered unit
            # vectors typically sit within R_MATCH of the cluster).
            parent_cid, _ = self.graph.write_on_surprise(
                representation=centroid,
                surprise=0.0,
                current_affect=mean_birth,
                name_hint=f"abstraction:k{cluster_idx}+{member_count - 1}",
                now=now,
                force_write=True,
            )

            for cid in members:
                self.graph.add_edge(
                    source_id=cid,
                    target_id=parent_cid,
                    type=EdgeType.IS_A,
                    weight=ABSTRACTION_PARENT_WEIGHT,
                    now=now,
                )

            abstractions_formed += 1

        return abstractions_formed
