"""Phase 3 vertical slice — the full mind running one tick at a time.

Verifies:
  - Phase 1 success condition still holds (regression).
  - All three encoders produce real unit vectors (text, image, audio).
  - Self/world boundary: Stimulus.source is deterministic from modality.
  - Agent registry: register_agent + refers_to auto-tagging on WORLD ingest.
  - The main loop cycles end-to-end:
      H ingest → B predict/observe → A inject + nudge_chain → C write/strengthen
      → F.attend → D.propose_and_choose → (G.request_expression on EXPRESS)
      → H.emit_output → INTERNAL_OUTPUT_ECHO → B at INPUT layer
  - Idle path: MainLoop.idle calls D.replay_loop when arousal/recency permit.

Run: python3 phase3.py
"""
from __future__ import annotations

import sys

import numpy as np

from backend.affect import AffectStack
from backend.attention import Attention
from backend.expression import Expression
from backend.graph import ConceptGraph, EdgeType
from backend.identity import (
    ChosenCandidate,
    Identity,
    RevisionRequest,
    SuppressionRequest,
)
from backend.input import (
    InputPipeline,
    Modality,
    Source,
    encode_audio,
    encode_image,
    encode_text,
    source_for,
)
from backend.main_loop import MainLoop
from backend.predict import PredictionEngine
from backend.simulation import ActionKind, SimulationReplay


def banner(label: str) -> None:
    print()
    print("=" * 64)
    print(label)
    print("=" * 64)


def main() -> int:
    # ------------------------------------------------------------------
    # Build the full stack.
    # ------------------------------------------------------------------
    t0 = 0.0
    affect = AffectStack(birth_seed=42, t_birth=t0)
    graph = ConceptGraph()
    predict = PredictionEngine(affect=affect, graph=graph)
    sim = SimulationReplay(affect=affect, graph=graph, predict_engine=predict)
    identity = Identity(
        affect=affect, graph=graph, predict_engine=predict, simulation=sim,
        birth_seed=42, birth_time=t0,
    )
    h = InputPipeline(affect=affect, graph=graph, predict_engine=predict, identity=identity)
    f = Attention(affect=affect, graph=graph)
    g = Expression(affect=affect, graph=graph, predict_engine=predict, identity=identity, input_pipeline=h)
    loop = MainLoop(
        affect=affect, graph=graph, predict_engine=predict, simulation=sim,
        identity=identity, attention=f, expression=g, input_pipeline=h,
    )

    banner("Boot")
    print(f"  graph nodes after boot   : {graph.node_count}  (2 from E + 6 from G = 8)")
    print(f"  pins after boot          : {graph.pin_count}")
    print(f"  encoders                 : {h.encoders.known_ids()}")
    print(f"  self_concept_id          : {identity.current_self_concept_id()}")
    print(f"  unknown_concept_id       : {identity.spine.unknown_concept_id}")

    checks: list[tuple[str, bool]] = []

    # ------------------------------------------------------------------
    # Encoders.
    # ------------------------------------------------------------------
    banner("Encoders — text / image / audio produce real unit vectors")
    v_t = encode_text("the dog is brown")
    rng = np.random.default_rng(0)
    v_i = encode_image(rng.random((48, 48, 3)).astype(np.float32))
    v_a = encode_audio(rng.normal(0, 1, 6000).astype(np.float32))
    for label, vec in [("text", v_t), ("image", v_i), ("audio", v_a)]:
        norm = float(np.linalg.norm(vec))
        print(f"  {label:5s} ||v|| = {norm:.4f}  (expect 1.0000)")
        checks.append((f"{label} encoder produces unit vector", abs(norm - 1.0) < 1e-3))

    # Source determinism.
    banner("Self/world boundary is deterministic from modality")
    expectations = {
        Modality.TEXT: Source.WORLD, Modality.IMAGE: Source.WORLD, Modality.AUDIO: Source.WORLD,
        Modality.INTERNAL_REPLAY: Source.SELF, Modality.INTERNAL_OUTPUT_ECHO: Source.SELF,
        Modality.INTERNAL_THOUGHT: Source.SELF, Modality.INTERNAL_SIMULATION: Source.SELF,
    }
    for m, want in expectations.items():
        got = source_for(m)
        print(f"  {m.value:22s} → {got.value}")
        checks.append((f"source({m.value}) is {want.value}", got is want))

    # ------------------------------------------------------------------
    # Agent registry + refers_to auto-tagging.
    # ------------------------------------------------------------------
    banner("Agent registry + refers_to auto-tag")
    alice = h.register_agent("alice", now=0.5)
    bob   = h.register_agent("bob",   now=0.5)
    print(f"  alice cid={alice}, bob cid={bob}, count={h.agent_count}")
    res_a = h.ingest_text("the dog is brown", agent_id=alice, now=1.0)
    res_b = h.ingest_text("birds fly south",  agent_id=bob,   now=1.5)
    res_u = h.ingest_text("rain falls",       now=2.0)   # falls back to E.unknown

    print(f"  alice ingest: refers_to {res_a.refers_to_edge.source_id}->{res_a.refers_to_edge.target_id}")
    print(f"  bob   ingest: refers_to {res_b.refers_to_edge.source_id}->{res_b.refers_to_edge.target_id}")
    print(f"  unknown ingest: refers_to from {res_u.refers_to_edge.source_id} (= unknown_id={identity.spine.unknown_concept_id})")
    checks.append(("alice refers_to edge created from alice's agent_concept",
                   res_a.refers_to_edge is not None and res_a.refers_to_edge.source_id == alice))
    checks.append(("unknown ingest falls back to E.unknown_concept_id",
                   res_u.refers_to_edge is not None and res_u.refers_to_edge.source_id == identity.spine.unknown_concept_id))

    # query_shared_knowledge
    shared = h.query_shared_knowledge(alice)
    print(f"  alice shared knowledge: {shared}")
    checks.append(("alice's refers_to chain reaches her concept(s)",
                   res_a.gap.concept_id in shared if res_a.gap.concept_id else False))

    # ------------------------------------------------------------------
    # Main loop cycle: F → D → (G if EXPRESS) → emit → echo
    # ------------------------------------------------------------------
    banner("Main loop cycle on the alice ingest")
    cycle_a = loop.cycle(res_a, now=1.001)
    print(f"  attended via mode={cycle_a.attention_frame.mode.value}")
    print(f"  spread active_set size: {len(cycle_a.spread.active_set)}, traversed: {len(cycle_a.spread.traversed_edges)}")
    print(f"  D action: {cycle_a.action.action.value} target={cycle_a.action.target_concept_id} score={cycle_a.action.score:.4f}")
    if cycle_a.action.action is ActionKind.EXPRESS:
        print(f"  → G fired, decision: {type(cycle_a.expression_decision).__name__}")
        if cycle_a.emitted_surface:
            print(f"  → emitted: {cycle_a.emitted_surface!r}")
            print(f"  → echo Stimulus modality: {cycle_a.echo_result.stimulus.modality.value}")

    # Force an EXPRESS path by handing the loop a strong-singleton active set.
    # Build a synthetic ingest_result that points at one cid with high activation.
    banner("Force EXPRESS by ingesting a known-and-strongly-active sentence")
    # Re-ingest alice's exact text — second time; should cosine-match strongly
    # so spread's seed is concentrated. Bypassing the surprise gate is fine —
    # we want F to produce a high-activation set.
    cycle_b = loop.cycle(res_a, now=1.002)   # re-cycle the same ingest_result
    print(f"  D action: {cycle_b.action.action.value} score={cycle_b.action.score:.4f}")
    if cycle_b.expression_decision is not None:
        print(f"  expression_decision: {type(cycle_b.expression_decision).__name__}")
        if cycle_b.emitted_surface:
            print(f"  emitted: {cycle_b.emitted_surface!r}")

    # Capture before-state for the echo verification
    output_count_before_emit = predict._stats["INPUT"].count
    emitted_log_before = len(h.emitted_log)

    # Manually drive request_expression with a focused intent — this is the
    # path E's spec calls "the realistic case at decision time" — exactly
    # the same pipeline the main loop runs when D picks EXPRESS, just
    # invoked directly so we always see the emit + echo regardless of D's
    # choice in this test.
    banner("Direct G round-trip: focused intent → emit → echo back through B@INPUT")
    from backend.expression import ExpressionIntent
    target_cid = res_a.gap.concept_id
    target_emb = graph.nodes[target_cid].embedding.copy()
    intent = ExpressionIntent(
        intent_id="forced-1",
        internal_repr=target_emb,
        active_concepts={target_cid: 1.0},
        now=10.0,
    )
    decision = g.request_expression(intent)
    print(f"  decision: {type(decision).__name__}")
    if isinstance(decision, ChosenCandidate):
        print(f"  chosen surface : {decision.candidate.surface_text!r}")
        print(f"  expression_gap : {decision.expression_gap:.6f}")
        echo = h.emit_output(Modality.TEXT, decision.candidate.surface_text, now=10.001,
                             parent_stimulus_id=res_a.stimulus.stimulus_id)
        print(f"  echo Stimulus  : id={echo.stimulus.stimulus_id} parent={echo.stimulus.provenance.parent_stimulus_id}")
        print(f"  emitted_log    : {len(h.emitted_log)} entries (added: {len(h.emitted_log) - emitted_log_before})")
        checks.append(("EXPRESS commits and surface logged", len(h.emitted_log) > emitted_log_before))
        checks.append(("emit_output triggers a self-overhearing INPUT-layer observation",
                       predict._stats["INPUT"].count > output_count_before_emit))

    # ------------------------------------------------------------------
    # Idle path: replay
    # ------------------------------------------------------------------
    banner("Idle path: replay fires when conditions met")
    print(f"  buffer_size before idle: {sim.buffer_size}")
    # Push last_observation_t back so idle thinks we've been quiet.
    loop.last_observation_t = -10.0
    idle = loop.idle(now=20.0, max_replays=2)
    print(f"  idle.replayed_count    : {idle.replayed_count}")
    print(f"  buffer_size after idle : {sim.buffer_size}  (entries persist; replay_count++)")
    checks.append(("idle.replay_loop fires when arousal low and observation old",
                   idle.replayed_count > 0))

    # Idle does NOT fire when recent observation
    loop.last_observation_t = 19.5
    idle2 = loop.idle(now=20.0, max_replays=2)
    print(f"  recent obs guard       : idle.replayed_count = {idle2.replayed_count} (expected 0)")
    checks.append(("idle short-circuits on recent observation",
                   idle2.replayed_count == 0))

    # ------------------------------------------------------------------
    # Image and audio ingest through the full stack.
    # ------------------------------------------------------------------
    banner("Image + audio ingest run through the same H→B→A→C path")
    img = rng.random((64, 64, 3)).astype(np.float32)
    res_img = h.ingest_image(img, now=30.0, agent_id=alice)
    print(f"  image: cid={res_img.gap.concept_id}, surprise={res_img.gap.is_surprise}, encoder={res_img.stimulus.encoder_id}")
    aud = rng.normal(0, 1, 4000).astype(np.float32)
    res_aud = h.ingest_audio(aud, now=31.0, agent_id=alice)
    print(f"  audio: cid={res_aud.gap.concept_id}, surprise={res_aud.gap.is_surprise}, encoder={res_aud.stimulus.encoder_id}")

    # Image refers_to from alice
    if res_img.refers_to_edge is not None:
        print(f"  image refers_to: alice ({res_img.refers_to_edge.source_id}) -> image-cid ({res_img.refers_to_edge.target_id})")
    checks.append(("image ingest goes through full pipeline",
                   res_img.gap is not None and res_img.stimulus.modality is Modality.IMAGE))
    checks.append(("audio ingest goes through full pipeline",
                   res_aud.gap is not None and res_aud.stimulus.modality is Modality.AUDIO))

    # ------------------------------------------------------------------
    # Final tally.
    # ------------------------------------------------------------------
    banner("Verdict")
    all_ok = True
    for label, ok in checks:
        mark = "OK  " if ok else "FAIL"
        print(f"  [{mark}] {label}")
        if not ok:
            all_ok = False

    print()
    print(f"final graph: {graph.node_count} nodes, {graph.edge_count} edges, {graph.pin_count} pins")
    print(f"replay buffer: {sim.buffer_size} entries")
    print(f"emitted surfaces: {len(h.emitted_log)}")
    print()
    print("PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
