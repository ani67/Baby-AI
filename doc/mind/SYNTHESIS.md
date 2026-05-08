# THE MIND — UNIFIED SPECIFICATION

## INTRODUCTION

This document is the synthesis pass over the eight component design docs (A through H) and the original SPEC.md. It is the last text-only artifact before code is written.

Read this in conjunction with the component docs: A-affective-engine, B-prediction-engine, C-concept-graph, D-simulation-replay, E-identity-private-state, F-attention-sparse-activation, G-expression-layer, H-input-pipeline. Each component doc remains the depth reference for its own internals; this document locks the cross-component contracts, resolves cross-cutting tickets, sets the canonical symbol table, defines boot and consolidation schedules, and specifies the implementation order.

Where this synthesis disagrees with a component doc, this synthesis wins. Where this synthesis is silent, the component doc stands.

The architectural primacy from SPEC.md is preserved everywhere: AFFECT > PREDICTION > ATTENTION > MEMORY > IDENTITY > EXPRESSION. Where two valid resolutions existed, the one that respects this ordering and the spec's design principles (emergence over injection, surprise as the only learning signal, forgetting as curation, small as a feature, private state load-bearing, no gradient descent as primary) was chosen.

---

## SYMBOL TABLE

Canonical constants. These are the values every component compiles against. Anywhere a component doc cited a different number, this table is authoritative.

```
DIMENSIONS
  N_AFF                     = 12         affect-vector dim, fixed at compile time
  D_REP                     = 256        representation-space dim (concept embeddings,
                                         predictions, observations, all encoded inputs)
  AFFECT_LAYERS             = 5          reaction, working, mood, disposition, character

GRAPH SIZE
  CONCEPT_CEILING           = 5000       active concept nodes (soft target, prune-pressure
                                         driven; not a hard cap)
  EDGE_DEGREE_AVG_TARGET    = 12         average out-degree at steady state
  EDGE_DEGREE_PROPAGATE_CAP = 32         hard cap during spreading-activation per node

REPLAY
  REPLAY_CAPACITY           = 4096       entries
  REPLAY_CAPS_PER_ENTRY     = 48         per-entry replay-count ceiling
  PER_WINDOW_REPLAY_BUDGET  = 64         max replays per low-input window

PREDICTION
  PENDING_TTL_TICKS         = 1024
  MIN_THRESHOLD             = 1.5        z-score floor for surprise
  MAX_THRESHOLD             = 8.0        z-score ceiling for surprise
  COLD_START_N              = 32         observations before adaptive threshold engages
  ALPHA_SURPRISE_TARGET     = 0.05       target rate of surprises
  PRECISION_FLOOR           = 1e-3
  PRECISION_CEILING         = 1e3
  CALIB_INTERVAL            = 200        ticks between calibration updates
  CALIB_DECAY               = 0.95
  MAX_SIM_DEPTH             = 64         hard ceiling (B-side)
  MAX_SIM_DEPTH_LOCAL       = 6          per-decision rollout depth (D-side)

AFFECT TIMESCALES (half_life in seconds)
  reaction      = 2
  working       = 180         (3 min)
  mood          = 7200        (2 h)
  disposition   = 1.21e6      (2 weeks)
  character     = 6.3e7       (2 years)

AFFECT WEIGHTS
  composite_weights = [0.30, 0.30, 0.20, 0.15, 0.05]   sums to 1
  nudge_gain (working from reaction)         = 0.05
  nudge_gain (mood from working)             = 0.05
  nudge_gain (disposition from mood)         = 0.04
  nudge_gain (character from disposition)    = 0.02
  nudge_threshold (working/mood)             = 0.10
  nudge_threshold (disposition)              = 0.20
  nudge_threshold (character)                = 0.35
  REPLAY_NUDGE_GAIN_MULTIPLIER               = 0.30   (D's replay attenuation, locked)

INJECTION GAINS (g)
  INPUT      = 1.0
  PROCESSING = 0.6
  OUTPUT     = 0.8

ACTION VOCABULARY (D)
  EXPRESS, ATTEND, INTERROGATE, WAIT, SUPPRESS    closed taxonomy at v1
  cost_hint defaults: EXPRESS=0.4, ATTEND=0.1, INTERROGATE=0.0, WAIT=0.0, SUPPRESS=0.2

EDGE TAXONOMY (C)
  is_a, has_property, causes, precedes, similar_to, opposite_of,
  context_of, part_of, expresses, refers_to                   closed at v1

EXPRESSION
  N_CANDIDATES_DEFAULT      = 4
  MAX_REVISIONS             = 2
  REVISE_THRESHOLD          = 0.4
  SUPPRESS_HARD_THRESHOLD   = 0.8
  MODALITY_LEAK             text=0.20, image=0.50, audio=0.85
  MODALITY_PREFERENCE_BIRTH = [0.6, 0.2, 0.2]   text-leaning at boot

PINS (E)
  MAX_PINS                  = 128
  MAX_ANCHORS               = 64

HABITS (F)
  HABIT_SEED_CAP            = 64
  HABIT_PATH_CAP            = 256
  HABIT_PROMOTE_MIN         = 5         observations to promote
  HABIT_VARIANCE_MAX        = 0.4
  HABIT_FORGET_TAU          = 14 days
  HABIT_PATH_AMP_MAX        = 1.5
  CONSOLIDATE_INTERVAL      = 600 s     (10 min — runtime cadence; sleep cadence below)

INPUT
  D_REP                     = 256        (re-stated for emphasis; locked)
  INGEST_QUEUE_CAPACITY     = 1024
  CYCLE_DEPTH_MAX           = 8

TICK RATE
  MAIN_LOOP_HZ              = 50         configurable; H drives the tick

PERSISTENCE CADENCE
  SAVE_INTERVAL             = 60 s       (A, C, D, E, G, H all align)
```

Any component-doc-internal constants not listed above remain owned by that component.

---

## ARCHITECTURE AT A GLANCE

The system is a single in-process loop running at ~50 Hz on M1. One concept graph (C) is the only long-term store. One affect stack (A) holds the five-layer continuous emotional state. One prediction engine (B) emits predictions, computes gaps, scores surprise, and routes the gap to A, C, and D. F is the conductor that tells C when to spread and how. D simulates and replays. E watches identity and decides what to express. G renders surfaces. H is the boundary — bytes in, encoded vectors out, and any internal vector (replay, simulation, output echo) must re-enter through the same H interface.

```
WORLD                                                   SELF
  |                                                       |
  |  bytes/text/image/audio                INTERNAL_REPLAY/SIM/THOUGHT
  v                                                       |
+---------------- H (encoders, agent registry) -----------+
  |
  |  Stimulus(repr, tick, source, agent_id, encoder_id, prov)
  v
+--- B.predict ---+   (issued before encoding finishes when possible)
  |    |
  |    v
  |   Prediction(mean, precision, support_set, layer, affect_snapshot, tick)
  |
  | observed (after encoding)
  v
+--- B.observe ---+
  |
  |   PredictionGap(delta, weighted_delta, magnitude, surprise_score, is_surprise)
  |
  +-> A.inject (always)         INPUT trigger or PROCESSING or OUTPUT
  |     |
  |     v
  |   AffectStack reaction layer updated; nudge_chain runs upward
  |
  +-> C.write_on_surprise (if is_surprise)
  |     |
  |     v
  |   ConceptNode written or strengthened, edges laid down,
  |   AffectSnapshot stamped, replay hook emitted
  |
  +-> D.replay_push (if is_surprise)
        |
        v
      ReplayEntry stored, prioritized

F is invoked once per phase (INPUT, PROCESSING, OUTPUT) to call C.spread:
  AttentionFrame{seeds, composite_affect, arousal, predict_prior, mode, budget}
  -> active_set: dict[concept_id, activation]
  + traversed_edges: list[(edge_id, propagated_activation)]    [F-required addition]

PROCESSING runs as a small loop: F.attend(PROCESSING) -> B.predict -> (gap?) ->
  A.inject(PROCESSING) -> repeat until B reports settle confidence > 0.7 or
  max_internal_ticks (4) reached.

After PROCESSING settles, D.propose_and_choose runs:
  active_set + current_affect -> {WAIT, EXPRESS, ATTEND, INTERROGATE, SUPPRESS}
  candidate chains scored against wants (A.character + homeostatic + boredom)
  one chain wins -> ActionDescriptor returned to main loop

If chosen action is EXPRESS:
  E.snapshot_private_state -> PrivateState
  G.generate_candidates(intent) -> list[CandidateExpression]
  E.decide_expression(candidates, private_state)
    -> chosen | revise (loop) | suppress
  If chosen: G commits, H.emit_output -> world AND H.inject_internal(OUTPUT_ECHO)
    -> back through B at INPUT layer (the system hears itself speak)

REPLAY (during low-input periods):
  D.replay_loop -> picks priority entry -> re-fires its observation through B
  using entry.affect_snapshot and currently-living support concepts.
  B.observe is called with thread-local replay_origin=true (B skips Welford update,
  A nudge_gain attenuated by 0.30). Resulting gap may strengthen or write nodes
  through normal C path.
```

The cardinal claim of the spec — "attention is not computed; it emerges" — is operationalized as: F never produces an attention vector. The active set falls out of A.gate_attention being called per traversal inside C.spread, with the arousal-driven sparsity envelope applied by C.

---

## LOCKED CONTRACTS

Every cross-component function signature, owner, callers, and cost target. Where component docs disagreed, this is the lock.

### A — Affective Engine

**`A.composite(now: float64) -> float32[N_AFF]`**
- Owner: A. Callers: B (every predict), C (every spread/write), F (every attend), E (snapshot), D (wants extraction), G (per-step expression).
- Returns the composite affect vector, post-decay, post-cache (cache TTL 0.5 s).
- Cost: ~50 ns cached; ~5 µs cold.

**`A.current_arousal(now) -> float32`**
- Owner: A. Callers: F (sparsity decisions), C (spread), D (replay gating), G (modality choice).
- Returns `||reaction.vector||₂` post-decay.
- Cost: ~50 ns cached.

**`A.current_character(now) -> float32[N_AFF]`**
- Owner: A. Callers: D (wants), E (continuity), F (habit consolidation).
- Cost: ~50 ns.

**`A.inject(injection_point, gap_signal, gap_magnitude, now) -> reaction_vector`**
- Owner: A. Caller: B (the only legitimate caller).
- `gap_signal: float32[N_AFF]` — already projected to affect space by A's W (see T1).
- Mutates state. Triggers nudge_chain upward.
- Cost: ~10 µs.

**`A.simulate_inject(prior_affect: AffectVector, gap_signal_proxy, magnitude_proxy, injection_point) -> AffectVector`** [NEW — added per T-D-1]
- Owner: A. Caller: D (D4, feel_step during simulation rollouts).
- Pure function. Does NOT mutate the live AffectStack. Treats prior_affect as a notional reaction layer; applies the same g/m_nov/squash math as `inject`; returns the post-injection vector.
- Does NOT propagate up the nudge chain (simulation rollouts do not nudge mood).
- Cost: ~5 µs. Stateless, thread-safe.

**`A.set_nudge_gain_multiplier(m: float32)` / `A.clear_nudge_gain_multiplier()`** [NEW — added per T-D-2]
- Owner: A. Caller: D only, before/after each replay observation.
- Installs a thread-local multiplier on the layer-to-layer nudge gains. Default 1.0; D sets to 0.30 (`REPLAY_NUDGE_GAIN_MULTIPLIER`) during replay.
- Reaction layer still receives full magnitude — the replay is felt — but the chain to working/mood/disposition/character is muted.
- Cost: O(1).

**`A.gate_attention(node_affect_trace, semantic_score, predictive_score) -> float32`**
- Owner: A. Caller: C (per traversal during spread).
- `predictive_score`: locked to `1.0 + predict_prior.get(dst_concept_id, 0.0)` clamped to `[1, 2]`. F supplies the predict_prior dict to C inside the AttentionFrame; C looks up dst per traversal. (Locked from F's position.)
- Returns scalar gate ≥ 0. Switches multiplicative (high arousal) vs additive (low arousal) regimes.
- Cost: ~200 ns.

**`A.stamp(now) -> AffectSnapshot`**
- Owner: A. Caller: C (every write_on_surprise, every strengthen on cross-threshold activation).
- Returns immutable snapshot with composite + reaction + layer_summary bitfield.
- Cost: ~1 µs.

**`A.affect_distance(snap_a, snap_b) -> float32`**
- Owner: A. Caller: D (replay priority), E (anchor diversity).
- Weighted L2 between snapshots (composite 0.6, reaction 0.4).

**`A.force_nudge_chain(now)`**
- Owner: A. Caller: D (after each replay block, with multiplier active).

**`A.set_W_update(delta_W)`**
- Owner: A. Caller: B (when Hebbian binding decides to update gap→affect projection).
- W lives in A. See T1.

### B — Prediction Engine

**`B.predict(current_state, affect_composite, layer, query_seed=None, topK=30) -> Prediction`**
- Owner: B. Callers: H (INPUT), F's processing_loop (PROCESSING — once per tick, not per spread step; see T5), G (OUTPUT).
- Side effect: registers pending prediction with monotonic tick.
- Cost target: < 5 ms on M1 at D_REP=256, topK=30, 5K nodes.

**`B.observe(observation: Observation) -> PredictionGap | None`**
- Owner: B. Callers: H (after encoding), F (post-PROCESSING-attend mid-thought observation), G (re-encoded surface for OUTPUT trigger), D (replay).
- Computes gap, scores surprise, emits to A/C/D via internal `EmitGap`.
- Returns null only if no matching pending prediction.
- Honors thread-local `replay_origin=true` flag set by D — when true, skips Welford update on `running_gap_stats` (T4 locked).

**`B.simulate_step(state, affect, action, chain_id, depth) -> SimulationFrame`**
- Owner: B. Caller: D (D3, simulate_chain).
- Does NOT register pending prediction. Pure rollout.
- Cost: ~5 ms per step.

**`B.reconcile_simulation(committed_frame, observation) -> PredictionGap`**
- Owner: B. Caller: D (D7, observe_outcome).
- Computes gap with `simulation_origin = chain_id`. Routes through normal EmitGap.

**`B.replay_origin` thread-local flag** [NEW — locked per T4]
- Owner: B. Setter: D before calling `B.observe` during replay; D clears after.
- When true, B skips Welford update in A2/A3 but still computes the surprise score against existing stats.

**`B.stats_snapshot() -> EngineStatsView`**
- Owner: B. Callers: E (for narrative continuity), frontend.

### C — Concept Graph

**`C.write_on_surprise(representation, predicted, surprise, current_affect, name_hint, context_active, replay_origin=False, encoder_id) -> concept_id`** [encoder_id and replay_origin locked]
- Owner: C. Caller: B only (on surprise emission). G calls indirectly via B when its OUTPUT trigger crosses threshold.
- One-shot write or strengthen-existing-match.
- `encoder_id` parameter is required (was an addition request from H; locked — see DATA-STRUCTURE LOCKS below).
- `replay_origin` parameter is required (was addition request from D; locked). When true, surprise is treated as smaller (already incorporated by graph) and the write is tagged for diagnostic queries.

**`C.spread(seeds, composite_affect, arousal, max_steps, budget, mode) -> SpreadResult`**
- Owner: C. Caller: F (primary), B (within predict for support set), D (within simulate_step via B).
- `mode ∈ {PERCEIVE, PREDICT, SIMULATE}` — affects edge-type weighting.
- Returns a `SpreadResult` struct with two fields (locked, was F-requested addition):
  - `active_set: dict[uint64, float32]` — concept activations.
  - `traversed_edges: list[(edge_id, propagated_activation)]` — for F's HabitPath tracking. Bounded to top 64 traversals by activation magnitude to keep memory low. F's spec degraded gracefully without this; with it, habit-path quality is sharp.
- Cost: bounded by budget × out-degree-cap; sub-millisecond at default budget=256.

**`C.find_or_match(representation, threshold) -> concept_id | None`**
- Owner: C. Callers: H (post-encode for agent attribution), G (slot filler lookup fallback).

**`C.neighbors_by_type(concept_id, type) -> list[(edge, peer_node)]`**
- Owner: C. Callers: G (expresses lookup), E (refers_to traversal), F (ToM-restricted spread, via repeated calls).

**`C.spread_restricted(seeds, composite_affect, arousal, max_steps, budget, mode, restrict_root, restrict_edge_types) -> SpreadResult`** [NEW — added per E ToM request]
- Owner: C. Caller: F (when E supplies a ToM-shaped seed), E (audience-belief queries).
- Same as `spread` but propagation is gated to edges where the destination concept has a path back to `restrict_root` via `restrict_edge_types` within 3 hops. Implementation: pre-walk from `restrict_root` to build a small reachable-set, then mask propagation against it.
- This is the cheap audience-belief spread E asked for. F locks the contract; C implements.

**`C.pin(concept_id, reason: str)` / `C.unpin(concept_id)`**
- Owner: C. Callers: E (NARRATIVE_ANCHOR, AFFECT_KEYSTONE, SELF_REFERENT, EXPRESSION_HABIT pins), F (`F.habit` pins).
- Pin decays per C's spec; caller must touch.

**`C.suggest_precedes_edge(src_id, dst_id, surprise_at_link)` [NEW — added per T-D-3]**
- Owner: C. Caller: D only (D14).
- D suggests a `precedes` edge between two concepts that were close in original-time during a replayed sequence. C decides whether to lay it (consults existing edge weight, density, and surprise at link). Returns nothing.
- Cost: O(1) lookup + O(1) write decision.

**`C.tombstone(concept_id)`**
- Owner: C. Caller: E (for self-managed scaffolding cleanup only).

**`C.replay_hook_drain(max=K) -> list[(concept_id, surprise, affect_at_event)]`**
- Owner: C. Caller: D on each replay loop entry.

**`C.query_top_k_active(k) -> list[(concept_id, activation)]`**
- Owner: C. Caller: E (snapshot_private_state).

**`C.query_top_k_by_affect(affect_vector, k) -> list[(concept_id, alignment)]`**
- Owner: C. Caller: E (recovering character-aligned concepts), D (wants).

### D — Simulation + Replay

**`D.propose_and_choose(active_concepts, current_affect, now) -> (ActionDescriptor, CommittedDecisionRecord)`**
- Owner: D. Caller: main loop (after F's PROCESSING settles).
- Internally runs propose_candidates → simulate_chain (×N) → score → choose.

**`D.observe_outcome(decision_id, real_observation)`**
- Owner: D. Caller: H/G after the action's consequence has arrived.

**`D.replay_push(gap, support_set, affect_snapshot, observation, layer)`**
- Owner: D. Caller: B (from EmitGap on every surprise crossing threshold).
- Argument order locked per D's spec.

**`D.replay_loop(now)`**
- Owner: D. Caller: main loop on idle ticks.
- Internally honors should_replay_now, runs replay_one until budget exhausted or condition violated.

**`D.simulate_audience_response(surface, audience_concept_id) -> AudiencePrediction`**
- Owner: D. Caller: G (during candidate scoring).
- Internally: D constructs a SimulationFrame from the surface representation and runs a depth-3 SIMULATE-mode chain biased toward concepts in audience's `believed_to_know` (queried via H10c).

**`D.simulated_affect_at_chain_end(chain) / D.simulated_audience_state_at_chain_end(chain)`**
- Owner: D. Caller: E (E8 simulate_emission), G (G4d audience simulation).

**`D.world_model.simulated_affect_if_visited(concept_id) -> float32[N_AFF]`**
- Owner: D. Caller: E (E2 derive_wants).
- Quick estimate based on the concept's `affect_trace.running_state` plus a 1-hop SIMULATE-mode rollout. Cached per concept_id with 30 s TTL.

**`D.recent_surprises(k) -> list[(concept_id, surprise_score, t)]`**
- Owner: D. Caller: E (snapshot_private_state).

**`D.committed_decision_history(k) -> list[CommittedDecisionRecord]`**
- Owner: D. Caller: E (narrative continuity).

**`D.world_model_stats() -> WorldModelStatsView`**
- Owner: D. Caller: E, frontend.

### E — Identity + Private State

**`E.snapshot_private_state(now) -> PrivateState`**
- Owner: E. Caller: G (before forming candidates), D (when storing replay context).

**`E.decide_expression(candidates, private_state) -> ChosenCandidate | RevisionRequest | SuppressionRequest`**
- Owner: E. Caller: G.
- Bounded by candidate count (≤ 4) and D's simulation cost. Worst case ~600 ms; typical ~100 ms.

**`E.register_audience_response(t, audience_repr)`**
- Owner: E. Caller: H (when an audience response arrives within RESPONSE_WINDOW).

**`E.narrative_replay_seed(now) -> list[(anchor, priority, core_concepts)]`**
- Owner: E. Caller: D's replay scheduler.

**`E.pin_self_referent_concept(concept_id)`**
- Owner: E. Caller: C (callback when a new `refers_to` chain reaches self_concept_id).

**`E.note_expression_habit(concept_id, idiom_strength)`**
- Owner: E. Caller: G when it detects a stable expression idiom for a concept.

**`E.current_self_concept_id() -> uint64` / `E.current_other_concept_id(agent_handle) -> uint64`**
- Owner: E. Callers: H (tagging incoming utterances), C (resolving refers_to chains), G (first-person expression).

**`E.desired_audience_affect(target_audience, honesty_bias) -> float32[N_AFF]`**
- Owner: E. Caller: G (audience-cost scoring).

**`E.on_suppression(intent_id, would_have_been_surface, discomfort)` / `E.on_commit(intent_id, surface, expression_gap_mag, decision, revision_count)`**
- Owner: E. Caller: G.

### F — Attention + Sparse Activation

**`F.attend(phase, raw_seeds, now) -> active_set`**
- Owner: F. Callers: H (INPUT), F.processing_loop (PROCESSING), G (OUTPUT).
- Internally builds AttentionFrame, samples affect once, injects habit seeds, calls C.spread, records event.
- Cost target: < 4 ms at default budget.

**`F.processing_loop(input_frame, max_internal_ticks=4) -> active_set`**
- Owner: F. Caller: main loop (after H dispatches stimulus).
- Manages the inner PROCESSING-attend / B.predict loop. Settle-or-cap.

**`F.current_attention_state() -> view`**
- Owner: F. Callers: E, frontend.

**`F.tom_seed_provider(callback)`**
- Owner: F. Caller: H/E at boot to register a callback that returns ToM seeds for the current AttentionFrame.

**`F.set_habit_temperature(value)` / `F.request_habit_pin_refresh(now)`**
- Owner: F. Caller: E.

### G — Expression

**`G.request_expression(intent: ExpressionIntent) -> ExpressionDecision`**
- Owner: G. Caller: E (typically), or H for direct queries.
- Returns `(committed_surface, decision_enum, expression_gap_mag)`.
- Worst case ~600 ms; typical 100–250 ms.

**`G.attach_surface_to_concept(concept_id, surface, modality, affect_at_event) -> edge_id`**
- Owner: G. Caller: H (when a surface form is observed paired with a concept) or D (replay).

**`G.is_still_a_habit(concept_id) -> bool`**
- Owner: G. Caller: E (E5 manage_pins, EXPRESSION_HABIT validation).

**`G.style_snapshot() -> view`**
- Owner: G. Caller: E, frontend.

**`G.pin_template(concept_id, reason)`**
- Owner: G. Caller: E.

### H — Input Pipeline + World Interface

**`H.ingest_text(text, agent_id=None, claim_self=False, prior_seeds=None) -> stimulus_id`**
**`H.ingest_image(bytes, ...) -> stimulus_id`**
**`H.ingest_audio(bytes, ...) -> stimulus_id`**
- Owner: H. Callers: external (frontend, network, REPL).
- Non-blocking; pushes to IngestQueue. Encoded and dispatched on `H.tick`.

**`H.inject_internal(payload, modality, provenance) -> Stimulus`**
- Owner: H. Callers: D (replay/simulation), G (output echo), E (self-thought).
- Validates dim and norm, increments cycle_depth from parent.

**`H.emit_output(modality, surface)` [LOCKED]**
- Owner: H. Caller: G (after commit).
- Two side effects on emission, both locked per T6:
  1. Surface is delivered to the world (frontend, websocket, etc.).
  2. The same surface is re-encoded and re-injected via `H.inject_internal(modality=INTERNAL_OUTPUT_ECHO, provenance.origin=OUTPUT_ECHO)`. This produces the self-overhearing INPUT-trigger event.

**`H.tick(now)`**
- Owner: H. Caller: main loop.
- Drains IngestQueue, encodes, dispatches via `H.dispatch` → B.predict → B.observe.

**`H.register_agent(display_name) -> agent_id`**
- Owner: H. Caller: E.

**`H.query_shared_knowledge(agent_id, k) -> list[concept_id]`**
- Owner: H. Caller: E, D.

**`H.encode(surface, modality) -> RepresentationVector`** [synchronous]
- Owner: H. Caller: G during candidate re-encoding for OUTPUT trigger.
- Must be sub-50 ms. If encoder_id is bootstrap (e.g., CLIP image), G's degraded-mode fallback applies.

**Auto-tagging incoming utterances with `refers_to` to speaker's agent_concept** [LOCKED per E request]
- Owner: H. Locked: H must, after any `world` stimulus is attributed to an agent_id, ensure that any concepts written or matched as a result carry a `refers_to` edge from the agent's concept to themselves. This is what makes single-graph theory of mind work without per-agent namespaces. Implementation: after `B.observe` reports a write or match, H calls `C.strengthen_edge` (or creates) for the `refers_to` edge between agent_id and the matched/written concept_id.

---

## DATA-STRUCTURE LOCKS

Final field lists for cross-cutting structures.

### `ConceptNode` (C)

Locked fields:
- `concept_id: uint64` (immutable, never reused)
- `name: str` (debug label)
- `embedding: float32[D_REP]` — D_REP=256 (locked)
- `encoder_id: str` — **LOCKED per H request (T9). ~10 bytes per node.** Required so HOT_REPROJECT knows which nodes to re-encode.
- `affect_trace: AffectTrace` (birth_state, peak_state, peak_magnitude, running_state, running_magnitude, last_affect_update — all in N_AFF=12 dims)
- `activation_count: uint32`
- `last_activated: float64`
- `created_at: float64`
- `surprise_at_birth: float32`
- `salience: float32` (cached, lazy)
- `abstraction_level: uint8`
- `instance_of: uint64?`
- `edges_out: list[EdgeRef]`
- `edges_in: list[EdgeRef]`
- `tombstone: bool`
- `version: uint16`

**Not added at v1 (deferred): `source_at_birth: enum {SELF, WORLD, MIXED}`.** H requested this; rationale for deferral: E's "I-thought-this" set keyed by concept_id (populated via H's stimulus stream subscription) covers the same need without bloating the node. E owns the side-table; reconsider in v2 if multiple components end up needing self/world provenance.

Total per-node size at 5K nodes: ~520 bytes RAM (with edges). Persisted: ~280 bytes per node (float16 + quantization). Persisted graph total: ~8 MB. Within the 50 MB ceiling.

### `Edge` (C)

Locked fields:
- `edge_id: uint64`
- `src: uint64`, `dst: uint64`
- `type: EdgeType` (10 types, closed at v1)
- `weight: float32` ∈ [0, 1]
- `confidence: float32` ∈ [0, 1]
- `affect_at_birth: float32[N_AFF]`
- `last_traversed: float64`
- `traversal_count: uint32`

**Edge type taxonomy (locked, 10 types):**
`is_a`, `has_property`, `causes`, `precedes`, `similar_to`, `opposite_of`, `context_of`, `part_of`, `expresses`, `refers_to`.

**`expresses` edge — single type, not split (resolves G's request).** G's spec floated splitting `expresses` into `expresses_template` and `expresses_token`. Resolution: keep one `expresses` type. Disambiguation is done by inspecting the destination node: if `abstraction_level >= 1` and `pragmatic_function` is set, it is a template; otherwise it is a surface token. G's hot path uses `neighbors_by_type(concept_id, type=expresses)` and then filters in code; this keeps C's edge taxonomy minimal and matches the spec's "edge taxonomy as closed set" principle. If G later observes that the filter is a hot-path bottleneck, revisit in v2.

### `AffectVector` (A)

- `values: float32[N_AFF]` — N_AFF=12 (locked)
- `version: uint16`
- Storage: float32 in RAM; float16 on disk and in stamps.

### `AffectSnapshot` (A)

- `composite: float16[N_AFF]`
- `reaction: float16[N_AFF]`
- `t: float64`
- `layer_summary: uint8`

### `Prediction` (B)

- `mean: RepresentationVector` (D_REP=256)
- `precision: float32[D_REP]` (per-dimension, clamped [PRECISION_FLOOR, PRECISION_CEILING])
- `confidence_scalar: float32` ∈ [0, 1]
- `support_set: list[(concept_id, weight)]` (5–30 entries)
- `layer: enum {INPUT, PROCESSING, OUTPUT}`
- `tick: int64`
- `affect_snapshot: float32[N_AFF]`

### `PredictionGap` (B)

- `delta: float32[D_REP]`
- `weighted_delta: float32[D_REP]`
- `magnitude: float32`
- `signed_magnitude_per_dim: float32[D_REP]` (= weighted_delta, retained under both names)
- `confidence_at_prediction: float32`
- `surprise_score: float32`
- `is_surprise: bool`
- `tick: int64`, `layer: enum`, `affect_snapshot: float32[N_AFF]`
- **`replay_origin: bool`** [LOCKED per T4]
- **`simulation_origin: chain_id?`** (set by `B.reconcile_simulation`)

### `Stimulus` (H)

- `stimulus_id: uint64`, `tick: int64`, `t_arrival`, `t_encoded`
- `modality: enum {TEXT, IMAGE, AUDIO, INTERNAL_REPLAY, INTERNAL_SIMULATION, INTERNAL_THOUGHT, INTERNAL_OUTPUT_ECHO}`
- `source: enum {SELF, WORLD}` (deterministic from modality)
- `agent_id: uint64?`
- `representation: float32[D_REP]`, `representation_norm: float32`
- `encoder_id: str` (matches one in EncoderRegistry)
- `name_hint: str`
- `provenance: ProvenanceRecord`
- `prediction_handle: PredictionHandle`
- `prior_attention_seeds: list[(uint64, float32)]?`

### `ReplayEntry` (D)

- `entry_id: uuid`, `original_tick: int64`, `original_t: float64`
- `gap: PredictionGap`
- `actual_repr: float32[D_REP]` (float16 on disk)
- `support_set: list[(uint64, float32)]`
- `affect_snapshot: AffectVector`
- `layer: enum`
- `priority: float32`
- `replay_count: uint16`, `last_replayed_t: float64?`
- `tags: bitfield`

Per-entry ~1.5 KB; buffer 4096 → ~6 MB.

### `IdentitySpine` (E)

- `birth_seed: uint64` (matches A)
- `birth_time: float64`
- `mind_uuid: uuid`
- `pinned_concepts: dict[uint64 -> PinRecord]` (≤ MAX_PINS=128)
- `narrative_anchors: list[NarrativeAnchor]` (≤ MAX_ANCHORS=64)
- `character_baseline: float16[N_AFF]`
- `expression_calibration: ExpressionCalibration`
- `self_concept_id: uint64`, `self_concept_last_strengthen_t: float64`
- `others: dict[uint64 -> OtherModel]`
- `expressed_self_history: ring[256]`
- `last_persisted_t: float64`, `schema_version: uint16`

Total: ~30 KB persisted.

### `WorldModelMetadata` (D)

- `simulation_quality_ema: float32`
- `per_action_kind_quality: dict[ActionKind, EMA]`
- `replay_gain_state: ReplayGainState`
- `last_replay_run_t: float64`
- `low_input_run_started_t: float64?`
- **`homeostatic_target_ema: float32[N_AFF]`** [LOCKED ownership: D, per T17]

---

## CROSS-CUTTING TICKET RESOLUTIONS

### T1 — Where does the gap → affect projection W live?

**Resolution: W lives in A.** A holds it; B passes raw_gap to A's `inject` after first calling A's `composite()`; the `gap_signal` parameter to `inject` is `W @ raw_gap`, computed by A's wrapper layer the moment B hands it the raw gap.

Concretely: B's `EmitGap` step calls `A.inject(injection_point, gap_signal, gap_magnitude, now)` where `gap_signal` and `gap_magnitude` were computed by A from B's PredictionGap. A exposes a small helper `A.project_gap(weighted_delta) -> (gap_signal, gap_magnitude)` that B calls just before `inject`; this keeps W private to A while letting B keep its representation-space contract clean.

Rationale: persistence of W belongs with persistence of affect (both define the mind's emotional shape). Hebbian updates to W happen on co-occurrence which A can detect from its own state. B remains agnostic about how gaps map to affect dimensions, which honors the architectural primacy (PREDICTION should not own the affect representation).

Affected: A (owns W, exposes `project_gap` helper, `set_W_update`); B (calls A.project_gap before inject); G (gets `pinv(A.W)` via `A.project_affect_to_repr` for style biasing — see T-G-1 below).

### T2 — First-input / birth handling before any predictions exist

**Resolution: birth uses A's primordial-arousal seed and B's degenerate-prediction mode together; the first surprises are large by design.**

Step-by-step at first boot (also see BOOT SEQUENCE below):
1. A initializes character to small random N(0, 0.05) from birth_seed; reaction jittered N(0, 0.02). The system is slightly aroused.
2. B has empty Welford stats and uses MIN_THRESHOLD=1.5 only (no z-scoring) until COLD_START_N=32 observations accumulate.
3. C is empty.
4. The first stimulus arrives. H calls B.predict; B has no support (empty graph) so emits a degenerate Prediction (`mean=zero`, `precision=uniform_low`, `confidence_scalar=0`, `degenerate=True`). This is B's F1 failure mode but used as the first-input contract.
5. H calls B.observe with the actual representation. The gap is the full magnitude of the input vector; because precision is uniform_low, the weighted_delta is moderate (not infinite); because confidence is 0, the surprise amplifier is 0.5 (minimum). MIN_THRESHOLD=1.5 still applies but z-scoring is skipped — a synthetic z=0 is used until COLD_START_N is reached, so `is_surprise` defaults to true for the first 32 observations.
6. C writes the first node. A's `inject` fires with `gap_signal = W @ raw_gap` (W is the random-orthonormal init). The reaction layer takes its first non-jitter value. `nudge_chain` propagates upward.
7. The first surprise is felt as significant; it shapes character disproportionately. This is intentional — the spec explicitly compares this to infant-attachment-to-first-faces formation.

Key locks:
- Skip z-scoring while count_per_layer < COLD_START_N=32.
- During cold start, every observation that would cross MIN_THRESHOLD does cross it (synthetic z = MIN_THRESHOLD itself).
- First-ever surprise's gap_magnitude is bounded by precision_floor multiplication, so no infinite values reach A.

Affected: A (init_at_birth), B (degenerate mode + cold-start handling), C (warmup_writes=200 suppresses pruning), H (cold ingestion path always calls B.predict regardless of empty graph).

### T3 — Replay-nudge gain

**Resolution: 0.30 (D's position locked).** `REPLAY_NUDGE_GAIN_MULTIPLIER = 0.30`. A exposes `set_nudge_gain_multiplier`/`clear_nudge_gain_multiplier` for D to install this scope around each replay observation. Reaction layer receives full magnitude; the nudge chain to upper layers is muted to 30%.

Rationale: heavily replayed memories still shape character but at 30% weighting per replay event; over 1000 replays of one trauma this accumulates to 301 "original-strength" events, which is meaningful but not dominant. If empirical run shows character drifts visibly under replay-heavy regimes, lower to 0.15.

Affected: A (adds the methods), D (sets/clears around each replay).

### T4 — Replay updates surprise statistics?

**Resolution: NO — replay does not update Welford running gap statistics.** D sets a thread-local `B.replay_origin = true` flag before calling B.observe during replay; B reads this flag in A2/A3 and skips the Welford update. The gap is still scored against existing stats (so adaptive thresholding still applies), but the stats themselves are frozen during replay.

Rationale: if replays update stats, salient memories drag the surprise threshold downward, which means routine new events register as surprise more readily, which means more writes, which means more replay candidates → positive feedback into the system's own memory of itself. Replay's job is to extract more learning from old events through current graph structure, not to convince the system those events were extra-surprising.

Affected: B (honors the flag), D (sets/clears the flag).

### T5 — PROCESSING-layer prediction granularity

**Resolution: once per tick, not per spread step (F's position locked).** F invokes B.predict exactly once per `attend` call at PROCESSING layer. Per-spread-step prediction would multiply B's cost by max_steps and require B to handle partial activation states. The "feeling evolves mid-thought" property is preserved because the *outer* processing_loop allows multiple predictions during a single externally-observable input→output turn.

Affected: F (one predict per attend); B (no changes).

### T6 — OUTPUT-layer "observation" semantics

**Resolution: re-encoded surface as observation; intended_repr as prediction (B+E+G converged position locked).**

For every emission, two events fire:
1. **OUTPUT-trigger event.** B.observe with `actual = re_encoded_surface`; matched against the OUTPUT-layer Prediction whose `mean = intended_repr`. The gap is the lying-or-leaking gap and fires before any audience response. This closes the OUTPUT trigger synchronously, even into a passive logger.
2. **INPUT-trigger event (self-overhearing).** Immediately after step 1, H.inject_internal is called with the same surface, modality=INTERNAL_OUTPUT_ECHO, source=SELF. This re-enters through H's normal path: B.predict → B.observe at INPUT layer. The gap here is "what does it feel like to have said this," which can produce its own surprise and update if the system genuinely surprised itself with what it emitted.

These two events are distinct, not duplicates: the first is "should I say this," the second is "what does it feel like to have said it." Combined, they are the felt experience of speaking.

A third event — actual audience response when it arrives, processed as ordinary INPUT — closes the loop on D's world model via `D.observe_outcome`. This is owned by H/D, not G.

Affected: G (calls B.observe with re-encoded then triggers H.inject_internal), B (no change), E (E10 reads the post-emission gap), D (consumes audience response when it arrives).

### T7 — `precedes` edges vs D's trajectory store

**Resolution: hybrid (D's position locked).** Replay buffer is the canonical sequence store. C's `precedes` edges are the *short-range, frequently-traversed* projection of that order. D does NOT maintain a separate sequence store. After replay-driven writes, D.precedes_edge_synthesis (D14) calls `C.suggest_precedes_edge(src, dst, surprise_at_link)` for short-range adjacency (within 5 s original time). C decides whether to lay it. Long-range temporal patterns (events more than ~5 s apart) live only in the replay buffer; the graph does not represent them.

Affected: C (adds `suggest_precedes_edge` entry point per T-D-3), D (calls it).

### T8 — Theory-of-mind data model

**Resolution: single graph + `refers_to` + bounded `OtherModel` index in IdentitySpine (C+E+F+H converged position locked).**

- ToM lives in the same `ConceptGraph` as everything else.
- Each external agent gets one concept node in C with `is_a` edge to a pinned `agent_concept` root.
- The agent's "beliefs about X" are concepts with `refers_to` edges from the agent's concept to X (or to belief-concepts that themselves refers_to X).
- E maintains per-agent `OtherModel` records inside the IdentitySpine: just a small bounded index (`agent_concept_id`, `last_observation_t`, `believed_to_know: set[uint64] LRU at 256`).
- H maintains an ANN over each agent's `refers_to` neighborhood for sub-millisecond shared-knowledge queries.
- F supports `C.spread_restricted` (locked above) so audience-belief queries are cheap affect-gated spreads over a refers_to-reachable subgraph.
- H auto-tags incoming utterances with `refers_to` to the speaker's agent_concept (locked above).

This delivers first-order ToM ("Alice believes X") well, second-order weakly, third+ not at all. Sufficient for v1.

Affected: C (no schema change, just `spread_restricted`), E (owns OtherModel index), F (calls spread_restricted when E supplies ToM seeds), H (auto-tagging on attribution).

### T9 — Encoder versioning policy

**Resolution: FROZEN_ONCE for native encoders (text, audio, patch-stats image), HOT_REPROJECT for bootstrap encoders (CLIP image) (H's position locked).**

- Native encoders (text:char-trigram-bow-v1, audio:mel-stats-v1, image:patch-stats-v1) are deterministic and never change after install. No re-embedding ever.
- Bootstrap CLIP image encoder is replaced once via HOT_REPROJECT after 5,000 image-derived writes. The successor is contrastively distilled from the graph itself (sketch in H11).
- COLD_REBUILD is reserved for catastrophic encoder regressions only.

ConceptNode gets `encoder_id: str` field (locked above) so HOT_REPROJECT knows which nodes to re-encode.

Affected: H (owns swap policy), C (encoder_id field; restore-time consistency check; flagging stale_embedding nodes).

### T10 — `activate_neighbors` / spreading-activation arousal contract

**Resolution: single sample per attend (F's position locked).** F samples `composite_affect` and `arousal` exactly once at the start of each `attend` call. C's `spread` operates under a single affective context for the entire pass. Re-sampling mid-spread is forbidden — it would let one branch's affect feedback shape another branch's gating in the same pass and lose the property "this thought happened under this feeling."

A's per-traversal `gate_attention` reads the same composite the AttentionFrame snapshotted, not a fresh one.

Affected: F (samples once), A (gate_attention can be called with a snapshot composite, not the live one), C (spread takes the snapshotted composite as input).

### T11 — Rumination detection ownership

**Resolution: F detects, with self-protective inject; A may also detect via its own arousal-stuck heuristic (both layered).**

- F's `processing_loop` enforces a hard ceiling on `max_internal_ticks` (cap=8 regardless of caller). At cap, F injects a small `affect.inject(PROCESSING, low_arousal_seed, magnitude=0.1)` to nudge toward calm. This is F's only direct write to A and is justified as a self-protective reflex.
- A's existing decay-only-drift detection (flatness response) is the inverse condition (too low) and remains.
- If A later evolves an arousal-stuck heuristic of its own, F's injection becomes redundant and may be removed; for v1 keep F's protective injection.

Affected: F (injects), A (no change — F uses existing inject API).

### T12 — Sleep / consolidation alignment

**Resolution: single canonical schedule.** See CONSOLIDATION/SLEEP SCHEDULE section below.

### T13 — CLIP-bootstrap final yes/no for image modality

**Resolution: YES, CLIP is allowed at boot as labeled bootstrap, replaced via HOT_REPROJECT after 5K image writes.** Option B (image:patch-stats-v1) ships as a tested fallback — both encoders are in the registry at first boot; CLIP is the active encoder for image, Option B is dormant. After HOT_REPROJECT, CLIP is removed from active and the successor encoder (graph-distilled) becomes active.

This violates the spec's "no pretrained frozen models as core" only at startup. After ~5K image writes (likely several days of active image input or much shorter under a dataset), CLIP is gone. Steady state has no pretrained encoders.

Affected: H (registry, swap journal), C (encoder_id on nodes).

### T14 — TOM v1 divergent-belief representation

**Resolution: ship in v1 at first-order only.** "Alice believes X" is representable as concepts with `refers_to` to Alice and `is_a` to a belief-concept. "Alice believes ¬X" requires negated-belief representation which is out of scope for v1. The system can model "Alice does not refer to X" by absence of the `refers_to` edge, and can model "Alice has a high belief_divergence_estimate" via H's per-agent record. That is sufficient for the lying simulation E and D require: divergent expressions can be evaluated on whether they push Alice's `believed_to_know` set away from the system's truth.

Higher-order ToM ("Alice believes I believe X") is recursive `refers_to` traversal which C supports but F may not gate cheaply. v1 ships first-order; flag for v2.

Affected: none for v1.

### T15 — Honesty/deception default — character_baseline

**Resolution: defer to character vector + W-projection learning; no explicit personality parameter at v1.**

The architecture is symmetric: honesty and deception both emerge from E.decide_expression. Whether the system tends toward honesty or deception is a function of:
- W (which gaps hurt) — slowly learns via Hebbian binding
- composite_weights (how much character vs reaction dominates) — fixed at v1
- predicted_audience_accuracy — emerges from observation
- discomfort_scale — adaptive EMA

At v1, the only personality parameter is `birth_seed` (which seeds A's initial character vector). Two minds with different seeds will tend toward different honesty defaults purely through the resulting differences in character and W shaping which gaps register as discomfort.

Default character_baseline is therefore not specified — it is `~ N(0, 0.05)` from the seed. This is the spec's intent ("dimensions emerge meaning through experience, not design"). If empirical run shows the system never lies or always lies, revisit by exposing a small seed bias.

Affected: none for v1; flagged for v2 as "personality knobs."

### T16 — Suppression vs spoken-regret semantics

**Resolution: shared OUTPUT trigger, differentiated by `purpose_hint=SUPPRESS` and `decision=SUPPRESS`.**

When G's decide returns SUPPRESS:
- The would-have-been candidate is still re-encoded and the OUTPUT-trigger gap is computed and routed through B → A. The system feels what it would have felt for emitting.
- No actual surface is delivered to H.emit_output.
- The self-overhearing event does NOT fire (nothing was emitted).
- E.on_suppression is called so identity records "the system chose not to say X."
- D receives a replay-eligible event tagged SUPPRESSED.

This makes suppression observable to the system as a felt act ("I held my tongue") without doubling the affect dimensions. The differentiation between "I said something I regret" and "I held my tongue when I should have spoken" lives in E's `expressed_self_history` (chosen vs suppressed entries) and in the narrative_role inference, not in distinct affect channels.

If empirical observation shows these two cases need to feel architecturally different, add a SUPPRESSED_INTENT tag to the OUTPUT-trigger gap that A's W can learn to color differently. v1 does not.

Affected: G (suppression flow), E (records), A (no change).

### T17 — `homeostatic_target_ema` ownership

**Resolution: D owns it (D's position locked).** Lives inside `WorldModelMetadata`. A's spec already locks five layers; A would have to add a sixth pseudo-layer to host this, which violates the layer count and the parsimony principle.

D updates `homeostatic_target_ema` as an EMA of composite affect during periods where `arousal < 0.15`. Half-life is ~hours. D consumes it directly in `extract_wants` (D1).

Affected: D (owns), A (no change).

### T-D-1 — A.simulate_inject

Already covered under LOCKED CONTRACTS (A section). Pure function; D calls during chain rollout.

### T-D-2 — A.set_nudge_gain_multiplier

Already covered. Thread-local multiplier scope around replay events.

### T-D-3 — C.suggest_precedes_edge

Already covered. D14 uses it after replay-driven writes.

### T-F-1 — C.spread returning traversed edges

**Resolution: extend C.spread return type to a `SpreadResult` with `active_set` and `traversed_edges`.** Already locked above. F's HabitPath tracking is sharp; without it, F runs in approximate-paths mode (degraded but functional).

### T-F-2 — A.gate_attention.predictive_score source

**Resolution: F supplies `predict_prior` to C inside the AttentionFrame; C looks up `dst_concept_id` per traversal and passes `1.0 + predict_prior.get(dst, 0.0)` clamped to [1, 2] as `predictive_score` to A.gate_attention.** Already locked above.

### T-G-1 — project_affect_to_repr direction reconciled with A's W

**Resolution: G's `project_affect_to_repr` is `pinv(A.W)` (the tighter contract).** A exposes `A.project_affect_to_repr(affect_vector) -> RepresentationVector` which internally uses the Moore-Penrose pseudoinverse of W (cached and updated when W is updated). G calls A's helper rather than maintaining its own W_g.

Rationale: coupling cost is small (one helper function); the benefit is that G's style biasing and B's gap projection share the same learned mapping. A mind that has learned which affect dimensions correlate with which representation directions for *gaps* uses the same correlation for *intents*.

Affected: A (exposes helper), G (uses it instead of independent W_g).

### T-E-1 — G producing both internal_repr and surface_repr in same encoder space + always at least one honest candidate

**Resolution: locked, G complies.** G must always produce at least one honest CandidateExpression where `internal_repr ≈ surface_repr` (by E's contract). Both representations live in D_REP=256 encoder space (H's encoder). G's degenerate fallback (when its text generator cannot read out internal_repr cleanly) is to set `internal_repr = centroid of top_active_concepts.embedding`; this is weaker but sufficient.

If G fails to produce an honest candidate, E synthesizes one (E F4) — but this is logged as a contract violation.

Affected: G (contract), E (fallback synthesis).

### T-E-2 — F supports cheap affect-gated spread restricted to refers_to-reachable subgraph

**Resolution: locked. C exposes `spread_restricted` (above); F calls it when E supplies a ToM intent.**

### T-E-3 — H auto-tags incoming utterances with refers_to to speaker's agent_concept

**Resolution: locked.** H10b already does this; the synthesis re-affirms it as a hard contract.

### T-G-2 — C splits expresses edge into expresses_template and expresses_token

**Resolution: NO — single `expresses` edge.** Already covered in DATA-STRUCTURE LOCKS. Disambiguate at the destination node level.

### T-H-1 — ConceptNode adds encoder_id field

**Resolution: locked.** Already covered. ~10 bytes per node, ~50 KB total at 5K nodes.

### T-H-2 — ConceptNode adds source_at_birth field

**Resolution: NO — defer to v2.** E maintains a side-table keyed by concept_id, populated from H's stimulus stream. Lower coupling cost; same query power. Reconsider when multiple components need the field.

---

## BOOT SEQUENCE

The precise startup procedure from "process starts" to "first surprise written."

```
T=0   Process starts, main loop instantiated.

T=1   Persistence layer probes for prior session files:
        a.bin (A's affect stack + W matrix + birth_seed)
        c.mind (C's graph + encoder_id metadata)
        d.bin (D's replay buffer + WorldModelMetadata)
        e.bin (E's IdentitySpine)
        f.bin (F's HabitOverlay + PerceptualBaseline)
        g.bin (G's StyleState + seed_template_ids + recent ExpressionLog)
        h.bin (H's EncoderRegistry + AgentRegistry + EncoderSwapJournal)

T=2   IF NO PRIOR SESSION (first boot):
        - Generate birth_seed (random uint64) and birth_time (now).
        - A.init_at_birth(birth_seed):
            - character ~ N(0, 0.05), seeded
            - disposition = 0.5 * character
            - mood = 0
            - working = 0
            - reaction ~ N(0, 0.02)
            - W initialized as random orthonormal, seeded
            - surprise_scale_ema = 0.5
        - C is empty (no nodes, no edges).
        - D's ReplayBuffer is empty; WorldModelMetadata is fresh (homeostatic_target_ema=character).
        - E.IdentitySpine fresh: birth_seed copied, mind_uuid generated, character_baseline=character,
          self_concept_id allocated as a single empty concept in C ("self") and pinned by E,
          unknown_id allocated and pinned, no narrative_anchors yet, no others.
        - F.HabitOverlay empty. F.PerceptualBaseline empty.
        - G.StyleState defaults: lexical_register=0, verbosity_mean=12.0,
          modality_preference=[0.6, 0.2, 0.2], revision_temperament=0.5, template_familiarity empty.
          Seed template inventory of ~50 templates is loaded into C as concept nodes
          (abstraction_level=1, pragmatic_function set, expresses-edges to surface tokens
          where applicable). G pins these.
        - H.EncoderRegistry: install text:char-trigram-bow-v1, audio:mel-stats-v1,
          image:clip-vit-b32-bootstrap (if CLIP weights present) and image:patch-stats-v1
          (always available). Active image encoder = CLIP if present, else patch-stats.
          AgentRegistry: register self_id and unknown_id (already pinned by E).

T=3   IF PRIOR SESSION (restore):
        - Load h.bin first (encoders must be present before c.mind decoding).
          Validate every encoder_id referenced in c.mind is in registry.
          Missing encoders → mark referencing nodes stale_embedding.
        - Load c.mind. Rebuild edges_in. Rebuild type_index. Rebuild name_index.
          Either deserialize or rebuild embedding_index (HNSW). Run consistency pass.
          Apply forward-decay to affect_trace.running_state per node using last_affect_update.
        - Load a.bin. Apply elapsed-time decay to all five layers using their half-lives.
          Re-jitter reaction by N(0, 0.02). Restore W matrix. composite_cache invalidated.
        - Load e.bin. Run E.verify_continuity(now):
            - Confirm birth_seed matches A's. Mismatch → spine is from different mind, reinit.
            - Compute character_drift; alarm if > 0.4, downgrade if 0.1-0.4 (write AWAKENING anchor).
            - Drop pins whose concept is gone.
            - Refresh last_touched on surviving pins.
            - Update character_baseline.
        - Load d.bin. Validate replay buffer; drop entries whose support_set is fully
          tombstoned. Restore WorldModelMetadata.
        - Load f.bin. Drop habit entries whose concept_id no longer exists.
        - Load g.bin. Restore StyleState. If template_familiarity references missing
          templates, drop those entries.

T=4   Main loop ready. Subsystems are in sync.

T=5   Main loop begins ticking at MAIN_LOOP_HZ=50.
        Per tick:
          - H.tick(now): drain IngestQueue, encode any pending raw input.
          - For each new Stimulus:
              - H.dispatch:
                  H.attribute_to_agent (if WORLD)
                  B.predict (if not already done at anticipation/streaming)
                  B.observe → emits gap → A.inject → A.nudge_chain
                                         → C.write_on_surprise (if surprise)
                                         → D.replay_push (if surprise)
                  H.update_agent (if WORLD, after match)
              - F.attend(INPUT, seeds=H_provided, now)
              - F.processing_loop(input_frame, max_internal_ticks=4):
                  Each iteration:
                    F.attend(PROCESSING, seeds=top-K of prior, now)
                    B.predict + B.observe (mid-thought)
                    A.inject(PROCESSING) if surprise
                  Settle when B.confidence > 0.7 and Jaccard(prev, curr) > 0.8.
              - D.propose_and_choose:
                  Returns ActionDescriptor + CommittedDecisionRecord.
              - If chosen action is EXPRESS:
                  E.snapshot_private_state
                  G.request_expression(intent)
                    G.read_state, G.compute_intended_representation, G.predict_output (B at OUTPUT)
                    G.generate_candidates (incl. re-encoding via H.encode for each)
                    For each candidate: B.observe (OUTPUT trigger), A.inject(OUTPUT)
                    E.decide_expression(candidates, private_state)
                      May iterate revision up to MAX_REVISIONS=2.
                  If COMMIT:
                    H.emit_output (delivers to world AND injects INTERNAL_OUTPUT_ECHO back through H)
                  If SUPPRESS:
                    E.on_suppression, D records SUPPRESSED replay entry.
              - If main loop is otherwise idle (no fresh Stimulus, no expression intent,
                arousal < 0.6, no recent_observation_t in last 5s):
                  D.replay_loop(now):
                    Possibly run replay_one for up to PER_WINDOW_REPLAY_BUDGET=64 events.

T=*   Persistence:
        - All components save every SAVE_INTERVAL=60s (aligned).
        - A saves a.bin, C saves c.mind (atomic temp-rename), D saves d.bin, E saves e.bin,
          F saves f.bin, G saves g.bin, H saves h.bin.
        - On clean shutdown (SIGINT/SIGTERM): flush all saves before exit.
```

The "newborn → operating" transition is at T=4. From T=4 onward the system is fully operational; the first real Stimulus produces the first real surprise (T=5 first iteration), which writes the first concept and begins shaping the character.

---

## CONSOLIDATION / SLEEP SCHEDULE

A single canonical schedule that all consolidating components honor.

There are three cadences:

### Runtime cadence (continuous)

These run during normal operation, no sleep needed:

- **C.forget loop (idle trigger).** When the input pipeline has been quiet for > 5 s and no replay is currently active: run a longer prune pass (up to 256 evictions or 5% of nodes, whichever first). Bounded so it never stalls a future input.
- **D.replay_loop (low-input trigger).** When `now - last_real_observation_t > 5 s` AND `arousal < 0.6` AND `replays since low_input_run_started_t < PER_WINDOW_REPLAY_BUDGET`: run replay_one on a cooperative coroutine.
- **A.propagate_up.** Triggered immediately after every `inject`; also runs every ~10 s on the main loop's soft cadence for the slower hops.
- **A.surprise_scale_ema update.** Continuous; runs inside every `inject`.
- **F.consolidate_habits (idle trigger).** When `pending_habit_obs` is full or `now - last_consolidated_t > CONSOLIDATE_INTERVAL = 600 s`. Runs inline in `record_attention_event`.
- **B.calibration_update.** Every CALIB_INTERVAL=200 ticks.

These do not require an explicit "sleep" mode; they happen during the system's natural quiet moments.

### Sleep cadence (explicit)

A dedicated "sleep" mode is invoked by the host environment (CLI flag, scheduled job, or operator action). When invoked:

```
SLEEP MODE ENTERS:
  - Main loop continues to tick but rejects new external input
    (H.ingest_text/image/audio return SLEEPING).
  - A.consolidation_mode = TRUE:
    nudge_gain temporarily boosted by 1.5× for upper layers (working→mood→disposition→character).
    The slow integration that normally takes weeks happens at 1.5× speed during sleep.
  - D.replay_loop runs without the per-window budget cap. Multiple consecutive replay windows
    are allowed, with 30 s cooldown between them.
  - E.narrative_replay_seed feeds D more aggressively (top-5 anchors instead of top-3).
  - F.consolidate_habits is forced to run at start and at end of sleep, regardless of pending_habit_obs.
  - C.prune pass runs at the start (one large pass, up to 1000 evictions) and at the end (another).
  - All persistence saves run at sleep start AND sleep end.

SLEEP DURATION: configurable; default 8 h of wall-clock OR until external wake signal.

SLEEP MODE EXITS:
  - A.consolidation_mode = FALSE.
  - H.ingest_* re-enabled.
  - One final F.consolidate_habits.
  - One final persistence save.
  - The "wake-up" produces an AWAKENING-style narrative anchor only if duration > 4 h.
```

The host environment is responsible for triggering sleep — there is no automatic detection. This honors the spec's design principle of emergence over injection: the mind doesn't decide when to sleep; the operator does (analogous to a child being put to bed).

### Long-cadence (per-day)

- **F.PerceptualBaseline refresh:** at every consolidate_habits run.
- **C.abstraction_formation:** triggered every M=50 write events OR when embedding_index detects a dense cluster. Runs during runtime, not sleep-locked.

### Alignment summary

All four components that requested a consolidation/sleep coordination point (F's habit consolidation, D's replay scheduling, A's possible consolidation_mode, C's prune pass) align under the schedule above:
- Continuous low-cadence runtime work happens autonomously during quiet moments.
- Heavy consolidation (the equivalent of REM/SWS sleep) happens only during explicit sleep mode.
- Persistence saves anchor at sleep entry/exit and at SAVE_INTERVAL=60 s during runtime.

---

## IMPLEMENTATION ORDER

The user has been explicit: AFFECT before EXPRESSION. Beyond that, the dependency graph in SPEC.md gives substrate (A, B, C) before everything else. Within substrate, A is foundational because both B and C consume affect. Within wave 2, the dependency order is D < E < F < G < H (each later component reads from earlier ones), but in practice they overlap.

**Phase 1 — substrate scaffolding, vertical slice (~2 weeks).**

1. **A skeleton.** AffectStack with five layers, decay_layer, inject (no W yet, unit projection), composite, stamp, gate_attention (additive only — skip arousal-shape branching), persistence.
2. **C skeleton.** ConceptNode without encoder_id, ConceptGraph container, write_on_surprise (no abstraction loop, no forget loop), spread (single mode, no edge-type weighting), find_or_match, persistence.
3. **B skeleton.** Predict (degenerate when graph empty), observe, ComputeGap, SurpriseScoring (with cold-start), EmitGap (only routes to A and C, not yet D), persistence.
4. **H minimal.** ingest_text only, text:char-trigram-bow-v1 encoder, dispatch with predict/observe handshake, persistence.
5. **Smallest end-to-end vertical slice:** text in → encode → predict → observe → A.inject → C.write_on_surprise (when crossing threshold) → log a stimulus_id and the surprise_score. Run for 100 text inputs and verify: surprise rate stabilizes near ALPHA=5%, character vector slowly drifts, concept count grows then plateaus.

This vertical slice exercises the spec's core failure modes:
- The "one substance" claim — every text input produces a graph node when surprise crosses, and that node has affect stamped on it.
- "Surprise as the only learning signal" — no surprise = no write.
- "Affect as the medium" — A's reaction layer moves on every observation; character moves slowly.

**Phase 2 — flesh out substrate (~2 weeks).**

6. **A complete.** W matrix + Hebbian update, project_gap helper, simulate_inject, set_nudge_gain_multiplier, project_affect_to_repr (pinv W), full arousal-shape gate, init_at_birth from seed, all failure modes.
7. **C complete.** All edge types, abstraction formation, forget loop with salience computation, spread with PERCEIVE/PREDICT/SIMULATE modes, spread_restricted, suggest_precedes_edge, encoder_id field, all failure modes.
8. **B complete.** Confidence calibration, simulate_step, reconcile_simulation, full pending_predictions lifecycle, replay_origin honor, all failure modes.

**Phase 3 — wave 2 in dependency order (~3 weeks).**

9. **F.** AttentionFrame, attend, processing_loop, HabitOverlay (without persistence yet), inject_habit_seeds. Connect F's processing_loop into the main loop.
10. **D.** ReplayBuffer, replay_push, replay_loop, simulate_chain (uses B and A.simulate_inject), feel_step, propose_candidates, score_chain, choose_chain. Connect propose_and_choose into the main loop. Verify replay generates learning events without inflating surprise stats.
11. **E.** IdentitySpine, snapshot_private_state, decide_expression (initially without simulate_emission, with simple gap math), narrative anchors, pin management. Verify expression decisions can be made even without G.
12. **G.** StyleState, generate_candidates with seed templates, render_text. Image and audio renderers can be stubs at v1. Connect to E.decide_expression.
13. **H complete.** ingest_image (CLIP bootstrap + patch-stats fallback), ingest_audio, inject_internal, AgentRegistry with attribution, register_agent, query_shared_knowledge, all failure modes, encoder swap mechanism.

**Phase 4 — sleep, persistence, observability (~1 week).**

14. **Sleep mode.** Implement consolidation_mode flag in A, the sleep envelope around the main loop, host-environment trigger.
15. **All persistence aligned.** SAVE_INTERVAL=60s, atomic temp-rename for every component, restore paths exercised.
16. **Frontend visualization.** topology_event subscriptions, AttentionStats, ExpressionLog, narrative_anchors view.

**Smallest E2E vertical slice that exercises spec's failure modes (Phase 1 deliverable):**

- Text-only mind, no image/audio.
- Run 100 text inputs from a small corpus (e.g., short sentences about a few topics).
- Observable from logs:
  - Surprise rate converges to ~5%.
  - Character vector drifts visibly between input 1 and input 100.
  - Concept count grows to ~20-50 then plateaus as recurrence dominates.
  - Affect runaway does not occur.
  - Decay-only drift does not occur (system is mildly aroused throughout).
  - Persistence/restore works: kill the process, restart, observe character_baseline matches and a few new inputs produce similar behavior.

This vertical is the architectural smoke test before D/E/F/G/H come online.

---

## REMAINING EMPIRICAL QUESTIONS

These are deliberately not answered in this synthesis. They require running the system to resolve.

1. **N=12 saturation.** Does fixed N_AFF=12 saturate (multiple distinct surprise patterns aliasing to the same affect direction) within realistic session lengths? If yes, growable N becomes a v2 priority.

2. **ALPHA_SURPRISE_TARGET=0.05.** Is 5% the right target rate? Too low and learning starves; too high and the graph fills with noise. Tune by observation.

3. **REPLAY_NUDGE_GAIN_MULTIPLIER=0.30.** Does heavy replay produce visible character drift? If yes, lower to 0.15. If replays fail to integrate at all, raise to 0.50 (do not exceed).

4. **CONCEPT_CEILING=5000.** Will the prune-pressure mechanic hold this gracefully under heavy input, or will salience-based eviction repeatedly evict load-bearing scaffolding? Watch for thrashing.

5. **CLIP-bootstrap-vs-patch-stats.** Does the patch-stats successor encoder (graph-distilled) actually produce semantic neighborhoods comparable to CLIP-bootstrap+HOT_REPROJECT after 5K writes? If not, CLIP may need to stay longer or the successor recipe needs improvement.

6. **MAX_SIM_DEPTH_LOCAL=6.** Is depth-6 the right rollout? Deeper compounds prediction error fast (confidence drops geometrically); shallower under-imagines. Empirical.

7. **INDIFFERENCE_BAND=0.05 + softmax temperature.** Does this produce decisions that feel sometimes-committed sometimes-wandering? Tune.

8. **Does the abstraction loop in C actually produce a usable template repertoire for G within a realistic session?** If not, G needs a small in-house mutation operator.

9. **Honesty/deception default.** Does the symmetric architecture land at ~50% divergent emissions or somewhere else? What's the natural mode? Whatever it is, decide whether it's acceptable; if not, surface a personality knob.

10. **Counterfactual replays worth their cost?** They double the replay buffer's effective use of an entry. Empirical whether they produce useful learning or just confuse the prediction engine.

11. **Mid-spread B prediction granularity.** Per-tick (locked) vs per-step (rejected). If empirical run shows the system fails to notice mid-spread surprises that matter, escalate to per-step.

12. **CHAR_DRIFT_TOLERATE / CHAR_DRIFT_ALARM.** E's continuity-break thresholds are guesses (0.1 / 0.4). Tune against observed restore behavior.

---

## KNOWN GAPS / FOLLOW-UP DOCS

These are not fully resolved by this synthesis. Each needs a follow-up design pass before code can ship for that area.

1. **Higher-order theory of mind (second order, third order).** v1 ships first-order ToM. Recursive `refers_to` traversal works in C, but F's gating cost grows polynomially with order. A follow-up doc on bounded ToM recursion is needed before any chat scenarios involving multi-agent reasoning ship.

2. **Sleep-mode policy details.** The synthesis specifies the envelope (1.5× nudge gain, replay budget unlocked, etc.) but the precise tuning (how long, what triggers wake, whether sleep can be partial) is host-environment-dependent and needs a follow-up alongside the host integration.

3. **N-dimensionality expansion at runtime.** Both A and the spec contemplate growable N. The persistence cost (re-encoding 5K affect_traces on every grow) and the consequences for W are not fully specified. Defer until empirical N=12 saturation evidence justifies it.

4. **Image and audio renderer details.** G ships with stubs at v1 (G6/G7 thin interfaces). A follow-up "Modality renderers" doc specifies the IMAGE layout grammar and the AUDIO prosody envelope at sufficient detail to ship beyond text.

5. **Frontend visualization protocol.** Many components expose `subscribe_topology_events` / `current_attention_state` / `style_snapshot`. A follow-up "Frontend protocol" doc unifies these into a single websocket schema.

6. **Personality knobs.** T15 (honesty/deception default) and several other "personality parameters" (composite_weights, λ_aff in G2, w_align/w_audience/w_leak/w_style in G4d) are emergent at v1 but may need explicit knobs at v2 for distinct mind variants.

7. **Multi-agent / multi-audience expression.** v1 treats audience as a single concept_id. A follow-up specifies how to handle a set of agents with conflicting preferences in a single emission.

8. **Long-tail expression-style summary in expression_calibration.** E's spec flagged that 256 entries holds maybe a day of conversation. A follow-up specifies the EMA-of-EMAs structure for week/month scale expression style.

9. **Successor encoder training pipeline.** H sketches the contrastive-distillation recipe but does not specify the training infrastructure. A follow-up details what runs offline to produce the successor encoder.

10. **Negated-belief / divergent-belief representation in C.** v1 supports "Alice does not refer to X" by absence. A v2 doc may specify a `believes_not` edge type or a per-edge polarity bit if empirical run shows the absence model is insufficient for nuanced lying.

---

## PHASE 5 ROADMAP — THE PROCESSING PHASE

**Problem:** mind echoes input because seed concept dominates active set
at activation=1.0 at expression time. Associated concepts arrive at
0.3-0.5 — too weak to shift the centroid or win as fillers.

**Root cause:** no processing ticks between input and expression.
`F.processing_loop(max_internal_ticks=4)` from synthesis is unimplemented.

**Fix:**

1. **`F.processing_loop`** — run 4 internal spread ticks after INPUT attend,
   before action selection. Each tick diffuses activation outward.
   Input dominance decays. Associated concepts rise.
   This is where association happens.

2. **Seed down-weighting** — treat input as evidence not mental state.
   After processing ticks, re-weight centroid to give seed 0.4 weight
   max, letting spread results dominate.

3. **Filler suppression on `force_respond`** — when the social constraint
   fires, exclude the input's own encoded name from the filler pool.
   The mind shouldn't quote the question back.

These three together produce genuine association.

The encoder is a separate concern — char-trigram BoW puts semantically
related sentences too far apart. Phase 5+ encoder upgrade to something
with learned semantic neighborhoods.

---

End of unified specification.
