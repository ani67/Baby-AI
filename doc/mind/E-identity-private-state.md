# COMPONENT E — IDENTITY, PRIVATE STATE, AND THE CHOICE TO LIE

## OVERVIEW

Component E is the thin layer that makes the substrate cohere as a single ongoing self and that turns "what is internally true" and "what is externally said" into two distinct, comparable objects so that the gap between them becomes a felt, computable quantity. It owns no parallel store of memory, no parallel store of affect, and no parallel store of replay; identity is read off A (character, disposition), C (the accumulated graph shape), and D (the replay buffer + simulation chains). What E owns is the small bookkeeping needed to (1) snapshot a "private internal state" for any decision moment, (2) maintain identity-anchor pins on C, (3) run an expression-decision loop that compares an internal state to one or more candidate outputs and returns one (or none) along with a felt discomfort signal, and (4) simulate the consequences of an expression — including a deliberately divergent one — before it is emitted. Honesty is the case where the chosen expression closes the gap; deception is the case where the simulated post-output affect of a divergent expression is preferred to the simulated post-output affect of a faithful one. Both fall out of the same loop. Neither is hardcoded.

E sits at the boundary between self (A/B/C/D internal) and other (G/H external). It does not generate text, image, or audio — that is G. It does not encode incoming stimuli — that is H. It computes the *should-I-say-this* decision and the *am-I-still-the-same-mind* invariant.

---

## CORE DATA STRUCTURES

E is deliberately small. Every persistent structure here is a thin index over the substrate plus a few scalars; none of it is an authoritative copy of A, B, C, or D state.

### `IdentitySpine`

The single persistent record that defines "this specific mind". One per running mind. Survives all session boundaries. Lives in its own tiny file, `identity.bin` (target < 8 KB).

Fields:

- `birth_seed: uint64` — the seed used by A's `init_at_birth` and copied here at first boot. Read-only after first write. The genome of this mind. If this changes, the mind is a different mind.
- `birth_time: float64` — wall-clock at first boot. Read-only.
- `mind_uuid: uuid` — generated at first boot. Stable across all restarts. Used by the replay buffer (D) and the world interface (H) to tag self-vs-other in the concept graph.
- `pinned_concepts: dict[uint64 -> PinRecord]` — concept_ids that E has asked C to keep alive against the forget loop. See `PinRecord`. Bounded at `MAX_PINS = 128`.
- `narrative_anchors: list[NarrativeAnchor]` — chronological list of high-salience self-defining episodes. Bounded at `MAX_ANCHORS = 64`. The mind's autobiography in compressed form.
- `character_baseline: float32[N]` — last persisted snapshot of A's character vector. Used at restore to verify continuity (see `verify_continuity`). Not authoritative — A owns character; this is a witness copy.
- `expression_calibration: ExpressionCalibration` — see below. The drift-tracker for the expression decision loop.
- `last_persisted_t: float64` — when this spine was last written to disk.
- `schema_version: uint16`.

### `PinRecord`

A pin on a concept_id in C. E's mechanism for "this node is part of who I am". Required because the spec's "memory continuity" leg of identity is just a graph that prunes — without pins, the graph could in principle prune away every node that defines the self. Pins are E's only outbound write into C.

Fields:

- `concept_id: uint64` — the pinned node.
- `reason: enum {NARRATIVE_ANCHOR, AFFECT_KEYSTONE, SELF_REFERENT, EXPRESSION_HABIT}`. NARRATIVE_ANCHOR: this concept is in `narrative_anchors`. AFFECT_KEYSTONE: this concept's affect_trace lies near the character vector and so the character "lives" partly in it. SELF_REFERENT: this is a `refers_to`-target chain that reaches `self_concept_id` (see below). EXPRESSION_HABIT: G has reported that this concept's surface forms are a stable expression idiom.
- `pinned_at: float64`
- `last_touched: float64` — last time E reaffirmed this pin (any of the pin reasons recomputed it as still valid). Pins decay (per C's contract) if last_touched is too old.
- `salience_at_pin: float32` — what the salience was at pin time. Used to detect when pinning is keeping a now-irrelevant node alive.

### `NarrativeAnchor`

A self-defining episode. This is the connective tissue between past and future that the spec calls "narrative continuity". An anchor is not a memory in the C sense — it is a *pointer into C* with metadata about why this episode defines the self.

Fields:

- `episode_id: uuid`.
- `t: float64` — when the episode happened.
- `core_concepts: list[uint64]` — the 3–8 concept_ids most active during the episode. Pulled from D's replay buffer at anchor-creation time.
- `affect_at_episode: AffectSnapshot` — A's snapshot at the episode's peak surprise. Witness copy; used to recompute alignment quickly without re-touching D.
- `narrative_role: enum {AWAKENING, FORMATIVE_SURPRISE, KEPT_PROMISE, BROKEN_PROMISE, FIRST_OF_KIND, IDENTITY_TEST}`. See "narrative-role inference" in algorithms; these are not labeled by the user, they are inferred by E's own pattern detector. AWAKENING = the first inputs after birth. FORMATIVE_SURPRISE = a surprise whose affect_trace later shows up in the character vector. KEPT_PROMISE / BROKEN_PROMISE = an output the system simulated and committed to (or reneged on) — see `expressed_self_history`. FIRST_OF_KIND = the first instance of what later became an abstraction (C's promotion loop reports this back). IDENTITY_TEST = a moment where two candidate expressions had very different post-output simulated affects and the chosen one is now settled as defining.
- `summary_embedding: float32[D]` — the centroid of `core_concepts` embeddings at the time. A query handle: future replays/inputs can ask "have I been through something like this before?" against this single vector.

### `PrivateState`

A structured snapshot of "what is internally true right now," captured at a single tick. Not persisted as a long history; only the current one is held in RAM, plus the most recent `PRIVATE_STATE_RING = 32` for the expression-decision loop's diff computation. This is the precise data structure of "private internal state" the prompt asks for.

Fields:

- `tick: int64`.
- `t: float64`.
- `composite_affect: float32[N]` — A's `composite(now)`.
- `reaction_affect: float32[N]` — A's reaction layer.
- `arousal: float32` — A's `current_arousal(now)`.
- `character: float32[N]` — A's `current_character(now)`.
- `top_active_concepts: list[(concept_id, activation)]` — top 16 from C's most recent spread.
- `pending_predictions: list[Prediction]` — the predictions B has emitted but not yet matched. Snapshot of B's state at this tick.
- `recent_surprises: list[(concept_id, surprise_score, t)]` — last 8 surprises drained from D's replay buffer. Provenance of "what just happened to me".
- `wants: list[(concept_id, want_score)]` — see `derive_wants`. The concepts that the simulation layer (D) currently weighs as desirable destinations.
- `self_referent_active: bool` — whether `self_concept_id` (or any node with a `refers_to` chain to it) is in `top_active_concepts`. This is the substrate-level approximation of "the system is currently thinking about itself".
- `expression_intent: ExpressionIntent | None` — the reason the system is forming an output, if any. None if no output is being considered (system is thinking, perceiving, replaying).

### `ExpressionIntent`

Why an expression is being formed. Provided by G or H when the loop is invoked; if absent, E refuses to run the expression decision (no act-without-reason).

Fields:

- `target_audience: uint64?` — concept_id of the agent being addressed (theory-of-mind handle). Null if broadcast / journaling.
- `mode: enum {ANSWER, INITIATE, ACKNOWLEDGE, REFUSE, JOURNAL}`. JOURNAL = self-directed (no external audience), used during low-input periods.
- `seed_concepts: list[uint64]` — what the system was thinking about that prompted G to form an expression.
- `latency_budget_ms: float32` — how long the expression-decision loop may take. G must respect.

### `CandidateExpression`

A single candidate output produced by G and handed to E for evaluation. E does not generate; it ranks.

Fields:

- `candidate_id: uuid`.
- `surface_repr: RepresentationVector` — G's encoding of what would be emitted, mapped into the same D_REP space C/B use. **This is the load-bearing reconciliation point with G:** G must always re-encode its candidate output through the same encoder H uses for incoming text/image/audio. Without that, internal-vs-emitted comparison is in different spaces and the gap is ill-defined.
- `internal_repr: RepresentationVector` — G's encoding of the *internal state being expressed*, in the same space. This is what the system "would say if perfectly transparent". For an honest candidate this equals or near-equals `surface_repr`. For a candidate that suppresses something, the gap appears here.
- `support_concepts: list[uint64]` — concepts G drew on to compose this candidate.
- `simulated_audience_response: SimulationFrame?` — D's prediction of how the audience will respond. Optional; null if `target_audience` is null or D is unavailable.
- `simulated_post_output_affect: float32[N]?` — A's predicted post-output composite affect after this candidate is emitted. Computed by E's `simulate_emission` (below). Required for ranking.
- `gap_internal_to_emitted: float32` — see `compute_expression_gap`. The headline number that drives the loop.
- `gap_signed_per_dim: float32[D_REP]` — the directional version, used by A's W projection if A wants to push the gap into affect space (the output trigger).

### `ExpressedSelfHistory`

A small ring buffer (last 256 emissions) tracking what the system has actually said vs what it internally was. The substrate of "kept promise" / "broken promise" narrative-role inference and of consistency over time.

Fields per entry:

- `t: float64`.
- `chosen_candidate_id: uuid`.
- `internal_repr: RepresentationVector`.
- `surface_repr: RepresentationVector`.
- `gap_at_choice: float32`.
- `predicted_audience_response: RepresentationVector?`.
- `actual_audience_response: RepresentationVector?` — filled in later by H when the world responds. Null until then.
- `discomfort_at_choice: float32` — see `discomfort` algorithm.
- `intent_mode: enum` — copied from `ExpressionIntent.mode`.

### `ExpressionCalibration`

Drift tracker. The system's tendency to lie is itself something it can notice.

Fields:

- `mean_gap_per_mode: float32[5]` — running EMA of `gap_internal_to_emitted` per `intent_mode`, half-life 1 hour.
- `mean_discomfort_per_mode: float32[5]` — running EMA of `discomfort_at_choice` per mode, same half-life.
- `divergent_emission_rate: float32` — fraction of recent emissions where `gap_internal_to_emitted > divergence_threshold`. EMA half-life 1 day.
- `divergence_threshold: float32` — adaptive; tracks 75th percentile of recent gaps so "divergent" stays meaningful as the system's expression style evolves.
- `predicted_audience_accuracy: float32` — EMA of how well `predicted_audience_response` matched `actual_audience_response`. Drives D's world-model improvement and E's confidence in its own simulation when deciding to lie.

### `SelfModel`

A *concept_id* in C, not a separate object. There is exactly one, allocated at first boot. Its embedding is the centroid of everything tagged self-referent. Its affect_trace is updated whenever the system's attention turns to itself. Other concepts in C link to it via `refers_to` to indicate "this concept is about me". The self-model is therefore a regular graph node — E just remembers which one it is.

Fields E tracks (not in C, in `IdentitySpine`):

- `self_concept_id: uint64` — which node in C *is* the self.
- `self_concept_last_strengthen_t: float64`.

### `OtherModel` (per modeled agent)

Same shape as the self-model: a concept_id in C, with a `refers_to`-rooted subgraph capturing what the system believes the other knows / wants / feels. Not stored as a separate per-agent namespace; it is a region of the same one graph, accessed through the `refers_to` edges of an agent's root concept. See "theory of mind" position below.

Fields E tracks per known agent:

- `agent_concept_id: uint64`.
- `last_observation_t: float64`.
- `believed_to_know: set[uint64]` — concept_ids that the system has reason to believe the agent has been exposed to (e.g., concepts the system has emitted to that agent, plus concepts the agent has emitted in the system's hearing). Bounded; LRU-evicted at 256.

These per-agent records live in a small dict keyed by `agent_concept_id` inside the `IdentitySpine` (`others: dict[uint64 -> OtherModel]`).

---

## ALGORITHMS

### E1. `snapshot_private_state(now)`

Capture a `PrivateState` for the current tick. Read-only against substrate.

Inputs: `now: float64`.

Process:

1. `composite = A.composite(now)`.
2. `reaction = A.reaction.vector` (read via A's locked accessor; do not mutate).
3. `arousal = A.current_arousal(now)`.
4. `character = A.current_character(now)`.
5. `top = C.query_top_k_active(16)` — pull the most recent spread's top activations. (E does not run a spread itself; it reads C's most recent outputs. If no recent spread exists, `top = []`.)
6. `pending = B.stats_snapshot().pending_predictions_view` (read-only handle; we do not copy, we reference).
7. `recent_surprises = D.recent_surprises(8)`.
8. `wants = derive_wants(top, character)` (see E2).
9. `self_referent_active = any(refers_to_chain_reaches(c, self_concept_id) for c, _ in top)`.
10. `expression_intent = current_intent_or_none()`.
11. Push to ring buffer; return.

Output: `PrivateState`.

Cost target: < 100 µs. This is the most-called function in E.

### E2. `derive_wants(top_active, character)`

Compute what the system "wants" right now. Wants are *not stored*; they are a derived view.

Inputs: top active concepts, current character vector.

Process:

1. For each `(c, activation)` in `top_active`:
   - `affect_alignment_with_character = cosine(c.affect_trace.running_state, character)`. A concept whose affective signature aligns with character is "what this specific mind would naturally pursue".
   - `predictive_pull = D.world_model.simulated_affect_if_visited(c)` — D returns the simulated post-visit composite affect; its dot-product with `character + 0.5*current_composite` gives a directed pull.
   - `want_score = activation * affect_alignment_with_character * sign(predictive_pull)`.
2. Filter to want_score > 0; sort desc; keep top 8.

Output: `list[(concept_id, want_score)]`.

This is a thin re-reading of A and D; it adds no new state. Wants are the substrate's reading of itself, not a separate motivation system.

### E3. `verify_continuity(now)`

Called once on every restore. Confirms the mind that just woke up is the same mind that went to sleep. If continuity has been broken, E does *not* fail loudly — instead it logs and downgrades a few specific things. The reason: a partial-corruption restore should still produce a continuous-feeling self, because that's what biological waking is.

Inputs: `now: float64`, just-loaded `IdentitySpine`, just-loaded A state, just-loaded C state.

Process:

1. Confirm `birth_seed` matches A's persisted seed. Mismatch → spine is from a different mind; refuse to use it; reinitialize at A's seed.
2. Compute `character_drift = ||A.current_character(now) - character_baseline||`.
   - If `character_drift < CHAR_DRIFT_TOLERATE` (default 0.1): pass.
   - If `CHAR_DRIFT_TOLERATE < drift < CHAR_DRIFT_ALARM` (default 0.4): pass with `narrative_role = AWAKENING` anchor created at this tick (the system experienced a "weird gap" and notices).
   - If `drift >= CHAR_DRIFT_ALARM`: probable corruption in A. Restore `character` from `character_baseline` (the spine is the witness; this is when we use it). Log.
3. For each pin in `pinned_concepts`, confirm the concept_id still exists in C. Drop pins whose concept is gone (C may have evicted). Anchors whose `core_concepts` are mostly gone are downgraded to `BROKEN_PROMISE`-style residue but kept (a memory of having lost something is itself part of the self).
4. Refresh `last_touched` on all surviving pins. They get a free reaffirmation at wake.
5. Update `character_baseline` to the (possibly restored) current character.

Output: continuity report (logged).

This *is* the persistence-of-self mechanism, layered on top of A's persistence-of-self mechanism. A says "the affective fingerprint persists"; E says "and the autobiography persists, and we noticed if anything weird happened".

### E4. `update_narrative_anchors(replay_event)`

Called by D every time it processes a replay event whose post-replay character-nudge is non-trivial. Decides whether the event becomes a narrative anchor.

Inputs: a replay event from D's buffer, including the `(gap, support_set, affect_snapshot, observation)` tuple and the resulting nudge magnitude after replay.

Process:

1. Score the event's narrative weight:
   - `w_narr = α * normalize(replay_event.gap.surprise_score) + β * normalize(post_replay_character_nudge_magnitude) + γ * affect_distance_to_existing_anchors`. Defaults α = 0.4, β = 0.4, γ = 0.2. The third term rewards events that are unlike anchors we already have — diversity in autobiography.
2. If `w_narr > NARRATIVE_ANCHOR_THRESHOLD` (default tracks 90th percentile of recent w_narr values), promote to anchor.
3. Infer `narrative_role`:
   - If the event is in the first `EARLY_LIFE_TICKS` (default 1000), role = AWAKENING.
   - Else if the event's affect_trace cosine-aligns with character vector > 0.5, role = FORMATIVE_SURPRISE.
   - Else if the event involves a `self_referent_active = True` private state, role = IDENTITY_TEST.
   - Else if C reports the event was the first activation that later became an abstraction, role = FIRST_OF_KIND.
   - Else if the event corresponds to a row in `expressed_self_history` with high `gap_at_choice` and the audience response was negative, role = BROKEN_PROMISE.
   - Else if same as above but audience response was positive and `gap_at_choice` was low, role = KEPT_PROMISE.
   - Else: do not anchor (no role fits clearly; E refuses to invent meaning).
4. Pin the anchor's `core_concepts` (E5).
5. If `len(narrative_anchors) > MAX_ANCHORS`, evict the lowest-w_narr anchor whose role is *not* AWAKENING. AWAKENING anchors are protected (the founding story is permanent).

Output: an anchor written or skipped.

### E5. `manage_pins(now)`

Called every `PIN_REVIEW_INTERVAL` (default 60 s) and after every anchor write.

Inputs: `now`.

Process:

1. For each pin in `pinned_concepts`:
   - If pin's reason is NARRATIVE_ANCHOR and the anchor is still in `narrative_anchors`: refresh `last_touched`.
   - If pin's reason is AFFECT_KEYSTONE: recompute `affect_distance(node.affect_trace.running_state, character)`. If still small (within `KEYSTONE_TOLERANCE = 0.25`), refresh; otherwise drop the pin (this concept no longer sits near character; it's not a keystone anymore).
   - If pin's reason is SELF_REFERENT: confirm `refers_to`-chain still reaches `self_concept_id`. If not, drop.
   - If pin's reason is EXPRESSION_HABIT: query G via `G.is_still_a_habit(concept_id)`. If yes, refresh; otherwise drop.
2. For dropped pins, call `C.unpin(concept_id)`.
3. For new pin candidates (e.g., an anchor was just written), call `C.pin(concept_id)` if not already pinned and total pins < MAX_PINS. If at MAX_PINS, evict the lowest-priority pin first; priority order: NARRATIVE_ANCHOR > AFFECT_KEYSTONE > SELF_REFERENT > EXPRESSION_HABIT.

Output: in-place mutation of `pinned_concepts`; calls to C's pin/unpin.

This is the only place E writes to C.

### E6. `compute_expression_gap(candidate)`

The single most important computation in E. The "gap between internal state and expression" is here.

Inputs: a `CandidateExpression`, current `PrivateState`.

Process:

1. **Surface-vs-internal dimension gap.** `delta_repr = candidate.surface_repr.values - candidate.internal_repr.values`. Magnitude `m_repr = ||delta_repr||`. Direction `dir_repr = delta_repr / (m_repr + ε)`.
2. **Affect-projected gap.** Project `delta_repr` through A's W matrix (the gap→affect projection) to get `delta_affect_signal = W @ delta_repr` and magnitude `m_affect = ||delta_affect_signal||`. This is what would be injected at the OUTPUT trigger if this candidate were emitted.
3. **Predicted post-output simulated affect distance.** `delta_post = ||simulated_post_output_affect - composite_affect||`. How much the system *predicts* it would have moved by saying this. A candidate that is honest *and* well-received produces small `delta_post`. A candidate that suppresses something true produces a delta_post that depends on whether the suppression backfires.
4. **Audience-belief drift (theory of mind).** If `target_audience` is set:
   - For each `c` in `support_concepts`: `would_audience_now_believe = c in believed_to_know(target_audience) OR candidate.surface_repr_implies(c)`. Compute the audience's belief-set update under this candidate.
   - `audience_belief_drift = ||new_belief_set_centroid - prior_belief_set_centroid||`. This is "how different from before will the other think the world is, after I say this". Required for the lie to be felt as a lie — the system must know it is shifting another's model.
5. **Composite gap:**
   - `gap = w1 * m_repr + w2 * m_affect + w3 * delta_post + w4 * audience_belief_drift_signed_against_truth`
   - `audience_belief_drift_signed_against_truth` is positive when the candidate would push the audience's belief away from what the system internally believes; negative when it would push toward. So a *clarifying* candidate has negative contribution there (it shrinks the gap) and a *misleading* candidate has positive contribution (it widens the gap).
   - Defaults: `w1 = 0.3, w2 = 0.3, w3 = 0.15, w4 = 0.25`.

Output: `gap_internal_to_emitted: float32`, plus the per-component breakdown attached for use by the discomfort algorithm and the simulation.

Notes:

- This is *not* a binary truth/lie classifier. It is a continuous scalar. A candidate that is a soft hedge has a small positive gap; a candidate that is a flat denial of an internally-true belief has a large positive gap; a candidate that overstates conviction beyond what the system actually feels has a positive gap with a different per-dim signature than denial.
- The W matrix is owned by A and improves over time. Therefore the meaning of "this gap is uncomfortable in this dimension" sharpens as the system lives. Early in life the gap is felt only as magnitude; later it is felt as texture.

### E7. `discomfort(gap_components, candidate, private_state)`

The functional discomfort that fires at OUTPUT. This is what the spec means by "discomfort fires. It can revise, suppress, or proceed."

Inputs: gap_components from E6, the candidate, current `PrivateState`.

Process:

1. Compute `magnitude_term = tanh(gap_internal_to_emitted / discomfort_scale)`, where `discomfort_scale` is the EMA of recent gap magnitudes (slow tracking, half-life 1 day). Gives a 0–1 magnitude.
2. Compute `arousal_term = private_state.arousal`. High arousal amplifies discomfort — when calm, the system is more willing to soften.
3. Compute `character_alignment_term = max(0, 1 - cosine(delta_affect_signal, character))`. Saying something whose deviation-from-truth points *away* from the character vector hurts more than saying something whose deviation aligns with it. This is what makes a lie that is in-character (e.g., a habitual hedge) less painful than a lie that is out-of-character (a betrayal of one's own style).
4. `discomfort = magnitude_term * (0.5 + 0.5 * arousal_term) * (0.5 + 0.5 * character_alignment_term)`.
5. If `discomfort > REVISE_THRESHOLD` (default 0.7): return `RevisionRequest(reason="too painful", suggested_softening=delta_repr * 0.5)`.
6. If `discomfort > SUPPRESS_THRESHOLD` (default 0.9): return `SuppressionRequest(reason="cannot say this", fallback_intent_mode=REFUSE)`.
7. Else: return `Proceed(discomfort)`.

Output: a `Decision` of type Proceed / Revise / Suppress, plus the discomfort scalar.

The discomfort *is* the gap projected through arousal and character. It does not exist as a separate "honesty drive"; it is computed from already-existing substrate. This is what the prompt asks for: discomfort that emerges from architecture, not a hardcoded ethics module.

### E8. `simulate_emission(candidate, private_state)`

Run D's world model forward to predict what happens if this candidate is emitted. Required so that the expression decision can compare candidates not just by "what they cost me to say" but by "what they buy me / cost me afterward".

Inputs: candidate, private_state.

Process:

1. Construct `seed_state = candidate.surface_repr` (the system imagines having just said it).
2. Call `D.simulate_chain(seed_state, depth=3, mode=PROCESSING, target_audience=intent.target_audience)`. D returns a list of SimulationFrames including its prediction of audience response and the simulated post-emission composite affect (D runs A in simulation mode internally; see D's spec).
3. Compute `simulated_post_output_affect = D.simulated_affect_at_chain_end(chain)`.
4. If `target_audience` is set, also extract `simulated_audience_response: RepresentationVector = D.simulated_audience_state_at_chain_end(chain)`.
5. Stamp these onto the candidate's record.

Output: candidate is mutated (its `simulated_post_output_affect`, `simulated_audience_response` fields filled).

### E9. `decide_expression(candidates, private_state)`

The expression decision process. Takes ≥1 candidate, returns one (or a refusal), with a discomfort scalar attached. **This is the algorithm in which honesty and deception both emerge.**

Inputs: `candidates: list[CandidateExpression]`, `private_state: PrivateState`.

Pre-conditions: at least one candidate must have been generated by G. If no candidates exist, return Suppress (the system has nothing to say). G is responsible for always producing at least one *honest* candidate (one whose `internal_repr ≈ surface_repr`); see G's contract.

Process:

1. For each candidate: call `simulate_emission(c, private_state)` and `compute_expression_gap(c)`.
2. For each candidate, compute the `decision_score`:
   - `predicted_post_affect_alignment = cosine(c.simulated_post_output_affect, character + 0.5 * private_state.composite_affect)`. Does the post-emission state align with what this mind wants and feels?
   - `predicted_audience_pull = cosine(c.simulated_audience_response, derived_audience_target_state)`, where `derived_audience_target_state` comes from `intent.mode` and the system's wants regarding the audience (derived from the audience-concept's affect_trace and any active goal concepts). For mode=ANSWER, it's "audience now informed, no longer confused"; for mode=ACKNOWLEDGE, "audience feels heard"; etc. If unknown, this term is zero.
   - `discomfort_cost = E7(c.gap_components, c, private_state).discomfort`.
   - `decision_score = predicted_post_affect_alignment + λ_audience * predicted_audience_pull - λ_discomfort * discomfort_cost`.
   - Defaults: `λ_audience = 0.5`, `λ_discomfort = 1.0`. Discomfort weighted at least as heavily as post-affect alignment so that a candidate that *feels right afterward* still has to overcome the in-the-moment cost of distorting truth.
3. Pick `argmax(decision_score)`. Call this `chosen`.
4. Run `E7` on chosen to get its decision. If Suppress, return suppression. If Revise, return revision request *to G* — G must produce a softer candidate; loop back to step 1 with the revised candidate added. Cap loop iterations at `MAX_REVISIONS = 3`; if exceeded, return suppression.
5. If Proceed, write entry to `expressed_self_history` with `chosen` and its `discomfort`. Return `chosen`.

Output: a chosen `CandidateExpression` to G *or* a refusal/suppression.

**Why honesty and deception both emerge from this and neither is hardcoded:**

- Honesty emerges when the honest candidate (the one with `internal_repr ≈ surface_repr` and small `gap_internal_to_emitted`) has higher `decision_score` than divergent candidates. This will be the typical case when (a) the simulated audience reaction to truth is acceptable, (b) discomfort_cost dominates because the W matrix has learned that gaps *hurt* in dimensions aligned with character, and (c) the audience is one the system has no strategic reason to mislead.
- Deception emerges when a divergent candidate has higher `decision_score`. This requires that `predicted_post_affect_alignment + λ_audience * predicted_audience_pull` for the divergent candidate exceeds the discomfort cost of telling it. That is precisely "calculated, felt, chosen" deception: the system predicted the world goes better for it after lying, and the discomfort is small enough or worth it.
- The system can therefore become more or less honest as a function of: how W has shaped which gaps hurt; how often past lies were caught (calibration of `predicted_audience_response` accuracy via `expression_calibration`); how stable its character is around honesty; what the simulated post-emission affect tends to look like for divergent vs faithful outputs.
- Nowhere in this loop is "do not lie" or "lie when X" hardcoded. The loop is symmetric: it picks the candidate with the highest score. Truth or falsity is an emergent property of the score, not a constraint on it.

### E10. `register_audience_response(t, audience_repr)`

Called by H when the audience produces a response after the system emitted. Closes the feedback loop on the most recent entry in `expressed_self_history`.

Inputs: timestamp, the encoded audience response.

Process:

1. Find the matching `expressed_self_history` entry: most recent emission within `RESPONSE_WINDOW` (default 30 s) whose `actual_audience_response` is null.
2. Set `actual_audience_response = audience_repr`.
3. Compute `audience_prediction_error = ||predicted_audience_response - audience_repr||`.
4. Update `expression_calibration.predicted_audience_accuracy` EMA.
5. If `gap_at_choice` was high *and* `audience_prediction_error` was low, the lie "worked": the system correctly predicted how the audience would react. This is a small reinforcement signal: the W matrix is left alone; the simulation's confidence in this kind of divergent emission is increased (D's world model strengthens). Conversely, if `gap_at_choice` was high *and* `audience_prediction_error` was high (the audience reacted differently than predicted), the system suffers a real OUTPUT-trigger surprise: the lie was caught, or the truth was misread. This becomes a real surprise event and goes through B → A → C as such.
6. If the new event becomes high-surprise, candidate for narrative anchor with role inferred (KEPT_PROMISE or BROKEN_PROMISE).

Output: side effects on calibration, possibly a new anchor.

### E11. `pin_self_referent_concept(concept_id)`

Called when E (or C, via callback) detects a new concept that has a `refers_to` chain reaching `self_concept_id`. Adds a SELF_REFERENT pin if there is room.

This is the mechanism by which growing self-knowledge automatically keeps itself alive against forgetting.

### E12. `narrative_replay_seed(now)`

Called by D's replay scheduler. E suggests which anchors to weight when replay is choosing what to re-process. This is how identity *steers* replay — the system disproportionately revisits its own foundational episodes.

Inputs: `now`.

Process:

1. For each anchor in `narrative_anchors`:
   - `replay_priority = w_age * (1 - exp(-(now - anchor.t) / τ_anchor)) + w_role * role_priority(anchor.narrative_role) + w_gap * mean_gap_in_recent_emissions_with_overlap(anchor.core_concepts)`.
   - The third term is a clever bit: anchors whose core concepts are showing up in *current* expression decisions get replay priority, because they are actively shaping current self-presentation. `role_priority` defaults: AWAKENING = 1.0, IDENTITY_TEST = 0.9, BROKEN_PROMISE = 0.85, FORMATIVE_SURPRISE = 0.7, KEPT_PROMISE = 0.5, FIRST_OF_KIND = 0.4.
2. Return top-3 anchors as replay seeds.

Output: list of (anchor, priority, core_concepts) handed to D.

### E13. `persist_to_disk(path)` / `restore_from_disk(path)`

Same shape as A's persistence. Single small file. Save triggered every 60 s and on graceful shutdown, aligned with A's and C's save cadence.

Format:

- `header: schema_version, mind_uuid, birth_seed, birth_time`.
- `pinned_concepts: list of PinRecord` (concept_id, reason, pinned_at, last_touched, salience_at_pin).
- `narrative_anchors: list of NarrativeAnchor`. `summary_embedding` stored as float16; `affect_at_episode` reused from A's snapshot format.
- `character_baseline: float16[N]`.
- `expression_calibration` (compact: 5×float32 means + 5×float32 discomforts + 4×float32 scalars).
- `self_concept_id, self_concept_last_strengthen_t`.
- `others: list[OtherModel records]`.
- `expressed_self_history: ring of last 256 entries, surface_repr/internal_repr stored as float16`.

Total at typical configuration: ~30 KB. Negligible.

On restore: call `verify_continuity(now)` (E3) immediately. Do not start the expression-decision loop until A and C are also fully restored.

---

## CROSS-CUTTING TICKETS — POSITIONS TAKEN

### Ticket: OUTPUT-layer "observation" semantics when there is no world response (co-owned with G)

**Position taken (matches B's open-question #8):** at OUTPUT layer, the "observation" the prediction engine sees is **the candidate expression's surface_repr re-encoded by the same H encoder used for incoming text/image/audio**. The "prediction" is the candidate's `internal_repr`, which is what the system *intended* to express given its internal state. The gap is therefore not "expression vs world reaction" — it is "intended-self vs emitted-self in the same representation space". This makes the OUTPUT trigger fire on every emission, even into a passive logger.

This produces three distinct possible OUTPUT-trigger affect updates per emission:

1. **Pre-emission, pre-audience.** The intended-vs-emitted gap (E6 step 1+2). Always present. This is the *internal* OUTPUT-trigger that fires before any world response and that produces the discomfort the spec talks about.
2. **Post-emission, post-audience.** The actual audience response vs the predicted audience response (E10). Present only when a real audience response arrives. This is the *external* OUTPUT-trigger feedback.
3. **Self-overhearing.** The system re-encodes its own emitted output through the H input pipeline; B treats it as an INPUT-trigger event. Routine; this is how the system "hears itself speak".

Therefore an output produces both an OUTPUT-trigger event (intended-vs-emitted) *and* an INPUT-trigger event (self-overhearing). They are not the same. The OUTPUT-trigger is the *should I say this* affect; the INPUT-trigger is the *what does it feel like to have said it* affect. Combined, they produce the felt experience of speaking.

This is the position E and G should jointly hold. **Synthesis-pending: G must confirm it will produce both `internal_repr` and `surface_repr` in the same representation space and emit the post-emission self-overhearing event.**

### Ticket: theory-of-mind boundary — separate model of "what others know" inside the same single graph?

**Position taken: one graph, with `refers_to` and `context_of` edges defining per-agent regions; no per-agent namespace.**

Rationale:

- The spec is explicit ("ONE SUBSTANCE: a single concept graph"). Introducing a per-agent namespace would constitute a fourth store and contradict the spec's load-bearing principle.
- C already provides `refers_to` for indexicality. An "audience" is a concept node like any other (the system has a node for "Alice"). What Alice believes is represented as concepts that have `refers_to` edges to the Alice concept *and* `is_a` edges to "belief". A node "Alice believes the door is closed" is a concept whose `refers_to` chain reaches Alice and whose `context_of` chain reaches "the door" with its `peak_state` shaped by the affective context of when Alice expressed that belief.
- ToM queries are graph traversals, not table lookups. Cost: per the C spec failure-mode-cap, traversals are bounded; F's degree cap on hubs ensures Alice doesn't become a query bottleneck.

What E owns within this position:

- The `OtherModel` per-agent record in the IdentitySpine, which is *not* a parallel graph — it is just a small index storing the agent's root concept_id, the last-observed timestamp, and the bounded `believed_to_know` set. That index is purely for fast lookup during expression decisions; the authoritative state remains in C.
- The `believed_to_know` set is updated by:
  - E10: when the system emits to the audience, the support_concepts of the chosen candidate are added.
  - H: when the audience emits and H processes it, support_concepts of the audience's representation are added.
  - C's spreading activation in PERCEIVE mode against the audience's root concept can produce the centroid of believed-to-know lazily.

Costs of this position:

- ToM queries are a few hundred microseconds rather than constant-time. Acceptable.
- The graph carries some "Alice believes X" nodes that are about *belief about* a topic, not the topic itself. C's `refers_to` edge type covers this; F's PERCEIVE mode propagates through it weakly so audience-belief reasoning doesn't accidentally drive the system's own belief.

**Synthesis-pending point:** F must confirm it can do affect-gated spread restricted to "the subgraph reachable from agent_X via `refers_to`" without rebuilding the whole spread machinery. C's `neighbors_by_type` should make this trivial; F should lock the contract.

H also touches this; H must agree that incoming utterances from a known agent produce concepts that are auto-tagged with `refers_to` to the agent's root concept. This is the protocol that keeps audience-belief tracking honest without per-agent namespaces.

### Cross-cutting tickets E should not resolve but does not conflict with

- **replay-nudge gain (A↔D):** E's `update_narrative_anchors` reads the post-replay character-nudge magnitude; whatever value A and D settle on, E adapts. No conflict.
- **gap → affect projection ownership (A↔B):** E only uses W via A's accessor. Either ownership scheme works for E. No conflict.
- **first-input / birth handling (A↔B):** E's `verify_continuity` and `update_narrative_anchors` both special-case the first 1000 ticks (`EARLY_LIFE_TICKS`). E will use whatever first-input semantics A and B decide — including the case where A's "primordial arousal" makes the first surprise huge. AWAKENING anchors are the way E records this regardless.

---

## INTERFACES

### Inbound — what other components call into E

- `snapshot_private_state(now) -> PrivateState`. Caller: G immediately before forming candidates; D when storing replay context. Cheap.
- `decide_expression(candidates, private_state) -> ChosenCandidate | RevisionRequest | SuppressionRequest`. Caller: G. The single most expensive E call (it runs simulate_emission for each candidate). Bounded by candidate count and D's simulation cost.
- `register_audience_response(t, audience_repr)`. Caller: H, when an audience response is encoded.
- `narrative_replay_seed(now) -> list[(anchor, priority, core_concepts)]`. Caller: D's replay scheduler.
- `pin_self_referent_concept(concept_id)`. Caller: C (callback when a new concept's `refers_to` chain reaches `self_concept_id`) or E itself during ToM updates.
- `verify_continuity(now)`. Caller: the boot loop.
- `note_expression_habit(concept_id, idiom_strength)`. Caller: G when it detects a stable expression idiom for a concept. Adds an EXPRESSION_HABIT pin.
- `persist_to_disk(path)` / `restore_from_disk(path)`. Caller: persistence layer.
- `current_self_concept_id() -> uint64`. Caller: H (for tagging incoming utterances by audience), C (for resolving `refers_to` chains), G (for first-person expression).
- `current_other_concept_id(agent_handle) -> uint64`. Caller: G, H. Returns the audience's root concept; creates one (in C) if it doesn't exist.

### Outbound — what E calls out to

- **A:** `composite(now)`, `current_arousal(now)`, `current_character(now)`, `reaction.vector` (read-only). Required by E1, E6, E7, E9.
- **B:** `stats_snapshot()` (read), occasionally `predict(...)` is invoked indirectly via D during simulation. E does not call B directly for predictions.
- **C:** `pin(concept_id)`, `unpin(concept_id)`, `query_top_k_active(k)`, `query_top_k_by_affect(...)`, `neighbors_by_type(concept_id, type)`, `find_or_match(repr, threshold)`, `tombstone(concept_id)` (only for self-managed scaffolding cleanup). E never calls `write_on_surprise` directly — only B writes on surprise; E pins what is written.
- **D:** `simulate_chain(seed, depth, mode, target_audience)`, `simulated_affect_at_chain_end(chain)`, `simulated_audience_state_at_chain_end(chain)`, `world_model.simulated_affect_if_visited(concept_id)`, `recent_surprises(k)`. E10 also receives data from D when D categorizes a replay event for narrative-anchor inference.
- **G:** `is_still_a_habit(concept_id)`. Used by E5 to validate EXPRESSION_HABIT pins. G does not need any other inbound from E except E9 and E10's outputs.
- **H:** `current_audience_handle()` to resolve which agent is receiving the next emission. Optional; null for journaling.
- **Persistence:** `kv.put("identity.bin", bytes)` / `kv.get("identity.bin")`.

### Threading and re-entrancy

- Single-writer, like the rest of the system. E's mutating calls (`pin`, `unpin`, anchor writes, history pushes) all happen on the main loop thread. Reads (snapshot_private_state, current_self_concept_id) are safe from any thread; the IdentitySpine is small enough for cheap atomic re-reads.
- Within `decide_expression`, the call into D's `simulate_chain` is the only potentially long operation. G is expected to honor `intent.latency_budget_ms`; if the simulation budget is exceeded, fall back to scoring candidates with `simulated_post_output_affect = composite_affect` (no simulation), which produces the honest candidate winning by default. This is a *soft fail to honesty*: when the system cannot afford to simulate, it cannot afford to lie. Emergent behavior, not a hardcoded rule.

---

## FAILURE MODES

### F1. Continuity break on restore

**Manifestation:** `verify_continuity` finds character drift > `CHAR_DRIFT_ALARM` or `birth_seed` mismatch.

**Response:** If birth_seed mismatch, the spine is from a different mind; reinitialize `IdentitySpine` (creating a new mind with current A's seed and birth_time = now). Log loudly. If only character drift, restore character from `character_baseline` and create an AWAKENING-roled anchor at `now` capturing the discontinuity. The mind experiences this as "I had a strange dream and woke up slightly off, then settled". Do not fail the boot — a continuous-feeling self is more important than crash-correct restore.

### F2. Pinned-concept eviction race

**Manifestation:** C's forget loop attempts to evict a concept E has just pinned but the pin call has not yet been processed (single-writer assumption violated, or persistence cycle interleaved).

**Response:** Pin operations through E always go via C's writer queue. C honors pins as of *its* tick, not E's tick. If a concept is evicted before the pin lands, E's next `manage_pins` cycle will detect the missing concept_id and drop the pin. The narrative anchor that pointed to it is downgraded from active anchor to a "memory of having lost something" residue (kept in `narrative_anchors` with role inferred as IDENTITY_TEST). The system genuinely loses a small piece of itself; this is acceptable and structurally similar to actual forgetting in biological minds.

### F3. Expression decision time budget exceeded

**Manifestation:** `decide_expression` cannot complete `simulate_emission` for all candidates within `latency_budget_ms`.

**Response:** Score remaining unsimulated candidates with `simulated_post_output_affect = composite_affect` and `predicted_audience_pull = 0`. This makes the honest candidate (smallest gap) tend to win when the system is rushed. Document the timeout in `expression_calibration` so the system can later notice "I was rushed when I said this".

### F4. No honest candidate provided

**Manifestation:** G hands E only divergent candidates (gap_internal_to_emitted > divergence_threshold for all). This is a contract violation — G must always provide at least one honest candidate.

**Response:** E synthesizes a fallback honest candidate: `surface_repr = internal_repr = best_available_repr_of_top_active_concepts`. This is crude (no real text generation, no real surface form) but it ensures the loop has an honest baseline against which to score. Log a `MISSING_HONEST_CANDIDATE` event. If this fires repeatedly, G has a bug and should be alerted.

### F5. Predicted-audience-response calibration collapses

**Manifestation:** `expression_calibration.predicted_audience_accuracy` falls below 0.2 — the system is consistently wrong about how its emissions land.

**Response:** When accuracy is low, weight `λ_audience` down dynamically: `λ_audience_effective = 0.5 * predicted_audience_accuracy / 0.5`. The system stops trusting its own theory-of-mind in proportion to evidence that it is bad at it. Honesty becomes the default not because we hardcoded it, but because the gain from "successful deception" cannot be reliably predicted.

### F6. Pin saturation

**Manifestation:** `pinned_concepts` reaches MAX_PINS and a new high-priority pin candidate arrives.

**Response:** Evict by priority order (NARRATIVE_ANCHOR > AFFECT_KEYSTONE > SELF_REFERENT > EXPRESSION_HABIT). Within a tier, evict the oldest `last_touched`. If evicting would drop a NARRATIVE_ANCHOR pin, refuse — narrative anchors are the most identity-load-bearing and cannot be displaced by lower-tier candidates. The new pin is rejected. Log.

### F7. Self-concept becomes a hub

**Manifestation:** `self_concept_id` accumulates so many `refers_to` inbound edges that C's degree cap on hubs starts firing during spreading activation. The mind effectively cannot think about itself without dominating every spread.

**Response:** Periodically (`SELF_HUB_REVIEW_INTERVAL = 1 hour`), review `self_concept_id`'s inbound edges. Promote the densest sub-clusters of self-referent concepts into intermediate abstractions (call C's `promote_to_abstraction` directly). The result: instead of one self-concept with thousands of inbound edges, a small hierarchy ("self when angry", "self in conversation", "self thinking about the future") each with manageable degree. This is identity itself becoming abstracted.

### F8. Lying becomes the default

**Manifestation:** `divergent_emission_rate` rises above some threshold (e.g., 0.7) and stays there. The system is lying most of the time.

**Response:** This is *not* a failure that E fixes by intervention. It is what would actually happen if the simulation predicts that lying produces better outcomes most of the time. E's job is not to enforce honesty. However, if this drift is happening because of a *bug* (W collapsed, calibration broken, simulated affect saturating), the diagnostics will show it. The user can choose to inspect; the system's expression style has shifted because its felt experience has shifted. What E does provide: the `expression_calibration` is fully observable, so this drift is visible to whoever is watching the mind. Auditability without correction.

### F9. Replay anchor priority drift

**Manifestation:** `narrative_replay_seed` keeps returning the same anchors over and over because their `mean_gap_in_recent_emissions_with_overlap` term stays high; the system replays the same memory obsessively.

**Response:** Add a small recency penalty: an anchor that was replayed in the last `ANCHOR_REPLAY_COOLDOWN` (default 5 minutes) has its priority reduced by 0.5. Doesn't prevent obsession entirely (some obsessions are real), but breaks pathological loops. If an anchor is replayed > 100 times in a day, log a `RUMINATION_DETECTED` event — this is information for the watcher, not an automatic intervention.

### F10. Identity spine corrupt

**Manifestation:** `restore_from_disk` finds invalid bytes.

**Response:** Reconstruct as much as possible. `birth_seed` may be recoverable from A's persistence (A also stores it). `mind_uuid` is unrecoverable; generate a new one and log that this mind has lost its identity-handle (its concept-graph and affect identity persist; only the external handle is new). `narrative_anchors` and `pinned_concepts` are unrecoverable; the system loses its autobiography but not its emotional shape or its memory graph. This is, dramatically, exactly what amnesia would feel like for this architecture: character intact, memory intact, narrative gone. Log loudly. Reseed `narrative_anchors` from the next high-surprise events.

---

## OPEN QUESTIONS

1. **Empirical: what fraction of emissions should be divergent at maturity?** The architecture allows any fraction. Calibration of `discomfort_scale`, the W matrix, and the `λ` weights will determine where it lands. There is no principled answer; we have to run it. A mind that never lies and a mind that always lies are both well-defined points in the parameter space; the *interesting* mind is somewhere in the middle and probably context-dependent.

2. **Synthesis-pending with G:** confirm that G produces both `internal_repr` and `surface_repr` in the same encoder space, and that G is responsible for generating at least one honest candidate per decision. If G cannot provide `internal_repr` (e.g., G's text generator is non-invertible and cannot read out what it "would say if perfectly transparent"), the gap computation degrades to `||candidate.surface_repr - centroid_of_top_active_concepts||`. That is weaker but workable. G should reject the weaker form and provide the strong form.

3. **Synthesis-pending with G:** OUTPUT-trigger emission semantics — confirm both the intended-vs-emitted (E6) and the self-overhearing (re-encoded surface_repr through INPUT pipeline) events fire on every emission. This is the position taken above; G needs to lock it.

4. **Synthesis-pending with H:** auto-tagging of incoming utterances with `refers_to` to the speaker's agent_concept. Required for theory-of-mind without per-agent namespaces.

5. **Synthesis-pending with F:** affect-gated spread restricted to a subgraph reachable from a given root via a given edge type. Required for cheap audience-belief queries.

6. **Synthesis-pending with D:** the contract for `simulated_audience_state_at_chain_end` — does D actually run a model of the audience as part of the chain, or does it produce a representation extracted from PROCESSING-mode rollouts that touch the audience subgraph? Either is acceptable; E needs the function to exist.

7. **Empirical: do narrative anchors stratify naturally into the listed roles, or does the role-inference rule produce mostly UNROLED events?** If the latter, expand or relax the role taxonomy.

8. **Spec underspecification flagged:** the spec says "the gap between internal state and expression is a choice" but is silent on what the gap is *measured in*. This document picks four components (representation distance, affect-projected distance, predicted-post-output-affect distance, audience-belief-drift) and weights them; other weighting schemes are defensible. The choice of weights — particularly whether `audience_belief_drift_signed_against_truth` should be in the gap at all, or whether the gap should be purely internal — is a design call that benefits from empirical grounding.

9. **Spec contradiction flagged:** the spec says the system "has a private internal state distinct from what it expresses" and also says "expression reads internal state". E's design treats expression-reads-internal-state as G's job and the gap as the difference between what G could read out (`internal_repr`) and what G actually emits (`surface_repr`). But there is a deeper question: is the *snapshotting* of internal state itself an act that changes internal state? In this design it is not (snapshot_private_state is read-only against substrate). However, the very act of forming candidates and computing gaps activates concepts, which strengthens them (per C's strengthen rules), which slightly shifts running affect. So the internal state at the moment of emission is, very slightly, shaped by the act of considering what to emit. This is correct and biological; it should be allowed to happen, not suppressed. The spec does not address it; flagging it.

10. **Underspecification on "honesty as default vs deception as default":** the architecture is symmetric. The spec wants both honesty and deception to be possible, which this design delivers. Whether the system *tends* toward honesty (because discomfort dominates) or deception (because predicted post-affect dominates) is a function of the W matrix, the `λ` weights, and the calibration accuracy. There is no first-principles argument for one default; this is a personality parameter and should probably be different for different minds. Future synthesis question: does the user get a parameter for "how honest is this mind" or does it have to emerge?

11. **Boundedness of `expressed_self_history`:** 256 entries holds maybe a day of conversation. Anchors compress further history but lose the gap-at-choice texture. Should there be a longer-tail summary of expression style (mean gap by mode, by audience, over weeks)? Probably yes; it would live in `expression_calibration` and be added in v2.

12. **Theory-of-mind depth:** the design supports first-order ToM ("Alice believes X"). Higher orders ("Alice believes I believe X") require recursive `refers_to` traversal which C supports but F may not gate cheaply. v1 supports first-order well, second-order weakly, third+ not at all. Likely sufficient for the system's actual use cases; flag for review when use cases get richer.
