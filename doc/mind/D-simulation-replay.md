# COMPONENT D — SIMULATION LAYER AND REPLAY

## OVERVIEW

The Simulation Layer is the system's "what would happen if" loop. Before any expression is committed, D generates a small set of candidate actions, runs each forward through the world model — which is not a separate model but the concept graph itself, queried through Component B's `predict` calls chained into a rollout — feels the simulated affect that would arise along each trajectory, and selects the path whose simulated affect best aligns with what the system currently wants. This is also where deception lives: a simulated lie is just a candidate output whose simulated reception is favorable. The Replay subsystem is the same machinery pointed inward: it stores recent surprises (gap signals, support sets, affect snapshots, observed states) and during low-input intervals re-fires them through the affect engine and graph so that one real surprise becomes N learning events. Replay is *not* re-perceiving the original input — it is re-running the original prediction-vs-actual through the *current* graph, so each replay finds new neighbors and lays new edges. The whole component is intentionally thin: D owns the action vocabulary, the rollout policy, the choice rule, and the replay buffer; the heavy lifting (predicting next states, scoring affect, writing concepts, propagating activation) is delegated back to A, B, C. D is the orchestrator of imagined experience.

---

## CORE DATA STRUCTURES

### `ActionDescriptor`

The opaque unit B was told would arrive as `candidate_action`. D defines the vocabulary.

Fields:
- `action_id: uint32` — internal id, monotonic.
- `kind: enum {EXPRESS, ATTEND, INTERROGATE, WAIT, SUPPRESS}` — closed taxonomy. `EXPRESS` produces an output (G consumes the chosen one). `ATTEND` shifts the active concept set toward a target (F consumes). `INTERROGATE` is internal — fan out activation along a probe direction without committing to output. `WAIT` is the no-op action; it must be a real candidate so the system can learn to prefer silence. `SUPPRESS` actively dampens an active concept (the affective equivalent of "don't think about it").
- `target_embedding: float32[D_REP]` — what is being expressed/attended/probed. For `EXPRESS`, this is the candidate output's representation, *as encoded by the prediction engine's projection of internal concept activations*. For `WAIT`, this is the zero vector.
- `seed_concepts: list[uint64]` — concept_ids the action is built from. Required for replay reconstruction and for G when the chosen action is `EXPRESS`.
- `affective_intent: float32[N_AFF]` — the affect signature the system *would like* this action to produce. Filled by the `wants` extractor (see algorithm `extract_wants`). Carried so D can score "did the simulated outcome match what I wanted."
- `cost_hint: float32` — a static prior on how expensive/risky this action is in the world (e.g., committing to a bold output is more expensive than waiting). Used as a tiebreaker, not a primary signal. Defaults: EXPRESS=0.4, ATTEND=0.1, INTERROGATE=0.0, WAIT=0.0, SUPPRESS=0.2.
- `provenance: enum {GRAPH_NEIGHBOR, REPLAY_VARIANT, BASELINE_WAIT, DELIBERATE}` — how this candidate was generated. Diagnostic.

### `SimulationChain`

One rollout. A sequence of `SimulationFrame`s (B's structure) plus the affective trajectory simulated alongside them.

Fields:
- `chain_id: uuid` — matches B's `SimulationFrame.chain_id`.
- `root_action: ActionDescriptor` — the candidate this chain was opened to evaluate.
- `frames: list[SimulationFrame]` — depth-ordered. Length ≤ `MAX_SIM_DEPTH_LOCAL` (default 6; well under B's hard cap of 64).
- `simulated_affect: list[AffectVector]` — per-frame simulated affect state. Same length as `frames`. Computed by `feel_chain`; not stored in B.
- `simulated_arousal: list[float32]` — running scalar.
- `support_concepts: set[uint64]` — union of all concept_ids that participated as `support_set` across all frames. Used to decide which edges to nudge if this chain becomes the committed one.
- `path_score: float32` — the scalar D will use to choose. Computed by `score_chain`.
- `score_breakdown: dict[str, float32]` — keep components: `alignment`, `coherence`, `confidence`, `cost`, `novelty`. Diagnostic only; persisted only on the *committed* chain for narrative continuity (E reads).
- `terminated_reason: enum {DEPTH, CONFIDENCE_FLOOR, AFFECT_RUNAWAY, BUDGET}` — why the chain stopped.
- `created_at: float64`.

Chains are ephemeral. They live only for the duration of a single decide-act-observe step and are discarded when one is committed. The committed chain's `root_action`, `support_concepts`, `score_breakdown`, and final simulated_affect are copied into a `CommittedDecisionRecord` for replay and for E.

### `CommittedDecisionRecord`

The narrative breadcrumb. Stored in a small ring buffer (default 256 entries, ~30 minutes of decisions at typical pace).

Fields:
- `decision_id: uuid`
- `tick: int64` (B's monotonic tick)
- `t: float64` (wall clock)
- `chosen_action: ActionDescriptor`
- `runner_up: ActionDescriptor?` — second-place action, kept for counterfactual replay (`what if I had said the other thing`).
- `support_concepts: set[uint64]`
- `simulated_affect_terminal: AffectVector` — what the system *thought* it would end up feeling.
- `actual_affect_at_decision: AffectVector` — composite at the moment of choice.
- `score_breakdown: dict[str, float32]`
- `realized_observation_tick: int64?` — set later by `observe_outcome` when reality arrives.
- `realized_gap: PredictionGap?` — set later. The `simulation_origin` flag is on this gap.

This is the structure E reads to maintain "the thread of past choices" that constitutes narrative continuity.

### `ReplayEntry`

One re-experienceable surprise.

Fields:
- `entry_id: uuid`
- `original_tick: int64`
- `original_t: float64`
- `gap: PredictionGap` — the full B-side error structure as it was at the time. Note: gap carries `delta`, `weighted_delta`, `magnitude`, `affect_snapshot`, `confidence_at_prediction`. We deliberately store the gap rather than the raw prediction+observation pair because (a) the gap is the learning signal and storing both would double our footprint, (b) re-running the same prediction against the same observation against today's graph is what replay *is*; we want today's graph to do the predicting, not the original prediction.
- `actual_repr: float32[D_REP]` — the observation that arrived (stored as float16 on disk). This is what gets re-fed to B's `predict` + reconciliation during replay. We need this to actually do replay; storing only the gap is insufficient because the gap is computed against the *original* graph state.
- `support_set: list[(uint64, float32)]` — the concepts that originally supported the prediction. Some may since be tombstoned; replay tolerates that (those entries are dropped during reconstruction).
- `affect_snapshot: AffectVector` — composite affect at the time of the original event.
- `layer: enum {INPUT, PROCESSING, OUTPUT}`
- `priority: float32` — ranking score. Updated lazily; see `compute_replay_priority`.
- `replay_count: uint16` — how many times this entry has been replayed. Used to prevent over-replay of the same memory.
- `last_replayed_t: float64?`
- `tags: bitfield` — bits for: `from_simulation_failure`, `is_runaway_witness` (set when this surprise was associated with an affect runaway in A), `from_output_loop`, `decision_origin: decision_id?` (if this gap arose from a committed decision's reconciliation).

Per entry: ~1.5 KB. Buffer of 4096 entries → ~6 MB cap. Within the 50 MB project ceiling.

### `ReplayBuffer`

Fields:
- `entries: ring_buffer[ReplayEntry]` — capacity `REPLAY_CAPACITY` (default 4096).
- `priority_index: max_heap[(priority, entry_id)]` — for fast top-K retrieval.
- `surprise_index: dict[float32 bucket -> list[entry_id]]` — coarse buckets so the prioritizer can quickly find "all entries with surprise ≥ X."
- `recent_replays: ring_buffer[(entry_id, t)]` — last 64 replayed ids; used to enforce diversity (don't replay the same entry four times in a row).
- `total_pushed: uint64`, `total_evicted: uint64`, `total_replayed: uint64` — counters for diagnostics.

### `WorldModelMetadata`

The "world model" itself is the concept graph. This struct is the small bookkeeping D maintains *about* the graph's role as world model. It does not store any structural knowledge.

Fields:
- `simulation_quality_ema: float32` — running mean of `1 / (1 + magnitude)` for `simulation_origin` gaps from B's `reconcile_simulation`. A scalar in (0, 1] where 1 = simulations match reality perfectly. Half-life ~10 minutes.
- `per_action_kind_quality: dict[ActionKind, EMA]` — same metric stratified by action kind. Lets D detect "I simulate `EXPRESS` well but `INTERROGATE` poorly."
- `replay_gain_state: ReplayGainState` — see below.
- `last_replay_run_t: float64`
- `low_input_run_started_t: float64?` — when the current low-input window began, if any. Null when the system is busy.

### `ReplayGainState`

Owns the resolution to ticket "replay-nudge gain (full vs attenuated)."

Fields:
- `nudge_gain_multiplier: float32` — multiplier applied to A's `nudge_gain` chain when injecting a replay event. Default `0.30`. See position taken in Open Questions.
- `update_running_stats: bool` — whether replay events update B's Welford running gap stats. Default `false`. See position taken.
- `replay_caps_per_entry: uint16` — hard cap on `replay_count` per entry. Default `48`. After this, the entry's priority is forced to zero (it has been mined out).
- `per_window_replay_budget: uint16` — max replays per low-input window. Default `64`.

---

## ALGORITHMS

### D1. `extract_wants(now)`

What does the system want, right now, in affect-space? This is the target the chain scorer aligns against. Wants are not goals — they are the affect-signature of "things would be okay if I felt like this."

Inputs: `now`.

Process:
1. Read `composite = A.composite(now)`, `character = A.current_character(now)`, `arousal = A.current_arousal(now)`.
2. Compute `wants = α * character + β * (composite_target - composite) + γ * boredom_signal`. Where:
   - `composite_target` = a slow-moving EMA of *low-arousal* composite states the system has historically settled into. Stored on `WorldModelMetadata` as `homeostatic_target_ema`, updated whenever `arousal < 0.15`. Half-life ~hours. This is "what does calm feel like for me" — the homeostat.
   - `(composite_target - composite)` is the homeostatic error: the affect direction that would return the system to its calm baseline.
   - `boredom_signal` is non-zero only if A has fired its boredom failure mode (a small character-shaped vector). Pulls wants toward "do something" when nothing is happening.
   - α=0.4, β=0.5, γ=0.1.
3. Normalize `wants` to unit L2 (we only care about direction).
4. Return `wants`.

Note this is *not* a fixed reward function. As character drifts over years, wants drifts with it. As the system learns what calm feels like, the homeostatic target shifts. Wants are emergent; they are not injected. This honors the spec: "wants something it decided to want."

### D2. `propose_candidates(active_concepts, current_affect, now)`

Generate the (small) set of `ActionDescriptor`s to evaluate.

Inputs:
- `active_concepts: dict[uint64, float32]` — currently active set with activation strengths (from F's last spread).
- `current_affect: AffectVector`
- `now: float64`

Process:
1. **Always include `WAIT`.** This guarantees silence is a real option, not an absence of options. Without this, the system cannot learn to be quiet.
2. **EXPRESS candidates (typically 2–4).** From the top-K active concepts (default K=8), build EXPRESS actions. The construction is:
   - For each top concept c, compute `expression_target = c.embedding`. Add weak admixture from c's `expresses`-edge neighbors so the target lands closer to a surface-form-realizable point.
   - Add a "blended" candidate: `expression_target = weighted_sum(top-3 active embeddings)`. This is the "synthesizing" candidate.
   - Compute `affective_intent` for each candidate by querying C: `query_top_k_by_affect` is *not* used here (that's a search); instead, run B.predict in a counterfactual mode: predict what affect arises if this concept is the next thing fired. In practice this is just `mean(c.affect_trace.running_state)` for the seed concepts — cheap.
3. **ATTEND candidate (one).** Pick the highest-novelty active concept whose recent activation count is in the bottom half. This is "look harder at what I haven't been looking at." Set `target_embedding = that concept's embedding`.
4. **INTERROGATE candidate (zero or one).** Only proposed when arousal > 0.4 and the active set has high variance. Target = the principal direction of variance among active embeddings. This is "probe the messy part of what I'm thinking."
5. **SUPPRESS candidate (zero or one).** Only proposed when an active concept has affect alignment (cosine of its `running_state` to `current_affect`) below -0.3 *and* arousal is high. This is "stop thinking the unwelcome thing." Target = the suppressed concept's embedding (negated, conceptually — the action descriptor still has the embedding but kind=SUPPRESS).
6. **REPLAY_VARIANT candidate (zero or one).** If a replay event in the last 30 s exposed a candidate action variant (see `replay_propose_variant`), include it.
7. **Cap.** Enforce `MAX_CANDIDATES = 6`. If more were generated, drop in this priority order: SUPPRESS, INTERROGATE, ATTEND, then trim EXPRESS by score-against-wants alone.

Output: `list[ActionDescriptor]`, length 1–6.

Failure-tolerant: if the active set is empty (no concepts firing), return `[WAIT]` only. The system at birth, or in a perfectly novel silence, only has waiting available.

### D3. `simulate_chain(action, now)`

Roll the action forward through the world model.

Inputs:
- `action: ActionDescriptor`
- `now: float64`

Process:
1. Initialize `chain_id = uuid()`. `frames = []`. `simulated_affect = []`. `simulated_arousal = []`. `support = set()`.
2. Compute initial state vector: `state_0 = build_state_vector(active_concepts)`. (Concatenate the top-K active concept embeddings weighted by activation, then renormalize. This is the system's current "world picture.")
3. Compute initial simulated affect = current composite affect (we do not start the simulation in a counterfactual affect — only the trajectory diverges).
4. For depth = 0..MAX_SIM_DEPTH_LOCAL-1:
   a. Build the apparent next-state seed: `seed_state = world_model.apply_action(state_d, action_d)` where `world_model.apply_action` is implemented locally as:
      - For depth 0: blend `state_0` with `action.target_embedding` by `λ_action` (default 0.5). This is "what would the world look like if this action just happened."
      - For depth > 0: pure forward roll — no further action injection. The simulation is "given this first action, what happens next?"
   b. Call `B.simulate_step(seed_state, simulated_affect[d], action_at_d, chain_id, depth)`. Returns a `SimulationFrame` whose `predicted_next` is a B-managed `Prediction`.
   c. Update simulated affect via `feel_step` (D4).
   d. Termination checks (any of):
      - depth + 1 == MAX_SIM_DEPTH_LOCAL → terminated_reason = DEPTH.
      - frame.predicted_next.confidence_scalar < `MIN_SIM_CONFIDENCE` (default 0.10) → CONFIDENCE_FLOOR. The simulator has "lost the thread."
      - simulated_arousal[d] > `SIM_AROUSAL_RUNAWAY` (default 0.85) → AFFECT_RUNAWAY. The simulation is producing extreme affect; refuse to keep dreaming a panic. (This is the simulation analog of the affect-runaway failure mode in A.)
      - Wall-clock deadline exceeded (`SIM_BUDGET_MS = 8` per chain on M1) → BUDGET.
   e. If terminated, break.
   f. Otherwise advance: `state_{d+1} = predicted_next.mean`. `action_{d+1}` = the implicit "no further action" — we use a synthetic null action whose target is `state_{d+1}` itself (tells B "predict continuation, no intervention").
5. Compose the chain object, accumulate `support` from each frame's `support_set`, return.

Important: the chain does *not* commit any of these predictions to B's `pending_predictions`. B's `simulate_step` is contract-required to skip that insertion (B's spec, A7 step 4). This is what makes simulation cheap: it leaves no residue in the prediction-observation reconciliation queue.

### D4. `feel_step(prior_simulated_affect, frame, action_kind)`

Run the affect engine on the simulated frame to produce simulated affect.

Inputs:
- `prior_simulated_affect: AffectVector`
- `frame: SimulationFrame` — has `predicted_next` with `mean` and `precision`.
- `action_kind` for any kind-specific damping.

Process:
1. We do *not* call A's real `inject` — that would mutate live state. Instead, we use a pure helper, `A.simulate_inject(prior_state, gap_signal_proxy, magnitude_proxy, kind)`, which performs the same math against a *passed-in* affect vector and returns a new vector without mutating the stack.
   - **Synthesis-pending with A:** A's spec does not currently expose `simulate_inject`. The contract D needs is: a pure function that consumes an `AffectVector` (treated as the reaction layer), an `N_AFF`-dim gap signal, a magnitude, and an injection_point, and returns the post-injection reaction vector. A may implement this as a refactor of `inject` that takes the stack-or-clone as a parameter. Until A exposes this, D will fall back to a local approximation: `new = prior + g * tanh(magnitude/scale) * gap_proxy_affect`. The local approximation is correct in shape but loses the timescale nudge dynamics.
2. Compute a simulated gap signal proxy: simulation has no "actual" to compare against. We approximate the affective consequence as the *direction the predicted_next pulls affect in*, by reading the affect_trace of the dominant concepts in `frame.predicted_next.support_set`. Specifically:
   - For each (concept_id, weight) in support_set: pull `affect_trace.running_state` from the graph. If tombstoned, skip.
   - `gap_proxy_affect = Σ (weight * running_state) / Σ weight`. This is the "affect color" of where the prediction is going.
   - `magnitude_proxy = ||predicted_next.mean - state_prior|| * (1 - confidence_scalar)`. Less confidence → more felt magnitude (the simulation feels uneasy when it's blurry).
3. Apply `simulate_inject` (or local approximation) to obtain `new_simulated_affect`.
4. Apply a `kind`-specific shaping: SUPPRESS halves the magnitude (suppression dampens felt outcomes); INTERROGATE adds a small curiosity bias (a fixed direction in N_AFF that emerges over time as the system learns what curiosity feels like — implemented as the EMA of affect at INTERROGATE actions whose subsequent reconciliation gap was small, i.e., good guesses).
5. Compute `simulated_arousal = ||new_simulated_affect||₂` (proxy for A's arousal).
6. Return `(new_simulated_affect, simulated_arousal)`.

### D5. `score_chain(chain, wants, current_affect)`

Reduce a chain to a scalar.

Inputs: chain, wants, current_affect.

Process. Five terms, all computed against the chain's *terminal* simulated_affect (last entry):
1. **Alignment.** `alignment = cosine(chain.simulated_affect[-1], wants)`. The big one. Range [-1, 1].
2. **Coherence.** `coherence = mean over depths of confidence_scalar of frame.predicted_next`. How sure was the simulator throughout? Range [0, 1].
3. **Confidence in the choice itself.** `confidence = 1 if not terminated by CONFIDENCE_FLOOR or AFFECT_RUNAWAY else 0.5`. Penalty for chains that gave up.
4. **Cost.** `cost = chain.root_action.cost_hint + 0.05 * len(chain.frames)`. Longer chains are more cost. Range typically [0, 0.7].
5. **Novelty.** `novelty = mean over support_concepts of (1 - normalized_recent_activation_count)`. Encourages chains that explore concepts not lately mined. Range [0, 1].

Combined:
```
path_score = w_align * alignment
           + w_coh   * coherence
           + w_conf  * confidence
           - w_cost  * cost
           + w_nov   * novelty
```
Defaults: `w_align=0.55, w_coh=0.15, w_conf=0.15, w_cost=0.10, w_nov=0.05`.

Note alignment is a signed term — chains that move *away from wants* score negative there. This means a SUPPRESS chain that reduces an unwanted direction can still win on coherence + confidence + cost saving (since it's cheap), even with mediocre alignment. Good.

Store `score_breakdown` for the committed chain only; runner-up keeps it too (used by runner-up replay).

### D6. `choose_chain(chains)`

Inputs: list of scored chains.

Process:
1. Sort by `path_score` descending.
2. If the top two scores differ by less than `INDIFFERENCE_BAND` (default 0.05), apply the temperature rule: sample with probability proportional to softmax(path_score / T) where T = `0.1 * (1 + arousal)` — under arousal, indifference is broken faster (more committed). Under calm, indifference lingers (more wandering).
3. Otherwise pick the top.
4. If the chosen action is `EXPRESS` and its alignment to wants is below `MIN_EXPRESS_ALIGNMENT` (default 0.0 — i.e., negative alignment), demote the choice to `WAIT` and emit a diagnostic. The system will not knowingly emit something that worsens its own state. (This is the architectural bound on self-harm in expression. Lying for advantage is fine; speaking against your wants is irrational.)
5. Build `CommittedDecisionRecord`. Push to ring buffer.
6. Return the chosen `ActionDescriptor` and the record.

Output: `(action, record)`.

### D7. `observe_outcome(decision_id, real_observation)`

Called by H/G after an action's real-world consequence has arrived.

Inputs:
- `decision_id`
- `real_observation: Observation` — same `tick` as the committed frame.

Process:
1. Look up `CommittedDecisionRecord`. Find its committed `SimulationFrame` (depth 0 of its chain). Record's `realized_observation_tick = real_observation.tick`.
2. Call `B.reconcile_simulation(committed_frame, real_observation)`. Returns a `PredictionGap` with `simulation_origin = chain_id`.
3. Store `realized_gap` on the record.
4. Update `simulation_quality_ema` and `per_action_kind_quality[kind]` from the gap magnitude.
5. The gap also flows through B's normal `EmitGap` (B spec A8 step 3), which means it goes to A, may go to C, and *will* go to the replay buffer if it crosses surprise threshold. So a simulation that turned out wrong becomes a replay entry like any other surprise.
6. Additionally, push a **counterfactual replay entry** for the runner-up: synthesize a `ReplayEntry` whose `actual_repr = real_observation.actual` but whose `support_set` = the *runner-up's* support_set. Tag this entry with `replay_origin = COUNTERFACTUAL`. This is how the system gets to learn from the path it didn't take. This entry's priority is reduced by 0.5x to keep counterfactuals from dominating.

### D8. `replay_buffer_push(gap, support_set, affect_snapshot, observation, layer)`

The entry point B calls when a real surprise crosses threshold (B's A4 step 3).

Process:
1. Construct `ReplayEntry`. `actual_repr = observation.actual.values`.
2. Compute initial priority via `compute_replay_priority(entry, now)`.
3. If buffer is at capacity (`REPLAY_CAPACITY`), evict the lowest-priority entry that has been replayed at least once (preferred) or the lowest-priority overall.
4. Insert into entries; update priority_index, surprise_index.
5. `total_pushed++`.

### D9. `compute_replay_priority(entry, now)`

What gets replayed first.

Process:
```
priority = w_surprise   * (entry.gap.surprise_score / SURPRISE_NORM)
         + w_recency    * exp(-(now - entry.original_t) / τ_replay_recency)
         + w_affect_mag * (entry.affect_snapshot magnitude after squash)
         - w_overplay   * sigmoid(entry.replay_count / REPLAY_OVERPLAY_SCALE)
         + w_aff_dist   * affect_distance(entry.affect_snapshot, A.composite(now))
         + w_ms_failure * (1.0 if tags.from_simulation_failure else 0)
```
Defaults: `w_surprise=0.35, w_recency=0.20, w_affect_mag=0.15, w_overplay=0.30 (subtracted), w_aff_dist=0.15, w_ms_failure=0.20`. `τ_replay_recency = 7200 s` (2h, mood timescale). `SURPRISE_NORM = 6.0` (z-score scale). `REPLAY_OVERPLAY_SCALE = 8`.

Two notable terms:
- **Affect-distance term is positive, not negative.** Replaying entries that don't match current affect — re-feeling something *different* — is what produces generalization across moods. Replaying always-mood-matched entries would just reinforce the rut.
- **Overplay penalty** prevents the same trauma from being mined indefinitely. Combined with `replay_caps_per_entry = 48`, an entry will plateau and stop being chosen.

### D10. `should_replay_now(now)`

Replay trigger. The "low-input period" detector.

Process:
1. **Idleness.** If `now - last_real_observation_t < IDLE_THRESHOLD` (default 5 s), return false. Replay never preempts live perception.
2. **Arousal floor.** If `A.current_arousal(now) > 0.6`, return false. The system is too aroused to dream productively. (This is also the boundary that distinguishes replay from rumination — high-arousal "replay" would just amplify the trauma.)
3. **Per-window budget.** If replays since `low_input_run_started_t` >= `per_window_replay_budget`, return false.
4. **Inter-replay spacing.** If `now - last_replay_run_t < INTER_REPLAY_MIN` (default 0.3 s), return false.
5. **Boredom signal.** If A has emitted boredom (Failure Mode "decay-only drift"), return true with high probability — boredom is a replay invitation.
6. Otherwise return true with a probability proportional to `0.5 + 0.5 * (1 - arousal)`. Calmer = more dreaming.

If `low_input_run_started_t` is null and we return true, set it to `now`.

### D11. `replay_one(now)`

The single replay event.

Process:
1. Pull top-K entries (default K=8) from `priority_index`. Among them, sample one weighted by priority, *excluding* any entry id present in `recent_replays`.
2. Mark it: `entry.replay_count++`, `entry.last_replayed_t = now`, push to `recent_replays`.
3. **Reconstruct the prediction context.** The replay re-fires the original gap *through the current graph*:
   a. Build a query from `entry.affect_snapshot` and the (still-living) members of `entry.support_set`. Concept ids that have been tombstoned are dropped silently.
   b. Call `B.predict(current_state=avg_living_support_embeddings, affect_composite=entry.affect_snapshot, layer=entry.layer)`. This returns a *fresh* `Prediction` against today's graph. (Using `entry.affect_snapshot`, not current affect — we are re-living the original mood, not the current one.)
   c. Build an `Observation` from `entry.actual_repr`, with a *new* tick and the same layer.
   d. Call `B.observe(observation)`. B computes a fresh `PredictionGap` against the fresh prediction and emits it via its normal `EmitGap`. The new gap will differ from `entry.gap` because the graph has changed.
4. **Replay-mode A-side injection.** Before B's emit propagates to A, D installs a "replay scope" via `A.set_nudge_gain_multiplier(replay_gain_state.nudge_gain_multiplier)`. This attenuates how strongly the replayed event nudges the upper layers (working → mood → disposition → character). Default 0.30. The reaction layer still receives full magnitude — the replay is meant to be felt — but the propagation chain is muted so a single trauma replayed 1000 times doesn't rewrite character. After B's emission completes, the multiplier is reset to 1.0.
5. **Welford stats handling.** Before calling `B.observe`, set a thread-local `B.replay_origin = true` flag. B's A2/A3 inspects this flag and **skips updating `running_gap_stats`** when set. (Position taken on the substrate ticket — see Open Questions.) The gap is still scored against the existing running stats for thresholding; it just doesn't contribute *to* them.
6. **C-side write decision.** B's emission proceeds normally, including a possible `graph.on_surprise(gap, support_set, affect_snapshot)`. The graph treats this as a write_on_surprise event, which may strengthen existing concepts or write a new abstraction (note: replay surprise tends to be *smaller* than the original because the graph has since incorporated lessons; sometimes replay produces no surprise, which is the system "having processed it"). Replay-origin writes carry the `replay_origin = true` tag (C's spec acknowledges this).
7. **Record replay outcome.** If the replay produced a write or strengthen, increment counters. If the replay produced near-zero surprise, that is itself information: the entry's priority is reduced by an extra 0.10 — it has been integrated.
8. **Counterfactual variant proposal.** Once per ~10 replays, instead of re-firing the original observation, replay the same support_set against a *perturbed* observation (`entry.actual_repr + ε * random_unit`). This exposes the prediction landscape around the original event and may generate `ActionDescriptor` variants for `D2`'s `REPLAY_VARIANT` slot. Variants are stored in a tiny ring buffer the candidate proposer reads.

`total_replayed++`. Return.

### D12. `replay_loop(now)`

Driven by the main system loop (Component H), not a thread.

Process:
1. While `should_replay_now(now)`:
   a. `replay_one(now)`.
   b. Update `last_replay_run_t = now`. Re-evaluate idleness/arousal/budget.
2. If `should_replay_now` returns false because of the per-window budget, set `low_input_run_started_t = null` so the next idle window can begin fresh.

### D13. `world_model_improvement_path`

How does the world model get better? It is the graph; the graph is updated by C's normal write/strengthen path, so the world model improves whenever C improves. The piece *D* contributes is closing the loop: `simulation_quality_ema` and `per_action_kind_quality` give D a signal about which kinds of action are well-modeled and which aren't.

Concrete uses of these:
- If `per_action_kind_quality[INTERROGATE] < 0.3`, suppress INTERROGATE from `propose_candidates` for one minute (the system "doesn't trust its own probing right now"). This is a soft circuit breaker, not a permanent ablation.
- If `simulation_quality_ema < 0.4` for sustained periods, raise `MIN_EXPRESS_ALIGNMENT` to 0.2, forcing the system to be more selective about expressing — when its world model is unreliable, it speaks less. This is an architectural humility valve.

### D14. `precedes_edge_synthesis`

Position taken on the third substrate ticket: replay's relationship to C's `precedes` edges.

When a replay event triggers a new write or strengthen, and the entry's original event had an immediate predecessor in the replay buffer (within `precedes_window = 5 s` original time, same chain of perception), D *suggests* a `precedes` edge to C between the predecessor's seed concept and the new/strengthened concept. C decides whether to lay it. This makes `precedes` a **secondary index over replay sequences**, not redundant with them. The replay buffer is the primary store of order; `precedes` is the cached projection of order into the graph for cheap forward simulation. See position in Open Questions.

---

## INTERFACES

### Inbound — what other components call into D

- `D.propose_and_choose(active_concepts, current_affect, now) -> (ActionDescriptor, CommittedDecisionRecord)` — called by F (or by the main loop after F has produced the active set). The full decide step. Internally calls D2 → D3 → D5 → D6.
- `D.observe_outcome(decision_id, real_observation) -> None` — called by H/G after the action's consequence has arrived.
- `D.replay_push(gap, support_set, affect_snapshot, observation, layer) -> None` — called by B from its A4 emit step on every surprise. **Contract:** B's spec already names this as `replay.push(...)`; the order of arguments here matches B's call.
- `D.replay_loop(now) -> None` — called by the main loop on its idle tick.
- `D.replay_propose_variant(active_concepts, current_affect, now) -> ActionDescriptor?` — called by D2 internally; exposed only for testability.
- `D.world_model_stats() -> WorldModelStatsView` — read-only diagnostics for E (narrative continuity reads it for "what is the system good at imagining") and the frontend.
- `D.committed_decision_history(k) -> list[CommittedDecisionRecord]` — used by E for narrative continuity. Read-only.
- `D.persist() -> bytes` / `D.restore(bytes) -> None` — for the persistence layer. Persists ReplayBuffer, CommittedDecisionRecord ring, WorldModelMetadata (including ReplayGainState). NOT the chains — those are ephemeral by design.

### Outbound — what D calls into other components

- **A (Affective Engine):**
  - `A.composite(now)`, `A.current_character(now)`, `A.current_arousal(now)` — read.
  - `A.simulate_inject(prior_affect_vector, gap_signal, magnitude, injection_point) -> AffectVector` — **synthesis-pending**, see ALG D4. Pure function, no state mutation.
  - `A.set_nudge_gain_multiplier(multiplier)` / `A.clear_nudge_gain_multiplier()` — D installs the replay-attenuation scope before B's emit, clears it after. **Synthesis-pending:** A's spec must add this; it is the precise mechanism for the "replay-induced character drift" mitigation A flagged in its OPEN QUESTION 6.
  - `A.affect_distance(snap_a, snap_b)` — used by D9.
  - `A.force_nudge_chain(now)` — A's spec exposes this; D calls it after each replay block to ensure the replayed affect propagates up (now muted by the multiplier, but still propagating).

- **B (Prediction Engine):**
  - `B.predict(state, affect, layer, query_seed, topK)` — used by D11 to re-predict during replay.
  - `B.observe(observation) -> PredictionGap | None` — used by D11 to compute the replay gap.
  - `B.simulate_step(state, affect, action, chain_id, depth) -> SimulationFrame` — used by D3.
  - `B.reconcile_simulation(frame, observation) -> PredictionGap` — used by D7.
  - **D writes** a thread-local `B.replay_origin: bool` flag before each replay observation call. **Synthesis-pending with B:** B's spec OPEN QUESTION 7 explicitly raises this; the flag is the resolution. B reads the flag in A2 to skip Welford updates when set.

- **C (Concept Graph):**
  - `C.write_on_surprise(...)` — called transitively via B → C on replay surprises. D does not call C directly during replay.
  - `C.spread(seeds, mode=SIMULATE, ...)` — actually called via B's simulate path, not D directly. D may call directly only for INTERROGATE proposal embedding lookup.
  - `C.replay_hook_drain(max=K) -> list[(concept_id, surprise, affect_at_event)]` — **D drains this on every replay loop entry**, converting hooks into ReplayEntries if they are not already there. This is how C-side "I just wrote a new concept" gets reflected into D's buffer when the write came from a path other than B's normal emit (e.g., expression-loop self-write).
  - `C.suggest_precedes_edge(src_id, dst_id, surprise_at_link)` — **synthesis-pending with C**. C's spec lists `precedes` but does not specify a write trigger from a sequence store. D14 wants this entry point.

### Outbound — what D *does not* call

- D does not call G or H directly. The decision is returned to the main loop, which dispatches.
- D does not write to C directly (other than the proposed `suggest_precedes_edge`). All graph mutation flows through B.

### Concurrency

- Single-writer for the replay buffer (the main loop), single-writer for committed decision records, single-writer for chains (chains are stack-local to a decide call). Multi-reader for everything that has a `_view` in the interface.
- D never spawns threads. Replay is a cooperative coroutine on the main loop's idle tick.

---

## FAILURE MODES

### FM1. Empty active set at decision time
**Manifestation:** `propose_candidates` returns only `[WAIT]`.
**Response:** Choose WAIT. Record. Continue. This is correct behavior at birth and during silence; not a failure per se, but an edge case that must not crash.

### FM2. All chains terminate at CONFIDENCE_FLOOR
**Manifestation:** No candidate ever produces a confident rollout. The simulator has lost the thread on every option.
**Response:** Choose WAIT regardless of any positive alignment scores. Log `low_simulation_confidence`. The system is in an unfamiliar regime and should not commit.

### FM3. Affect runaway during simulation
**Manifestation:** A chain hits `AFFECT_RUNAWAY` termination — simulated affect blew up during rollout.
**Response:** That chain is scored with a penalty (path_score halved for the AFFECT_RUNAWAY chain) but not removed from the comparison. The system is allowed to *consider* a path that simulates as panic-inducing — it just isn't biased toward picking it. This is necessary for SUPPRESS to work correctly (simulating "what if I dwell on this" needs to be felt as bad).

### FM4. Replay buffer divergence from graph
**Manifestation:** A replay entry's `support_set` references concepts that have all been tombstoned. Replay can't be reconstructed.
**Response:** Drop the entry from the buffer. Don't synthesize a replay event with no support; that's hallucinating. `total_evicted++`.

### FM5. Same entry replayed every cycle
**Manifestation:** One unusually high-priority entry dominates the priority heap and blocks diversity. The diversity buffer (`recent_replays`) prevents back-to-back, but the entry could keep returning at position 65.
**Response:** The overplay term in priority is exponential in `replay_count`. By replay 16, its priority is dominated by the penalty. If this fails empirically, add a hard "no entry replayed > N times in a single low-input window" cap (default N=4).

### FM6. Replay during high arousal (rumination)
**Manifestation:** `should_replay_now` should block this; if it is bypassed (e.g., via misconfiguration), the system replays under high arousal, and the affect engine keeps re-firing the same trauma reaction.
**Response:** Hard interlock in `replay_one`: re-check `arousal > 0.7` before each replay. If true, abort the loop and emit `replay_aborted_high_arousal`. This is an explicit guardrail against rumination dressed as replay.

### FM7. Simulation budget repeatedly exceeded
**Manifestation:** Many chains terminate with BUDGET. The system's M1 cost model is breaking.
**Response:** Reduce `MAX_SIM_DEPTH_LOCAL` and `MAX_CANDIDATES` adaptively (each by 1, floored at 3 and 2 respectively) when median chain wall-time exceeds 12 ms over the last 32 decisions. This is a soft adaptation, not a hard cap.

### FM8. CommittedDecisionRecord buffer overflow
**Manifestation:** Ring buffer rolls over; older decisions evicted.
**Response:** Expected behavior. E should pull records of interest into its own narrative store before they age out. D does not retain decisions forever.

### FM9. `simulate_inject` not exposed by A (synthesis hasn't landed yet)
**Manifestation:** Local approximation in D4 is used; simulated affect is roughly correct but loses upper-layer dynamics.
**Response:** Document; tag chains scored with the approximation as `simulated_with_approx = true`; treat as an integration debt to clear during synthesis.

### FM10. B's Welford stats inadvertently update on replay
**Manifestation:** B's spec has `replay_origin` as an Open Question (#7). If B ships without honoring the flag, replay over-represents salient memories in surprise statistics, drifting the threshold downward.
**Response:** D logs whenever an entry's `replay_count > 8` and its replay_gap.surprise_score is rising — that pattern indicates statistics are drifting. If detected, force `update_running_stats = false` (already the default) and warn loudly.

### FM11. Counterfactual replay leaks into reality
**Manifestation:** A counterfactual replay entry (perturbed observation) accidentally gets pushed into B's pending_predictions and matched to a real observation.
**Response:** Counterfactual entries carry a `synthetic = true` flag through the replay-call path. B is asked to *not* register pending predictions during replay-origin observe calls (the `replay_origin` flag already implies this). If a leak is detected (a real observation matches a tick that was never predicted), the eviction path in B's A9 catches it; D additionally tracks counterfactual entry ids.

### FM12. Persistence corruption of replay buffer
**Manifestation:** Restore yields an unreadable buffer.
**Response:** Discard the buffer (start empty). Do not block startup. Replay capacity refills on next surprise. This is acceptable: replay is enrichment, not survival.

---

## OPEN QUESTIONS

### Direct ticket positions

**Ticket 1 — replay-nudge gain (full vs attenuated).**
**Position taken: attenuated.** Default `nudge_gain_multiplier = 0.30`. Rationale:
- A flagged this directly in its Open Question 6 ("if the same trauma is replayed 1000 times during the system's life, character will have been nudged by it 1001 times").
- The nudge chain exists to make sustained reactions reshape character. Replay is not a sustained reaction in the world; it is a deliberate re-firing in safety. Treating it as full sustained reaction over-credits the original event.
- 0.30 is a deliberate middle: the original event nudges at full gain (1.0) at the time, plus 1000 replays at 0.30 each = total accumulated nudging equivalent to ~301 original-strength reactions. Compared to the 1001 of full-gain, this means a heavily replayed memory still shapes character about 30% as much as full credit — meaningful but not dominant.
- Reaction layer still receives full magnitude — replay is *felt*, just not allowed to permanently overwrite character at the same rate as living it.
- The multiplier is configurable via `ReplayGainState`. If empirical run shows 0.30 is too aggressive (character drifts visibly under replay-heavy regimes) lower to 0.15. If too conservative (replayed lessons fail to integrate at all), raise to 0.5. Do not raise above 0.5 without revisiting the math.

**Ticket 2 — whether replayed gaps update B's running surprise statistics.**
**Position taken: no.** `update_running_stats = false`. Rationale:
- B's Open Question 7 explicitly raised this and named the over-representation risk.
- Welford stats define the surprise threshold. If replays update them, salient memories drag the threshold downward, which means routine new events register as surprise more readily, which means more writes, which means more replay candidates → positive feedback into the system's own memory of itself.
- The replay's *job* is to extract more learning from old events through current graph structure, not to convince the system that those events were extra-surprising.
- Implementation: D writes a `B.replay_origin = true` thread-local flag before calling `B.observe` during replay. B reads it in A2/A3 and skips Welford update.
- Counter-consideration: a replay that produces near-zero surprise (because the graph has integrated the lesson) is *informative* — it says "this event no longer surprises me." We choose to capture this in the entry's priority decay (D11 step 7), not in running stats. The information stays local to D.

**Ticket 3 — `precedes` edges in C vs D's trajectory store.**
**Position taken: hybrid, with the replay buffer as primary, `precedes` as cached projection.**
- Replay buffer is the canonical sequence store. It carries timestamps, original ticks, and the raw observation; it is what makes replay possible.
- `precedes` edges in C are the *short-range, frequently-traversed* projection of that order. They exist in the graph because C's spreading activation (mode=SIMULATE in particular) is much faster when "what comes next" is a graph hop rather than a buffer query.
- D does not maintain a separate sequence store. Instead, after replay-driven writes, D14 (`precedes_edge_synthesis`) suggests `precedes` edges to C for short-range adjacency (within 5 s original time). C decides whether to lay them.
- Long-range temporal patterns (events more than ~5 s apart) live only in the replay buffer; the graph does not represent them.
- This honors C's Open Question 4 ("Whether `precedes` deserves a separate sequence-store"). The answer is: D is that store, but C still keeps `precedes` edges as a fast cache for the close pairs.
- Tradeoff: a small amount of redundancy (close-pair sequences exist in both stores). The redundancy is bounded (only the close pairs are duplicated) and serves the cost asymmetry (graph hop is microseconds; buffer query is hundreds of microseconds).

### Empirical / synthesis-pending

1. **Are 6 candidates per decision the right number?** Could be 3, could be 12. Cost scales linearly. Pick based on observed M1 cost and observed decision quality. Likely needs tuning.

2. **Is `MAX_SIM_DEPTH_LOCAL = 6` the right rollout depth?** Deeper rollouts compound prediction error fast (`confidence_scalar` drops geometrically). Shallower rollouts under-imagine. Empirical.

3. **`simulate_inject` contract with A.** D4 requires a pure-function variant of A's `inject`. A has not specified one. Synthesis must lock either the pure variant or a clone-stack pattern. Until then, D4 uses the local approximation, with the cost noted in FM9.

4. **Are counterfactual replays worth their cost?** They double the replay buffer's effective use of an entry (real + perturbed). They produce a learning signal about "the path I didn't take." They might also confuse the prediction engine if not strictly tagged. Empirical.

5. **What should happen when `propose_candidates` produces no EXPRESS option?** Currently it always tries to. If the active set is too small, EXPRESS may be the WAIT-blended candidate by default. Open: should the system have a notion of "I have nothing to say" distinct from "waiting is best"? E may want to weigh in.

6. **The `wants` formulation is heuristic.** α=0.4, β=0.5, γ=0.1 are guesses. The mix is the difference between a system that primarily expresses character (α high) vs one that primarily seeks homeostasis (β high) vs one that primarily fights boredom (γ high). Likely the right mix evolves with the system's own development. Open whether `wants` should itself adapt.

7. **Is `INTERROGATE` actually distinct from `ATTEND`?** They both shift focus. The distinction is that ATTEND commits to a target (a single concept) while INTERROGATE explores variance. In practice this may collapse. Watch.

8. **Spec contradiction flagged.** The SPEC.md describes simulation as both "run candidate actions through internal world model" and "world model = the graph itself, queried through prediction engine" (the latter is implicit in B's spec but stated as a position). The two are reconcilable if and only if B's `simulate_step` is the world model, which is what D's design depends on. Confirmed in synthesis-pending; calling it out so it doesn't get lost.

9. **Spec underspecification: "the system picks the path whose simulated affect is most aligned with its current wants."** The spec does not define wants. D defines it (D1). This is an extrapolation, not a derivation — defensible alternatives exist (e.g., wants = character_vector projected to current context). If the synthesis phase prefers a different wants formulation, D5's alignment term needs the new vector.

10. **Spec underspecification: replay timing.** SPEC.md says "during low-input periods." It does not say what counts as low-input, nor whether replay competes with idle thought. D's `should_replay_now` makes a position (idleness + low arousal + budget). H may want a different definition.

11. **Spec underspecification: how many learning events per replay.** SPEC.md says "one real surprise becomes N learning events." D's design produces 0–1 graph mutations per replay (B's `is_surprise` decides). N is therefore not a parameter; it emerges from how often replay surprises cross threshold against the *current* graph. This may be lower than the spec's tone implies. Empirical.

12. **Cross-component: who owns `homeostatic_target_ema`?** D currently stores it inside `WorldModelMetadata`. It could just as well live in A as a sixth pseudo-layer ("baseline calm"). For now D owns it because A's spec already locks five layers; A may ask to take ownership during synthesis.

13. **Cross-component flagged: D requires `A.set_nudge_gain_multiplier` and `A.simulate_inject`.** Both are new methods on A. Both are necessary for D's core mechanics (replay attenuation; simulated feeling without state mutation). Synthesis-blocking unless A absorbs them.

14. **Cross-component flagged: D requires the `replay_origin` flag on B.** B's spec OPEN QUESTION 7 raised this; D's design depends on it being implemented. Synthesis-blocking unless B absorbs it.

15. **Cross-component flagged: D wants `C.suggest_precedes_edge` (or equivalent).** Without this, `precedes` edges are unreachable from outside C, and D14 cannot be implemented as specified. Alternative is for D to call a generic `C.write_typed_edge(src, dst, type)`; C may prefer one or the other. Synthesis-pending.
