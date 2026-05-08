# COMPONENT F — ATTENTION AND SPARSE ACTIVATION

## OVERVIEW

Attention is not a separate computation in this system. It is the visible footprint of the affect state pressing into the concept graph: the subset of nodes that fire, the speed they fire at, the breadth of their fan-out, and the order in which they reach the prediction engine. The substrate components have already done most of the heavy work — A computes the composite vector and the per-node `gate_attention` math; B emits prediction priors; C exposes a `spread` algorithm with PERCEIVE/PREDICT/SIMULATE modes and an arousal-driven sparsity envelope. F is the conductor: it owns the loop that decides *when* to call C, *with which seeds*, *in which mode*, *with what budget*, and *how to fold A's gate and B's prior into a single attention signal that C consumes per edge*. F also owns the read-time formation of perceptual habits — the slow stable patterns that make this mind attend to the same kinds of things in the same way over years. Habits are not a new data store; they are a thin overlay on the graph (a salience boost on a small set of "habitual seed concepts") plus a slow-decaying multiplier on a small set of edge-traversal patterns. Personality, perceptually, is what F has habituated to.

The cardinal claim of the spec — "attention is not computed — it emerges from the affect state touching the graph" — is operationalized here as: F never produces an attention vector. F constructs the *call site* (seeds, mode, budget, gate function) and lets C's spread plus A's gate produce the activation pattern. The pattern *is* the attention.

---

## CORE DATA STRUCTURES

F owns a small amount of state. Most of the system's attentional intelligence is in A and C; F's persistent state is the perceptual-habit overlay and a few cached structures the loop reuses.

### `AttentionFrame`

A short-lived per-call descriptor passed to `C.spread`. Built fresh on every attention pass; not persisted.

Fields:
- `phase: enum {INPUT, PROCESSING, OUTPUT}` — which of the three lifecycle points triggered this call. Maps directly to A's `injection_point` and B's `layer`.
- `mode: enum {PERCEIVE, PREDICT, SIMULATE}` — passed to C. Determined by `phase` (see ALGORITHMS / phase_to_mode).
- `seeds: dict[uint64, float32]` — concept_ids and initial activation. Sum-normalized to 1 before handing to C.
- `composite_affect: float32[N]` — snapshot of `A.composite(now)` taken once at frame construction. Never re-sampled mid-spread.
- `arousal: float32` — `A.current_arousal(now)`, also taken once.
- `predict_prior: dict[uint64, float32] | None` — per-concept prior bias supplied by B (which concepts B's last `predict` named in its support set, with their weights). Used to bias edge selection toward predicted-then-confirmed paths. None at INPUT phase before B has run.
- `budget: uint16` — per-call hard ceiling on node touches. Cost-controlled (see ALGORITHMS / budget_for).
- `max_steps: uint8` — propagation depth for the spread.
- `top_k_override: uint16 | None` — if set, F fixes the sparsity envelope explicitly rather than letting C derive it from arousal.
- `tick: int64` — global tick from B's counter. Stamped on the frame so post-hoc analysis can correlate F's attention pass with B's prediction.
- `habit_seed_ids: set[uint64]` — concepts injected by the habit overlay this call. Tracked separately from `seeds` so F can attribute later.

### `HabitOverlay`

The persistent layer that turns repeated attention patterns into a perceptual fingerprint. Owned by F. Survives across sessions.

Fields:
- `habit_seed_table: dict[uint64, HabitEntry]` — concept_ids that have demonstrated repeated, sustained, character-aligned activation. Bounded at `HABIT_SEED_CAP = 64`. Eviction policy below.
- `habit_path_table: dict[(uint64, uint64, EdgeType), HabitPath]` — recurring `src -> dst` traversals. Bounded at `HABIT_PATH_CAP = 256`.
- `last_consolidated_t: float64` — last time `consolidate_habits` ran. Drives the daily/long-cadence consolidation.
- `habit_temperature: float32` — global multiplier on how strongly habits inject and amplify. Range [0, 1]; default 0.5. Allows the system to suppress its own habits during extreme novelty (high working-affect deviation from character → temperature drops, novelty rules).

`HabitEntry` fields:
- `concept_id: uint64`
- `seed_strength: float32` — base seed activation injected when habit fires. [0, 1].
- `affect_signature: float32[N]` — the composite affect at which this concept tends to be relevant. Used to gate whether to fire the habit at this moment.
- `affect_tightness: float32` — how narrow the affective context is. Low = fires across many moods; high = fires only in a specific affective regime.
- `fires_in_phase: uint8` — bitfield over {INPUT, PROCESSING, OUTPUT}.
- `n_observations: uint32` — how many consolidation cycles have observed this pattern.
- `last_fired_t: float64`.
- `creation_t: float64`.
- `pin_request_active: bool` — whether F has asked C to soft-protect this concept from forgetting. F never hard-pins; that is E's privilege.

`HabitPath` fields:
- `src: uint64`, `dst: uint64`, `edge_type: EdgeType`
- `traversal_amplifier: float32` — multiplier (≥1, capped at `HABIT_PATH_AMP_MAX = 1.5`) applied during PERCEIVE/PREDICT spread when this exact src/dst/type triple is followed.
- `affect_signature: float32[N]`, `affect_tightness: float32` — same role as in seeds.
- `n_observations: uint32`, `last_fired_t: float64`.

### `AttentionStats`

In-RAM running counters used both for cost-throttling and for the diagnostic surface E and the frontend read.

Fields:
- `frames_per_phase: int64[3]`
- `mean_budget_used_per_phase: float32[3]` — Welford running mean.
- `mean_active_set_size_per_phase: float32[3]`
- `mean_arousal_per_phase: float32[3]`
- `predict_prior_overlap_running: float32` — fraction of the post-spread active set that was named by B's `predict_prior` for the same tick. Tracks how attentionally aligned F and B are.
- `last_runaway_t: float64` — last time a budget-exhaust event was logged.
- `processing_recursion_depth_max_seen: uint8` — defensive: how deep PROCESSING-phase re-entry has gotten.

### `PerceptualBaseline`

A tiny, slow-evolving structure that captures "what this mind tends to attend to in the absence of strong signal." One vector. Used as a soft seed when input is sparse.

Fields:
- `baseline_affect_target: float32[N]` — the affect state under which the system most often attends productively. EMA over consolidation cycles.
- `baseline_seed_ids: list[uint64]` — small set (≤8) of concepts derived from `habit_seed_table`'s top entries by `n_observations`, restricted to those whose `affect_signature` is close to `baseline_affect_target`.
- `last_refresh_t: float64`.

This is the closest thing F has to "what the system thinks about when it isn't thinking about anything." It's used for the boredom/idle path (see Component A's flatness response).

---

## ALGORITHMS

### F1. attend(phase, raw_seeds, now)

The single top-level entry point. Called by H at INPUT, by F's own processing loop at PROCESSING, by G at OUTPUT.

Inputs:
- `phase: enum {INPUT, PROCESSING, OUTPUT}`
- `raw_seeds: dict[uint64, float32] | None` — caller-supplied seeds. May be None for PROCESSING (F seeds itself from the prior frame's active set) or for an idle PROCESSING tick (F seeds from `PerceptualBaseline`).
- `now: float64`

Process:
1. Sample `composite_affect = A.composite(now)` and `arousal = A.current_arousal(now)`. **Sample once.** This is the cardinal claim: a single attention pass operates under a single affective context. Re-sampling mid-spread would let affect feedback from one branch shape another branch's gating in the same step, which makes spread non-deterministic and loses the property that "this thought happened under this feeling."
2. Compose the seed set:
   - Start with `raw_seeds` if provided, else empty.
   - Inject habit seeds via `inject_habit_seeds(composite_affect, arousal, phase)`. This is additive, not replacing; caller seeds always remain.
   - Sum-normalize.
3. Compute mode: `mode = phase_to_mode(phase)` — INPUT→PERCEIVE, PROCESSING→PREDICT, OUTPUT→SIMULATE. (See "phase semantics" subsection below for the rationale; OUTPUT is SIMULATE because G is asking "what would happen if I said this?".)
4. Pull the latest `predict_prior` from B if available for the current tick (B's last `Prediction.support_set`); else None.
5. Compute `budget = budget_for(phase, arousal)` and `max_steps = steps_for(phase, arousal)`.
6. If `top_k_override` is set on the frame (rare; only F's own loop sets it), pass through; else let C derive top_k from arousal.
7. Build `AttentionFrame` and call `C.spread(seeds, composite_affect, arousal, max_steps, budget, mode)` with the per-node `gate_attention` callback wired to A.
8. Receive `active_set: dict[concept_id, activation]`.
9. **Per-step prediction granularity decision.** F invokes C exactly once per `attend` call; intermediate spread steps do *not* fire B predictions individually. (See the Tickets section for full reasoning.)
10. Run `record_attention_event(frame, active_set, now)` to update HabitOverlay and AttentionStats.
11. Return `active_set` to the caller.

Cost target on M1: under 4 ms per call at default budget=256, max_steps=3, 5K node graph. Dominated by C.spread's edge traversals (~256 traversals × ~200 ns per traversal incl. gate eval) plus A.composite cache hit (free) plus habit injection (≤ 64 hash lookups, ~5 µs).

### F2. phase_to_mode(phase)

Trivial map but documented because the choice is load-bearing.

- INPUT → PERCEIVE. The system is meeting the world; activate broadly through associative/contextual edges to ground the input. PERCEIVE in C weights `similar_to`, `context_of`, `has_property` highest.
- PROCESSING → PREDICT. The system is "thinking forward"; activate along causal/temporal/taxonomic edges so B's next predict has good support. PREDICT in C weights `causes`, `precedes`, `is_a` highest, inverts `opposite_of`.
- OUTPUT → SIMULATE. The system is asking "what does this output produce in the world?"; activate along forward causal/temporal edges, lightly through taxonomy. SIMULATE in C weights `causes`, `precedes` highest.

### F3. budget_for(phase, arousal) and steps_for(phase, arousal)

The cost model. Budget caps total node touches per spread; steps caps depth.

```
phase     arousal_band     budget    max_steps
-------   --------------   ------    ---------
INPUT     low (<0.15)        320         3       broad sweep into context
INPUT     mid                256         3
INPUT     high (>0.6)        128         2       narrow, fast lock-on
PROCESS   low                256         3       diffuse forward thinking
PROCESS   mid                192         3
PROCESS   high               160         4       narrow but deeper — ruminate
OUTPUT    low                192         3
OUTPUT    mid                160         3
OUTPUT    high               128         2       say less, say it surely
```

Linear interpolation across arousal between bands so the cost is a smooth function. The pattern: high arousal narrows breadth (cuts budget) but lets PROCESSING go deeper (more steps), reflecting "narrow rumination". Low arousal widens breadth (raises budget), keeps depth modest, reflecting "associative drift."

Total worst-case cost per outer tick (one INPUT + several PROCESSING + one OUTPUT): 320 + 4×320 + 192 ≈ 1800 node touches at low arousal, ~1.5 ms total spread cost on M1. Comfortable.

### F4. inject_habit_seeds(composite_affect, arousal, phase)

The mechanism by which the habit overlay turns into seed concepts.

Process:
1. For each `HabitEntry` in `habit_seed_table`:
    - Skip if `phase` bit not set in `entry.fires_in_phase`.
    - Compute `affect_distance = ||composite_affect - entry.affect_signature||` (already squashed, both bounded).
    - Compute `affect_match = exp(-affect_distance / max(entry.affect_tightness, 0.05))`.
    - Skip if `affect_match < HABIT_FIRE_THRESHOLD = 0.3`.
    - Compute `inject_strength = entry.seed_strength * affect_match * habit_temperature`.
    - Multiply by `(1 - 0.5 * arousal)` — high arousal suppresses habit injection so genuine novelty is not buried under habit.
    - Add `entry.concept_id` to seeds with `inject_strength`. If already in seeds, take max.
2. If habit_seed_table is empty (cold session, or after consolidation reset) and seeds is also empty for this PROCESSING tick, inject `PerceptualBaseline.baseline_seed_ids` at strength 0.05 each.
3. Return augmented seeds.

Cost: O(|habit_seed_table|) ≤ 64 hashmap lookups + arithmetic. < 5 µs.

### F5. record_attention_event(frame, active_set, now)

Run after every `attend`. Updates HabitOverlay's running counts and AttentionStats. The habit *promotion* itself is deferred to `consolidate_habits` (slow, periodic); `record_attention_event` only collects evidence.

Process:
1. Update `AttentionStats`:
    - `frames_per_phase[frame.phase] += 1`
    - Welford update of `mean_budget_used_per_phase[frame.phase]` with `actual_touches` (returned alongside `active_set` by C — synthesis-pending: C must surface this; if C does not, F approximates via `len(active_set) * average_out_degree_estimate`).
    - Welford update of `mean_active_set_size_per_phase`, `mean_arousal_per_phase`.
    - Increment `predict_prior_overlap_running` via EMA against `|active_set ∩ frame.predict_prior| / |active_set|` if `predict_prior` exists.
2. Update last-fired timestamps in matched habit entries. For each `concept_id in frame.habit_seed_ids ∩ active_set`, set `entry.last_fired_t = now`.
3. Stage candidate habit observations into a small ring buffer (`pending_habit_obs`, capacity 1024):
    - Top 8 concepts by activation in `active_set` are candidates for `HabitEntry` reinforcement.
    - Top 8 traversed edges (provided by C; synthesis-pending — see INTERFACES) are candidates for `HabitPath` reinforcement.
    - Each candidate is staged with `(concept_id_or_edge_key, composite_affect, phase, now)`.
4. If `pending_habit_obs` is full or `now - last_consolidated_t > CONSOLIDATE_INTERVAL = 600 s` (10 min), call `consolidate_habits(now)`.

Cost: small. Welford updates are O(1) per field; staging is O(top_8). < 10 µs.

### F6. consolidate_habits(now)

The slow, periodic function that turns recent attention patterns into stable habits. The "memory of attention." This is what makes personality emerge from perception.

Process:
1. **Drain `pending_habit_obs`** into per-concept and per-edge tally dicts:
    - For each observation, `tally[concept_id].count += 1`; running mean of `composite_affect` and running variance for tightness; bitfield of phases observed.
    - Same for edges.
2. **Promote candidates to habits.** A concept becomes a `HabitEntry` if its tally count ≥ `HABIT_PROMOTE_MIN = 5` *and* its affect-signature variance is ≤ `HABIT_VARIANCE_MAX = 0.4` (i.e., it consistently fires under similar affective contexts; a concept that fires under any mood is not a habit, it is just a hub). Same for paths.
3. **Reinforce existing habits.** For each promoted candidate that is already in `habit_seed_table`:
    - `entry.n_observations += tally.count`
    - Smoothly update `entry.affect_signature` and `entry.seed_strength`:
      - `affect_signature = 0.9 * affect_signature + 0.1 * tally.affect_mean`
      - `seed_strength = clamp(seed_strength * 0.95 + 0.05 * normalized_tally_count, 0.05, 0.5)`
    - `entry.affect_tightness` updated as EMA of tally affect-stddev.
    - `entry.fires_in_phase |= tally.phase_bits`.
4. **Insert new habits.** If table not at cap, insert. If at cap, evict the entry with lowest score:
    - `score = n_observations * exp(-(now - last_fired_t) / HABIT_DECAY_TAU) * affect_match_with_character`
    - `affect_match_with_character` is cosine of `entry.affect_signature` vs `A.current_character()`. Habits aligned with character get protected; habits aligned with transient mood do not.
5. **Decay all habits.** Every entry: `seed_strength *= 0.99`; entries with `seed_strength < 0.02` and `now - last_fired_t > HABIT_FORGET_TAU = 14 days` are evicted.
6. **Pin/unpin requests to C.** For habits with `n_observations ≥ 50` and `seed_strength ≥ 0.2`, set `pin_request_active = True` and call `C.pin(concept_id)` with `pin_reason = "F.habit"`. C's pin-decay rule means F must touch (re-pin) habit pins every consolidation cycle to keep them. For habits below threshold, `C.unpin`.
7. **Refresh `PerceptualBaseline`**: pick top 8 `habit_seed_table` entries by `n_observations` whose `affect_signature` is within 0.3 of `A.current_character()`; copy concept_ids into `baseline_seed_ids`. Update `baseline_affect_target` as EMA toward the average of these entries' affect signatures.
8. **Adjust `habit_temperature`**: high working-affect deviation from character (large `||A.working - A.character||`) → drop temperature toward 0.2 (let novelty rule). Calm working-affect → drift back toward 0.5.

Cost: amortized. Runs at most every 10 min, processes ≤1024 staged observations, touches ≤64 habits. Total < 1 ms even at session-end.

### F7. processing_loop(input_frame, max_internal_ticks)

Manages the PROCESSING-phase recursion. After INPUT settles, the system can re-enter PROCESSING multiple times before producing OUTPUT — this is "thinking before speaking."

Inputs:
- `input_frame: AttentionFrame` — the result of the INPUT phase.
- `max_internal_ticks: uint8` — caller-imposed cap (G or H decides when to stop). Default 4.

Process:
1. `current_active = input_frame.active_set`
2. For `i in 0..max_internal_ticks`:
    a. Pick top-K (K=8) concepts from `current_active` as next seeds.
    b. Call `attend(PROCESSING, seeds=top_k, now)`. This produces a new active set.
    c. Ask B: `predict(state_from_active_set, A.composite(), PROCESSING)`. Predict once per processing tick, not per spread step. The state vector handed to B is the activation-weighted mean of node embeddings in the new active set.
    d. If B returns `confidence > PROCESSING_SETTLE_CONFIDENCE = 0.7` and the active set overlaps the prior tick's active set by > 0.8 (Jaccard), break early: the system has settled.
    e. If B returns a high-surprise gap on this internal tick (mid-thought surprise), let A inject as PROCESSING; the next tick's `composite_affect` will reflect that. This is the "feeling evolves mid-thought" mechanism.
3. Return the final active set.

Cost: bounded. 4 ticks × 4 ms per tick = 16 ms worst case for the entire processing phase. Acceptable.

**Position on PROCESSING granularity ticket:** B predicts once per `attend` call (i.e., once per processing tick), not once per spread step inside C. Per-step prediction would multiply B's cost by max_steps (up to 4×) and require B to handle partial activation states, which complicates B's representation-space contract for marginal benefit. Per-tick prediction preserves the "feeling evolves mid-thought" property because the *outer* loop allows multiple predictions during a single externally-observable input→output turn. If empirical run shows the system fails to notice mid-spread surprises that matter, escalate to per-step. v1 ships per-tick.

### F8. theory_of_mind_attention_shaping(other_agent_id, current_frame)

Optional augmentation called by H when the system has an active model of an external agent and wants to "prepare" for what the other will likely attend to. Synthesis-pending with E and H.

Provisional design:
1. From C: fetch concepts the system believes the other-agent attends to. These are concepts reachable from `other_agent_id` via `refers_to` and `part_of` chains, filtered by their own `running_state` affect-trace (the system's model of how the other feels).
2. Project them into seed-strength contributions at a reduced amplitude (`TOM_AMPLITUDE = 0.3`).
3. Merge into `current_frame.seeds` *before* habit injection so habits can still suppress weak ToM seeds.
4. Tag these seeds in `current_frame.habit_seed_ids` separately (`tom_seed_ids`) for stat tracking.

**Position on ToM ticket:** YES, the system's model of another agent's attention should bias its own — but at reduced amplitude and only when H/E has actually built a non-trivial agent model. The mechanism is just additional seeds; F does not need a separate algorithm for ToM beyond the seed-injection step. Detailed ownership of the agent model itself is E/H. F provides the hook (`theory_of_mind_attention_shaping`) and the call site (in `attend` step 2, after habit injection, before normalization). Marked synthesis-pending: E/H must specify how to enumerate "what the other attends to." If E/H produce nothing for v1, F's behavior degrades to no ToM bias — no failure, just a missing capability.

### F9. arousal_attention_shape(arousal)

A specification, not a separate function. Rather than a single function, "arousal shapes attention" is realized at three levels, all derived from `A.current_arousal(now)`:

1. **In A's `gate_attention`**: A already switches between multiplicative (high arousal, narrow/intense) and additive (low arousal, broad/diffuse) gating. F does not duplicate this.
2. **In C's `spread`**: C already sets `top_k = round(k_base + k_arousal * (1 - arousal))`. F does not duplicate this either.
3. **In F's `budget_for` and `steps_for`**: F adjusts call-level cost (budget, depth) by arousal band as tabulated in F3.

The three layers compose: high arousal → A gates multiplicatively (only well-aligned nodes pass) → C narrows top_k → F shrinks budget and may extend depth. The cumulative effect at arousal=0.8: a small intense beam, exhausted in a few hops. At arousal=0.1: a wide diffuse cloud, decaying gracefully. **This is the only arousal contract F adds beyond what A and C already provide.**

### F10. attention_at_input vs processing vs output — phase semantics

Distinct enough to spell out concretely.

**At INPUT:**
- Mode = PERCEIVE.
- Seeds come from the input pipeline H (concepts matched by `find_or_match` on the encoded input) plus habit injection plus optional ToM seeds.
- Budget skews wide; depth shallow. Goal: ground the input in context.
- B's prediction prior is *not yet available* for this specific input (B's INPUT-phase predict happens *before* H produces the final encoding, against H's pre-encoding stub state). F's INPUT call happens after B has predicted and after H has handed the actual encoding to F. The active set produced here is what B compares against.
- Affect injection at INPUT is large because surprise is computed first: A's `inject(INPUT, ..., g=1.0)` runs *before* F's INPUT attend in the lifecycle, so the composite F samples already reflects the input's affective signature.

**At PROCESSING:**
- Mode = PREDICT.
- Seeds come from the prior frame's top-K active concepts, plus habit injection. No new external input.
- Budget moderate; depth slightly higher. Goal: extend the chain forward.
- B's prediction is run *after* each PROCESSING attend (one per tick), and B may detect mid-thought surprise → A injects PROCESSING (g=0.6). The next tick's composite reflects the shift.
- This is where mood-vs-reaction asymmetry shows up: a calm mood with a sharp reaction injection produces a brief narrow thought; sustained reaction produces sustained narrow processing.

**At OUTPUT:**
- Mode = SIMULATE.
- Seeds come from G — specifically the concepts G is preparing to express. Habit injection muted (`habit_temperature * 0.5`) at OUTPUT because output-side habits are stylistic, not gating, and G owns those.
- Budget skews narrow; depth shallow. Goal: ask "if I express *this*, what does the world activate in response?" — a simulation rollout, briefly.
- The active set produced is what G compares against the system's *current* internal state to detect the discomfort gap (the lying signal). F does not detect the gap; F only produces the simulated post-output state. E/G computes the divergence.
- After OUTPUT, A's `inject(OUTPUT, ..., g=0.8)` fires, driven by E's gap measurement.

### F11. cost_summary

Every per-tick cost on M1, in microseconds, expected:

```
A.composite()         cached, ~50 ns
A.current_arousal()   cached, ~50 ns
inject_habit_seeds    < 5 µs
build AttentionFrame  < 1 µs
C.spread (budget=256) ~ 1500 µs
B.predict             ~ 5000 µs (per B's spec; runs ONCE per tick, not per step)
record_attention_event< 10 µs
-----------------------------------
per-tick total        ~ 6.5 ms
```

A full input→processing×4→output sequence: ~ 35 ms. Comfortably under the 10 Hz tick rate the spec implies. Headroom for visualization, ToM enumeration, and the occasional consolidation pass.

---

## INTERFACES

### Inbound (others call F)

- `attend(phase, raw_seeds, now) -> active_set: dict[uint64, float32]`
  - Caller: H (INPUT), F's own processing loop (PROCESSING), G (OUTPUT).
  - Side effect: AttentionFrame recorded in stats; habit observations staged; possible habit consolidation triggered.
- `processing_loop(input_frame, max_internal_ticks) -> active_set`
  - Caller: H or G — whichever owns the outer tick scheduler. Typically H drives the input→processing handoff.
- `current_attention_state() -> { active_set, frame_metadata }`
  - Read-only view for the frontend visualization and for E (identity needs to read what is currently attended-to).
- `request_habit_pin_refresh(now) -> None`
  - Trigger an out-of-cadence consolidate so pins held by F do not lapse. Called by E when it notices F's pinned set drifting.
- `tom_seed_provider(callback)` — registration. H/E supplies a callback that, given a current `AttentionFrame`, returns ToM seeds. F invokes it inside step 2 of `attend`. Optional; if unregistered, ToM is a no-op.
- `set_habit_temperature(value)` — tuning hook for E during major identity transitions (e.g., a deliberate "open mind" mode).
- `persist() -> bytes` and `restore(bytes) -> None` — for HabitOverlay and PerceptualBaseline.

### Outbound (F calls others)

From A:
- `A.composite(now) -> float32[N]` — once per attend call. (Spec already defines.)
- `A.current_arousal(now) -> float32` — once per attend call.
- `A.current_character() -> float32[N]` — only during `consolidate_habits`.
- F does not call `A.inject` — A's injections are driven by B based on prediction gaps. F does not produce gaps.

From B:
- `B.predict(state, affect, layer) -> Prediction` — called from `processing_loop` after each PROCESSING attend (and also wrapped around INPUT/OUTPUT by H/G respectively, not by F).
- F reads B's `Prediction.support_set` to populate `predict_prior` on subsequent frames.

From C:
- `C.spread(seeds, composite_affect, arousal, max_steps, budget, mode) -> dict[uint64, float32]` — the workhorse. F is the primary caller.
- `C.pin(concept_id, pin_reason="F.habit")` and `C.unpin(concept_id)` — habit pins.
- C's per-edge gate (synthesis-pending detail): C's spread internally calls `A.gate_attention` per traversal, with `semantic_score = activation * edge.weight * edge.confidence`, `predictive_score = predict_prior_lookup_for_dst(predict_prior, dst_concept_id, default=1.0)`, and the node's own `affect_trace.running_state` as `node_affect_trace`. **Synthesis-pending with C and A:** the precise wiring of `predictive_score` from F's `predict_prior` into A's gate function is something F supplies but A's gate consumes; the field exists in A's signature but A's spec does not say where the value comes from. F's claim: `predictive_score` is `1 + predict_prior.get(dst_concept_id, 0)` clamped to [1, 2]. Concepts B predicted strongly get a 2× boost; all others pass through unchanged (1×). This makes "predicted then confirmed" paths slightly preferred without suppressing novelty.

From C also (synthesis-pending — C does not currently surface these but F needs them for habit_path tracking):
- The list of edges traversed during a spread, ideally with their final propagated activation, returned alongside `active_set`. If C does not expose this, F infers paths heuristically by sampling the top-N (concept_id, neighbor) pairs after the spread, which is approximate and noisy. **Flag: requires a small extension to C.spread's return signature.**

From E (optional):
- `E.tom_seeds_for(other_agent_id, current_frame) -> dict[uint64, float32]` — provides ToM seeds when the system has an active agent model. Optional; F handles None gracefully.

### Threading

F is single-loop with C and A: all calls happen on the main perception thread. F never spawns threads. `consolidate_habits` runs inline at the end of `record_attention_event` calls when triggered, accepting the occasional 1 ms hiccup every 10 minutes.

---

## FAILURE MODES

### F-FM1. Attention saturation (active set explodes)

**Manifestation:** Active set returned by C exceeds expected size (e.g., > 200 in a 5K graph). Subsequent processing chokes; B's predict gets noise.

**Detection:** F checks `len(active_set)` against `max_active_set_per_phase` (default INPUT=128, PROCESSING=96, OUTPUT=64). Logs `ATTENTION_SATURATION` with phase and size.

**Response:** Truncate to top-N by activation before returning to caller. Do not retry the spread with a smaller budget — the cost is already paid. Increment `last_runaway_t`. If saturation rate > 5% of frames over a 5-minute window, raise `habit_temperature * 0.8` (suppress habits — they may be over-injecting).

### F-FM2. Empty active set

**Manifestation:** C returns no concepts above `θ_activate`. Caller has nothing to process.

**Detection:** `len(active_set) == 0` after spread.

**Response:** Inject `PerceptualBaseline.baseline_seed_ids` directly as the active set (with strength 0.05 each) and return. This is the "blank stare" fallback — the system always has something to attend to even if the input had no graph anchor. If no baseline exists either (cold start), return an empty dict; H/G will call A's flatness response.

### F-FM3. Habit overlay corruption (NaN in affect signatures, invalid concept_ids)

**Manifestation:** Habit injection introduces NaN seeds, or seeds reference tombstoned concepts.

**Detection:** Per-frame finiteness check on injected seeds; `C.spread` will reject NaN. Per-load: scan all `habit_seed_table` entries against `C.nodes`, drop entries whose `concept_id` no longer exists.

**Response:** Drop the offending entry, log `HABIT_CORRUPTION`. The overlay is rebuildable from observation; losing a habit is recoverable.

### F-FM4. Predict-prior staleness

**Manifestation:** F uses a `predict_prior` from a tick that no longer reflects current state (e.g., several attends happened without B predicting, so F is biasing toward an old prediction).

**Detection:** `predict_prior_tick` is older than `now_tick - PREDICT_PRIOR_TTL = 4 ticks`.

**Response:** Drop the prior; treat as None. F does not synthesize a new prior — that is B's job.

### F-FM5. Processing recursion runaway

**Manifestation:** `processing_loop` keeps running because B never returns settle confidence and the active set keeps churning. CPU spikes.

**Detection:** Hitting `max_internal_ticks` repeatedly (>5 in a 30 s window).

**Response:** F enforces a hard ceiling on `max_internal_ticks` (cap = 8 regardless of caller request). At cap, F returns whatever active set is current and logs `PROCESSING_RECURSION_CAP`. Critically: F also injects a small `affect.inject(PROCESSING, low_arousal_seed, magnitude=0.1)` to nudge the system toward calm — this is the closest F gets to driving A directly, and it is justified as a self-protective reflex against rumination loops. **Synthesis-pending with A:** A may prefer to detect rumination itself via its own arousal-stuck heuristic; if so, F removes this injection.

### F-FM6. Habit pin lapse

**Manifestation:** F holds pins on habit concepts but `consolidate_habits` doesn't run for a long time (e.g., system suspended). C's pin-decay degrades the pins to soft boosts; the habit concept gets pruned.

**Detection:** On consolidation, F checks `pin_request_active` entries against C's pin status; missing pins indicate decay happened.

**Response:** Re-pin if the entry is still valid; log `HABIT_PIN_LAPSED`. The habit is *not* lost — its evidence is still in `n_observations` — but the concept it pointed at may be gone, in which case the entry is dropped.

### F-FM7. Per-edge gate disagreement

**Manifestation:** F's `predict_prior` says concept X should be boosted, but C's spread reaches X via an edge whose `affect_alignment` is near zero. The gate produces near-zero activation; X never fires despite being predicted.

**Detection:** Diagnostic only; observable as low `predict_prior_overlap_running`.

**Response:** This is the system *correctly* refusing to attend to predicted-but-not-felt concepts. No fix; it's working as designed. If overlap stays below 0.1 for extended periods, flag for E to investigate identity drift — F is no longer following B.

### F-FM8. Persistence corruption (HabitOverlay)

**Manifestation:** Restore fails or yields invalid entries.

**Response:** Reset HabitOverlay to empty; log loudly. Habits will rebuild over the next ~24 hours of normal use. PerceptualBaseline rebuilds within minutes once habits start re-accumulating. The mind continues without a perceptual fingerprint for a day; not catastrophic.

### F-FM9. Cold-start (no habits, no baseline)

**Manifestation:** First session, first hour. `inject_habit_seeds` injects nothing.

**Response:** Expected. F runs purely on caller-supplied seeds and C's structural spread. Habits will appear within ~6–8 consolidation cycles (1 hour of active use). This is correct: a newborn mind has no perceptual habits yet.

### F-FM10. Affect-shaped attention starvation

**Manifestation:** Composite affect points in a direction with no matching `affect_trace` in any concept (e.g., a brand-new affective regime). Every `gate_attention` returns near zero. Active set collapses.

**Detection:** Active set size < 4 for several consecutive frames despite seeds being injected.

**Response:** F temporarily relaxes the gate by widening C's spread to include `affect_alignment` of zero (effectively additive even at high arousal). One-shot per affect regime — once nodes start accumulating affect traces in this direction, gating returns to normal. F achieves this by passing a special `top_k_override = max_top_k` and accepting the cost. Logs `AFFECT_REGIME_NOVEL`. This is the architectural answer to "how does the system *bootstrap* attention into a new emotional state?"

---

## OPEN QUESTIONS

1. **Should `consolidate_habits` integrate sleep cycles?** Component A's spec mentions a possible `consolidation_mode(true/false)` flag for sleep. If A introduces it, F's consolidation cadence should align — habits crystallize during sleep, not during waking attention. Synthesis-pending with A and D.

2. **Does the predict_prior boost (1× to 2×) need to vary by phase?** At PROCESSING the boost makes sense; at INPUT B's prior is from the *prior* tick and may not be relevant; at OUTPUT B's prior is about the proposed expression. The current design uses the same boost everywhere. Empirical: measure overlap separately by phase and consider phase-specific multipliers.

3. **HabitPath amplifier — does it actually do anything if PERCEIVE/PREDICT spreading dominates?** Habit paths only matter if the same `(src, dst, type)` triple is repeatedly traversed under similar affect. In a richly-connected graph, lots of paths qualify. The cap of 256 may be too small or too large; needs tuning.

4. **What happens during identity transitions?** If E forces a character shift (an unusual event but possible per the spec's "ten thousand moods" mechanism), F's habits become misaligned with the new character. Current design lets habits decay naturally over 14 days. E may want to trigger an explicit habit-flush via `set_habit_temperature(0)` for a transition window. Synthesis-pending with E.

5. **ToM amplitude calibration.** `TOM_AMPLITUDE = 0.3` is a guess. Too high and the system becomes paranoid (its own attention is dominated by its model of others'); too low and ToM doesn't influence attention enough to be useful. Empirical.

6. **C's return signature for traversed edges.** F needs a list of traversed edges to track HabitPaths. C's current spec returns `dict[uint64, float32]` (active set only). Either C extends its return type, or F approximates paths heuristically. **Synthesis-required with C.** I have argued the extension is small and worth it; if rejected, F runs in approximate-paths mode and HabitPath quality degrades.

7. **Does F need its own "what is being attended to right now" cache for E's identity reads?** Currently F exposes `current_attention_state()` returning the latest frame. If E reads attention many times per tick (unlikely), a cache helps. If E reads at human-perceptible rates, the latest-frame approach is fine.

8. **Spec contradiction worth surfacing.** The spec says "attention is not computed — it emerges." But it also says "what propagates is determined by the intersection of: semantic relevance × affective relevance × predictive relevance." The intersection of three signals *is* a computation. The reconciliation in this design: F computes nothing of its own; the three signals are computed by A (affective), B (predictive), and C (semantic), and F merely chooses the call site. The "emergence" claim holds at the level of F; it does *not* hold at the level of A's gate, which is explicit math. This is fine but worth flagging — readers expecting a fully implicit attention will not find one. Attention is computed, just not by F.

9. **Does `arousal` need to be sampled more than once per attend at very high arousal?** At arousal=0.9, the reaction layer might shift meaningfully within a single attend's duration (sub-200 ms). Re-sampling mid-attend would introduce non-determinism but might better track rapidly evolving feeling. Decision in v1: do not re-sample. Trade-off: a "frozen" affect during a single attention pass means a single thought has a single feeling, even if reality is shifting underneath. This matches phenomenology and simplifies analysis. Revisit if behavior looks too sluggish under acute affect.

10. **Underspecification flag: F's relationship to G at OUTPUT phase.** The spec says output attention is its own thing, but G's spec has not been written yet. F provides an OUTPUT mode that returns an active set. What G does with that active set — whether it directly samples from it for expression or runs another layer of selection — is G's call. F's contract: at OUTPUT, the active set represents "what the world might activate if this output is produced." G interprets it. Synthesis-pending with G.
