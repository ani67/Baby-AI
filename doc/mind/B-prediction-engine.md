# COMPONENT B — THE PREDICTION ENGINE

## OVERVIEW

The Prediction Engine is the substrate component that gives The Mind a reason to learn anything at all. Before any input is processed, before any concept activates, before any output forms, the engine produces an expectation — a vector in representation space that anticipates "what comes next." When reality arrives, the engine compares prediction to actuality and produces a *prediction gap*: a signed, multi-dimensional error signal. That gap is the only thing in the entire system that triggers learning. Confirmation costs nothing; surprise costs and earns everything. The engine fires at three layers (input, processing, output), drives the Affective Engine through gap magnitude and direction, gates writes to the Concept Graph through a surprise threshold, and serves as the kernel of the Simulation Layer (a simulation is a chain of predictions). It is intentionally cheap, stateless between predictions, and improves not by gradient descent but by the same mechanism every other part of the system uses: when its own predictions are wrong, the graph it predicts from is updated, which makes its next prediction better. The Prediction Engine does not know things. It guesses, measures how wrong it was, and hands the wrongness to the rest of the system.

## CORE DATA STRUCTURES

Every structure here is owned by the Prediction Engine. Anything that lives elsewhere (the affect vector, concept nodes) is referenced via the interfaces specified later.

### `RepresentationVector`
A fixed-dimensional dense vector that is the universal currency of prediction. Every prediction and every encoded input/state must produce one of these so that gap computation is well-defined.
- `values: float32[D_REP]` — the vector itself.
- `D_REP: int` — global constant. Recommended D_REP = 256. Rationale: large enough to encode rich semantics (concept embeddings live here), small enough that L2 distance and cosine on M1 are sub-millisecond for batches of <100. Must equal the embedding dimension used by the Concept Graph (synthesis-phase reconciliation point with C).
- `norm_hint: float32` — cached L2 norm; lets cosine similarity skip a sqrt.
- `source: enum {INPUT_ENCODED, GRAPH_PREDICTED, SIMULATED, OBSERVED, REPLAYED}` — provenance, used for diagnostics and to route the gap appropriately.

### `Prediction`
The output of a prediction call, before reality has arrived. This is a *distribution*, not a point, because prediction confidence must be readable.
- `mean: RepresentationVector` — the expected representation.
- `precision: float32[D_REP]` — per-dimension inverse variance. High precision = "I'm confident on this dimension." Low precision = "I have no idea on this dimension." Stored per-dimension rather than as a full covariance matrix to keep memory and arithmetic O(D_REP) instead of O(D_REP^2).
- `confidence_scalar: float32 in [0,1]` — a single summary derived from precision (mean of normalized precision values). Cached so the Affective Engine and attention layer don't have to recompute.
- `support_set: list[(concept_id, weight)]` — the (typically 5–30) concepts that contributed to this prediction, with their contribution weights. Needed by the Concept Graph for selective edge-weight updates and by the Simulation Layer to know what was being assumed.
- `layer: enum {INPUT, PROCESSING, OUTPUT}` — which of the three injection points emitted this prediction.
- `tick: int64` — global monotonic time index at the moment of prediction. Lets the engine correlate prediction with the matching observation.
- `affect_snapshot: float32[N_AFF]` — copy of the composite affect vector at prediction time. Required because the prediction was *conditional* on this affect; the gap can only be interpreted if we remember what mood produced the guess. Synthesis-phase reconciliation with A: the Affective Engine must expose a read-only `composite()` returning this vector.

### `Observation`
What actually arrived, paired against a prior prediction.
- `actual: RepresentationVector`
- `tick: int64` — must match a `Prediction.tick` for gap computation.
- `layer: enum {INPUT, PROCESSING, OUTPUT}` — must match the prediction's layer.

### `PredictionGap`
The error signal. This is the only thing that flows downstream as a learning trigger.
- `delta: float32[D_REP]` — `actual - mean`, dimension-wise. Direction matters: positive on a given dimension means "more of this than I expected," negative means "less."
- `weighted_delta: float32[D_REP]` — `delta * precision`. This is the precision-weighted error: surprise on a dimension you were confident about counts more than surprise on a dimension you knew nothing about. Mahalanobis-style but cheap.
- `magnitude: float32` — L2 norm of `weighted_delta`. The single scalar surprise score.
- `signed_magnitude_per_dim: float32[D_REP]` — same as `weighted_delta`; kept under both names because the Affective Engine consumes "direction" while the Concept Graph consumes "magnitude," and clarity at the interface beats one fewer field.
- `confidence_at_prediction: float32` — copy of `Prediction.confidence_scalar`. Needed for the arousal mapping (high-confidence wrongness produces more arousal than low-confidence wrongness).
- `surprise_score: float32` — final surprise after threshold and confidence weighting. Computed by the SurpriseScoring algorithm below. This is the value compared to the write threshold.
- `is_surprise: bool` — `surprise_score >= SURPRISE_THRESHOLD`.
- `tick: int64`, `layer: enum`, `affect_snapshot: float32[N_AFF]` — carried through.

### `PredictionEngineState`
Persistent state owned by the engine. Survives across sessions.
- `pending_predictions: dict[int64 -> Prediction]` — predictions that have been emitted but not yet matched to an observation. Bounded; entries older than `PENDING_TTL_TICKS` are evicted.
- `running_gap_stats: WelfordStats` — online mean and variance of `weighted_delta.magnitude` per layer. Used to set the adaptive surprise threshold without storing history.
  - `count_per_layer: int64[3]`
  - `mean_per_layer: float32[3]`
  - `m2_per_layer: float32[3]` (running sum of squared deviations, Welford)
- `confidence_calibration: ReliabilityHistogram` — for each of K=10 confidence buckets, tracks how often predictions in that bucket were actually accurate. Used to detect over- or under-confidence and to recalibrate `confidence_scalar` before it leaves the engine.
  - `bucket_count: int32[10]`
  - `bucket_correct: int32[10]` (where "correct" = `magnitude < bucket-specific accuracy band`)
- `tick_counter: int64` — global monotonic counter incremented on every prediction.
- `last_persisted_tick: int64` — last tick at which state was flushed to disk.

### `SimulationFrame` (shared with Component D)
The Prediction Engine produces these; the Simulation Layer consumes and chains them.
- `prior_state: RepresentationVector` — the pre-action state.
- `candidate_action: ActionDescriptor` — opaque to this component; D defines.
- `predicted_next: Prediction` — the engine's forward prediction conditioned on the action.
- `chain_id: uuid` — groups frames belonging to one rollout.
- `depth: int8` — how many steps deep into the rollout this frame is.

## ALGORITHMS

### A1. Predict
**Inputs:** `current_state: RepresentationVector`, `affect_composite: float32[N_AFF]`, `layer: enum {INPUT, PROCESSING, OUTPUT}`, optional `query_seed: RepresentationVector` (used at OUTPUT layer to predict how a candidate output will land), optional `topK: int = 30`.

**Process:**
1. Compute the conditioning vector `q = concat(current_state, affect_composite, query_seed_or_zeros)`. Project to D_REP via a fixed (non-learned) random projection matrix `P_q` initialized once at engine birth and persisted. Rationale: deterministic, cheap, no gradient training; the projection is a hashing trick that maps state+affect into the same space as concept embeddings.
2. Query the Concept Graph for the top-K most similar concepts to `q` using the graph's existing similarity index. This call is `graph.activate_neighbors(q, k=topK, affect_gate=affect_composite)`. The graph applies its affect-gated spreading activation and returns a list of `(concept_id, embedding, activation_strength)`.
3. Compute the predicted mean as a precision-weighted average of returned embeddings:
   - For each returned concept, the weight is `activation_strength * affect_alignment(concept.affect_trace, affect_composite)`. `affect_alignment` is cosine similarity in affect space, clamped to [0,1]. This means concepts with affect traces similar to current affect contribute more — the system predicts what it expects given how it feels.
   - `mean = normalize(sum(weight_i * embedding_i) / sum(weight_i))`.
4. Compute precision per dimension:
   - For each dimension d, `var_d = weighted_variance(embedding_i[d])`. Low variance across the top-K → high precision on that dimension. High variance → low precision.
   - `precision[d] = 1 / (var_d + EPSILON)` then clamped and normalized so the mean precision across dimensions equals 1.0. Normalizing keeps the magnitude of `weighted_delta` on a comparable scale across predictions.
5. `confidence_scalar = sigmoid(log(mean(precision)) - confidence_calibration.shift)`, where `confidence_calibration.shift` is updated by the calibration algorithm (A6). This is the single user-facing confidence number, post-calibration.
6. Build the `Prediction`, stamp `tick`, capture `affect_snapshot`, store the support set (concept_id + final weight), and insert into `pending_predictions[tick]`.
7. Return the `Prediction`.

**Outputs:** `Prediction` (and a side effect: entry added to `pending_predictions`).

**Cost target:** under 5ms on M1 for D_REP=256, topK=30, current graph size ≤5K nodes. The dominant cost is the graph similarity query, which the Concept Graph must serve from an index (synthesis with C: requires `activate_neighbors` to be sub-2ms).

### A2. ComputeGap
**Inputs:** `Observation`, lookup of the matching `Prediction` from `pending_predictions[observation.tick]`.

**Process:**
1. If no matching prediction exists (e.g., observation arrived for a tick that was never predicted, or the prediction was evicted by TTL), emit a `MISSED_PREDICTION` failure event and return null. Do not fabricate a gap.
2. Verify `prediction.layer == observation.layer`. Mismatch is a programming error — fail loud.
3. `delta = observation.actual.values - prediction.mean.values` (per-dimension).
4. `weighted_delta = delta * prediction.precision`.
5. `magnitude = L2_norm(weighted_delta)`.
6. Update `running_gap_stats` for this layer using Welford's online update.
7. Construct `PredictionGap` with all fields populated except `surprise_score`/`is_surprise` (filled by A3).
8. Remove the prediction from `pending_predictions`.

**Outputs:** `PredictionGap` (without surprise scoring yet).

### A3. SurpriseScoring
**Inputs:** `PredictionGap` (without `surprise_score`), `running_gap_stats` for the relevant layer, `confidence_at_prediction`.

**Process:**
1. Compute the z-scored magnitude relative to the layer's running statistics:
   - `mean_mag, var_mag = running_gap_stats.layer_stats(layer)`
   - `z = (magnitude - mean_mag) / sqrt(var_mag + EPSILON)`
2. Apply the confidence amplifier:
   - `surprise_score = z * (0.5 + confidence_at_prediction)`
   - Rationale: being wrong when you were certain is much more surprising than being wrong when you had no idea. The amplifier ranges from 0.5 (zero confidence: gap counts half) to 1.5 (full confidence: gap counts 1.5x). Linear is intentional — non-linear curves invite tuning that can't be justified empirically yet.
3. `is_surprise = surprise_score >= SURPRISE_THRESHOLD`. The threshold is *adaptive*:
   - `SURPRISE_THRESHOLD = max(MIN_THRESHOLD, target_z_for_top_alpha_percentile)`
   - The engine targets that approximately `ALPHA = 5%` of observations cross the threshold. `target_z_for_top_alpha_percentile` is read from the layer's running distribution (stored as a coarse 20-bin histogram alongside Welford stats; see Failure Mode F4 for what happens if the histogram is empty).
   - `MIN_THRESHOLD = 1.5` (z-score floor) so that the first few observations of a session don't trigger spurious "surprise" before statistics stabilize.
4. Stamp `surprise_score` and `is_surprise` on the gap.

**Outputs:** complete `PredictionGap`.

**Why adaptive threshold:** a fixed threshold gives the wrong answer when the input distribution shifts. A novel modality, a noisy environment, or a predictable one all change the baseline gap distribution. The system should learn from the top X% of surprises *relative to its own current life*, not against a hardcoded number.

### A4. EmitGap
**Inputs:** `PredictionGap`.

**Process:**
1. Always send to the Affective Engine via the `affect.on_gap(gap)` interface. Even non-surprise gaps inform affect (small gaps = small affect updates = the texture of normal experience).
2. If `is_surprise`, also send to the Concept Graph via `graph.on_surprise(gap, prediction.support_set, prediction.affect_snapshot)`. The graph decides whether to write a new node, strengthen edges, or both.
3. If `is_surprise`, push to the Replay Buffer (Component D) via `replay.push(gap, support_set, affect_snapshot, observation)`.
4. Update `confidence_calibration`: bucket the prediction by its `confidence_scalar`, increment `bucket_count[bucket]`, and increment `bucket_correct[bucket]` if `magnitude < bucket-specific accuracy band` (band derived from running stats).

**Outputs:** none (side effects only).

### A5. ConfidenceToArousal
**Inputs:** `Prediction.confidence_scalar`, optional context (whether a gap has been computed yet).

**Process:** Pre-observation, the engine emits an *arousal seed* to affect that scales with confidence: high confidence → low pre-arousal (system is settled), low confidence → high pre-arousal (system is on edge). Post-observation, the actual gap drives the affect update; confidence then determines amplification (A3 step 2). The relationship is therefore biphasic:
- Before the answer arrives: `pre_arousal_seed = (1 - confidence_scalar) * AROUSAL_GAIN_PRE`. Sent to affect as a small, transient nudge.
- After the answer arrives: confidence amplifies how much the gap moves affect (handled in A3 and the Affective Engine).

**Outputs:** `pre_arousal_seed: float32`, sent to affect via `affect.on_pre_prediction(seed)`.

This is what produces the felt difference between "I am unsure and waiting" (rising arousal) and "I am sure and waiting" (calm). Arousal is not predicted; it falls out of confidence.

### A6. CalibrationUpdate
**Inputs:** `confidence_calibration` (current), recent observations.

**Process:** Periodically (every `CALIB_INTERVAL = 200` ticks):
1. For each confidence bucket, compute observed accuracy = `bucket_correct / bucket_count`.
2. Compare to the bucket's nominal confidence (e.g., bucket 7 represents confidence ~0.7).
3. If mean (observed - nominal) across populated buckets exceeds `CALIB_DRIFT_THRESHOLD`, set `confidence_calibration.shift` so that future `confidence_scalar` values are squashed or stretched accordingly. This is a single scalar correction — anything richer is gradient descent in disguise.
4. Decay `bucket_count` and `bucket_correct` by `CALIB_DECAY = 0.95` so calibration stays current rather than dominated by ancient data.

**Outputs:** updated `confidence_calibration`.

### A7. SimulationStep
**Inputs:** `current_state: RepresentationVector`, `affect_composite`, `candidate_action: ActionDescriptor`, `chain_id`, `depth`.

**Process:**
1. Component D supplies a function `world_model.apply_action(state, action) -> seed_state`. The Prediction Engine treats this as a black box that returns a hypothesized next state.
2. Run A1 (Predict) with `current_state = seed_state`, `affect_composite` carried forward (or simulated forward by the Affective Engine if D requests it), `layer = PROCESSING`.
3. Wrap the returned `Prediction` into a `SimulationFrame` with the chain_id and depth.
4. Do **not** insert into `pending_predictions` — simulated frames are not awaiting a real observation. They live in the simulation chain only.
5. Return the frame.

**Outputs:** `SimulationFrame`.

This is the entire engine-side contribution to simulation: a simulation rollout is a chain of A1 calls with no observations to match against. The Simulation Layer (D) decides which chain to commit to and how simulated affect is felt.

### A8. SimulationVsRealityReconciliation
**Inputs:** committed `SimulationFrame` (the one whose `candidate_action` was actually executed), real `Observation` of what the world produced.

**Process:**
1. Compute a `PredictionGap` between the committed frame's `predicted_next` and the real observation, using A2.
2. This gap is tagged with a `simulation_origin = chain_id` flag and routed to Component D's `world_model.on_simulation_gap(gap, frame)` so that the world model can be improved.
3. The same gap is *also* routed normally through A4 (affect, graph, replay), because from the engine's perspective a simulation that turned out wrong is just another wrong prediction. Surprise from a failed simulation is real surprise.

**Outputs:** `PredictionGap`, plus calls into D and standard A4 emission.

### A9. PendingEviction
**Inputs:** `tick_counter`, `pending_predictions`.

**Process:** Every tick, evict any prediction with `prediction.tick < tick_counter - PENDING_TTL_TICKS`. Default `PENDING_TTL_TICKS = 1024`. Each eviction emits a `STALE_PREDICTION` event for diagnostics. The eviction is silent on affect/graph (no synthetic gap is created) — we did not observe what was predicted, so we have no learning signal.

**Outputs:** none (side effect: state pruned).

## INTERFACES

### What this component exposes (others call in)

- `predict(current_state, affect_composite, layer, query_seed=None, topK=30) -> Prediction`
  - Caller: Component H (Input Pipeline) at INPUT layer; Component F (Attention) at PROCESSING layer; Component G (Expression) at OUTPUT layer; Component D (Simulation) via A7.
  - Side effect: registers a pending prediction.
- `observe(observation: Observation) -> PredictionGap | None`
  - Returns null if no matching prediction. Otherwise returns the full computed gap and emits it (A4) before returning. Caller can read the gap without re-fetching it.
  - Caller: Component H delivers the encoded actual representation back to the engine after input is fully encoded; G delivers the post-expression observation; F delivers post-activation actuals during PROCESSING.
- `confidence(prediction_id) -> float32`
  - Convenience accessor for callers that hold a Prediction reference and want only the calibrated scalar.
- `simulate_step(state, affect, action, chain_id, depth) -> SimulationFrame`
  - Caller: Component D.
- `reconcile_simulation(frame, observation) -> PredictionGap`
  - Caller: Component D, after the world delivers a real observation following an executed simulated action.
- `stats_snapshot() -> EngineStatsView`
  - Read-only view of running stats and calibration. Used by the frontend for visualization and by Identity (E) for narrative continuity (the *shape of what surprised the system* is part of identity).
- `persist() -> bytes` and `restore(bytes) -> None`
  - Serialization of `PredictionEngineState`. Called by the persistence layer at session boundaries.

### What this component calls out to (others must expose)

- **Concept Graph (C):**
  - `graph.activate_neighbors(query_vector, k, affect_gate) -> list[(concept_id, embedding, activation_strength)]` — used by A1 to gather support for the prediction. Must be sub-2ms for k=30 over 5K nodes.
  - `graph.on_surprise(gap, support_set, affect_snapshot) -> None` — fired when a surprise crosses threshold. Graph decides whether to write a new node and which edges to update.
  - **Synthesis-phase reconciliation:** the embedding dimension used by the graph must equal `D_REP`. The graph's `activate_neighbors` must accept an `affect_gate` argument; if Component C decides affect-gating happens via a different call path, A1 step 2 must be adjusted.

- **Affective Engine (A):**
  - `affect.composite() -> float32[N_AFF]` — read-only snapshot used at A1 step 1 and stored on every Prediction.
  - `affect.on_pre_prediction(seed) -> None` — small arousal nudge before observation.
  - `affect.on_gap(gap) -> None` — magnitude and signed-per-dim direction drive the affect update. Component A is responsible for translating `weighted_delta` and `magnitude` into N_AFF-dimensional affect movement; the Prediction Engine does not interpret affect dimensions.
  - **Synthesis-phase reconciliation:** Component A must define how `signed_magnitude_per_dim` (D_REP-dimensional) projects into the N_AFF affect dimensions. Two reasonable contracts: (a) A maintains a learned/projection matrix `M : D_REP -> N_AFF`; (b) A consumes only `magnitude` and a small set of summary statistics. The Prediction Engine produces both and lets A choose.

- **Simulation + Replay (D):**
  - `world_model.apply_action(state, action) -> RepresentationVector` — used by A7.
  - `world_model.on_simulation_gap(gap, frame) -> None` — used by A8.
  - `replay.push(gap, support_set, affect_snapshot, observation) -> None` — used by A4 on surprise.

- **Persistence layer (system-level):**
  - Standard read/write of bytes; not specific to any component.

## FAILURE MODES

### F1. Prediction has no support (empty top-K)
**Manifestation:** A1 step 2 returns an empty list (concept graph is empty or no concept passes affect-gating).
**Response:** Emit a `Prediction` with `mean = zero_vector`, `precision = uniform_low` (so any observation will register as moderate, not infinite, surprise), `confidence_scalar = 0`. Tag the prediction with `degenerate = True`. Downstream consumers treat degenerate predictions as informational. This must work — it is the system's state at birth, before anything has been written to the graph.

### F2. NaN or inf in prediction or gap
**Manifestation:** Numerical overflow from a runaway precision (var_d → 0) or a malformed input vector.
**Response:** Clamp precision to `[PRECISION_FLOOR, PRECISION_CEILING]` with `PRECISION_FLOOR = 1e-3`, `PRECISION_CEILING = 1e3`. If a NaN still appears anywhere in the pipeline, drop the prediction/gap, log loudly, and increment a `NUMERIC_FAULT` counter. Never propagate NaN to affect or graph — those subsystems can be poisoned for the rest of the session by a single bad value.

### F3. Mismatched layer between prediction and observation
**Manifestation:** Caller routed an OUTPUT-layer observation against an INPUT-layer prediction (or similar). This is always a programming error.
**Response:** Refuse to compute a gap. Raise an explicit `LayerMismatchError`. Do not soft-fail — silent miscategorization here would corrupt running statistics permanently.

### F4. Cold-start running statistics
**Manifestation:** First N observations of a fresh session have no Welford stats to z-score against, so `surprise_score` is undefined or extreme.
**Response:** While `count_per_layer[layer] < COLD_START_N` (default 32), use `MIN_THRESHOLD` only and skip z-scoring. Once warm, switch to adaptive. Persistence carries Welford state across sessions, so cold-start really only happens at first birth.

### F5. Pending predictions accumulate (observation never arrives)
**Manifestation:** Caller predicts but never observes. `pending_predictions` grows unbounded, memory leaks.
**Response:** A9 (PendingEviction) runs every tick. Hard cap on `len(pending_predictions)` at 4096; if exceeded, evict oldest regardless of TTL and log `PENDING_OVERFLOW`. This indicates a contract violation by a caller and should surface in diagnostics.

### F6. Calibration histogram becomes degenerate
**Manifestation:** All buckets empty except one (the system always emits the same confidence). Calibration update produces zero or extreme shift.
**Response:** If fewer than 3 buckets have `count >= CALIB_MIN_PER_BUCKET = 20`, skip calibration update for this cycle. Calibration only runs when the data supports it.

### F7. Surprise rate collapses or explodes
**Manifestation:** Adaptive threshold tracks ALPHA=5%, but if the input distribution suddenly shifts (e.g., a new modality is introduced), surprise rate may briefly hit 50% or 0%.
**Response:** Clamp the dynamic threshold to `[MIN_THRESHOLD, MAX_THRESHOLD = 8.0]` so that even during a regime shift, surprise scores remain bounded. The 5% target is a soft goal, not a constraint.

### F8. Simulation chain depth blows up
**Manifestation:** D requests deep rollouts and the engine produces hundreds of frames per choice.
**Response:** The engine itself does not cap depth — that's D's policy. But the engine refuses any single `simulate_step` call that arrives with `depth > MAX_SIM_DEPTH = 64` and emits `SIM_DEPTH_EXCEEDED`. This is a backstop, not a feature.

### F9. Persistence corruption
**Manifestation:** `restore()` receives bytes that don't deserialize cleanly.
**Response:** Reset `PredictionEngineState` to fresh-birth state. Log loudly. Do *not* attempt partial recovery — running stats are tightly correlated and a partial restore is worse than a clean start.

### F10. Confidence drift from real wrongness
**Manifestation:** The system becomes systematically over- or under-confident over long timescales (dispositional drift in calibration).
**Response:** This is what A6 is for. If A6 is itself failing (e.g., the drift threshold is being chased indefinitely), the calibration shift is hard-clamped to `[-2.0, +2.0]` so confidence cannot collapse to 0 or saturate at 1.

## OPEN QUESTIONS

These are deliberately unresolved. Some require empirical data; others depend on Components A, C, D being specified.

1. **What exactly is the "representation space" beyond "D_REP-dim float vector"?** I have specified the *container*. The semantics of axes are emergent — concept embeddings populate the space, prediction averages over them. But whether D_REP=256 is enough, whether there should be modality-specific subspaces (vision vs text vs audio occupying different regions), and whether the projection matrix `P_q` should ever be re-seeded — these are empirical. I picked 256 because it works well on M1 and matches typical small embedding sizes, but it is not principled.

2. **Should precision be per-dimension or full-covariance, or a low-rank approximation?** I chose per-dimension because of cost. A rank-r covariance might capture correlations between dimensions that genuinely matter (e.g., "if dim 12 is high, dim 47 is usually low too — being wrong on 47 given 12 is a different kind of surprise"). This may need to be revisited if the graph evolves correlated structure that per-dim precision can't see.

3. **The exact mapping from `signed_magnitude_per_dim` to N_AFF affect dimensions is owned by Component A.** The Prediction Engine produces a D_REP-dim error vector and an N_AFF-dim affect snapshot, and hands both to A. Whether A uses a fixed projection, a slowly-learned projection, or only consumes magnitude — that is A's choice. Synthesis phase must lock this contract.

4. **What is the correct ALPHA (target surprise rate)?** I picked 5%. Cognitively, this means the system experiences a real surprise about once every twenty inputs, which feels plausible. But this number wants to be tuned by watching real behavior — too low and learning starves, too high and the graph fills with noise.

5. **Should confidence's effect on arousal be linear (as I have it) or curved?** Linear is honest; it admits we don't know. A sigmoid would feel more biological but invites parameter-tuning theater. Empirical question.

6. **Does PROCESSING-layer prediction make sense as a single call, or as one prediction per spreading-activation step?** I have treated PROCESSING as a single layer for now. But spreading activation in C may produce multiple intermediate states, each of which could in principle be predicted ahead of time. If C exposes intermediate ticks, this engine could fire mid-spread predictions. That is more biological but more expensive. Defer to Component F's attention/sparsity design.

7. **How do replayed gaps interact with running statistics?** A replayed surprise is not a new world event. Should it update Welford stats? My current design says yes — the replay is going through the engine fresh and the system is genuinely re-experiencing the gap given a (now-different) graph. But this could cause statistics to over-represent salient memories. Alternative: tag replayed observations and exclude from running stats. Component D should weigh in.

8. **What is the engine's behavior at OUTPUT layer when there is no real observation of the world's reaction?** For text output to a passive logger, there is no "world reaction" arriving back. The engine cannot compute a gap and so cannot drive the output-loop affect trigger that the spec calls for. The spec's "output trigger" gap is between *internal state* and *expression*, not between *expression* and *world response*. Resolution: at OUTPUT layer, the "observation" is the system's own generated representation re-encoded, and the "prediction" is what the system *intended to express* given its internal state. The gap is the lying-or-leaking gap. This needs to be confirmed in synthesis with G and E.

9. **Spec contradiction to flag:** the spec says "Prediction is cheap — it runs forward through the concept graph using current activation patterns and affect state" and also "the system predicts what it expects to see next in representation space." These are compatible, but the spec leaves ambiguous whether the prediction is "what I expect the *next input* to be" (forward-in-time prediction of the world) or "what I expect *this current input* to look like in representation space before it's fully encoded" (top-down perceptual prediction). My design supports both: at INPUT layer, the prediction is top-down (what does this input look like, given context?) and at PROCESSING/OUTPUT layers it is forward-in-time (what comes next?). This dual interpretation should be confirmed in synthesis.

10. **Spec underspecification:** "How the prediction engine improves over time" is asked of this component, but the spec provides no mechanism beyond "the graph it predicts from gets better." I have honored this — the engine has no learnable parameters of its own except the calibration shift. If after empirical run we find prediction quality doesn't improve fast enough, a candidate next step is adding a slow Hebbian update to `P_q` (the conditioning projection). But this would introduce a learnable parameter and should be deferred.
