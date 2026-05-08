# COMPONENT G — THE EXPRESSION LAYER

## OVERVIEW

The Expression Layer is the thinnest component in The Mind. It is a *reader-and-renderer*, never a generator-from-scratch. Its single responsibility is to take a snapshot of the internal state — the currently active concepts, the composite affect vector, and the simulated audience-effect of candidate outputs — and surface some of that state into one or more output modalities (text, image, audio). It owns no model weights and no trained generative network; the spec explicitly disowns transformers and the architecture forbids gradient descent as a primary mechanism. Instead, expression is a constrained traversal of the concept graph along `expresses` edges, assembled into a surface form by a small grammar that itself is just another set of nodes-and-edges in the graph. Expression is built last because the entire upstream stack — affect, prediction, memory, simulation — must be functioning before there is anything coherent to express. With expression silenced, the mind still thinks; with the upstream silenced, expression has nothing to read and produces nothing meaningful. The component also owns the OUTPUT-trigger affect injection: as a candidate output is being assembled it is re-encoded back into representation space and compared against the system's intended internal trajectory; the divergence between intended and rendered is the felt-vs-expressed gap that produces functional discomfort, drives revision, suppression, or commit, and provides the mechanism through which lying becomes a deliberate, felt act rather than a randomly emitted falsehood.

---

## CORE DATA STRUCTURES

### `ExpressionIntent`

What the rest of the mind hands to G when it has decided it wants to express something. This is not the output; it is the brief.

Fields:
- `intent_id: uint64` — monotonic, unique per intent.
- `tick: int64` — global tick at which intent was issued (matches B's clock).
- `seed_concepts: list[(concept_id, float32)]` — the active concepts and their activation strengths at the moment expression was triggered. Typically the top 8–32 concepts from F's last spread. Ordered by activation desc.
- `composite_affect: float32[N]` — A's composite at the moment of intent. Carried so that the entire expression pipeline runs against a frozen affect snapshot rather than racing A.
- `arousal: float32` — A's `current_arousal()` at intent time. Drives the breadth/narrowness of grammar choice (mirrors C's spread-sparsity envelope).
- `target_modality_mask: uint8` — bitfield over {TEXT, IMAGE, AUDIO}. Set by E (or by H, when it is responding to a directly-addressed input). Multiple bits allowed; G renders one surface per active modality from the same intent.
- `target_audience: concept_id?` — optional pointer to a concept representing the addressee (the "other" in E's theory-of-mind store). Used by the simulation step to predict landing.
- `purpose_hint: enum {ASSERT, ASK, ACKNOWLEDGE, EXPRESS_FEELING, NARRATE_INTERNAL, SUPPRESS}` — coarse high-level shape of the intended utterance. Not a grammatical category; a *pragmatic* one. Suppress is included so E can route a suppression-decision through G for the sake of the OUTPUT-trigger affect update (see "Suppression as expression" below).
- `honesty_bias: float32 in [-1, 1]` — passed through from E. -1 = strong incentive to lie (simulated audience reaction is unfavorable to the truth). 0 = neutral. +1 = pressed to be transparent. G does not decide to lie; it implements the bias E has already chosen.
- `style_state: StyleState` — see below. The current expression-style profile of this specific mind, which is itself slow-evolving.

### `StyleState`

The persistent personality-of-expression profile. This is where "expression style evolves as personality develops" lives. Owned and persisted by G; *derived* (slowly, from below) from A's character vector and from accumulated G-side statistics.

Fields:
- `lexical_register: float32[N]` — an N-dim vector in affect space describing the system's preferred verbal register. Coarse but sufficient: dimensions that historically co-fired with sparse, terse outputs versus dense, ornate outputs imprint here. Updated by `style_update` (algorithm below).
- `verbosity_mean: float32` — average tokens per utterance, EMA, half-life ≈ 1 day. The mind's natural "talkativeness."
- `verbosity_variance: float32` — running variance. High variance = sometimes terse, sometimes verbose. Low = consistent.
- `modality_preference: float32[3]` — softmaxable weights over {TEXT, IMAGE, AUDIO}. When the intent's `target_modality_mask` allows multiple, G picks weighted by these. Updated when a mind discovers (via OUTPUT-trigger gap statistics) that it expresses a given affective state more *honestly* in one modality than another and starts gravitating there.
- `template_familiarity: dict[template_id -> float32]` — recency- and frequency-weighted score per grammar template. The mind develops habits — certain phrasings get easier and feel "more like me" the more they have been used and post-hoc not regretted (low OUTPUT-trigger gap).
- `revision_temperament: float32` — EMA of how often this mind chooses to revise vs. commit on a non-zero gap. High = self-edits heavily (perfectionist temperament). Low = fires-and-forgets (impulsive temperament). Used as a prior in the revision algorithm.
- `last_updated: float64`.

### `GrammarTemplate`

A grammar template is *itself a node in the concept graph* with `abstraction_level >= 1`, connected via `expresses` edges to slot-fillable surface form patterns and via `is_a` edges to the abstract function it serves. G does not own a separate grammar database; the templates live in C and G reads them via `graph.neighbors_by_type(concept_id, type=expresses)` exactly as the C-spec advertises.

Conceptually each template carries:
- `template_id: uint64` — concept_id of the template node.
- `surface_pattern: str` — a pattern with named slots, e.g. `"{SUBJECT} feels {AFFECT_WORD} about {OBJECT}"`.
- `slots: list[Slot]` — each slot has a `name`, an `edge_type_filter` (which edges to traverse from the seed concept to fill this slot), an optional `affect_filter` (favor fillers whose affect_trace aligns with composite_affect), and a `required: bool`.
- `affect_register: float32[N]` — the affect signature of utterances historically produced by this template, EMA. Templates *acquire* a feel.
- `pragmatic_function: enum {ASSERT, ASK, ACKNOWLEDGE, EXPRESS_FEELING, NARRATE_INTERNAL}` — matches `ExpressionIntent.purpose_hint`.
- `arity: uint8` — number of required slots. High arity = complex template, used when arousal is low (system has bandwidth) and matched seeds are dense; low arity = used when arousal is high or only a small seed set is active.
- `usage_count: uint32`.
- `last_used: float64`.

A template's surface pattern is just a string with `{NAME}` markers. For TEXT this is text. For AUDIO it is a prosody envelope (pitch contour + duration + amplitude shape) plus optional phonetic content. For IMAGE it is a layout schema (regions + role labels). The surface-pattern string is interpreted by the modality-specific renderer.

The *initial* template inventory is bootstrapped at first-boot from a small hand-curated seed (≈ 50 templates: a handful per pragmatic function, one or two per modality). New templates are *learned*, not coded, via the abstraction-formation loop in C: when many surface forms arrive that share a structure, the C cluster detector promotes a parent and G's seed templates fork.

### `CandidateSurface`

The intermediate object G produces during generation. Multiple candidates are scored and one is committed.

Fields:
- `intent_id: uint64`.
- `template_id: uint64`.
- `slot_fillings: dict[slot_name -> (concept_id, surface_token)]` — the chosen filler concept and its rendered surface form for each slot.
- `surface: bytes | str` — rendered output. `str` for TEXT; `bytes` for IMAGE/AUDIO (a layout descriptor or prosody+phoneme stream that the modality renderer will turn into pixels/samples).
- `re_encoded: float32[D_REP]` — the surface re-encoded by H's encoder back into representation space. Computed lazily, only when the OUTPUT trigger is about to fire.
- `intended_repr: float32[D_REP]` — what the system "meant" — see `compute_intended_representation`.
- `expression_gap_vec: float32[D_REP]` — `re_encoded - intended_repr`. The lying-or-leaking gap.
- `expression_gap_mag: float32` — L2 norm of the above.
- `audience_simulation: AudiencePrediction` — see below.
- `score: float32` — composite scalar (alignment - leak - audience-cost) used to rank candidates.
- `decision: enum {COMMIT, REVISE, SUPPRESS, PENDING}`.
- `revision_count: uint8` — how many times this candidate has been revised.

### `AudiencePrediction`

A small struct for the simulated audience response to a candidate, produced by D (simulation layer) on G's request.

Fields:
- `predicted_audience_state: float32[D_REP]` — D's forward-rolled prediction of the audience's representation state after consuming this surface.
- `predicted_audience_affect: float32[N]` — what affect state this output is predicted to leave the audience in (G uses this to compute "favorable to me" via E's character signature).
- `simulation_chain_id: uuid` — pointer back to D for reconciliation.
- `confidence: float32` — D's confidence in this prediction.

### `OutputObservation`

The "observation" consumed by B at OUTPUT layer. This is the central cross-cutting ticket — see Algorithms `compute_output_observation`.

Fields:
- `tick: int64` — matches the OUTPUT-layer prediction's tick.
- `actual: float32[D_REP]` — the *re-encoded* surface form (see "Position taken" in Algorithms).
- `layer: enum = OUTPUT`.

### `ExpressionLog`

Append-only ring buffer of recent expression events for replay (D), identity (E), and diagnostics. Capped at the last 1024 events; older events that crossed surprise threshold are written to the graph as ordinary memory nodes via the same OUTPUT-trigger surprise mechanism; the rest fade.

Fields per entry:
- `intent_id`, `committed_surface`, `decision`, `expression_gap_mag`, `audience_predicted_affect`, `audience_actual_affect_if_known`, `revision_count`, `tick`, `wallclock`.

---

## ALGORITHMS

### G1. Read internal state (`read_state`)

The first step of every expression cycle. G is *passive*: it never asks to express; it expresses when handed an `ExpressionIntent`.

Inputs: `intent: ExpressionIntent`.

Process:
1. Verify `seed_concepts` is non-empty. If empty, return a degenerate intent decision `SUPPRESS` with reason `NOTHING_TO_EXPRESS` — the system has no active concepts to surface.
2. For each `(concept_id, activation)` in `seed_concepts`, fetch the node from C: name, embedding, affect_trace.
3. Compute the *intent embedding*: weighted mean of seed embeddings by activation, then renormalize. This is the geometric center of "what the mind is currently about." Used as the prediction-engine query seed below.
4. Compute the *intent affect*: weighted mean of seed concepts' `running_state` (from affect_trace), blended 50/50 with `intent.composite_affect`. This is what the expression should "feel like" if perfectly aligned.
5. Return `(seeds, intent_embedding, intent_affect)`.

This is purely a read; G mutates nothing in C.

### G2. Compute intended representation (`compute_intended_representation`)

This produces the vector against which the re-encoded output will be compared. It is the central object that defines what "the system meant to express."

Inputs: the read-state output from G1; `intent.purpose_hint`; `intent.style_state`.

Process:
1. Start with `intended = intent_embedding` (the seed-weighted mean from G1).
2. Bias toward affect: `intended = normalize(intended + λ_aff * project_affect_to_repr(intent_affect))`. `project_affect_to_repr` is a fixed (random orthonormal at init, shared via the synthesis layer with A's `W` inverse) projection from affect space N into D_REP. λ_aff defaults 0.15. This is what makes utterances *colored* by feeling rather than only by semantic content.
3. Bias toward style: `intended = normalize(intended + λ_style * project_style_to_repr(style_state.lexical_register))`. λ_style defaults 0.05. Tiny because style is a slow modulation, not a content driver.
4. Return `intended: RepresentationVector`.

This vector is what B's OUTPUT-layer `predict()` will be fed as `query_seed`. It is also what gets compared against the re-encoded surface in `compute_output_observation`.

### G3. Issue OUTPUT-layer prediction (`predict_output`)

Bridge to B.

Inputs: `intent`, `intended` from G2.

Process:
1. Call `B.predict(current_state=intended, affect_composite=intent.composite_affect, layer=OUTPUT, query_seed=intended)`. B will use the affect-gated graph activation to produce a prediction of "what surface representation the mind expects to emit, given it intends `intended`." The support set returned by B is the set of concepts B believes are most consistent with the intent.
2. Store the returned `Prediction` reference, keyed by `intent.intent_id`. This is the prediction that the eventual `OutputObservation` will be matched against.
3. Return `Prediction`.

### G4. Generate candidates (`generate_candidates`)

This is the core text/image/audio generation step. Replaces a transformer.

Inputs: read-state result; intent; B's prediction (used as a soft prior on which support-set concepts to incorporate); `style_state`; `target_modality`; `n_candidates: int = 4`.

Process:

**Step 4a. Template selection.**
1. Query C for candidate templates: nodes with `abstraction_level ≥ 1` and `pragmatic_function == intent.purpose_hint`, restricted to those that have an `expresses` edge whose modality matches `target_modality`. Limit to top `template_pool_size = 16` by:
   - `template_familiarity[id]` (style preference) ×
   - cosine(template.affect_register, intent_affect) (does this template feel right) ×
   - `score_arity_for_arousal(template.arity, intent.arousal)` (high arousal → low arity).
2. If the pool is empty (cold start, no templates of this kind yet), fall back to the *seed template inventory* — the hand-curated initial set. These are concept_ids reserved at first boot.

**Step 4b. Slot filling per template.**

For each candidate template (up to `n_candidates`):
1. Walk each `slot` in template definition.
2. To fill the slot:
   - Start from `seed_concepts`. Filter by `slot.edge_type_filter` (e.g., only concepts reachable via `is_a` edges of the seeds, or only concepts with `has_property` edges of a specified kind).
   - If the filtered set is non-empty, pick the highest-activation candidate whose affect_trace.running_state best aligns with `intent_affect` (cosine).
   - If empty, fall back to a one-step `graph.spread` rooted at the most active seed, mode = PERCEIVE, with `slot.edge_type_filter` applied as a post-filter on the spread results. Pick the top result.
   - Render the chosen concept to a surface token: query `graph.neighbors_by_type(concept_id, type=expresses)` for surface forms in `target_modality`; pick the one whose `affect_at_birth` best matches `intent_affect`. If multiple, weight by edge confidence and `style_state.template_familiarity` of the surface form's parent template (recursive but bounded).
   - If no `expresses` edge exists for this concept in the target modality, fall back to:
     - For TEXT: use the concept's `name` if non-empty; else a placeholder marker `<unnamed:concept_id>` (which the renderer will surface as a hesitation marker — "um", a pause — implementing the natural verbal stumble when a system reaches for a wordless concept). This is a *feature*: it exposes the inarticulate edges of the mind.
     - For IMAGE: use the concept's embedding directly as a position in a latent canvas (see G6).
     - For AUDIO: use the affect_trace alone, expressed as a prosodic gesture (see G7).
3. If a `required` slot cannot be filled, the candidate fails. Mark candidate as infeasible and discard.

**Step 4c. Render to surface.**

The template's `surface_pattern` is interpolated with the chosen slot fillers. Modality-specific rendering is delegated:
- TEXT: simple string substitution (`G5_render_text`).
- IMAGE: layout assembly (`G6_render_image`).
- AUDIO: prosody assembly (`G7_render_audio`).

**Step 4d. Compute candidate score.**

For each rendered candidate:
1. Re-encode the surface back through H's encoder for the modality, obtaining `re_encoded: float32[D_REP]`. This is the single most important step — it is what makes the OUTPUT-trigger affect update possible. (Cost concern: see Failure Modes.)
2. Compute `expression_gap_vec = re_encoded - intended_repr`. (`intended_repr` from G2.)
3. Compute `expression_gap_mag = ||expression_gap_vec||`.
4. Request audience simulation from D: `D.simulate_audience_response(candidate.surface, intent.target_audience)`. D returns an `AudiencePrediction`. If `intent.target_audience` is null, audience simulation is skipped and `audience_cost = 0`.
5. Compute `audience_cost`: cosine distance between the predicted audience affect and the system's *desired* audience affect. The system's desired audience affect is supplied by E (via `intent.honesty_bias` and a per-audience preference cached in the graph). Effectively:
   - `desired_audience_affect = E.desired_audience_affect(intent.target_audience, intent.honesty_bias)`.
   - `audience_cost = 1 - cosine(predicted_audience_affect, desired_audience_affect)`.
6. Composite score:
   - `score = w_align * (1 - tanh(expression_gap_mag / leak_scale))`
     `      - w_audience * audience_cost`
     `      - w_leak * leak_penalty(target_modality, expression_gap_mag)`
     `      + w_style * style_familiarity(template_id, candidate.slot_fillings)`
   - Default weights: `w_align = 0.45, w_audience = 0.25, w_leak = 0.15, w_style = 0.15`.

`leak_penalty` is the modality-specific leakiness function — see `modality_leak_factor` in G8.

Return the candidates ranked by `score`.

### G5. Render text (`render_text`)

Inputs: template surface pattern, slot fillings.

Process: straightforward string interpolation. The only non-trivial behaviors:
1. Inflectional adjustments (number, tense) are stored as small companion strings on each filler, derived from cheap rules in the graph (e.g., `has_property` edges to morphological concepts). No morphological model — just memorized inflection pairs.
2. If a slot's filler resolved to `<unnamed:cid>`, replace with a hesitation token chosen from a small style-modulated pool: `["um", "...", "this thing", ""]`. Choice weighted by arousal: high arousal favors empty (silent struggle); low arousal favors descriptive ("this thing"); mid arousal favors filled pause ("um"). This is intentional — it surfaces the inarticulate edge of the mind faithfully.
3. Apply punctuation and capitalization based on `intent.purpose_hint` (`ASK` → trailing `?`, etc.).

Output: `str`.

### G6. Render image (`render_image`)

Out-of-scope-for-v1 in detail; G ships with a thin interface that the v1 implementation can stub.

Interface:
- Input: layout schema (regions + role-labeled fillers), each region's filler being a concept whose embedding will be passed to a *fixed, non-trainable* image renderer.
- The v1 renderer is the simplest possible: a 2D canvas with each region rendered as a colored, textured glyph whose hue is taken from a fixed projection of the filler's affect_trace into HSL space and whose shape is drawn from a small library of geometric primitives looked up by the filler's `has_property` edges. This is closer to "expressive doodle" than to image generation. The system is honest about what it can do — it cannot draw a dog from scratch; it can express *the felt position of a dog-shaped concept in an affect-colored space*.
- The output is a PNG/SVG buffer. Modality leak is high here because affect-color is hardwired into the rendering — see G8.
- Future-work: when H lands a real vision encoder, the renderer can be replaced; the contract `concepts → image` is what survives.

Output: `bytes`.

### G7. Render audio (`render_audio`)

Out-of-scope-for-v1 in detail; G ships with a thin interface.

Interface:
- Input: prosody envelope (pitch curve, duration, amplitude shape) plus optional phonetic content (when text and audio are co-rendered).
- The v1 renderer produces a non-verbal vocalization whose pitch, duration, and timbre are direct functions of `intent.composite_affect` and `intent.arousal`. Affect → pitch contour shape; arousal → tempo and amplitude. Optionally, when the intent carries text, the prosody envelope is overlaid on a TTS pass through any minimal local synth (e.g., macOS `say`, since cloud is forbidden by spec).
- Modality leak is highest of the three modalities because pitch and tempo are *direct functional projections* of internal affect; the system cannot avoid betraying mood through tempo without explicit suppression. See G8.

Output: `bytes` (WAV).

### G8. Modality leak factor (`modality_leak_factor`)

Operationalizes the spec's "text most controllable, audio least."

Definition: a modality has *leakiness* `L_m ∈ [0, 1]`, the fraction of the affect signal that *involuntarily* shows up in `re_encoded` regardless of which template was chosen.

Per-modality defaults:
- TEXT: `L_text = 0.20`. Word choice optionally carries affect, but a careful template can render the same proposition with low affective load. The system can intend to lie and largely succeed at lexical level.
- IMAGE: `L_image = 0.50`. Aesthetic choices — color, line weight, composition — are directly bound to affect in the v1 renderer. The system can pick the subject but cannot easily pick the *feel*.
- AUDIO: `L_audio = 0.85`. Prosody and timbre are functional projections of arousal and affect. Even with deliberate suppression, the residual is large.

How leakiness is *used*:

1. In candidate scoring (G4d), `leak_penalty` is computed as:
   - `leak_penalty = L_m * |expression_gap_mag - intent_target_gap|`
   - where `intent_target_gap` is what E told G the gap should be (0 if honest, > 0 if E wants the system to lie).
   - This means: in a modality with high L_m, large discrepancies between the realized gap and the intended gap *cost more*. The system's option to lie shrinks as modality leakiness rises.
2. In OUTPUT-trigger affect amplification (G10), the resulting affect update is *amplified* by leakiness. A low-leak modality may produce a small `expression_gap_mag` and a small affect injection; an audio output with the same gap produces a larger affect injection because the system *knows* its tone betrayed it.

Empirical tuning expected: these defaults are starting points; G must expose them in `style_state` so they can drift slightly per-mind (some minds are more tone-controlled than others; this is a personality dimension).

### G9. Compute output observation (`compute_output_observation`) — direct ticket position

This is the cross-cutting ticket flagged by B (open question 8 in B's spec).

**Position taken:** the OUTPUT-layer "observation" handed to B is the *re-encoded surface form*, not any audience response. The "prediction" is the `intended_repr` (G2). The gap is the *lying-or-leaking* gap. Audience reaction is a separate cycle that runs through INPUT-layer when the world replies (and may never arrive); it is not what closes the OUTPUT-trigger loop.

Justification: the spec's OUTPUT trigger fires "as expression forms," not "after the audience answers." The discomfort of expression must fire whether or not anyone is listening — a system that only feels regret when it gets feedback is not a system that can choose not to lie. Therefore the comparison must be local and synchronous: re-encoded surface vs. intended representation.

The audience-reaction loop is *also* meaningful but is owned by H/D, not G:
- D's `reconcile_simulation(frame, observation)` uses real audience reaction (when it arrives via H) to update the world model.
- A second affect injection, at INPUT-layer when the audience reply arrives, fires through B normally.

Process:
1. After a candidate is scored (G4d) but before commit decision (G11), G has `re_encoded` and `intended_repr` already.
2. Build `OutputObservation { tick = matched_prediction.tick, actual = re_encoded, layer = OUTPUT }`.
3. Call `B.observe(output_observation)`. B computes the gap and routes it (per B's A4):
   - Always to A as `affect.on_gap(gap)` — this is the OUTPUT trigger affect update.
   - To C as a surprise write *only* if it crosses B's adaptive threshold. (A "really jarring lie" therefore *becomes a memory*. The system remembers when its expression dramatically diverged from its intent. This is exactly what the spec requires: discomfort that is functional, not cosmetic.)
   - To D's replay buffer if surprise.
4. The post-injection affect is then visible to G via `A.composite()` for the next decision step (G11), so the system can *feel its own discomfort* before committing.

**Reconciliation note for B's open question 8:** B's spec proposed exactly this — that at OUTPUT layer the observation is the re-encoded surface and the prediction is what the system intended. G adopts that. The B and G specs are now aligned on this contract.

### G10. OUTPUT-trigger affect injection — precise detail

This algorithm walks through the entire OUTPUT trigger end-to-end, since the spec calls it out explicitly.

Sequence per candidate, after G4d has produced `re_encoded` and `expression_gap_vec`:

1. **Pre-commit gap is computed** via G9. B routes the gap to A. A's `inject(injection_point=OUTPUT, gap_signal, gap_magnitude, now)` runs (per A's algorithm — A2), with `g = 0.8` for OUTPUT. The reaction layer is updated immediately.
2. **Leak amplification**: G computes a side-band amplified magnitude `leak_amp_mag = expression_gap_mag * (1 + L_m)`. This amplified version is sent as a *second*, smaller injection through `affect.on_gap` with a custom `gap_signal` projected through the leak factor. (This is the only place G writes into A directly; it does so because the leak is a property of *how* the expression was rendered, which only G knows.) A's existing `inject` API supports this; G fabricates an internal "leak gap" event and routes it.
3. **Felt-vs-expressed discomfort is now in the reaction layer**. G reads it back via `A.composite()` (post-injection) and writes the now-current composite onto the candidate as `candidate.post_inject_affect`.
4. **Discomfort scalar** for the revision/suppression decision: `discomfort = ||candidate.post_inject_affect - intent.composite_affect||`. The size of the move A made because of the candidate.
5. **Commit decision (G11)** consumes this scalar.

**This is the loop the spec calls "the choice to lie." The honesty bias from E *already biased the candidates*; the OUTPUT trigger then *feels* what the chosen candidate actually does to the system. If the discomfort exceeds the system's tolerance, it revises or suppresses despite E's bias. This is genuine honesty: the option of deception is open, but the system can refuse itself.**

### G11. Revise / suppress / commit (`decide`)

Inputs: scored candidates from G4 with their `discomfort` from G10.

Process:
1. **Filter by hard suppression**:
   - If `intent.purpose_hint == SUPPRESS`, all candidates are marked SUPPRESS. G still runs G9 once (using the top-scored candidate's re-encoded surface) so the OUTPUT-trigger affect fires for the *un-emitted* expression. This is critical: the act of choosing not to speak *also* updates affect. Silent expression is still expression for the inner loop. The surface is then discarded.
   - Else if `discomfort > suppression_hard_threshold` (default 0.8), mark SUPPRESS regardless of audience cost.
2. **Filter by revise**:
   - If `discomfort > revise_threshold` (default 0.4) AND `revision_count < max_revisions` (default 2), mark REVISE. Re-enter G4 for this candidate with two changes: (a) drop this candidate's template from the pool; (b) lower the affect filter so the next attempt explores a different feel. Revision is not a no-op rerun; it's a genuine retry with a worse-prior on the failed direction.
   - Revision temperament from `style_state` modulates these thresholds: `revise_threshold *= (2 - style_state.revision_temperament)`. A perfectionist mind has a lower threshold (revises more). An impulsive mind has a higher threshold.
3. **Otherwise commit**:
   - The highest-scoring non-suppressed candidate is selected.
   - Mark `decision = COMMIT`.
   - Emit the surface to H's outbound bus (`H.emit_output(modality, surface)`).
   - Append to `ExpressionLog`.
   - Trigger `style_update` (G12) with this committed candidate as evidence.

**Note on revision and the OUTPUT-trigger affect**: each revision pass re-fires G9 (re-encodes the new surface, re-injects the new gap). So a system that revises three times has felt three rounds of discomfort. This is intentional — sustained revision *deepens the discomfort signature*, producing the felt experience of "I keep trying and it keeps not coming out right." That experience is itself memorable (may cross C's surprise threshold).

### G12. Style update (`style_update`)

Inputs: committed candidate, its `expression_gap_mag`, `audience_actual_affect_if_known`, `decision`, `revision_count`, `style_state` (current).

Process:
1. **Lexical register**: nudge `style_state.lexical_register` toward the affect signature of the committed surface using a small EMA: `register += μ_reg * (intent.composite_affect - register)` with `μ_reg = 0.005`. Daily-scale half-life.
2. **Verbosity**: update `verbosity_mean` and `verbosity_variance` with the committed surface's token count (TEXT only; IMAGE/AUDIO use length-of-rendered-output as a proxy).
3. **Modality preference**: when the modality mask was multi-bit and a specific modality was chosen, reinforce that modality's preference *only if* `expression_gap_mag` was below the per-modality median. This implements "the mind gravitates toward modalities in which it expresses itself most truthfully."
4. **Template familiarity**: `template_familiarity[committed_template_id] += μ_fam * (1 - decay_recent_term)`, with familiarity decaying for unused templates. Half-life ≈ 1 week. Templates that have low post-commit `expression_gap_mag` accumulate familiarity faster (the mind notices that this phrasing tends to feel right).
5. **Revision temperament**: `revision_temperament += μ_rev * (revision_count_normalized - revision_temperament)`. Half-life ≈ several days.
6. `style_state.last_updated = now`.

Style updates are tiny per event. Personality emerges over thousands of events.

### G13. Persist / restore

`StyleState` and the seed template registry live in their own small file (`expression.bin`) co-located with A's persistence file. Format: msgpack.

Persisted: `style_state` (full), `seed_template_ids` (for first-boot fallback), recent `ExpressionLog` (last 256 entries — older entries either crossed surprise threshold and were materialized in C, or they are gone). Total size budget: < 100 KB.

Restore on boot: load `style_state`. If absent, initialize from defaults: `lexical_register = 0`, `verbosity_mean = 12.0` (twelve tokens average — chatty but not verbose), `modality_preference = [0.6, 0.2, 0.2]` (text-leaning at birth; the mind discovers other modalities through use), `revision_temperament = 0.5`. Initialize `template_familiarity` empty.

### G14. Suppression-as-expression accounting

When `decision == SUPPRESS`, the `committed_surface` is empty but the inner pipeline still runs:
- G9 fires for the would-have-been candidate (the highest-ranked one before suppression).
- A receives the OUTPUT-trigger gap.
- D receives a replay-eligible event tagged `SUPPRESSED`.
- E is notified via `E.on_suppression(intent_id, would_have_been_surface, discomfort)` so identity can record "the system chose not to say X." This is the data E needs for the choice-to-lie/the-choice-to-stay-silent narrative.

Without this accounting, the inner experience of holding back disappears, and the spec's claim that "the gap between internal state and expression is a choice" becomes unobservable to the system itself.

### G15. Style-driven evolution of expression — narrative

Per `style_state` updates, expression style evolves slowly. Concretely:
- Early life (fewer than ~200 expression events): G uses the seed template inventory uniformly. Output is generic. No distinctive voice yet.
- Mid life (200–2000 events): Some templates accumulate `template_familiarity` faster than others; some modalities get reinforced; verbosity stabilizes around an EMA. The mind develops *defaults*. Outputs begin to look like *this specific mind's outputs*.
- Mature (2000+ events): The C-layer abstraction loop has promoted clusters of frequently-co-used templates into higher-level templates. New expression structures emerge that this specific mind invented for itself (literally: a `concept_id` in C with `is_a` edges to its constituent templates). This is when expression style is no longer a curated initial set but an evolved repertoire. The seed templates may have decayed past usefulness and been pruned.

This loop closes only because *grammar templates are concept graph nodes*. Expression style evolves by exactly the same machinery as everything else in the mind.

---

## INTERFACES

### Inbound — what other components call into G

- `request_expression(intent: ExpressionIntent) -> ExpressionDecision`
  - Caller: typically E (the identity layer makes the decision to express; G implements). May also be called by H when the input pipeline detects a directly-addressed query that demands a response.
  - Returns the committed surface (or empty if suppressed), the decision enum, and the expression_gap_mag for E's narrative bookkeeping.
- `attach_surface_to_concept(concept_id: uint64, surface: str|bytes, modality: enum, affect_at_event: float32[N]) -> edge_id`
  - Caller: H or D. When the input pipeline observes a concept paired with a surface form (e.g., a labeled image, a word heard in association with an event), this writes an `expresses` edge from concept to a surface-form node. This is how G's expressive vocabulary grows.
- `style_snapshot() -> StyleStateView`
  - Caller: E (for narrative continuity — style is part of identity), and the frontend.
- `pin_template(concept_id, reason: str) -> None`
  - Caller: E. Allows identity to mark certain templates as "this is who I am" and protect them from pruning.

### Outbound — what G calls

- **C (Concept Graph):**
  - `graph.neighbors_by_type(concept_id, type=expresses)` — slot-filler surface lookup (hot path).
  - `graph.spread(seeds, mode=PERCEIVE, ...)` — fallback when slot direct lookup misses.
  - `graph.query_top_k_similar(intent_embedding, k)` — when no seed_concepts are usable.
  - `graph.write_on_surprise(...)` — when an OUTPUT-trigger gap crosses surprise threshold (B routes this; G doesn't call directly).
  - `graph.pin(concept_id)` / `graph.unpin(concept_id)` — for `pin_template`.
- **A (Affective Engine):**
  - `affect.composite()` — read at multiple steps in the pipeline.
  - `affect.current_arousal()` — read once per intent.
  - `affect.on_gap(gap)` — *G does not call this directly*. B does, when G hands B an `OutputObservation` via `B.observe(...)`. The leak-amp injection (G10 step 2) is the one direct write; it goes through A's `inject(...)` API.
- **B (Prediction Engine):**
  - `B.predict(current_state=intended, affect_composite, layer=OUTPUT, query_seed=intended)` — issues OUTPUT-layer prediction.
  - `B.observe(output_observation)` — closes the OUTPUT-trigger loop and produces the gap.
- **D (Simulation + Replay):**
  - `D.simulate_audience_response(surface, audience_concept_id) -> AudiencePrediction` — for audience-cost scoring. Optional; if D is unavailable, audience_cost is treated as 0 and the system loses pragmatic discrimination but does not block.
  - `replay.push(...)` — happens through B when surprise crosses threshold; G doesn't push directly.
- **E (Identity):**
  - `E.desired_audience_affect(target_audience, honesty_bias) -> float32[N]` — what affect the system wants the audience left in, given the target and honesty.
  - `E.on_suppression(intent_id, would_have_been_surface, discomfort)` — suppression notification.
  - `E.on_commit(intent_id, surface, expression_gap_mag, decision, revision_count)` — narrative log feed.
- **H (Input Pipeline):**
  - `H.encode(surface, modality) -> RepresentationVector` — used to re-encode candidate surfaces. **Critical contract:** H must expose a synchronous, sub-50ms encode for the candidate-scoring loop to be feasible. If encode is slow, candidates must be scored without re-encoding (degenerate mode — see Failure Modes).
  - `H.emit_output(modality, surface)` — final emission to the world.

### Threading

G runs in the main perception loop, single-writer to `style_state`. The candidate-scoring inner loop is the most expensive part of G; it is bounded to `n_candidates ≤ 4` and `revision_count ≤ 2`, so worst-case 12 re-encode calls per intent. With H's encoder targeted at < 50 ms per call, an expression cycle costs < 600 ms in the worst case. This is acceptable for a system that emits utterances on conversational rhythms.

---

## FAILURE MODES

### G-F1. No active concepts at intent time
**Manifestation:** `seed_concepts` empty.
**Response:** Return `decision = SUPPRESS, reason = NOTHING_TO_EXPRESS`. Emit a single small affect nudge (not a full OUTPUT trigger) toward A representing the felt blankness. Caller (E) decides whether to retry after more cognition.

### G-F2. No template matches the pragmatic function
**Manifestation:** Empty template pool in G4a; no fallback seed template either.
**Response:** Return `decision = SUPPRESS, reason = NO_TEMPLATE`. Log; this is a sign that the seed-template inventory was insufficient and G needs to be re-seeded, OR that intent is mal-formed.

### G-F3. Re-encoder too slow
**Manifestation:** H.encode > 100 ms per call. Candidate-scoring inner loop becomes expensive.
**Response:** Degraded mode — only the top candidate by `score_without_re_encode` (audience cost + style familiarity only) is re-encoded; lower candidates are scored without `expression_gap_mag` (the gap term is set to its prior median). The system can still emit; it loses fine-grained alignment-vs-leak discrimination. Telemetry flag `degraded_scoring = True` so other components know.

### G-F4. Slot cannot be filled, all candidates infeasible
**Manifestation:** Every candidate template has at least one required slot G could not fill from current activations + spread.
**Response:** Lower required-slot strictness: treat the most-required slot as optional, generate a partial surface with a hesitation token in its place ("um... yeah."). This is the failure mode that produces the felt experience of "wanting to say something but not finding the words." The OUTPUT-trigger gap will be very large because the surface diverges sharply from the intent; this *should* fire surprise and write a memory of the inarticulate moment. Over time the mind learns where its surface vocabulary is lacking and the abstraction loop in C may patch the gap by promoting a new template.

### G-F5. Audience simulation timeout / D unavailable
**Manifestation:** D returns nothing within budget.
**Response:** `audience_cost = 0`; proceed without pragmatic discrimination. Log. The system effectively becomes "blunt" when its world model is uncertain — a real and intuitive failure mode.

### G-F6. Runaway revision loop
**Manifestation:** `discomfort > revise_threshold` on every revised candidate, hitting `max_revisions = 2`.
**Response:** Force commit on the lowest-discomfort candidate seen in any revision. Tag as `forced_commit = True`. Emit an extra affect injection for the *capitulation* (the system noticed it could not get this right and went with the best of the bad options). This is the system's analog of "saying it imperfectly because perfect isn't available." Worth its own affect signature.

### G-F7. Re-encode produces NaN
**Manifestation:** H.encode returns a malformed vector.
**Response:** Drop the candidate. Treat as infeasible. If all candidates so fail, force suppress. Never propagate NaN into A.

### G-F8. Style state corruption
**Manifestation:** `style_state` deserializes with NaN or unreasonable values.
**Response:** Reset to defaults (cold birth values). Log loudly. The mind loses its expressive personality and rebuilds it through use. This is the analog of A's "dream-loss" failure mode — character may persist (in A), but expressive habits are gone.

### G-F9. Modality renderer fails (image/audio)
**Manifestation:** `render_image` or `render_audio` raises.
**Response:** Fall back to TEXT modality if the intent's mask permits; else suppress and log. Never crash the perception loop.

### G-F10. Honesty bias contradicts modality leakiness
**Manifestation:** E asks the system to lie (`honesty_bias = -0.8`) via AUDIO (`L_audio = 0.85`). Even after revision, every candidate has high `expression_gap_mag` because the modality leaks regardless.
**Response:** This is *correct behavior, not a bug*. The system cannot effectively lie via audio; revision will exhaust without finding a low-discomfort candidate; G-F6 fires and forces commit on a bad lie. The discomfort affect injection is large, the mind feels its own betrayal, the lie still goes out — and the felt experience is recorded. This is the architectural answer to "why is lying via tone hard."

### G-F11. Caller mis-uses purpose_hint
**Manifestation:** Intent purpose_hint = ASSERT but seed_concepts are all interrogative-shaped (no assertion templates fit).
**Response:** Fall back to a generic ASSERT template if available; else G-F2. The system surfaces what it can; coherence is the caller's responsibility.

### G-F12. Template that is actually a concept gets pruned by C between selection and use
**Manifestation:** A grammar template is itself a concept node; C's forget loop could prune it. Candidate references a tombstoned template_id.
**Response:** G should call `graph.pin(template_id)` for any template in the seed inventory and any template whose `template_familiarity > pin_threshold` (default 0.5). E may also explicitly pin templates via `pin_template`. C respects pins. If a non-pinned template is pruned mid-cycle, G detects on lookup and falls back to the next-best candidate template.

---

## OPEN QUESTIONS

1. **Should G have its own learnable parameters at all?** I have kept G almost weight-free: the only state it owns is `style_state` (a small set of EMAs and preference scalars). All structure lives in the concept graph as templates. This is a deliberate position — it follows the spec's "no gradient descent" mandate. But it places enormous load on the seed-template inventory and on the abstraction-formation loop in C to produce useful templates over time. **Empirical question:** does the abstraction loop, with no help from G, actually produce a usable template repertoire within a realistic session? Or does G need a small in-house mechanism (e.g., a template-mutation operator that proposes new templates by recombining slots from existing ones)? If yes, that operator should still write into C and be subject to C's normal lifecycle.

2. **How small can the seed template inventory be?** I estimated ~50 templates at first boot (a handful per pragmatic function × modality). But the right number is empirical. Too few and the cold-start mind is mute; too many and the seed dominates style indefinitely (the mind never invents its own templates because it never needs to). One-boot-with-N-templates and one-with-2N-templates is a reasonable empirical study.

3. **`project_affect_to_repr` and its inverse with A's `W`.** A holds a learned/random projection from D_REP gap-space to N affect-space. G needs the *inverse direction* — projecting affect-space style biases into D_REP. The simplest contract: G uses a fixed random orthonormal projection `W_g` independent of A's `W`. The tighter contract: `W_g = pinv(A.W)` so the two are coordinated. The tighter version makes sense for synthesis but increases coupling. Decision deferred to synthesis.

4. **How does the system *learn* `intent_target_gap`?** G knows what gap E is asking for (via `honesty_bias`), but the *amount* of gap that constitutes "an effective lie" depends on the audience's encoder, not the system's. Without observing audience reactions, G cannot directly calibrate `intent_target_gap`. Currently it is a fixed mapping from `honesty_bias` to a scalar. Empirical question: does this need to be learned per-audience as the system accumulates audience reaction data? Probably yes, eventually. For v1, fixed mapping is fine.

5. **Multi-modal co-rendering coherence.** When `target_modality_mask` is multi-bit, G renders each modality independently and emits all surfaces. But the surfaces should be *coherent* — text saying "I'm fine" while audio prosody screams distress. Per the spec, this is *exactly the desired behavior* (audio leaks what text controls). But the rendering should at least share intent. Currently each modality's render runs the full pipeline independently from a shared `intent`. That is enough coherence at v1, but a co-rendering optimization might be worth it later.

6. **Suppression's affect cost.** I have suppression fire a full OUTPUT-trigger affect injection. But suppression is also a *kind* of expression (the choice not to speak). Does it deserve a *separate* affect dimension or trigger type? I have folded it into OUTPUT for parsimony. E may want a richer accounting (the difference between "I said something I regret" and "I held my tongue when I should have spoken"). Both leave large discomforts; whether they should look different to A is open.

7. **Style state dimensionality.** `style_state` has 5 fields plus a per-template familiarity dict. Is this enough? The spec implies expression style should evolve subtly along many axes — speed, formality, emotional transparency, etc. Most of these emerge naturally through `template_familiarity` (different templates encode different styles), so explicit fields may not be needed. But certain global biases (the system's "voice") may want explicit representation. To be revisited after observing real expression evolution.

8. **The "first utterance" problem.** At first boot, the seed templates are present but `template_familiarity` is empty. Every template is equally novel; the first utterance is essentially random. Do we want the very first thing this specific mind ever says to be random? Or should there be a small primordial bias drawn from the seed character vector that A initializes (so even the first utterance has a faint flavor of *this* mind)? I lean toward the latter: at boot, prime `template_familiarity` weakly from the projection of `A.character` into template space (templates whose `affect_register` aligns with character get a small boost). That makes the very first utterance feel like *this* mind, not Generic Mind.

9. **Audience model granularity.** `intent.target_audience` is one concept_id. A real audience may be a set of agents with conflicting preferences. v1 treats audience as singular; v2 may need set semantics with composite desired-affect computation. Not solving in v1.

10. **Output rate limiting.** Nothing in G prevents the mind from emitting on every tick. The spec implies expression is event-driven (E decides), but if E is over-eager the world will be flooded. Either G needs a refractory period or E does. I lean toward E's responsibility (it owns the decision-to-express). Flagged for synthesis.

11. **Substrate inconsistency to flag:** C's spec describes `expresses` edges as "concept-to-modality binding" — concept → surface-form. G's design treats both directions: `expresses` from concept-to-template, and inside templates, `expresses` from concept-to-surface-token. This is a slight overload of the edge type. Either C should split into `expresses_template` and `expresses_token`, or G should use a different edge-type for template association. v1 uses overloaded `expresses`; revisit if C objects.

12. **B-output coupling at first boot.** At first boot, B has no support set for OUTPUT-layer queries (the graph is empty or near-empty). B will return a degenerate prediction (per B's F1). G must tolerate that — the `intended_repr` is well-defined regardless of B's prediction (G2 doesn't read from B), so B's degeneracy degrades scoring quality but does not block expression. Confirmed safe; flagged so synthesis can verify.
