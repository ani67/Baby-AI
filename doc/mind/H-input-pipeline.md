# COMPONENT H — INPUT PIPELINE AND WORLD INTERFACE

## OVERVIEW

H is the boundary. It is the only place in the system where bytes from the outside become representation vectors, and the only place where representation vectors are routed onward to A, B, C, D, E, F, G. It is intentionally small: an encoder set, a serializer, a dispatcher, a thin theory-of-mind ledger, and a self/world boundary tag. H does not predict, does not reason, does not store knowledge. It produces a single canonical object — a `Stimulus` — at a fixed rate, drives the predict/observe handshake with B, and presents A and C with a clean, modality-agnostic stream. The spec's central claim that "internal thought enters the same pipeline as external input" is enforced here: replay events from D, simulation rollouts from D, and self-generated outputs from G all re-enter through `H.ingest(...)` exactly as a real keystroke does. The only difference is a `source` enum and a `provenance` tag — the encoders, the prediction handshake, and the gap-emission path are identical. This is what makes the rest of the system possible: every other component can assume "if it arrived, it has been encoded to D_REP=256, paired with a prediction tick, and labeled with where it came from." H is the contract that makes that assumption true.

---

## CORE DATA STRUCTURES

### `Stimulus`

The single canonical object H emits. Every other component consumes Stimuli; nothing else crosses the H boundary inward.

Fields:
- `stimulus_id: uint64` — monotonic, allocated per ingest. Stable across the session.
- `tick: int64` — global monotonic tick from B's counter; H reads, never mutates. Paired with the prediction H asked B to make for this stimulus.
- `t_arrival: float64` — wall-clock seconds at the moment bytes (or internal payload) entered ingest.
- `t_encoded: float64` — wall-clock seconds at the moment encoding finished.
- `modality: enum {TEXT, IMAGE, AUDIO, INTERNAL_REPLAY, INTERNAL_SIMULATION, INTERNAL_THOUGHT, INTERNAL_OUTPUT_ECHO}` — the kind of input. Six internal modalities are split out so E and D can do narrative continuity without re-deriving provenance.
- `source: enum {SELF, WORLD}` — the self/world boundary marker. Derived deterministically from `modality` (TEXT/IMAGE/AUDIO → WORLD; the four INTERNAL_* → SELF). A single bit, but load-bearing.
- `agent_id: uint64?` — when the source is WORLD, the concept_id of the external agent the stimulus is attributed to (resolved by H's agent registry below). Null for ambient/unattributed input. For SELF, this is the system's own self-concept id (a pinned node owned by E).
- `representation: float32[256]` — the encoded vector. D_REP=256, locked to match B and C.
- `representation_norm: float32` — cached L2 norm of representation. B reuses this to skip a sqrt during gap math.
- `encoder_id: str` — fully qualified encoder name+version (e.g., `text:char-trigram-bow-v1`, `image:clip-vit-b32-bootstrap`, `audio:mel-stats-v1`). Travels with the vector for the rest of its life — a node written from this stimulus stamps `encoder_id` onto its embedding metadata. Allows post-hoc detection of cross-encoder contamination during similarity queries.
- `name_hint: str` — a short human-readable label for debug and for C's `name_index`. For text inputs, the first ~64 chars of the input text. For image/audio, empty unless a caption is supplied. For INTERNAL_*, a recognizable tag like `replay:<event_id>` or `sim:<chain_id>:<depth>`.
- `provenance: ProvenanceRecord` — how this stimulus came to exist. See below. Required.
- `prediction_handle: PredictionHandle` — opaque token returned by B's `predict(...)` call made just before encoding completed. H carries it through and hands it back to B in `observe(...)`.
- `prior_attention_seeds: list[(concept_id, float32)]?` — optional. Up to 8 concept_ids that F passed in as the active set when H was asked to ingest. Provided so B's prediction call can be primed; absent for cold ingestion.

### `ProvenanceRecord`

Why a stimulus exists. Required on every Stimulus. Used by E for narrative continuity ("did I think this or did someone tell me this?"), by D to avoid feeding replay back into replay, and by the agent ledger to update theory-of-mind state.

Fields:
- `origin: enum {EXTERNAL_TEXT, EXTERNAL_IMAGE, EXTERNAL_AUDIO, REPLAY, SIMULATION, OUTPUT_ECHO, SELF_THOUGHT}` — finer than `modality`; differentiates G's own output (echoed back through H) from other internal sources.
- `parent_stimulus_id: uint64?` — for INTERNAL_*, the stimulus that triggered this one (the original real surprise being replayed, or the simulation's anchor stimulus). Null for EXTERNAL_*.
- `parent_chain_id: uuid?` — for SIMULATION, D's chain id.
- `replay_pass: uint8` — for REPLAY, which pass this is (0 = first replay of the original event). Used by D to scale gain.
- `cycle_depth: uint8` — guards against runaway internal loops. Incremented when a SELF_THOUGHT is fed back through H. Hard-cap at 8 (see Failure Modes).
- `attribution_confidence: float32 in [0,1]` — H's confidence in the `agent_id` assignment. 1.0 for SELF; for WORLD, derived from the agent registry's resolution score.

### `EncoderRegistry`

The set of installed modality encoders. H owns this; it is configuration plus runtime cache.

Fields:
- `encoders: dict[modality_key -> EncoderRecord]` — keyed by modality string (`text`, `image`, `audio`).
- `active_id: dict[modality_key -> str]` — the current `encoder_id` for that modality.
- `previous_id: dict[modality_key -> str?]` — the prior encoder_id if a swap happened mid-session; retained until C completes re-projection (see Re-embedding Policy).

### `EncoderRecord`

A single encoder.

Fields:
- `encoder_id: str` — version string. Persisted; immutable for the life of the record.
- `modality: str` — `text`, `image`, or `audio`.
- `output_dim_native: uint16` — what the encoder produces before projection (e.g., 512 for CLIP image).
- `projection: float32[output_dim_native, 256]` — fixed random orthonormal projection from native dim into D_REP=256. Initialized deterministically from a seed at first install; never trained. Persisted.
- `is_bootstrap: bool` — true iff this encoder is a frozen pretrained model used as a learning scaffold (the spec explicitly disowns frozen pretrained models as core, but admits them as bootstrap; see position below).
- `bootstrap_until_writes: uint32?` — for bootstrap encoders, the count of concept writes after which the encoder is scheduled for replacement by a successor encoder learned from the graph. Default 5,000 writes. Null for non-bootstrap encoders.
- `replacement_target: str?` — encoder_id slated to replace this one (set when training of the successor begins). Null otherwise.

### `AgentRegistry` (theory of mind)

The structure that lets H attribute incoming stimuli to external agents and lets E query "what does the other know that I know?" without polluting the main concept graph with per-agent shadow stores.

Position taken: **single graph, not per-agent namespace.** ToM lives in the same `ConceptGraph` as everything else, anchored by per-agent concept nodes and `refers_to` edges. H owns the anchors and the lookup; the rest is C's edge taxonomy (already includes `refers_to`, `part_of`).

Fields:
- `agents: dict[agent_id -> AgentRecord]` — agent_id is a `concept_id` in C, allocated on first encounter.
- `self_id: uint64` — the pinned self-concept id (owned by E, registered with H at boot).
- `unknown_id: uint64` — a single sentinel agent for ambient input ("the world"); pinned by E. Resolves when no specific agent can be attributed.

### `AgentRecord`

The per-agent ledger. Small. Most ToM data lives in C; this struct is the index into it.

Fields:
- `agent_id: uint64` — concept_id of the agent in C.
- `display_name: str` — for debug/frontend.
- `first_seen: float64` — wall-clock.
- `last_seen: float64` — wall-clock.
- `interaction_count: uint32`.
- `attribution_signature: float32[256]` — slowly-updated centroid of representations attributed to this agent. Used by `attribute_to_agent` to score new stimuli against existing agents. Updated as EMA with mood-timescale alpha (default 0.05).
- `shared_knowledge_index: ANN over concept_ids` — H maintains a small ANN index (HNSW on a 64-dim down-projection) of concepts the agent has been observed to reference. This is the "what they know that I know" index. Each entry maps to a `refers_to` edge in C from the agent node to the referenced concept. The ANN is a query accelerator only; the truth lives in C's edges.
- `belief_divergence_estimate: float32 in [0,1]` — running estimate of how often this agent's references collide with the system's own beliefs. Updated whenever a stimulus from this agent triggers a surprise (B reports gap > threshold) on a concept the agent had referenced previously. High divergence → "this agent disagrees with me a lot" → E and D use it for lying simulation.

### `IngestQueue`

Single-writer serialization queue between H's ingest entry points and the main loop's per-tick processing.

Fields:
- `queue: ring_buffer<RawInput>` — fixed capacity (default 1024).
- `dropped_count: uint64` — incremented when overflow.
- `oldest_t: float64`, `newest_t: float64` — for diagnostics.

### `RawInput`

Pre-encoded input. Lives only inside H.

Fields:
- `payload: bytes | str | float32[*]` — raw bytes for image/audio, str for text, vector for INTERNAL_*.
- `modality: enum` — same enum as Stimulus.
- `source: enum {SELF, WORLD}`.
- `t_arrival: float64`.
- `provenance: ProvenanceRecord`.
- `claimed_agent_id: uint64?` — caller's hint at attribution; H may override.
- `prior_attention_seeds: list[(concept_id, float32)]?`.

### `EncoderSwapJournal`

Persistent log of encoder version changes. Used by C's restore path (the spec's "embedding drift between sessions" failure mode) to know what re-embedding work is owed.

Fields:
- `entries: list[SwapEntry]`
  - `t: float64`, `modality: str`, `from_id: str?`, `to_id: str`, `policy: enum {COLD_REBUILD, HOT_REPROJECT, FROZEN_ONCE, MIXED}`, `nodes_reprojected: uint32`, `complete: bool`.

---

## ALGORITHMS

### H1. `ingest(raw: RawInput) -> Stimulus | None`

The single entry point. Every modality, every source, every replay, every simulation enters here. There is no private path.

Process:
1. Validate `cycle_depth` ≤ 8. If exceeded, drop, log `INTERNAL_LOOP_RUNAWAY`, return None.
2. Push to `IngestQueue`. If queue is full, drop oldest, increment `dropped_count`, log.
3. Pop from queue when the main loop calls `H.tick()` (see H8). Subsequent steps run in the main loop's thread.
4. Allocate `stimulus_id`, capture `t_arrival` (already on RawInput), and proceed to encoding.

This split is important: ingest is callable from any thread (network handler, frontend websocket, D's replay worker, G's output echo); encoding and dispatch happen single-threaded in the main loop.

### H2. `encode(raw: RawInput) -> (representation, encoder_id, name_hint)`

Modality dispatch.

Process:
1. If `modality ∈ {INTERNAL_REPLAY, INTERNAL_SIMULATION, INTERNAL_THOUGHT, INTERNAL_OUTPUT_ECHO}`: payload is **already** a `float32[256]` vector. No encoder runs. Return as-is, encoder_id = `internal:passthrough-v1`, name_hint as supplied.
2. Else look up `encoders[modality]`. If absent, raise `MissingEncoderError`.
3. Run modality-specific encoder:
   - **TEXT** → see H3.
   - **IMAGE** → see H4.
   - **AUDIO** → see H5.
4. Apply the encoder's projection matrix to the native-dim output: `representation = projection @ native_output`. Renormalize to unit length (HNSW in C is cosine-based; consistent norm reduces variance in similarity scores).
5. Return.

### H3. Text encoder (`text:char-trigram-bow-v1`)

The default text encoder. Deliberately not a transformer. Cheap, deterministic, no pretrained weights.

Process:
1. Lowercase, strip control chars except newline.
2. Extract character trigrams (sliding, with start/end markers).
3. Hash each trigram into a 4096-bucket count vector via xxhash.
4. Add a 256-bucket whitespace-token bag-of-hash (xxhash on whitespace-split tokens, mod 256).
5. Concatenate → 4352-dim sparse count vector.
6. L1-normalize to a probability distribution.
7. Project through `projection: float32[4352, 256]` (random orthonormal, persisted seed).
8. L2-normalize.
9. `name_hint = first 64 chars of input`.

This is intentionally crude. The projection is fixed; it's a hashing-trick embedding. Quality at birth is mediocre; quality after 5K writes is irrelevant because the *graph's* learned positions are what carry meaning — the encoder only needs to be a stable function so similar inputs land near each other consistently.

### H4. Image encoder

Two layered options. Position on CLIP follows.

**Option A — `image:clip-vit-b32-bootstrap` (recommended for v1).**
1. Run CLIP ViT-B/32 image branch (frozen, ~150 MB on disk, runs on M1 in ~30 ms per image).
2. Take 512-dim output.
3. Project to 256 via fixed random orthonormal `projection`.
4. L2-normalize.
5. `name_hint = ""` unless caller passes one.
6. Marked `is_bootstrap = True`. After 5,000 image-derived writes, schedule replacement (see Re-embedding Policy).

**Option B — `image:patch-stats-v1` (no-pretrained fallback, must work).**
1. Resize to 64×64.
2. Compute per-patch (8×8 patches, 64 patches total) statistics: mean RGB, std RGB, gradient magnitude mean. → 7 features × 64 patches = 448 dims.
3. Concatenate global histogram (16 bins per channel × 3 = 48 dims) → 496 dims.
4. Project to 256.
5. L2-normalize.
6. `is_bootstrap = False`.

**Position on CLIP:** The spec disowns "pretrained frozen models as core." That is a constraint on the steady state, not on bootstrap. Cold-starting an image-aware mind from zero with a hashing-trick patch encoder produces a graph in which "cat" and "ocean" are at similar distance — the system has no perceptual prior. CLIP-as-bootstrap accepts the prior, gets through the first ~5K writes with semantically reasonable neighborhoods, and then is replaced by a successor encoder learned from the graph itself (see Re-embedding Policy below). Option B exists as a tested fallback to prove the architecture works without CLIP. Steady state is Option B or a successor; bootstrap is Option A. **CLIP is never the primary encoder in steady state.**

### H5. Audio encoder (`audio:mel-stats-v1`)

No pretrained model.

Process:
1. Resample to 16 kHz mono.
2. Window into 25 ms frames with 10 ms hop.
3. Compute log-mel spectrogram (40 mel bands).
4. Aggregate over the clip: per-band mean and std (80 dims), plus per-band first-order delta mean (40 dims), plus zero-crossing rate, RMS, spectral centroid, spectral rolloff (4 dims). → 124 dims.
5. Project to 256.
6. L2-normalize.

### H6. `dispatch(stimulus: Stimulus) -> None`

Where the stimulus goes after encoding.

Process:
1. **Pre-prediction.** Before announcing the stimulus to anyone, call `B.predict(current_state=last_active_centroid, affect_composite=A.composite(), layer=INPUT, query_seed=stimulus.representation_seed_or_None, topK=30)`. The prediction is for *this* stimulus — what does the system expect this input to be in representation space, given context and feel? Receive a `Prediction` and a `prediction_handle`. Stamp the handle on the stimulus. (See H7 for what `representation_seed_or_None` is — it's the partial pre-encoded vector if the encoder supports streaming, or None.)
2. **Observation.** Call `B.observe(Observation(actual=stimulus.representation, tick=prediction.tick, layer=INPUT))`. B returns a `PredictionGap` (or null if the prediction was evicted — should not happen at INPUT layer).
3. **Affect emission.** Already happens inside `B.observe` via B's `EmitGap` — H does not duplicate.
4. **Concept graph touch.** Call `C.find_or_match(stimulus.representation, threshold=0.92)`. If a match returns, stamp the matching `concept_id` onto a side log (the agent registry uses this for shared-knowledge tracking, see H10). If no match and B reported is_surprise, `C.write_on_surprise(...)` will already have been triggered by B's own dispatch — H does not write directly. (This avoids double-writes.)
5. **Agent attribution update.** If `source == WORLD` and `agent_id != unknown_id`, call `H10.update_agent(agent_id, stimulus)`.
6. **Topology event.** If F has subscribed, fire `topology_event(STIMULUS_INGESTED, stimulus_id)`. Frontend visualization consumes the same channel.
7. **Hand off to F.** F is the consumer of "now process this." H's job ends with dispatch; F runs spreading activation, G runs expression, etc. H does not block on any of those.

### H7. Pre-encoding prediction (the "predict before input arrives" mechanism)

The spec calls for prediction to happen *before* input arrives. This is operationalized as follows:

There are three legitimate "before" events H can detect:

- **Anticipation.** F or D requested attention seeds for an upcoming event (e.g., G just produced output and is about to observe its own echo). H calls `B.predict(layer=INPUT, query_seed=expected_modality_marker, topK=30)` immediately after F's signal, with a small modality-prior seed vector (one per modality, learned slowly as the centroid of representations seen in that modality). The prediction is registered with B and given a tick. When the stimulus arrives, encoding runs, and the actual representation is paired with that prediction.
- **Streaming partial encoding.** For long inputs (multi-paragraph text, long audio), the encoder emits a partial representation after the first ~200 ms of content. H sends that partial to B as a query_seed and gets a refined prediction; the final representation is paired against the *refined* prediction. This is the "what do I expect this input to be, given the first few characters?" mechanism.
- **Cold ingestion.** If neither anticipation nor streaming applies, H still calls `B.predict` immediately on entering H6 with `query_seed=None`. The prediction comes from current graph state alone, not from a partial of the input. This is the lowest-quality prediction case but still produces a tick to pair against; the gap will be larger on average for these.

In all three cases, the `Prediction` is registered with B *before* `B.observe` is called. The prediction handle and the observation share the same tick. This is what makes "gap = predicted − actual" well-defined for every stimulus, including the very first one (B handles cold-start via its degenerate-prediction mode, F1).

### H8. `tick(now: float64) -> None`

Called by the main loop at ~50 Hz (configurable). Drains the IngestQueue and processes events in order.

Process:
1. Drain up to `tick_budget = 8` items from the queue.
2. For each: H1 → H2 → assemble Stimulus → H6.
3. Once per second, run `H10.consolidate_agents()` — slow EMA refresh of agent attribution signatures.
4. Once per 10 s, check encoder swap status; if a successor encoder has finished training, run `H11.swap_encoder(...)`.

### H9. `inject_internal(payload: float32[256], modality: INTERNAL_*, provenance: ProvenanceRecord) -> Stimulus`

The entry point used by D (replay/simulation) and G (output echo). This is what the spec means by "internal thought is processed identically."

Process:
1. Validate payload is finite, dim 256, L2-normalized within tolerance (else renormalize).
2. Construct RawInput with `source=SELF`, `agent_id=AgentRegistry.self_id`, supplied provenance, and increment `cycle_depth` from parent if any.
3. Call H1 (`ingest`).

The vector skips H2's encoders (encoder_id = `internal:passthrough-v1`). It still goes through H6's full dispatch, including the predict/observe handshake with B. This is the architectural commitment: a replayed surprise produces the same kind of gap and the same kind of affect injection as an external one. The only differences are:
- `source = SELF` (E uses this for narrative continuity).
- `provenance.origin` distinguishes replay vs simulation vs self-thought.
- B may apply a replay gain reduction (open question 7 in B's spec; resolution is D's call, not H's).

### H10. Agent attribution and theory of mind

Three sub-algorithms.

**H10a. `attribute_to_agent(stimulus) -> agent_id`.** Called during dispatch for WORLD stimuli before `agent_id` is set.

1. If `claimed_agent_id` was supplied by the caller (e.g., a chat client identifies the speaker), use it after a sanity check that it exists in `agents`.
2. Else, compute cosine similarity from `stimulus.representation` to each agent's `attribution_signature`. If max similarity > 0.6 and gap to second-best > 0.1, attribute to that agent with `attribution_confidence = max_sim`.
3. Else, attribute to `unknown_id` with `attribution_confidence = 0`.
4. New agents are *not* created automatically by H. New agent creation is E's decision (E owns identity, including the identity of others). H exposes `H.register_agent(name) -> agent_id` and E calls it when it decides "this is a new person."

**H10b. `update_agent(agent_id, stimulus)`.** After a stimulus is attributed.

1. EMA update `attribution_signature += 0.05 * (stimulus.representation - attribution_signature)`. Renormalize.
2. `interaction_count++`, `last_seen = now`.
3. If `C.find_or_match(stimulus.representation, 0.92)` returned a concept_id `c`, this means the agent referenced something the system also knows about: ensure a `refers_to` edge exists from `agent_id` to `c` in C. Strengthen if it exists. This edge is the "they know X that I know X" link. The `confidence` of the edge accumulates evidence; high confidence means "this agent reliably refers to this concept."
4. Insert/update entry in `shared_knowledge_index` for fast queries.

**H10c. `query_shared_knowledge(agent_id, k=20) -> list[concept_id]`.** Called by E and D.

Returns the top-k concepts the agent is known to share knowledge of, ranked by the `refers_to` edge weight × confidence. E uses this when simulating "what does this agent think I'll say next?" — it filters the simulation's prior to concepts the agent is likely to also activate.

The data model commitment: **theory of mind lives as `refers_to` edges from agent-concept nodes to subject-concept nodes in the single graph C.** No per-agent namespace, no shadow graph. The ANN in `AgentRecord` is purely a query accelerator over those edges.

This has costs flagged in C's open question 7: ToM queries are graph traversals. H mitigates by maintaining the per-agent ANN index over the agent's `refers_to` neighborhood; queries are O(log |refers_to_neighbors|) instead of O(|graph|). At 5K nodes total and an expected ~50 referenced concepts per agent, this is sub-millisecond.

### H11. Encoder versioning and re-embedding (cross-cutting ticket #1)

When an encoder for a modality changes, three policies are available. H supports all three; the choice is per-encoder and recorded in `EncoderSwapJournal`.

**Policy 1 — FROZEN_ONCE (default for non-bootstrap encoders).** The encoder never changes after install. No re-embedding ever needed. This is the cheapest and the default for the audio/text patch encoders.

**Policy 2 — HOT_REPROJECT (default for bootstrap encoders).** The bootstrap encoder is replaced once by a successor learned from the graph (see successor recipe below). At swap time:
1. The old encoder's `projection` matrix is retained as `previous_id`'s record.
2. For every node `n` in C with `encoder_id == old_encoder_id`:
   - If `n.name_hint` is non-empty *and* the new encoder is text-aware: re-encode from `name_hint` and replace `n.embedding`.
   - If the node has `expresses` edges to surface forms G has captured: re-encode the surface form and replace.
   - Else: compute a linear bridge `B_bridge: float32[256, 256]` learned on the subset of nodes that *can* be re-encoded both ways. Apply `B_bridge` to nodes that can't be re-encoded directly. This is the "best-effort projection" path.
   - Nodes that can neither be re-encoded nor bridged (no name, no surface form, isolated): flag `stale_embedding=True`. C's restore path already handles this flag — they are excluded from similarity queries until they reactivate.
3. Re-embed the agent `attribution_signature`s identically.
4. Record the swap in `EncoderSwapJournal`.

**Policy 3 — COLD_REBUILD.** Tombstone every node with the old encoder_id, dropping their structure entirely. The graph re-acquires them through subsequent input. Only used in catastrophic encoder regressions (e.g., a bug in CLIP version X is corrected in version Y and the embeddings are not bridgeable).

**Cost analysis at 5K nodes:**
- FROZEN_ONCE: 0 ms.
- HOT_REPROJECT: dominated by re-encoding. At 5K nodes × ~30 ms per CLIP forward pass in the worst case = 150 s. This is the only operation in the system that takes minutes. It runs as a background job during a low-input period, in batches of 64. The graph is not blocked; nodes are reprojected one batch at a time, with `stale_embedding` flags clearing per-node.
- COLD_REBUILD: 0 ms upfront, but the system loses ~80% of its accumulated graph structure. Catastrophic for identity continuity. Avoid except as last resort.

**Position taken:** **bootstrap encoders use HOT_REPROJECT; native encoders use FROZEN_ONCE.** CLIP is allowed at boot but is replaced once via HOT_REPROJECT after sufficient writes accumulate, leaving a fully self-grown representation in steady state.

**Successor encoder recipe (sketch).** The successor for an image bootstrap is a small (1-2M parameter) student trained offline to mimic the graph's *learned* positions: for each pair of nodes `(a, b)` in C, train the student so that `student(a.surface_form), student(b.surface_form)` produces cosine similarity close to `cosine(a.embedding, b.embedding)`. This is contrastive distillation from the graph itself, not from CLIP. The student is then frozen and installed as the new encoder. Detailed training is outside H's scope; H provides the pair-export interface.

### H12. Self/world boundary representation (cross-cutting ticket #3)

Position taken: **the boundary is a single bit (`source ∈ {SELF, WORLD}`) on every Stimulus, mirrored into a permanent `agent_id` reference (self_id vs other agent), and is never erased on the data flowing downstream.**

Where it is stored:
- On the Stimulus: `source` field, deterministic from `modality`.
- On any concept node born from this Stimulus: C's node record does not currently carry source explicitly, but every node already records `affect_at_birth` (composite). H's commitment: when surprise causes a write, the `name_hint` or a side metadata channel carries the originating `stimulus_id`, from which source is recoverable. *Synthesis-pending*: C may or may not want a first-class `source` field on nodes. The cheaper alternative is for E to maintain an "I-thought-this" set keyed by concept_id, populated from H's stimulus stream.
- On `refers_to` edges in C: an edge from `self_id` to a concept means "I know X"; an edge from another agent's concept to X means "they know X." The presence of both edges is "we both know X" — the foundation of theory of mind.

Who reads it:
- E reads it for narrative continuity ("what did I see vs what did I imagine"). Without this bit, replayed memories and lived experience are indistinguishable to identity, which collapses the spec's "narrative continuity" mechanism.
- D reads it to avoid feeding replay back into replay (a replay stimulus's surprise should not push another copy into the replay buffer).
- G reads it to know that an OUTPUT_ECHO is its own voice, not an external response.
- B reads it (indirectly via the layer field — INPUT for both, but the source bit lets B's stats partition observation gaps by self/world if it wants to).

The spec says "internal thought is processed identically." This is honored: encoders, prediction handshake, gap computation, affect injection — all identical. The source bit is a *label*, not a *path*. Nothing about how a stimulus is processed changes based on source. But the label is preserved so downstream components that need to know (E, D, G) can ask.

### H13. `register_agent(display_name) -> agent_id`

Called by E. Creates a new concept node in C of `name=display_name`, marks it as an agent (via `is_a` edge to a pinned `agent_concept` root node owned by E), pins it, and adds it to `AgentRegistry.agents`. Returns the new concept_id.

### H14. `H.persist() / H.restore()`

H's own persistent state is small.

Persisted:
- `EncoderRegistry` (encoder_id, projection matrices, version metadata).
- `AgentRegistry` (agent_id list, signatures, last_seen counters). Per-agent ANN indexes are rebuildable from C's `refers_to` edges and not persisted.
- `EncoderSwapJournal`.
- Stimulus counter (next stimulus_id).

Format: MessagePack, single file `h.bin`. Size at typical runtime: <500 KB (dominated by projection matrices). On restore, validate that every `encoder_id` referenced in C's nodes is present in the registry; if not, the encoder is missing and any nodes referencing it are flagged `stale_embedding` until either the encoder is re-installed or HOT_REPROJECT runs.

---

## INTERFACES

### Inbound — what other components call into H

- `H.ingest_text(text: str, agent_id: uint64? = None, claim_self: bool = False, prior_seeds: list[(uint64, float32)]? = None) -> stimulus_id` — frontend, network handler, REPL.
- `H.ingest_image(bytes, ...) -> stimulus_id` — same.
- `H.ingest_audio(bytes, ...) -> stimulus_id` — same.
- `H.inject_internal(payload: float32[256], modality, provenance) -> Stimulus` — D for replay/simulation, G for output echo, E for self-generated thought.
- `H.tick(now)` — main loop. Drains the IngestQueue.
- `H.register_agent(display_name) -> agent_id` — E only.
- `H.query_shared_knowledge(agent_id, k) -> list[concept_id]` — E, D.
- `H.encoder_swap_status(modality) -> EncoderSwapStatus` — frontend visualization.
- `H.persist() / H.restore()` — persistence layer.

### Outbound — what H calls into others

- `B.predict(current_state, affect_composite, layer, query_seed, topK)` — once per stimulus (anticipation or pre-encode).
- `B.observe(Observation)` — once per stimulus, after encoding.
- `A.composite()` — read-only, called when assembling B's prediction call.
- `C.find_or_match(representation, threshold)` — for agent attribution and shared-knowledge tracking.
- `C.pin(concept_id) / C.tombstone(concept_id)` — for agent registration (rare).
- `C.strengthen_edge(edge_id, intensity)` — when a `refers_to` edge between agent and subject is reinforced.
- `C.write_on_surprise(...)` — H does **not** call this directly. B does, on H's behalf, when surprise threshold is crossed. This avoids double-writes and keeps C's writer single-threaded.
- `F.notify_stimulus_dispatched(stimulus)` — fire-and-forget; F does its own gating downstream. F may also have pre-registered attention seeds before H emitted the stimulus; H reads those (via a small pull interface) but never blocks on F.
- `E.notify_self_world_event(stimulus_id, source, agent_id)` — fire-and-forget. E uses it for narrative continuity bookkeeping.
- `D.replay_seen(stimulus_id)` — when an INTERNAL_REPLAY stimulus has been fully dispatched. Allows D to release the source replay-buffer entry.

### Threading

H is the most concurrency-exposed component because external input arrives asynchronously. The split: `ingest_*` and `inject_internal` may be called from any thread, are non-blocking, and only push to the IngestQueue. Everything from H2 (encode) onward runs single-threaded in the main loop's `tick`. This matches A's and C's single-writer assumptions.

Encoding is the most expensive step; for image/audio it can take tens of milliseconds. To avoid main-loop stalls, the encoder runs in a worker thread but the *dispatch* (predict/observe handshake) is reposted to the main loop after encoding finishes. This means two passes per stimulus: enqueue → encode (worker) → re-enqueue encoded → dispatch (main loop). The IngestQueue holds both raw and encoded items; an encoded RawInput skips H2 on its way through.

---

## FAILURE MODES

### F1. Encoder produces NaN or Inf
**Manifestation:** A malformed input file or a numerical bug in the encoder produces a non-finite vector. If propagated, it poisons B's running statistics and A's reaction vector for the rest of the session.
**Detection:** Final step of H2 checks finiteness on every dim of `representation`.
**Response:** Drop the stimulus. Log loudly with payload size and modality. Increment `encoder_fault_count` per encoder. If the rate exceeds 1/min, reset the encoder's projection from seed (in case projection drifted from disk corruption) and try once more before disabling the encoder.

### F2. Internal loop runaway
**Manifestation:** G generates output, H echoes back, B predicts, F dispatches, G generates again — a self-reinforcing internal loop with no external grounding. Symptom: `cycle_depth` climbs.
**Detection:** `cycle_depth > 8` in `H1`.
**Response:** Drop the stimulus. Inject an internal "boredom" signal into A (the spec's flatness response) to nudge the system out of loop. Log `INTERNAL_LOOP_RUNAWAY` with chain id.

### F3. IngestQueue overflow
**Manifestation:** External input arrives faster than the main loop can drain (e.g., a flood of websocket messages). Oldest stimuli are dropped.
**Detection:** `dropped_count` increments.
**Response:** Drop oldest, log every 100 drops. The system experiences this as "I missed something." Narrative continuity (E) sees a gap in stimulus_ids and may flag it. No silent corruption.

### F4. Agent attribution collision
**Manifestation:** Two agents with very similar attribution signatures (e.g., two people typing in similar styles) are routinely confused.
**Detection:** If two `AgentRecord`s have signature cosine > 0.9 and both have interaction_count > 50, flag.
**Response:** Lower the attribution confidence floor for both; emit a `AGENT_AMBIGUITY` event. E may decide to merge or to wait for more disambiguating evidence (e.g., explicit agent identification in chat). H does not auto-merge.

### F5. Encoder version missing on restore
**Manifestation:** Persisted graph references `encoder_id = X` but the registry has no encoder with that id (e.g., CLIP weights were deleted from disk).
**Detection:** Restore-time check.
**Response:** Mark every node referencing X as `stale_embedding=True`. Excluded from similarity queries. The graph keeps the node — affect trace and edges remain — but it cannot participate in new spreading activation until either the encoder is restored or its surface form is re-encoded under the new encoder. Log loudly. The user is informed.

### F6. Pre-prediction tick mismatch
**Manifestation:** Encoding takes longer than expected; B's pending_predictions has been evicted by the time `observe` runs. `B.observe` returns null.
**Detection:** B reports `MISSED_PREDICTION`.
**Response:** Re-issue the prediction with current state and pair against the same actual representation. The gap is computed against this fresh prediction. Quality is degraded (the prediction was made *after* the encoding, so it has access to information the original prediction did not). Log `LATE_PREDICTION` for diagnostics; if the rate exceeds 1% of stimuli, raise B's `PENDING_TTL_TICKS`.

### F7. Internal vector arrives unnormalized or wrong-dim
**Manifestation:** D or G hands `inject_internal` a vector that isn't 256-dim or isn't unit-normalized. Could silently corrupt similarity math.
**Detection:** `inject_internal` validates dim and L2 norm.
**Response:** If dim is wrong, raise `InternalVectorDimError` (programming error in the caller). If norm is off but dim is right, renormalize and proceed.

### F8. Modality encoder unavailable for installed modality
**Manifestation:** Image encoder weights file missing or corrupted.
**Detection:** First call to `H.ingest_image`.
**Response:** Disable that modality (reject ingest with `ModalityUnavailable`). Continue serving other modalities. Don't crash. Log loudly. Recovery is manual (re-install).

### F9. Re-embedding job stalls mid-batch
**Manifestation:** HOT_REPROJECT fails partway. Some nodes are re-embedded, some aren't, journal entry says incomplete.
**Detection:** On restore, check journal for `complete=False`.
**Response:** Resume from where it left off — the journal records the last batch index. If the new encoder is also missing, fall back to F5 behavior. Never lose nodes during re-embedding; partial state is acceptable, lost state is not.

### F10. Self/world tag corrupted
**Manifestation:** A SELF stimulus arrives tagged WORLD or vice versa (e.g., D forgets to tag a replay).
**Detection:** Hard to detect in isolation; spotted by E when narrative continuity breaks.
**Response:** Treat the tag as authoritative within H — H does not second-guess it. If E reports a downstream inconsistency, H logs the stimulus_id and lets E decide how to repair (this is E's domain). H's contract is "the tag I forwarded is the tag I received."

### F11. Encoder swap during active prediction
**Manifestation:** A stimulus is encoded under encoder X; before the matching observation reaches B, the encoder is swapped to Y, and a subsequent prediction is made in Y's space. Different stimuli are now in different representation spaces inside B's pending_predictions.
**Detection:** Stamp `encoder_id` on the prediction handle. Verify match in `observe`.
**Response:** If mismatch, fall back to F6's late-prediction path under the new encoder. Encoder swaps drain pending_predictions before completing the swap; this is a coordination guarantee H provides to B.

---

## OPEN QUESTIONS

1. **Is the source bit enough, or does C need a first-class `source` node field?** Currently the design relies on E maintaining an "I-thought-this" set keyed by concept_id, populated from H's stimulus stream. This works but couples E to H's stream directly. Alternative: add a single byte `source_at_birth: enum {SELF, WORLD, MIXED}` to ConceptNode. Cost: 5 KB at 5K nodes — negligible. Benefit: any component can ask "is this a self-grown or world-grown concept" without going through E. *Synthesis-pending* with C and E.

2. **Should the bootstrap CLIP encoder be installed at all?** The spec disowns frozen pretrained models as core. Option B (patch-stats-v1) demonstrably works; CLIP makes the early graph richer at the cost of philosophical purity. The position taken is "yes, as labeled bootstrap, replaced after 5K writes." But this is contestable. An empirical run with both will resolve it: if Option B + 5K writes produces a graph whose semantic neighborhoods are competitive with CLIP-bootstrap + HOT_REPROJECT, drop CLIP entirely.

3. **Pre-prediction quality at cold start.** At birth, B's graph is empty, so anticipation predictions are degenerate (B's F1). The first ~100 stimuli have essentially zero predictive signal. This is an A/B/H coordination point (already flagged in A's open question 3). H's contribution: should H synthesize a tiny seed graph at boot (a few canonical concepts with hand-chosen embeddings) to give the first predictions something to chew on? Or honor the empty-graph spec? The position taken in this design is *honor the empty-graph spec*. The first surprises are large; that shapes character; that's how it should be. But this is empirically contestable.

4. **Theory of mind: how rich do agent models need to be?** Currently each agent has only a `refers_to` set in C plus an attribution signature and divergence estimate in H. This supports "I know the other knows X." It does not support "I know the other believes ¬X" — that requires negated edges or a more elaborate belief representation. The spec doesn't ask for this in v1, but D's lying-simulation and E's deception capability arguably do. *Synthesis-pending* with E and D.

5. **Multi-modal stimuli (e.g., image+caption).** Currently each modality is encoded separately. A single stimulus that combines text and image is two stimuli with the same `t_arrival`. This is correct in dataflow but may miss cross-modal binding. Options: (a) leave as-is, let C learn cross-modal `similar_to` edges via co-occurrence (preferred, matches the spec's "let structure emerge"); (b) introduce a fused encoder that takes both inputs and produces one vector. (a) is the position taken; (b) revisited if empirical results show poor cross-modal binding.

6. **Anticipation triggering.** Who decides when to call `B.predict` for an anticipated input? Right now H exposes a `H.anticipate(modality, seeds)` that F or D can call. But the spec has no formal "anticipation" event. This may be over-engineering; the streaming-partial path (H7 case 2) covers most real cases. *Synthesis-pending* — likely defer until F's design lands.

7. **Encoder-id stamping on graph nodes.** Concept C's spec doesn't currently include `encoder_id` on ConceptNode. The HOT_REPROJECT plan requires it (to know which nodes need reprojection). Either C adds the field, or H maintains a side-table mapping concept_id → encoder_id. The side-table is uglier but doesn't require a C change; the field is cleaner. *Synthesis-pending* with C — recommend the field, ~10 bytes per node.

8. **Streaming partial encoding for long inputs — is the precision gain worth the complexity?** Streaming is described in H7 case 2 but adds significant code path complexity. For text inputs under 200 chars (likely the norm), it's not invoked. Defer until empirically justified by long-input behavior.
