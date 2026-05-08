# COMPONENT C — THE CONCEPT GRAPH

## OVERVIEW

The Concept Graph is the substrate of the mind. It is a single, in-memory, on-disk persisted directed multigraph in which each node is a discrete concept and each edge is a typed, weighted relationship between two concepts. The graph is the only long-term store the system has: there are no separate tables for "memories", "personality traits", "perceptual filters", "world model entities", or "expression tokens". Those are five readings of the same nodes through different access patterns. A concept is born by a one-shot write at the moment of surprise, carries with it the affective state at the time of writing (its "affect trace"), accumulates activation history, gets connected by typed edges to neighbors, and is eventually pruned when its salience falls below the cost of keeping it. Spreading activation, gated by the current composite affect vector supplied by Component A and the prediction priors supplied by Component B, is the mechanism by which the graph "thinks". The graph caps at roughly 5,000 active nodes; depth comes from edge density and abstraction, not node count. Total persistent footprint targets under 50 MB. This component owns the data, the write-on-surprise mechanism, the spreading activation algorithm, the forget loop, the abstraction-formation loop, and the persistence layer. It owes other components a stable query surface and a stable write surface; it depends on A for affect vectors, on B for surprise signals and prediction targets, on D for replay-driven re-writes, and on F for the precise gating math of attention.

---

## CORE DATA STRUCTURES

### `ConceptNode`

The atomic unit of the graph. One node = one concept. Every node carries the entire payload below; there are no node subtypes (a "fact node" vs. a "rule node" is a distinction expressed through edges, not through schema variants).

Fields:

- `concept_id: uint64` — monotonically allocated unique id. Never reused after deletion. Stable across sessions. The graph never references concepts by name internally; name collisions are allowed.
- `name: str` — human-readable label, primarily for debug, frontend visualization, and log/replay traces. Not authoritative. Empty string allowed for purely sub-symbolic concepts that never received a verbal label.
- `embedding: float32[D]` — position in representation space. D is set by Component H's encoders (working assumption: D = 256). This is the vector all similarity computations and prediction-gap computations use. Stored at full precision in RAM, persisted as float16 to disk.
- `affect_trace: AffectTrace` — see below. The affective signature of the concept. Non-null for every node.
- `activation_count: uint32` — total times this node has been activated above the activation-record threshold across its lifetime. Saturates at 2^32-1.
- `last_activated: float64` — wall-clock epoch seconds of the most recent above-threshold activation. Used by the forget loop and recency-weighted query patterns.
- `created_at: float64` — wall-clock epoch seconds at write time. Immutable.
- `surprise_at_birth: float32` — the prediction-gap magnitude that caused this concept to be written. Conserved for the lifetime of the node. Used by the forget loop as a "this earned its place" signal.
- `salience: float32` — a derived, cached scalar that the forget loop reads. Recomputed lazily (see Salience algorithm). Stored so that prune passes can sort without recomputing for every node.
- `abstraction_level: uint8` — 0 for pure instances; incremented when the abstraction loop creates a node by generalizing over multiple lower-level instances. The level is informational only; it does not change semantics.
- `instance_of: uint64?` — optional pointer to the abstraction node this concept was promoted to (if any). Redundant with the `is_a` edge but accessed in hot paths where edge traversal is too expensive. May be null.
- `edges_out: list[EdgeRef]` — outbound edges. See `Edge`.
- `edges_in: list[EdgeRef]` — inbound edges. Maintained as a denormalization for cheap reverse lookup; rebuilt from `edges_out` if corrupt.
- `tombstone: bool` — soft-delete flag. A tombstoned node is invisible to all query and activation paths but is retained for one persistence cycle so that incoming dangling edges can be cleaned without a stop-the-world pass.
- `version: uint16` — increments on every mutation. Used by replay (D) and prediction (B) caches to invalidate stale references.

### `AffectTrace`

The affect imprint of a concept. This is what makes a concept "vivid" or "faded". It is the load-bearing piece of the spec — without it, nodes are inert and the graph collapses back into a knowledge base.

Fields:

- `birth_state: float32[N]` — the composite affect vector (Component A) at the moment the concept was first written. Immutable. N = 8–16, fixed at session start, supplied by A.
- `peak_state: float32[N]` — the affect vector recorded at the activation event with the largest affect magnitude this concept has ever participated in. Updated only when a new event exceeds the prior peak by more than a small hysteresis (default 5%). Acts as the "high-water mark" of how strongly this concept has ever felt to the system.
- `peak_magnitude: float32` — cached L2 norm of `peak_state`, for fast comparison.
- `running_state: float32[N]` — exponentially weighted moving average of the affect vector at every above-threshold activation of this concept. Decay rate matched to the `mood` timescale of A. This is what the activation loop primarily reads to decide whether the concept's "current felt color" matches the current felt color of the system.
- `running_magnitude: float32` — cached L2 norm of `running_state`.
- `last_affect_update: float64` — wall-clock epoch seconds. Used to apply forward decay before reading `running_state`.

### `Edge`

A typed, directed, weighted connection between two nodes.

Fields:

- `edge_id: uint64` — unique id. Allocated separately from concept_ids.
- `src: uint64` — source concept_id.
- `dst: uint64` — destination concept_id.
- `type: EdgeType` — see taxonomy below.
- `weight: float32` — strength of the relationship. Bounded [0, 1]. Initialized at write time; updated by reinforcement and decay rules below.
- `confidence: float32` — orthogonal to weight: how reliable this edge is, given the variance of the surprises that strengthened it. Bounded [0, 1]. An edge can be strong (high weight) but low confidence (the system has seen counter-evidence) or weak but highly confident.
- `affect_at_birth: float32[N]` — the composite affect at the moment this edge was first laid down. Smaller, lower-precision sibling of node affect trace, retained because relationships have feel.
- `last_traversed: float64` — wall-clock epoch seconds of the most recent activation propagation across this edge. Edge forgetting uses this independently of node forgetting.
- `traversal_count: uint32` — number of times the edge has carried activation above propagation threshold.

`EdgeRef` in `edges_out` / `edges_in` is the in-RAM compact representation: `(edge_id, peer_id, type, weight, confidence)`. The full edge record lives in a flat `edge_table` keyed by `edge_id`.

### `EdgeType`

The closed taxonomy of edge kinds. The spec lists `is_a, has_property, causes, precedes, similar_to, opposite_of, context_of`. I keep these and add three to make the substrate sufficient for what D, E, F, G, H need.

- `is_a` — taxonomy / abstraction. Required for the abstraction-formation loop.
- `has_property` — attribution. Lets concepts carry features without inflating the embedding.
- `causes` — directional causal claim. Asymmetric. Required for the world model in D.
- `precedes` — temporal order. Asymmetric. Required by D and by replay sequencing.
- `similar_to` — symmetric associative tie. Required by F's spreading activation as the default fallback path when no typed edge applies.
- `opposite_of` — symmetric contrastive. Different from `similar_to` because predictions made along an `opposite_of` edge are actively negated by B.
- `context_of` — situational co-occurrence. The "this concept tends to fire near that concept under these conditions" backbone.
- `part_of` — mereological. Distinct from `is_a` because parts inherit no taxonomy. Required by H to represent agents-with-features (theory of mind needs a concept of "the other's belief is part of the other"). Justification: collapsing this into `has_property` loses the directional decomposability D needs to simulate.
- `expresses` — concept-to-modality binding. The link between an abstract concept and a verbal label, image fragment, or audio gesture (G). Justification: G needs a fast lookup from "this concept is active" to "what surface forms have historically attached to it"; folding this into `has_property` makes G's hot path slow.
- `refers_to` — the self-reference and indexical edge. Used by E to point internal-state concepts at the concepts they are about (a memory of "I felt embarrassed when I saw X" has `refers_to` to X). Justification: indexicality without this collapses into ambiguous `context_of` chains.

The set is closed at v1. New edge types may emerge through the spec's "growth" provision; if so, they are added by an explicit migration, not invented at runtime. Runtime invented types break query stability for B, D, E, F, G, H.

### `ConceptGraph` (the container)

Fields:

- `nodes: dict[uint64, ConceptNode]` — primary concept store, keyed by concept_id.
- `edges: dict[uint64, Edge]` — primary edge store, keyed by edge_id.
- `name_index: dict[str, set[uint64]]` — many-to-many. A name may resolve to multiple concept_ids (homonyms). Empty-string entries excluded.
- `embedding_index: ANNIndex` — approximate-nearest-neighbor structure (HNSW recommended) over `embedding`. Used by B to find prediction targets and by the write path to find candidate parents for a new concept. Rebuilt incrementally; full rebuild allowed only during persistence cycles.
- `type_index: dict[EdgeType, set[uint64]]` — edge ids grouped by type, for fast "give me all `causes` edges from X" queries without scanning.
- `next_concept_id: uint64`
- `next_edge_id: uint64`
- `node_count_active: uint32` — cached; tombstoned and pending-delete excluded.
- `prune_pressure: float32` — derived gauge in [0, 1]; how close we are to the ceiling. Drives the forget loop's aggressiveness.
- `affect_dim: uint8` — N. Frozen at session start.
- `embedding_dim: uint16` — D. Frozen at session start.
- `schema_version: uint16` — for persistence migration.

### `ReplayHookQueue` (boundary structure for D)

A small ring buffer of `(concept_id, surprise, affect_at_event)` tuples that the graph emits whenever a write or strengthen event crosses surprise threshold. D consumes; the graph never blocks on D. If D is not running, the queue overflows and old entries are dropped silently — the graph's correctness must not depend on D draining it.

---

## ALGORITHMS

### 1. One-Shot Write (`write_on_surprise`)

The single most important operation. Triggered by Component B when a prediction gap exceeds the surprise threshold.

Inputs:
- `representation: float32[D]` — the actual representation that arrived (from H or from the active processing layer).
- `predicted: float32[D]` — what B predicted.
- `surprise: float32` — the gap magnitude (B's responsibility to compute and threshold).
- `current_affect: float32[N]` — composite vector from A.
- `name_hint: str?` — optional label from H, may be empty.
- `context_active: list[uint64]` — currently active concept_ids, ordered by activation strength, capped at K (default K = 16). Provided by F or the calling layer.

Process:

1. **Find candidate match.** Query `embedding_index` for the nearest existing node to `representation` within radius `r_match` (default cosine sim ≥ 0.92). If a match exists:
    - Treat this as a strengthen event, not a write. Increment `activation_count`, refresh `last_activated`, update `running_state` of `affect_trace`, possibly update `peak_state`. Recompute `salience` lazily.
    - If `name_hint` is non-empty and not already in `name_index` for this concept, add it.
    - Return existing concept_id.
2. **No match → write a new node.**
    - Allocate `concept_id = next_concept_id++`.
    - Construct `ConceptNode` with `embedding = representation`, `created_at = now`, `surprise_at_birth = surprise`, `activation_count = 1`, `last_activated = now`.
    - `affect_trace.birth_state = current_affect`, `peak_state = current_affect`, `running_state = current_affect`, magnitudes cached.
    - `abstraction_level = 0`. `instance_of = None`.
3. **Lay down structural edges.** For each `peer_id` in `context_active`:
    - Compute `affinity = cosine(node.embedding, peer.embedding) * activation_strength(peer)`.
    - If `affinity > θ_edge_min` (default 0.4), create a `context_of` edge from `peer_id` to the new node, weight = `affinity`, confidence = 0.5 (we have one observation), `affect_at_birth = current_affect`.
    - If the strongest 1–3 of these peers have cosine similarity above `θ_similar` (default 0.75), additionally create a `similar_to` edge.
    - At write time, no other edge types are speculatively created. Other types must be earned through observation (causes, precedes, opposite_of, etc.). Speculative typed edges produce a polluted graph quickly.
4. **Insert into indices.** name_index, embedding_index (incremental insert), type_index for each new edge.
5. **Push prune budget.** Increment `prune_pressure` by `1 / target_ceiling`. If `prune_pressure > 1.0`, schedule an immediate forget pass (see Forget loop).
6. **Emit replay hook.** Push `(concept_id, surprise, current_affect)` into `ReplayHookQueue`.
7. **Return concept_id.**

The write is atomic at the data structure level (single-threaded write loop assumed; see Failure Modes for the multi-writer case). No partial writes are observable to readers.

Notes on the "one-shot" claim: the *node* is one-shot. Edges around it are continuously revised. There is no contradiction — the spec says "concept written or strengthened" and "edge weights updated".

### 2. Strengthen (`strengthen_node`, `strengthen_edge`)

When an existing concept fires (above activation threshold):

- `activation_count += 1`.
- `last_activated = now`.
- `running_state += α * (current_affect - running_state)`, where α is mapped to A's mood timescale (e.g., α = 0.05).
- If `||current_affect|| > peak_magnitude * (1 + hysteresis)`, update peak.
- Mark salience as dirty (lazy recompute).

When activation propagates across an edge:

- `weight ← clip(weight + η_strengthen * activation_intensity, 0, 1)`.
- `confidence ← confidence + (1 - confidence) * γ_obs`, where γ_obs is small (e.g., 0.02). Bounded growth, never reaches 1.0 deterministically.
- `last_traversed = now`. `traversal_count++`.
- If subsequent observation contradicts the edge (B reports a prediction made along the edge was wrong), `confidence ← confidence * (1 - γ_contra)` with γ_contra = 0.1, and `weight ← weight * (1 - η_contra)` with η_contra = 0.05. Edges decay faster on contradiction than they grow on confirmation, deliberately.

### 3. Spreading Activation (`spread`)

The "thinking" operation. Called by F (attention) and by B (prediction) and by D (simulation rollouts). Affect-gated.

Inputs:
- `seeds: dict[uint64, float32]` — initial activation per concept_id, sum normalized to 1.
- `composite_affect: float32[N]` — current composite from A.
- `arousal: float32` — derived scalar from A (norm of fast-timescale layers, see A's spec).
- `max_steps: uint8` — propagation depth (default 3).
- `budget: uint16` — maximum total node touches (default 256). Hard cap for M1 cost.
- `mode: enum{PERCEIVE, PREDICT, SIMULATE}` — affects gating and edge type weights.

Process:

1. **Sparsity envelope.** Compute `top_k = round(k_base + k_arousal * (1 - arousal))`. High arousal → small top_k (narrow, intense). Low arousal → large top_k (broad, diffuse). Defaults: k_base = 4, k_arousal = 28, so top_k ∈ [4, 32].
2. **Active set initialization.** Active = seeds. Visited = {}.
3. **For each step in max_steps, while budget > 0:**
    a. For each (concept_id, activation) in Active sorted by activation desc, take top_k:
        - For each outbound edge of concept_id:
            - Skip if edge.weight * edge.confidence < `θ_propagate` (default 0.1).
            - Compute affect alignment: `align = 1 - normalized_distance(edge.affect_at_birth, composite_affect)`. Range [0, 1]. This is how the graph "feels" along the edge.
            - Compute type weight `w_type` from a per-mode lookup table:
                - PERCEIVE mode: similar_to, context_of, has_property weighted high; causes, precedes weighted moderate; opposite_of weighted low.
                - PREDICT mode: causes, precedes, is_a weighted high; opposite_of inverted (propagates as a negative prior to B); context_of moderate.
                - SIMULATE mode: causes, precedes weighted highest; is_a moderate; the rest low.
            - Propagated activation = `activation * edge.weight * edge.confidence * align * w_type * decay_step`.
            - Decay_step = 0.6^step (path-length damping).
            - If propagated > θ_activate (default 0.05) and concept_id not over its per-step refractory limit, add to Next.
            - budget -= 1. If budget ≤ 0, break.
    b. Active = Next merged into Active by max(); Visited ∪= keys(Next).
4. **Emit final activations.** Return `dict[uint64, float32]` of all concepts that exceeded θ_activate at any step, with their max activation across steps.
5. **Side effects.**
    - For each concept that crossed the activation-record threshold, call `strengthen_node`.
    - For each edge that carried activation > θ_traversal_record, call `strengthen_edge` (the propagation pathway, not contradiction).
    - Emit no replay hooks here; that is B's responsibility based on prediction gap.

Cost: bounded above by budget * average_out_degree, guaranteed O(budget) on M1. With budget = 256 and average out-degree ≈ 8 in a healthy 5K graph, a spread is comfortably sub-millisecond.

### 4. Forget Loop (`prune_pass`)

Continuous, but triggered rather than scheduled. Two trigger modes:

- **Pressure trigger.** When `prune_pressure > 1.0`, run an immediate, bounded prune pass before returning from the write that caused the overflow. Bounded to `prune_budget = 32` evictions per call to keep write latency predictable. Pressure decremented per eviction.
- **Idle trigger.** When the input pipeline has been quiet for `idle_threshold` (default 5 s) and no replay is currently active, run a longer pass (up to 256 evictions or until 5% of nodes inspected, whichever first).

Salience computation per node:

```
salience =
   w_surprise   * normalize(surprise_at_birth)
 + w_affect     * peak_magnitude_of_affect_trace
 + w_recency    * exp(-(now - last_activated) / τ_recency)
 + w_frequency  * log1p(activation_count) / log1p(activation_count_global_p95)
 + w_density    * normalize(num_edges_with_weight_above_threshold)
 + w_uniqueness * (1 - max_cosine_similarity_to_neighbors_within_2_hops)
```

Defaults: w_surprise = 0.20, w_affect = 0.25, w_recency = 0.15, w_frequency = 0.10, w_density = 0.15, w_uniqueness = 0.15. These weights are tunable hyperparameters; they encode the spec's KEEP IF clauses directly.

A node is eligible for eviction iff:

- It is not pinned (E may pin nodes that are part of the character vector or stable identity scaffolding; see Interfaces).
- It is not currently active (not in any spread's active set in the last `protect_window`, default 30 s).
- Its salience falls in the bottom percentile relevant to current pressure: bottom 1% under low pressure, bottom 10% under high pressure.
- Or it satisfies any of the unconditional eviction conditions:
    - It has zero edges (isolated) and `now - created_at > τ_isolated_grace` (default 60 s — one-shot writes get a chance to acquire neighbors before being culled).
    - It has been subsumed by an abstraction (a node with `instance_of != None` whose abstraction parent has high confidence; the instance can be retired and its edges rerouted to the parent if doing so does not destroy uniqueness).
    - It is a duplicate of a higher-salience node (cosine sim > 0.97 to a node with strictly higher salience).

Eviction process per node:

1. Mark `tombstone = True`. Do not delete from indices yet.
2. For each inbound edge, decrement the source's edge list and emit a "neighbor lost" weak signal (some edges may be rerouted to an abstraction parent if one exists).
3. For each outbound edge, mark for deletion at the next persistence cycle.
4. Decrement `node_count_active`, decrement `prune_pressure` by `1 / target_ceiling`.

Hard physical deletion happens at the next persistence write (see Persistence). Tombstoned nodes invisible to all reads; this gives B's caches one cycle to invalidate.

### 5. Abstraction Formation (`promote_to_abstraction`)

Abstractions are how depth replaces width. Triggered:

- Periodically (every M write events, default M = 50), or
- When the embedding_index detects a dense cluster of low-abstraction nodes (≥ k_cluster_min, default 4) within a tight radius (cosine sim ≥ θ_cluster, default 0.85).

Process:

1. **Identify cluster.** Set of candidate concept_ids C, all `abstraction_level = 0` (or all the same level — abstractions can be promoted recursively).
2. **Synthesize parent.**
    - `embedding = mean(C.embedding)` then renormalize.
    - `affect_trace.birth_state = mean(c.affect_trace.birth_state for c in C)`. This is the "what does this category feel like on average" signature.
    - `peak_state = the c.peak_state with greatest peak_magnitude`. Abstractions inherit the most vivid member's peak.
    - `running_state = mean(running_states)`.
    - `name = ""` initially; G may attach a label later via `expresses` edges if a verbal regularity emerges across members.
    - `abstraction_level = max(c.abstraction_level for c in C) + 1`.
    - `surprise_at_birth = mean(c.surprise_at_birth for c in C) * (1 - 0.2)` — abstractions are slightly less "earned" than their members, so they are easier to prune if they don't get reinforced.
3. **Wire `is_a` edges.** From each member to the new parent, weight = cosine(member, parent), confidence = 0.5.
4. **Set `instance_of` pointer** on each member for fast access.
5. **Edge inheritance (lazy).** Do not copy member edges to the parent at promotion time. Instead, when a member's edge is traversed N_inherit times (default 3), promote a copy of that edge to the parent. This is how "all dogs bark" emerges from "this dog barked, that dog barked, that dog barked" without runaway edge replication.
6. **Members are not retired** at promotion time. Retirement is the forget loop's job, only when an instance is genuinely subsumed.

### 6. Salience Recompute (`refresh_salience`)

Lazy. Every node has a `salience_dirty` flag (implicit; tracked by a small dirty-set on the graph). Salience is recomputed when:

- The forget loop is about to inspect the node.
- A query explicitly asks for top-N by salience.

Computation is the formula in Forget Loop. Cost: O(degree) for the density and uniqueness terms; rest is O(1). On a node with average out-degree 8, recompute is ~10µs.

### 7. Persistence (`snapshot`, `restore`)

The graph persists as a versioned binary snapshot, written atomically on a configurable cadence (default: every 60 s if there have been writes; always on graceful shutdown).

Format: a single `.mind` file with the following layout (length-prefixed sections, all little-endian):

- `header`: magic bytes, `schema_version`, `affect_dim`, `embedding_dim`, `node_count_active`, timestamps.
- `nodes_section`: packed records. Embeddings stored as float16 (4× space saving with negligible activation cost — recovered to float32 on load). Affect trace fields stored as float16.
- `edges_section`: packed records. Weight, confidence as uint8 quantized [0, 255] mapped to [0, 1]. `affect_at_birth` of edges stored as int8 quantized (lossy but acceptable — edge affect is a tiebreaker, not an anchor).
- `index_section`: serialized HNSW index. Optional; if absent on load, rebuilt in background.
- `metadata_section`: `next_concept_id`, `next_edge_id`, `prune_pressure`, mood/disposition/character handoff to A (A persists its own state; this section just records the handshake epoch).

Atomicity: write to `.mind.tmp`, fsync, rename. The previous snapshot is retained as `.mind.prev` for one cycle as a fallback.

Hard physical deletion of tombstoned nodes/edges happens during snapshot write — they are simply not serialized. This is the only point at which a concept_id becomes truly retired.

Size budget at 5K nodes:
- Node base record (~64 bytes excluding embedding/affect): ~320 KB.
- Embeddings (5K × 256 × 2 bytes float16): 2.5 MB.
- Affect traces (5K × 16 × 2 bytes × 3 vectors): 480 KB.
- Edges: at average out-degree 12 → 60K edges. Each ~32 bytes packed → 1.9 MB.
- HNSW index: ~3 MB at this scale.
- Total: ~8 MB. Comfortably under 50 MB ceiling. Headroom intentional — leaves space for replay buffer (D) and identity layer (E) to share the ceiling.

### 8. Restore on Boot

1. Open `.mind`, validate header. If corrupt, fall back to `.mind.prev`. If both corrupt, start from empty graph and log loudly.
2. Stream in nodes; allocate next_concept_id past max seen.
3. Stream in edges; rebuild `edges_in` denormalization.
4. Rebuild type_index from edges.
5. Rebuild name_index from nodes.
6. Either deserialize or rebuild the embedding_index. Rebuild is acceptable up to 5K nodes (HNSW build at this scale is ~1 s).
7. Run an immediate consistency pass: prune any edges referencing missing concepts, decrement `node_count_active`. Log discrepancies.
8. Apply forward decay to all `running_state` affect traces using the persisted `last_affect_update` so the system "wakes up" with continuous affect rather than paused affect.
9. Open the graph for reads and writes.

---

## INTERFACES

The concept graph exposes a narrow, stable surface. Other components read and write only through these.

### Outbound contracts (the graph requires these from others)

**From A (Affective Engine):**
- `A.composite_affect() -> float32[N]` — synchronous, must be cheap. Called by every spread and every write. **Synthesis-phase reconciliation point:** the dimension N and the meaning of "composite" are A's call. This component requires only that the call returns a stable-dimension vector and is safe to call at any time.
- `A.arousal() -> float32` — scalar in [0, 1] for the spread sparsity envelope. If A does not expose this directly, the graph will derive it from the norm of the reaction-timescale layer. Flagged for synthesis: A may want to define this canonically.
- `A.notify_affect_event(concept_id, intensity, vector_at_event)` — the graph fires this on activations that change a node's running affect state significantly. A may use it to update its own state. Fire-and-forget; A must not block.

**From B (Prediction Engine):**
- `B.surprise(predicted, actual) -> float32` — returns gap magnitude. The graph never decides what counts as surprise; it consumes the number.
- `B.is_above_threshold(surprise) -> bool` — single source of truth for whether a write should fire.
- The graph emits `B.on_traversal(edge_id, propagated_activation, predicted_target)` so B can later compare against actual and call back with `strengthen_edge`/contradiction. **Synthesis-phase reconciliation point:** B's prediction-target representation must be expressible as either an embedding (graph vector) or a node id (graph reference) — the graph supports both.

**From D (Simulation + Replay):**
- The graph exposes `replay_hook_drain()` that D pulls from. The graph does not push.
- D calls `graph.spread(seeds, mode=SIMULATE, ...)` for rollouts.
- D may call `graph.write_on_surprise(...)` for replay-driven writes; replay writes are tagged with a `replay_origin: bool` and carry slightly reduced surprise (B's job to compute, not the graph's).

**From E (Identity, Private State):**
- E may call `graph.pin(concept_id)` and `graph.unpin(concept_id)` to mark identity-anchor nodes immune to forgetting. The graph respects pins absolutely.
- E may call `graph.query_by_affect_signature(signature, k)` to find concepts that "feel like" a given affect — used to reconstruct character.

**From F (Attention):**
- F is the primary caller of `spread`. F may also override the default sparsity envelope by passing explicit top_k.

**From G (Expression):**
- G calls `graph.neighbors_by_type(concept_id, type=expresses)` to find surface forms.
- G calls `graph.write_on_surprise(...)` when its own output produces an output-loop affect that itself crosses surprise threshold (via B).

**From H (Input Pipeline):**
- H produces `representation` vectors and `name_hint`s. H is the canonical source of embeddings.
- H may call `graph.find_or_match(representation, threshold)` for fast existence checks before deciding whether to invoke B.

### Inbound contracts (the graph offers these)

- `write_on_surprise(representation, predicted, surprise, current_affect, name_hint, context_active) -> concept_id`
- `strengthen_node(concept_id, current_affect)` — used when an external caller knows it just observed a confirming activation but did not cross surprise threshold.
- `spread(seeds, composite_affect, arousal, max_steps, budget, mode) -> dict[concept_id, activation]`
- `neighbors(concept_id, type_filter=None) -> list[(edge, peer_node)]`
- `neighbors_by_type(concept_id, type) -> list[(edge, peer_node)]`
- `find_or_match(representation, threshold) -> concept_id | None`
- `query_top_k_similar(representation, k) -> list[(concept_id, similarity)]`
- `query_top_k_by_affect(affect_vector, k) -> list[(concept_id, alignment)]`
- `query_top_k_salient(k) -> list[concept_id]`
- `pin(concept_id) / unpin(concept_id)`
- `tombstone(concept_id)` — explicit eviction for E's use only (to retire identity scaffolding deliberately).
- `snapshot(path) / restore(path)`
- `replay_hook_drain(max=K) -> list[(concept_id, surprise, affect_at_event)]`
- `subscribe_topology_events(callback)` — F and the frontend visualization use this for incremental updates without polling.

All write methods are serialized through a single writer thread. All read methods are safe to call from any thread; readers see a consistent snapshot at a given activation step (no torn reads of node fields).

---

## FAILURE MODES

**Embedding collisions / near-duplicates.** Two distinct concepts arrive with embeddings within `r_match`. Manifestation: the second is silently merged into the first; the second concept's name and affect are absorbed. Correct response: the find-or-match step keeps an audit trail (last 100 merges in a debug ring buffer). If E or H later detects that the merged concepts should have been distinct (e.g., via name disambiguation), the audit trail enables a forced split. Without that, the system mostly tolerates merging — it's a feature of one-shot writes that ambiguity collapses.

**Embedding drift between sessions.** H's encoder changes (model retraining, version bump). Old persisted embeddings are now in a different space from new ones. Manifestation: all spreads return nonsense for old-vs-new concept comparisons. Correct response: schema_version bump triggers a re-embedding pass at load: every persisted node is fed back through H's encoder using its `name` (if non-empty) and any `expresses`-edge surface forms. Nodes with no recoverable surface representation are retained but flagged "stale_embedding" and excluded from similarity queries until reactivated. **Synthesis-phase reconciliation point:** H must publish encoder version in its handshake with the graph; the graph must check it on restore.

**Prune pressure spike.** Many writes in a short window push pressure well above 1. Manifestation: write latency degrades because every write triggers a bounded prune. Correct response: the bounded budget per write (32 evictions) caps degradation. If pressure stays above 1.5 for more than `pressure_alarm_window` (default 10 s), the graph escalates: temporarily raises `θ_match` to merge more aggressively at the write step (less new nodes), and reduces `θ_activate` to spread faster (forces more strengthen-vs-write outcomes). Recovery is automatic; the alarm clears when pressure < 0.9.

**Affect dimension mismatch.** A is restarted with a different N than the persisted graph. Manifestation: catastrophic — affect vectors no longer align. Correct response: refuse to load. The graph treats `affect_dim` as part of the schema; mismatch is a hard error. The user is forced to either restore the old A state or accept rebuilding the graph (keeping nodes, dropping affect traces). Better than silent corruption.

**Edge contradiction storms.** A frequently traversed edge keeps being contradicted (B reports the prediction along it failing). Manifestation: weight collapses, confidence collapses, but the edge keeps being re-strengthened by spreading activation on co-occurrence. Correct response: when an edge's confidence falls below `θ_confidence_dead` (default 0.05) while weight is non-trivial, the edge is moved to a "quarantine" sub-store: it is preserved (so the system remembers it tried this hypothesis) but excluded from PREDICT-mode propagation. PERCEIVE mode still uses it weakly. This implements "I used to think X caused Y, now I'm not sure" without losing the relationship.

**Replay hook queue overflow.** D is not draining. Manifestation: surprises stop being relayed for replay. Correct response: drop oldest. Log a warning if drops exceed 10/min. The graph's correctness is independent of D; D's learning is the casualty, which is acceptable.

**Persistence write fails mid-flight.** Disk full, or process killed. Manifestation: `.mind.tmp` orphaned, `.mind` either present (old, fine) or absent (catastrophic). Correct response: on next boot, ignore `.mind.tmp`; load `.mind` if present. If `.mind` is missing but `.mind.prev` is present, load it and emit a "lost up to one snapshot interval" warning. If everything is gone, start empty and log loudly. Never crash on startup over data loss — the mind continues, in whatever state it can be reconstructed.

**Pinned node referenced by orphaned context.** E pins a node and forgets to unpin. Manifestation: salience floor on a node that no longer matters. Correct response: pins decay. Pin records carry a `pin_reason: str` and a `pin_age: float`. If pin_age exceeds 30 days without E reaffirming via touch, the pin is downgraded to a salience boost rather than absolute immunity, and the node becomes prunable. E is responsible for re-pinning anything that should be permanent. The graph refuses to keep a hidden permanent-set without E's continuing acknowledgment.

**Spreading activation runaway.** Pathological graph topology (one hub node connected to thousands) causes spread to consume its full budget on the hub's neighborhood. Manifestation: all spreads degenerate to "the hub fires". Correct response: degree cap on propagation. When taking outbound edges from a node, only the top `degree_cap` (default 32) by `weight * confidence * affect_alignment` are considered per spread. Hubs still exist; they just don't dominate every thought.

**Abstraction over-collapse.** The cluster detector creates an abstraction node from members that should have stayed distinct. Manifestation: queries return the abstraction where the system meant the instance; expression flattens to generalities. Correct response: members are not retired at promotion. If subsequent activations consistently surface a member with high specificity (defined as activation of the member without activation of the parent), the parent's confidence drops and may be tombstoned, leaving members intact. Symmetric: if members are never re-activated after the parent is created, the parent absorbs them gracefully.

---

## OPEN QUESTIONS

1. **Embedding dimension D.** Tentatively 256. The right number depends on H's encoder, which is being designed in parallel. If H lands on D ≪ 256 (e.g., 128), HNSW becomes faster but match thresholds need re-tuning. If D ≫ 256, the 50 MB budget tightens. Empirical: try D = 128 and 256, measure retrieval quality and persistence size.

2. **Affect dimension N.** Spec says 8–16, A picks. The graph is parameterized on N and indifferent. But affect-trace storage cost scales with N: at N = 16 we are at ~480 KB total for 5K nodes; at N = 32 we are still under budget. If A finds 16 dimensions insufficient to "distinguish states that matter differently" (the spec's phrase), the graph can absorb the increase.

3. **Sparsity envelope constants.** k_base, k_arousal, θ_propagate, θ_activate. Picked by gut at write time. These need empirical tuning against real spreading-activation behavior; they will look wrong on the first run.

4. **Whether `precedes` deserves a separate sequence-store.** Temporal chains of length > 5 are awkward in a graph that propagates by depth-bounded BFS. D may need a sidecar trajectory store. **Synthesis-phase reconciliation point:** if D builds its own sequence store, `precedes` may be redundant in the graph proper and could be demoted.

5. **Quantization tolerance for embeddings.** Float16 in-memory works for 5K nodes on M1; below ~10K nodes the precision loss in cosine similarity is empirically noise-level. Float8 / int8 quantization would buy us space we don't currently need; only adopt if the budget tightens.

6. **Whether `expresses` edges should live in the main edge table.** They are touched only by G and by the frontend. A separate sidecar table would speed G's hot path and keep the core graph cleaner. Decision deferred to G's spec.

7. **Theory-of-mind concept duplication.** When the system models another agent (H/E concern), it stores concepts representing the *other's* model of the world. These look identical to its own concepts, distinguished only by `refers_to` chains. Is one graph the right structure, or should there be a per-agent namespace? The spec says one graph. I am preserving that. But the cost is that ToM queries are graph traversals rather than table lookups; this may bite F. Flagged for synthesis.

8. **Pin semantics.** I proposed decaying pins. E may want absolute pins. The trade-off: absolute pins permit hidden long-tail accumulations; decaying pins force E to take ownership. I lean decaying; will defer.

9. **Multi-writer concurrency.** Currently single-writer assumed. If D's replay runs in a worker thread and writes back to the graph, we need either a queue funnel or a real lock. Queue funnel is simpler and matches the rest of the architecture (everything happens in the affect/predict main loop). Decision deferred until D specifies its concurrency model.

10. **Underspecification in the source spec — flagged.** The spec says "edge weights updated" but does not say what the update *rule* is, nor whether confidence is even a concept (I introduced it). The spec also says "abstractions form from specific instances over time" but does not specify the trigger. Both of these are design choices I made above. The spec's KEEP/FORGET clauses are also enumerated as bullet points without weights; the salience formula above is one principled blending of them. Other blendings are defensible. Finally, the spec says "concept written or strengthened in graph (one-shot)" — the word "one-shot" applies cleanly to writes but is ambiguous for strengthens; I have treated strengthens as continuous, not one-shot, on the grounds that one-shot strengthen would defeat affect trace's purpose (running_state needs more than one observation to mean anything). If the spec author meant one-shot strengthen, this design is wrong about that and should be revisited at synthesis.

11. **Contradiction in the spec — flagged.** The spec says "Confirmation costs nothing. Surprise costs everything — and earns everything", which strongly implies that activations on confirmation produce no graph mutation. But it also says "frequently reactivated" is a KEEP IF criterion, and "activation_count" is a node field. Reactivation must therefore mutate at least the activation_count and last_activated fields, even on confirmation. I have treated these mutations as below the level of "learning" — they are bookkeeping, not graph structural change. Flagged for synthesis.

12. **Initial state.** What does the graph contain at birth? Empty? A small seed set bootstrapped by H? The spec's affect-engine question ("what does the system feel at birth?") has a sibling here: "what does it know at birth?" If empty, the first 100 inputs all produce surprise (everything is novel), and the prune pressure mechanic must tolerate a very rapid initial growth phase. I default to empty + a "warming-up" flag that suppresses pruning for the first `warmup_writes` (default 200) writes. Flagged for synthesis with A and H.
