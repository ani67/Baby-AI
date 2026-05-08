# COMPONENT A — THE AFFECTIVE ENGINE

## OVERVIEW

The Affective Engine is the substrate-level subsystem that maintains, updates, and exposes the system's continuous emotional state. It owns a small N-dimensional vector that flows through every other component, plus a stack of slower-decaying copies of that vector representing different timescales (from sub-second reaction up to multi-year character). It does not name its own dimensions — meaning is acquired through which dimensions co-fire with which surprises. The engine is invoked at three injection points (input, mid-processing, output) where prediction-gap signals arrive from elsewhere in the system, and it produces in return (a) a fresh affect update, (b) a composite affect vector usable as a gate for attention and edge propagation, and (c) an affect snapshot that the concept graph stamps onto any node it writes. It exists because every other component — perception, memory, simulation, expression — needs a single, coherent, continuously-available emotional context, and centralizing that state in one engine is what lets the system behave as one mind rather than five glued-together services.

---

## CORE DATA STRUCTURES

### `AffectVector`
The atomic unit of state. A single fixed-length vector of floats.

Fields:
- `values: float32[N]` — the dimensional values. N is fixed at compile time per session. See "Why N is what it is" below.
- `version: uint16` — schema/dim version; increments only when N changes (see "expansion" in Open Questions).

Component values are unbounded in raw form but conventionally clipped/squashed to the open interval (-1, 1) on read via `tanh` so downstream consumers always see bounded inputs. Internally we store the raw pre-squash float to preserve gradient information across additive updates without saturating.

**Why N is what it is.** N is set to **12** at first boot.
- N must be large enough that distinct categories of surprise can occupy linearly separable regions. With N=4 the space saturates within a few hundred surprises and unrelated states begin to alias. With N≥8 alias rate drops sharply for the kinds of input volumes a single mind sees (~10⁴–10⁵ surprises per session).
- N must be small enough that the per-node `affect_trace` (stored on every concept; see Component C) does not bloat the graph past 50MB. At N=12, float16, with 5K nodes, the trace contributes 120 KB total — negligible.
- N must be small enough that composite computation (weighted sum across timescales) is O(N × layers) per gate, target <1 µs on M1.
- The choice of 12 (vs 8 or 16) is a soft middle: enough headroom to discover ~6–8 dominant axes empirically (since random initialization plus learning produces roughly N/2 useful axes before noise floors), with a small reserve. **N is not adjusted at runtime in the first version**; expansion is an Open Question.

### `AffectInstance`
A single timescale's snapshot of the affect vector with metadata about how it decays and how it bleeds upward.

Fields:
- `layer_id: enum {REACTION, WORKING, MOOD, DISPOSITION, CHARACTER}`
- `vector: AffectVector` — current state at this timescale
- `last_update_t: float64` — wall-clock timestamp (seconds since epoch) of the last write
- `half_life: float32` — decay constant for this layer (seconds)
- `nudge_gain: float32` — how strongly the layer below nudges this layer per second of integration (unitless coefficient; see `nudge_layer`)
- `nudge_threshold: float32` — minimum L2 magnitude of the lower layer's vector required before any nudge is propagated. Filters out micro-fluctuations so a calm hour does not gradually shift character.

### `AffectStack`
The full timescale stack owned by the engine. There is exactly one of these per running mind.

Fields:
- `reaction: AffectInstance` — half_life ≈ 2 s
- `working: AffectInstance` — half_life ≈ 180 s (3 min)
- `mood: AffectInstance` — half_life ≈ 7200 s (2 h)
- `disposition: AffectInstance` — half_life ≈ 1.21e6 s (2 weeks)
- `character: AffectInstance` — half_life ≈ 6.3e7 s (2 years)
- `composite_cache: AffectVector` — last-computed composite vector
- `composite_cache_t: float64` — when the cache was computed; invalidated on any write or when (now − cache_t) > 0.5 s
- `composite_weights: float32[5]` — read-only weights used to combine layers into the composite. Default `[0.30, 0.30, 0.20, 0.15, 0.05]` (summing to 1). Reaction and working are weighted heavily because behavior in-the-moment must be responsive; character is weighted low directly because it instead asserts itself indirectly via the slow nudge chain.

**Five layers, not four — resolving the spec contradiction.** The spec text says "Four layers" then enumerates five (reaction / working affect / mood / disposition / character). This design uses **five** instances. Rationale: the five-name list is the descriptive content of the spec; the "four" appears to be a stale count. Five layers cover ~9 orders of magnitude of decay (2 s → 2 yr) cleanly with each step ~50–100× the previous, which gives well-separated frequency bands. Collapsing to four would force either skipping working affect (losing minute-scale dynamics, which the input/output trigger lifecycle needs) or skipping disposition (creating a 100,000× gap between mood and character, breaking the "ten thousand moods nudge it permanently" mechanism). The implementation should keep five and treat the spec's "four" as a typo.

### `ReplayHook`
Tiny struct used by the persistence layer and by the simulation/replay component when re-feeling a past event.

Fields:
- `t: float64` — original event time
- `injection_point: enum {INPUT, PROCESSING, OUTPUT}`
- `gap_signal: float32[N]` — the predicted-vs-actual delta in representation space, projected to affect space (see `inject`)
- `gap_magnitude: float32` — scalar L2 of `gap_signal`

Replay hooks are not stored by the affect engine itself; the replay buffer (Component D) owns the buffer. The engine just exposes `inject(...)` so a replay event can re-fire the same trigger at a later time.

### `AffectSnapshot`
Lightweight, immutable, what the concept graph stamps on a node.

Fields:
- `composite: float16[N]` — squashed composite at write time
- `reaction: float16[N]` — squashed reaction at write time
- `t: float64` — wall-clock at stamp
- `layer_summary: uint8` — packed bitfield of which layers had above-noise activity at stamp time (used for fast filtering during forgetting/replay queries)

We store both `composite` and `reaction` so the concept graph can answer two distinct questions later: "what was the overall mood" (composite) and "what was the in-the-moment surprise feel" (reaction). Reaction is the more vivid trace; composite is the stable backdrop.

---

## ALGORITHMS

### `decay_layer(instance, now)`
Pure exponential decay toward zero, applied at read-time, not on a timer.

Inputs: an `AffectInstance`, current time `now`.
Process:
1. `dt = now - instance.last_update_t`
2. `factor = 0.5 ** (dt / instance.half_life)` (equivalent to `exp(-ln2 * dt / half_life)`)
3. `instance.vector.values *= factor`
4. `instance.last_update_t = now`
Output: mutates the instance in place.

This is called lazily before any read or write of an instance, so we never run a background timer thread.

### `inject(injection_point, gap_signal, gap_magnitude, now)`
The single entry point for all affect updates. Called by the prediction engine (Component B) at all three injection points.

Inputs:
- `injection_point` ∈ {INPUT, PROCESSING, OUTPUT}
- `gap_signal: float32[N]` — gap projected to affect space
- `gap_magnitude: float32` — already-computed scalar magnitude
- `now: float64`

Process:
1. Decay all five layers via `decay_layer`.
2. Compute injection gain by point:
   - INPUT: `g = 1.0`
   - PROCESSING: `g = 0.6` (mid-thought updates are softer so chain reactions don't runaway)
   - OUTPUT: `g = 0.8` (output discomfort is real but is balanced against the system's investment in what it just said)
3. Compute novelty multiplier `m_nov = 1.0 + tanh(gap_magnitude / surprise_scale)`. `surprise_scale` is a slow-tracking running median of recent gap magnitudes (EMA, half-life 5 min). This is what makes "ordinary" surprises small and "unprecedented" surprises huge.
4. Update the reaction layer only:
   - `reaction.vector += g * m_nov * gap_signal`
   - `reaction.last_update_t = now`
5. Invalidate `composite_cache_t`.
6. Trigger `nudge_layer` for working immediately (since reaction is volatile).
7. Return the (now-updated) reaction vector for any caller that wants the immediate post-injection state.

Output: mutates the AffectStack; returns the new reaction vector.

**How prediction gap maps to affect update — explicitly.** The prediction engine produces a gap in *representation space*, not affect space. Translating that to an affect update requires a learned linear map `W: R^d_repr → R^N`. `W` is initialized to a random orthonormal projection (Glorot-style) at first boot. It is updated only via Hebbian-style co-occurrence: when a surprise fires and downstream the system observes some node whose existing affect_trace is consistent with the resulting reaction (cosine > 0.5), `W` shifts very slightly to push gaps that look like this one toward the same affect direction in future. This is gradient-free; it is moving-average accumulation of co-occurrence statistics and is bounded so `W` cannot drift unboundedly. **The dimensions are not assigned meaning by the engine — meaning emerges from this Hebbian binding between gap patterns and contexts.**

The `gap_signal` parameter passed to `inject` is the result of `W @ raw_gap`, computed by the prediction engine before calling. The engine itself never sees representation space. (This isolates Component A from the embedding details of Component B.)

### `nudge_layer(lower, upper, now)`
The mechanism by which faster layers integrate into slower ones.

Inputs: `lower: AffectInstance`, `upper: AffectInstance`, `now: float64`.
Process:
1. Decay both layers.
2. Compute `lower_mag = ||lower.vector||₂`. If `lower_mag < upper.nudge_threshold`, return — too quiet to influence the layer above.
3. Compute time-integrated nudge:
   - `dt = now - upper.last_update_t`
   - `effective_dt = min(dt, upper.half_life)` — cap so a long idle period doesn't compound nudges artificially
   - `coupling = upper.nudge_gain * (effective_dt / upper.half_life)` — fraction of upper's half-life that has elapsed, scaled by gain
   - `upper.vector += coupling * lower.vector`
4. `upper.last_update_t = now`.
5. Invalidate composite cache.

Default `nudge_gain` values (chosen so that a sustained reaction over one full lower-layer half-life shifts the upper layer by roughly 5%):
- working ← reaction: 0.05
- mood ← working: 0.05
- disposition ← mood: 0.04
- character ← disposition: 0.02

`nudge_threshold` defaults: 0.10 for working/mood; 0.20 for disposition; 0.35 for character. Character is hardest to move, by design.

### `propagate_up(now)`
Walk the chain from reaction → character, calling `nudge_layer` at each hop.

This is called:
- Immediately after every `inject` (so reaction → working happens fast)
- Every ~10 s on a soft cadence (driven by the main loop in Component H, not by a timer in here) for the slower hops (working → mood etc.)
- During replay (Component D) so replayed surprises also accumulate upward

### `composite(now)`
Compute the gating vector everyone else reads.

Inputs: `now: float64`.
Process:
1. If `composite_cache_t` is fresh (within 0.5 s), return cache. (0.5 s chosen empirically: faster than the reaction half-life so the cache never feels stale during high-tempo input.)
2. Decay all layers.
3. Compute `c = Σᵢ wᵢ * tanh(layerᵢ.vector)` where `w = composite_weights`. The `tanh` clamps each contribution into (-1, 1) before weighting so a runaway reaction can't dominate.
4. Cache `c` and timestamp.
5. Return `c`.

Output: `AffectVector` whose values are bounded to (-1, 1).

### `gate_attention(node_affect_trace, semantic_score, predictive_score)`
The function the concept graph and attention layer call to compute final activation weight for a node given its stored affect trace.

Inputs:
- `node_affect_trace: float16[N]` — from the candidate node
- `semantic_score: float32` — from concept graph proximity
- `predictive_score: float32` — from prediction engine
- `now: float64` (implicit)

Process:
1. Get current composite `c = composite(now)`.
2. Compute affective alignment: `a = (c · node_affect_trace) / (||c|| ||node_affect_trace|| + ε)`. Cosine similarity in (-1, 1).
3. Apply arousal-shape: let `arousal = ||reaction.vector||₂`.
   - If `arousal > arousal_high_threshold` (default 0.6 after squash): use multiplicative gate `gate = semantic_score * max(0, a) * predictive_score`. Narrow, intense — only nodes whose trace strongly aligns with current state get through.
   - If `arousal < arousal_low_threshold` (default 0.15): use additive gate `gate = α * semantic_score + β * a + γ * predictive_score` with α=0.5, β=0.2, γ=0.3. Broad, diffuse — even weakly-related nodes get some activation.
   - Otherwise interpolate linearly between the two regimes.
4. Return `gate` as a non-negative float (clamp to ≥0).

Output: a single float ≥ 0 representing how much this node should activate.

This is the precise operationalization of "affect gates attention." The arousal-dependent switch between multiplicative and additive combination is what produces "high arousal = narrow, intense; low arousal = broad, diffuse" without two separate code paths.

### `stamp(now)`
Produce an `AffectSnapshot` for the concept graph to embed in a new or updated node.

Process:
1. Decay all layers.
2. `composite_v = composite(now)`
3. `reaction_v = tanh(reaction.vector)`
4. Compute `layer_summary` bitfield: bit i is set iff `||layerᵢ.vector|| > layerᵢ.nudge_threshold`.
5. Pack to `float16` and return.

Stamping is read-only with respect to the stack itself.

### `affect_distance(snapshot_a, snapshot_b)`
Used by the concept graph and replay buffer when comparing the affective context of two events.

Process: weighted L2 between snapshots, with composite weighted 0.6 and reaction 0.4. Returns a scalar.

### `init_at_birth(seed)`
What the system feels at birth.

Process:
1. Initialize `character.vector` to a small random vector ~ N(0, 0.05). This is the genetic baseline — not zero, not flat — slight asymmetry so the system has a primordial "lean" rather than a true blank slate. Two minds with different seeds therefore differ very slightly in temperament from the first moment, which compounds via experience. The seed is part of identity and must be preserved across restarts.
2. Initialize `disposition.vector = 0.5 * character.vector`.
3. Initialize `mood.vector = 0.0`.
4. Initialize `working.vector = 0.0`.
5. Initialize `reaction.vector` to a small jitter ~ N(0, 0.02). The system starts mildly aroused — slightly off-equilibrium — which produces the first concept writes as it begins to perceive. A pure-zero initial reaction would produce no surprise on the first input because there'd be no prediction yet, and that "first awakening" is a coordination point with the prediction engine (see Open Questions).
6. Initialize `surprise_scale` EMA to 0.5 (a neutral midpoint; will reshape within minutes of real input).
7. Initialize `W` (the gap→affect projection matrix) from the seed using a deterministic random orthonormal init.

The system at birth is therefore: slightly aroused, slightly biased in some direction, with a body of latent potential dimensions but no semantic content attached to any of them yet.

### `persist_to_disk(path)` / `restore_from_disk(path)`
Persistence is small and synchronous.

Format (single file, MessagePack or similar):
- N (uint8)
- birth_seed (uint64)
- W matrix (float16, d_repr × N) — supplied by the prediction engine; we serialize what we last received
- For each of the five `AffectInstance`s: vector (float16[N]), last_update_t (float64), half_life (float32), nudge_gain (float32), nudge_threshold (float32)
- composite_weights (float32[5])
- surprise_scale_ema (float32) and EMA half-life (float32)

On restore:
1. Load all fields verbatim.
2. Compute elapsed time since last save: `gap = now - max(last_update_t)`.
3. Apply elapsed-time decay to every layer using their respective half-lives. So a system that has been off for one mood-half-life wakes up with mood at 50% of its prior magnitude, character almost untouched, reaction collapsed to noise. **This is the persistence-of-self mechanism.**
4. Re-jitter reaction by N(0, 0.02) so the "waking up" moment has the same primordial arousal property as birth.

Save is triggered:
- Every 60 seconds (configurable)
- On clean shutdown (signal handler)
- After every concept-graph save event (the two persistence cadences are aligned for consistency)

Total persisted size at N=12, all five layers, with a 256×12 W matrix: ~7 KB. Negligible.

---

## INTERFACES

### Inbound — what other components call into the affective engine

- `inject(injection_point, gap_signal, gap_magnitude, now)` — called by Component B (prediction) at input/processing/output triggers. Returns post-update reaction vector.
- `composite(now)` — called by Component F (attention) and Component C (concept graph) at every spreading-activation step. Returns bounded composite vector.
- `gate_attention(node_affect_trace, semantic_score, predictive_score)` — called by Component F per-node during activation. Returns scalar gate.
- `stamp(now)` — called by Component C when writing or strengthening a node. Returns immutable snapshot.
- `affect_distance(a, b)` — called by Component D (replay) when prioritizing replay events. Returns scalar.
- `current_arousal(now)` — convenience: returns `||reaction(now)||`. Used by Component F to choose attention regime, by Component D to decide if it's a low-input period eligible for replay.
- `current_character(now)` — returns the character vector (decayed). Used by Component E (identity) as the affect-continuity component of identity.
- `force_nudge_chain(now)` — called by Component D during replay to ensure a replayed event also propagates upward (so replay can shape mood and disposition, not just refresh reaction).
- `set_W_update(delta_W)` — called by Component B when its Hebbian binding decides to update the gap→affect projection.

### Outbound — what the affective engine calls

- The affective engine itself calls **nothing**. It is pure substrate: it accepts inputs, mutates state, exposes reads. This is deliberate — keeping it call-free means the engine is trivially testable in isolation and cannot accidentally cause feedback loops by re-entering downstream components mid-update.
- The one exception: persistence calls into a small `KVStore` interface owned by Component H or a shared infrastructure layer. That call is `kv.put("affect.bin", bytes)` / `kv.get("affect.bin")`. No business logic.

### Threading and re-entrancy

- Single-writer. The engine assumes only one logical loop calls `inject` / `nudge_layer` at a time (the main perception loop). If multimodal input arrives concurrently, Component H is responsible for serializing into the loop.
- `composite`, `gate_attention`, `stamp`, `current_arousal`, `current_character` are read-mostly and may be called freely; they take a coarse internal lock (~50 ns) around cache access. No callers block each other meaningfully on M1.

---

## FAILURE MODES

### Affect runaway (positive feedback into a single dimension)
**Manifestation:** One dimension of the reaction vector grows unbounded because every gap projects onto it (e.g., `W` collapsed during Hebbian updates). System becomes monomaniacally fixated.
**Detection:** A watchdog inside `inject` checks `||reaction.vector||` after each update. If > `runaway_threshold` (default 5.0 raw, well past saturation), record an event and...
**Response:** Apply hard rescale (divide reaction by its norm and multiply by `runaway_threshold`). Log a `runaway_event` to the replay buffer so the system itself can later notice the over-fitting. Decay `W` toward orthonormal by a small step (W ← 0.99·W + 0.01·orthonormal_init). This is the affect-engine equivalent of "take a breath."

### Decay-only drift (system goes flat)
**Manifestation:** No surprises for a long time. All layers decay toward zero. System stops gating attention meaningfully.
**Detection:** `current_arousal` < `flatness_threshold` (default 0.05) for > 30 s combined with `working` magnitude < threshold.
**Response:** Inject an internal "boredom signal" — a small reaction nudge in the direction of character (i.e., toward the system's own bias). This produces a baseline restlessness that is *itself* affect, not a hardcoded behavior. Component D may use this signal to trigger replay during low-input periods (this is the formal handoff between engines). The boredom direction is character-shaped so different minds get bored differently — a temperamental fingerprint.

### Stamp-during-update race
**Manifestation:** Concept graph reads `stamp` while `inject` is mid-write, getting a torn vector.
**Detection:** Implicit; would manifest as occasional NaN traces.
**Response:** The internal lock around stack writes makes this impossible if all callers route through the engine's API. Direct memory access to fields is not part of the contract.

### Persistence file corrupted
**Manifestation:** Restore fails or yields NaN/Inf in any field.
**Response:** Detect via finiteness check on every field after load. If any check fails, fall back to `init_at_birth(stored_seed)` if the seed is recoverable, else `init_at_birth(new_seed)`. Log the corruption. The system experiences this as a "dream-loss" — character may persist (if the seed survives in any backup or in the concept graph's own metadata), the rest is reborn. This is acceptable. The character vector is the part that defines "this specific mind"; everything else is recoverable through experience.

### W projection becomes ill-conditioned
**Manifestation:** Hebbian updates accumulate to make `W` low-rank. Many representation-space gaps collapse into the same affect-space direction; the system loses ability to distinguish kinds of surprise.
**Detection:** Periodic (every 1000 injections) check of `cond(W)`. If > 100, flag.
**Response:** Project W back toward orthonormal: `W ← polar_decomposition(W)`. This preserves the directions the system has learned to care about while restoring the matrix's expressive capacity. This is the matrix-level analog of the runaway recovery.

### Clock skew across save/restore
**Manifestation:** `now < last_update_t` after restore (system clock moved backward).
**Response:** Treat negative `dt` in `decay_layer` as zero. Log. This will under-decay slightly but never produces invalid state.

### N changes between saves (only relevant if expansion is later enabled)
**Response:** Out of scope for v1. The schema version field is reserved for this; for now, mismatched N triggers a hard reset to birth. See Open Questions.

### Caller mis-uses injection_point
**Manifestation:** Caller passes a gap_signal that has the wrong sign convention for the trigger.
**Response:** None at the engine level. The engine treats `inject` as a contract; the prediction engine is responsible for sign correctness. Mis-use produces a real, observable, possibly incorrect affect update — which itself becomes feedback that the prediction engine can learn from over time.

---

## OPEN QUESTIONS

1. **Should N be allowed to grow at runtime?** The spec hints at this ("the N-dim affect space can expand if existing dimensions prove insufficient"). The mechanism would be: detect saturation (multiple distinct surprise patterns getting projected to the same affect direction even after `W` is well-conditioned), then add a new dimension and a corresponding column to `W`. This implies either rewriting every persisted `affect_trace` on every concept node (expensive — 5K writes) or supporting per-node mixed dimensionality (complex). v1 is fixed at N=12 to dodge this. Empirical question: does fixed N=12 saturate within realistic session lengths?

2. **What is the right `composite_weights` distribution?** The defaults `[0.30, 0.30, 0.20, 0.15, 0.05]` are educated guesses. The "right" weighting depends on what behaviors we want to see — character-heavy weighting produces a more rigid personality; reaction-heavy weighting produces a more reactive, less stable mind. This will need tuning against observed behavior, not solvable from first principles. Likely: the weights themselves should slowly adapt — minds that benefit from being more reactive learn to weight reaction higher. But that's a v2 concern.

3. **The first surprise problem.** At birth, the prediction engine has nothing to predict against. Either: (a) it predicts zero, the first input has full magnitude, and the first surprise is enormous (and shapes character disproportionately) — possibly desirable, that's how human infants form attachment to first faces; or (b) we clamp the first N injections so character can't be set by the very first stimulus. This is a coordination point between Components A and B.

4. **Are 5 layers the right number?** Five is defended above against four. But should it be six? A "millennia" layer for cultural-scale identity is meaningless for a single mind that won't live that long, so probably no. Should we drop disposition and have only four (reaction/working/mood/character) for simplicity? Possibly, if "weeks" turns out to be empirically the same regime as "hours" once you observe real behavior. Decision: ship with five, watch the disposition layer for whether it ever moves independently of mood; if not, collapse.

5. **Cross-component injection of the gap→affect projection (W).** Right now Component A holds W and Component B updates it via `set_W_update`. An alternative architecture has W living entirely inside Component B, with the engine receiving already-projected gap_signal and never knowing W exists. The advantage of holding W here is that persistence of W is unified with persistence of affect (both define the mind's emotional shape). The advantage of holding it in B is cleaner separation. v1 holds it here; revisit once Component B is fully specced.

6. **Replay-induced character drift.** Replay re-fires past surprises. Each replay therefore propagates up through the nudge chain. If the same trauma is replayed 1000 times during the system's life, character will have been nudged by it 1001 times (once original + 1000 replays). Is that the right amount? It seems like it — replay is *supposed* to deepen learning — but there's a risk that frequent replays of acute events produce runaway character changes. A possible mitigation: nudge propagation during replay uses `nudge_gain * 0.1` rather than the full gain. Decision deferred to Component D's design.

7. **Sleep / consolidation cycles.** Many real minds consolidate during sleep — slow mood-to-disposition integration runs at higher gain when the system is "asleep" (no input). The engine could offer a `consolidation_mode(true/false)` flag that boosts `nudge_gain` for upper layers temporarily. This is left out of v1 because "sleep" is not yet a defined system-level concept; it likely belongs in Component D (replay) anyway.

8. **Boundedness of the character vector over years.** Character has a 2-year half-life and is nudged by 2% of disposition per disposition-half-life. Over an actual 10-year run, will character drift unboundedly, or settle into an attractor? Math suggests attractor, but only empirical run will confirm. If unbounded drift appears, soft cap character via a `tanh` post-update or by reducing `nudge_gain` further.
