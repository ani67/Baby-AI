# Baby AI — Failure Point Analysis

Deep audit of the codebase from the perspective of an ML researcher.
Organized by severity: what will break training, what will corrupt results,
what will crash in production, and what limits the architecture.

---

## CRITICAL: Training-Breaking Issues

### 1. Batch/Single Path Signal Mismatch

The model learns different things depending on which code path runs.

```
_step_single() path:                 _step_batch() path:
─────────────────────                ────────────────────
Stage 0 signal:                      Stage 0 signal:
  step % 3 != 0                        (step + len(samples)) % 3 != 0
  → deterministic per step             → SAME value for ALL 32 samples
                                       → entire batch is positive or negative

Stage 1+ threshold:                  Stage 1+ threshold:
  Adaptive median of recent sims       Hardcoded sim > 0.0
  Adjusts to model's distribution      Any positive cosine = positive
  Can drop to 40th percentile          → almost everything is positive
                                       → model never gets negative signal

Positive rate tracking:              Positive rate tracking:
  Appended per step                    delta_summary always says True
  Affects threshold adaptation         → positive_history not updated
                                       → health monitor sees wrong data
```

**Impact**: In batch mode, the model gets almost exclusively positive signal.
It never learns to distinguish. This is why clusters saturate — every update
pushes weights in the same direction.

**File**: `orchestrator.py` lines 597-609 vs 268-276, 737-744

### 2. FF Update Accumulation Without Normalization (Partially Fixed)

The batch LR scaling (`lr / batch_size`) was added, but there's a deeper issue:

```python
# node.py line 47
update = sign * magnitude * self._last_input * (1 - activation ** 2)
self.weights = self.weights + update
```

The term `(1 - activation²)` is the tanh derivative. When activation ≈ ±1
(saturated node), this term → 0 and learning stops entirely. But the
saturation decay in `forward()` reduces weights by 0.97-0.99×, which
causes activation to drop, which causes `(1 - act²)` to spike, which
causes a large update, which causes re-saturation.

```
Cycle:
  activation → 0.95 → (1-0.95²) = 0.0975 → small update
  saturation decay → weights × 0.97
  activation → 0.70 → (1-0.70²) = 0.51 → LARGE update
  activation → 0.95 → saturated again → repeat
```

**Root cause**: Multiplicative decay (saturation handler) fighting additive
updates (FF rule) creates an oscillation. These are two different optimization
objectives competing.

**Fix needed**: Either remove saturation decay and let the FF rule's
built-in `(1-act²)` damping handle it, or use weight normalization
(L2-normalize weights after each update) instead of multiplicative decay.

### 3. Weight Snapshot Distance Scales With Node Count

```python
# baby_model.py line 749
def _cluster_weight_snapshot(self, cluster):
    return torch.cat([n.weights.detach().clone() for n in cluster.nodes])
```

This concatenates ALL node weights into one vector. For 8 nodes × 512 dims
= 4096-dim vector. The distance `torch.dist(before, after)` scales with
dimensionality. A cluster with 8 nodes shows 2.8× more "change" than a
cluster with 1 node, even if per-node change is identical.

**Impact**: Growth monitor decisions (BUD triggers) are biased toward
larger clusters appearing more active. This is subtle but affects which
clusters get split.

---

## HIGH: Correctness Issues

### 4. Resonance Gate Lets 30-35% of Graph Through

```python
# baby_model.py line 32
self.resonance_threshold = 0.05
self.resonance_min_pass = 12
```

With 334 clusters, the `min_pass = 12` guarantee means at least 12 always
activate. But cosine similarity of random 512-d vectors averages ~0.04, so
threshold 0.05 barely filters anything. Most clusters pass.

```
Expected: selective activation, sparse forward pass
Actual:   88-97 clusters fire per step (30-35% of graph)
          → O(n²) co-firing pairs → 4000+ pairs per step
          → health monitor sees "active_per_step > 40" → adjusts threshold
          → threshold can't go higher than 0.15 (LIMITS dict)
```

**Fix**: Resonance should be relative, not absolute. Use top-K (e.g., top 20)
instead of threshold-based filtering. Or normalize by the distribution of
similarities for that specific input.

### 5. Hebbian Edge Update Has No Normalization

```python
# graph.py Edge.hebbian_update()
delta = 0.01 * from_activation * to_activation - decay
self.strength = max(0.0, min(1.0, self.strength + delta))
```

The `0.01 * from_act * to_act` term is always positive when both clusters
fire (activations are `abs()` of node activations → always positive).
The decay is 0.001. So edges monotonically strengthen toward 1.0 as long
as both clusters fire, which they do because resonance lets everything through.

```
After 1000 steps where both fire:
  strength += 1000 × (0.01 × 0.5 × 0.5 - 0.001)
           = 1000 × 0.0015
           = 1.5 → clamped to 1.0

All edges converge to 1.0 → graph loses structure → everything connects
```

**Fix**: Hebbian update should be competitive — normalize across outgoing
edges so total outgoing strength stays constant. Or use Oja's rule which
has a built-in normalization term.

### 6. Co-firing Buffer Can OOM

```python
# orchestrator.py line 100
self._cofiring_buffer: list[tuple[str, str]] = []
```

With 64 active clusters per step: C(64,2) = 2016 pairs per step.
Buffer flushes every 50 calls. That's 50 × 2016 = 100,800 tuples.

With 280 clusters: C(280,2) = 39,060 pairs × 50 = ~2 million tuples
in memory before flush. Each tuple is ~100 bytes → 200MB.

In batch mode: flush every 50 batch calls, but each batch still only
adds one step's worth of pairs. So 50 × 39,060 = ~2M tuples still.

### 7. Thread Safety: Projector Writes Node Positions Unsafely

```python
# projector.py line 96
node.pos = [float(cx + ox), float(cy + oy), float(cz + oz)]
```

This mutates `node.pos` from a thread pool executor while the main thread
reads `node.pos` in `graph.to_json()`, `cluster.forward()`, and viz emit.
No lock, no copy-on-write. Can produce half-written positions:

```
Thread pool:  node.pos = [1.0, ...]   ← writing x
Main thread:  reads node.pos          ← sees [1.0, old_y, old_z]
Thread pool:  node.pos = [1.0, 2.0, ...]  ← writing y
```

---

## MEDIUM: Architecture Limitations

### 8. Forward-Forward Is Fundamentally Limited Here

The FF rule learns by pushing weights toward positive examples and away from
negative ones. But the "positive" signal is cosine similarity between model
output and CLIP embedding of the teacher's answer. This means:

```
What FF learns:  "make my output vector point toward this CLIP vector"
What it can't:   learn compositional structure, hierarchies, or abstractions
```

The model can only learn to produce vectors similar to CLIP embeddings it
has seen. It cannot generalize beyond the CLIP embedding space. It's
essentially doing nearest-neighbor retrieval dressed up as neural learning.

**Deeper issue**: The 512-d identity vector of each cluster converges to
the mean of all inputs that activated it. With cosine similarity resonance,
clusters specialize by direction in embedding space. But cosine similarity
in CLIP space doesn't map cleanly to semantic categories — "dog" and "puppy"
might be close, but "dog" and "bone" (semantically related) are not.

### 9. Growth Operations Don't Respect Semantic Structure

BUD splits by k-means on weight vectors. But weight vectors are pulled in
many directions by FF updates from different inputs. The split doesn't
consider whether the cluster's activation pattern is truly bimodal in a
meaningful way.

```
Cluster c_05 fires for both "dog" and "car" inputs
  → weights are mean of dog and car embeddings
  → k-means might split into "do" and "car-g" (arbitrary split)
  → or "positive examples" vs "negative examples" (meaningless)
```

INSERT uses PCA of residuals between connected clusters. This makes more
sense mathematically, but the residuals are between cluster outputs, not
between input representations. The new cluster learns to model the
difference between two cluster outputs, which is an indirect signal.

### 10. Stage Transitions Are Fragile

```
Stage 0 → 1: step ≥ 800 AND ≥ 60 active clusters
Stage 1 → 2: step ≥ 3000 AND > 120 clusters AND positive_rate > 55%
```

These are hardcoded thresholds with no fallback. If the model happens to
have 59 active clusters at step 800, it stays in Stage 0 forever.
If positive_rate hovers at 54%, it stays in Stage 1 forever.

There's no timeout: "if you haven't advanced by step 5000, advance anyway."
There's no regression: once at Stage 2, you can't go back even if the
model catastrophically forgets.

### 11. Identity Texture Variance Is a Poor Split Signal

```python
# graph.py QuadTile.compute_variance()
return self._texture.var().item()
```

The identity texture is a 512-d vector tiled into 64×64 pixels. The
variance of this texture measures the variance of the identity vector
itself — how "spread out" the 512 dimensions are.

But this doesn't measure what we care about: whether the cluster
represents multiple distinct concepts. A cluster could have high variance
(weights have diverse values) while being perfectly specialized (always
fires for "dog"). Or low variance (uniform weights) while being confused
(fires for everything).

**Better metric**: Use activation bimodality (already computed on clusters)
or output coherence as the split signal, not texture variance.

---

## LOW: Code Quality Issues

### 12. Pickle Checkpoint Security

```python
# store.py line 257
data = pickle.load(f)
```

Arbitrary code execution if checkpoint files are tampered with. Not a
realistic threat in a local research tool, but worth noting.

### 13. Silent Exception Swallowing

Seven places where exceptions are caught and silently discarded:
- `emitter.py` connect/disconnect/broadcast
- `projector.py` reproject
- `bridge.py` health_check/list_models
- `encoder.py` SVD fallback

Each of these hides potential errors. The system "works" but with
degraded functionality that's hard to diagnose.

### 14. Unbounded SQLite Queries

```python
# store.py get_cofiring_pairs() — no LIMIT
# store.py prune_old_snapshots() — fetches ALL snapshots
# store.py export_dialogue_csv() — fetches ALL dialogues
```

With 334 clusters: C(334,2) = 55,611 co-firing pairs.
After 10K steps: 10K snapshots loaded into memory for pruning.

---

## Architectural Failure Modes (What Happens Over Time)

### Mode 1: Convergence Collapse
```
All clusters → similar identity vectors
  → resonance lets everyone through
  → Hebbian strengthens all edges to 1.0
  → inhibition suppresses everyone equally
  → all outputs are the same averaged vector
  → model outputs "generic thing" for all inputs
```

### Mode 2: Saturation Oscillation
```
FF updates → weights grow → activations saturate
  → saturation decay → weights shrink → activations drop
  → FF updates resume → weights grow again → repeat
  → no net learning, just oscillation
```

### Mode 3: Growth Explosion
```
Clusters fire frequently → BUD triggers (bimodality from noise)
  → more clusters → more co-firing pairs → more CONNECT edges
  → more edges → more pathways → more clusters fire
  → positive feedback loop → graph grows unboundedly
  → forward pass slows → system becomes unusable
```

### Mode 4: Stale Teacher Signal
```
Precomputed curriculum: same 5K COCO captions recycled forever
  → model memorizes the 5K embedding directions
  → BUD splits along memorized boundaries
  → no new concepts can form after memorization complete
  → graph structure fossilizes
```

---

## Priority Fix List

| # | Issue | Impact | Effort | Fix |
|---|-------|--------|--------|-----|
| 1 | Batch signal mismatch | Training broken | Medium | Port adaptive threshold to batch path |
| 2 | Saturation oscillation | No learning | Medium | Replace decay with weight normalization |
| 3 | Resonance too permissive | O(n²) cost | Low | Use top-K instead of threshold |
| 4 | Hebbian edge saturation | Graph collapses | Medium | Normalize outgoing edge strengths |
| 5 | Co-firing buffer OOM | Crash | Low | Cap buffer size or flush more often |
| 6 | Projector thread safety | Corrupt positions | Low | Copy positions, don't mutate nodes |
| 7 | Stage transition fallback | Stuck forever | Low | Add timeout-based auto-advance |
| 8 | Weight snapshot scaling | Biased growth | Low | Use per-node mean distance |
