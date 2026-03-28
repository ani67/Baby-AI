# Baby AI — Next Architecture
### From Fixed-Graph Forward-Forward to Resonance-Driven Quadtree Computation

*Written after sessions ending at step ~11,255. Covers what was built, what was learned, what broke, and the complete redesign emerging from those lessons.*

---

## Table of Contents

1. [What Exists Now](#1-what-exists-now)
2. [What Was Learned from Running It](#2-what-was-learned-from-running-it)
3. [The Core Insight: What Went Wrong Architecturally](#3-the-core-insight-what-went-wrong-architecturally)
4. [The New Architecture: Overview](#4-the-new-architecture-overview)
5. [Component 1: Resonance-Based Activation](#5-component-1-resonance-based-activation)
6. [Component 2: Quadtree Tile Computation](#6-component-2-quadtree-tile-computation)
7. [Component 3: Fractal Hierarchy](#7-component-3-fractal-hierarchy)
8. [Component 4: GPU Texture Forward Pass](#8-component-4-gpu-texture-forward-pass)
9. [Component 5: Real-Time Delta Visualization](#9-component-5-real-time-delta-visualization)
10. [Migration Path](#10-migration-path)
11. [What Stays the Same](#11-what-stays-the-same)
12. [Open Questions](#12-open-questions)
13. [Bugs to Fix Before Migration](#13-bugs-to-fix-before-migration)

---

## 1. What Exists Now

### 1.1 System Architecture (Current)

```
seed_data.py → data/stage0/ (images + concepts.txt)
     ↓
curriculum.py → picks item (image or text concept)
     ↓
question_gen.py → generates question for teacher
     ↓
teacher/bridge.py → Ollama/LLaVA answers question
     ↓
encoder/clip_mlx.py → encodes answer as 512-dim CLIP vector
     ↓
baby_model.py → forward pass through cluster graph
     ↓
orchestrator.py → _compute_is_positive() → adaptive percentile threshold
     ↓
baby_model.py → Forward-Forward weight update
     ↓
growth.py → BUD / INSERT / EXTEND / PRUNE / DORMANT
     ↓
viz/emitter.py → WebSocket diff → Three.js 3D point cloud
```

### 1.2 Key Data Structures (Current)

**Node** — weights(512,), bias(1,). Activates via tanh(w·x + b). Updates via Forward-Forward local rule.

**Cluster** — 8 nodes. Receives incoming edge signals. Returns weighted sum activation. Has cluster_type: Integration / Transformation / Arbitration / Routing / Dormant.

**Graph** — registry of clusters + directed edges. Edges have strength (0-1), coactivation history, steps_since_activation. Traversal is layer-by-layer BFS.

**BabyModel** — owns the graph. Runs forward pass. Triggers growth every 50 steps. Checkpoints every 100 steps. Hard cap: 64 active clusters.

### 1.3 Learning Signal (Current, Working)

Forward-Forward: no backpropagation, no global loss function. Each node updates locally.

```
positive pass: weights move toward input  (goodness increases)
negative pass: weights move away from input (goodness decreases)
```

`is_positive` determined by cosine similarity between model prediction and teacher answer vector, compared against adaptive 50th-percentile threshold over last 25 steps.

**This part works correctly as of step ~11,255.**

### 1.4 Training Data (Current)

- ~500+ images across 60+ categories (expanded from original 20)
- 342 text concepts
- Sourced from loremflickr.com
- Teacher: LLaVA 7B via Ollama

### 1.5 All Bugs Fixed So Far

See `fixes-log.md` for the full list. Critical ones:

| Bug | Impact | Status |
|-----|--------|--------|
| `is_positive` never called (input_vector always None) | 1800 steps of fake signal | Fixed |
| Weight normalization after every update | Learning erased each step | Fixed |
| Growth cap checked once outside BUD loop | Blew past 64, reached 94+ | Fixed |
| Cleanup ran before checkpoint restore | Reconnection overwritten | Fixed |
| WebSocket set iteration race condition | 3D view going black | Fixed |
| Projector `offset[2]` index out of bounds | Log spam every 10 steps | Pending |
| Text concepts using image question templates | "Name this: [IMAGE: a yesterday]" | Pending |

---

## 2. What Was Learned from Running It

### 2.1 What the Logs Told Us

**Similarities rose from 0.39 → 0.65-0.70 over 11,000 steps.** Real learning happened. The model got measurably better at predicting teacher responses.

**But the variance stayed huge:**
```
step=11156: sim=0.7196  ← good
step=11157: sim=0.3861  ← terrible
step=11158: sim=0.6864  ← good
```

This is not noise. It means the model handles familiar input categories well and completely fails on unfamiliar ones. No generalization yet.

**Cluster activations kept saturating** — same clusters (c_08, c_22, c_38, c_00) always dominated, always crept toward 1.0, always needed dampening. The dampening worked but it was a treadmill: knock one down, it recovers, another rises. After 11,000 steps all clusters were still orange (Arbitration). Zero specialization.

**The M outputs stayed semantically unrelated** — "sells store dark dangerous" for dessert image. The decoder finds nearest CLIP vocab words to the model's output vector, but the model hasn't learned to point its output at the right area of CLIP space consistently.

### 2.2 What the 3D View Told Us

The blob shape persisted. What it meant: all 64 active clusters were homogeneous in activation, type, and behavior. Different inputs activated the same clusters at nearly the same levels. The graph had no internal differentiation.

Genuine specialization — where one cluster reliably activates for animals and another for vehicles — requires many repeated exposures to the same category with consistent signal. With 60+ categories and only 64 clusters, no single category gets enough dedicated cluster attention to force differentiation.

### 2.3 The Fundamental Tension

The hard cap of 64 active clusters was wrong in conception. It conflates two separate things:

- **Computational load** — how much work happens per forward pass
- **Memory capacity** — how much structure the model can build

Capping total clusters at 64 caps memory capacity. A cluster that starts specializing for "animals" gets pruned before it consolidates, replaced by a fresh generic one. The system never develops lasting specialist structure.

---

## 3. The Core Insight: What Went Wrong Architecturally

### 3.1 The Wrong Constraint

External structural constraints imposed from outside:
- 64 cluster cap
- 8 layers max
- Fixed nodes-per-cluster
- Fixed edge strength initialization

These are engineering decisions pretending to be architectural ones. They created a system that could never develop rich structure because the structure was pre-capped before it could form.

### 3.2 The Right Principle

**Structure should emerge from dynamics, not be imposed by design.**

The three actual rules a brain-like system needs:

1. **Activate if the input resonates with you** — no fixed cap on who activates
2. **Suppress neighbors who represent the same thing** — lateral inhibition limits simultaneous redundant activation
3. **Strengthen connections to clusters you consistently fire with** — Hebbian learning builds structure

Everything else — how many clusters, how deep, what types form — emerges from these three rules applied consistently over time.

### 3.3 The Performance Insight

The cluster count isn't the performance bottleneck. The edge count is. At 64 clusters, 11,000 steps produced 10,000+ coactivation pairs. With 512 clusters and the same edge density that would be 80,000+ edges — CPU array operations can't traverse that at training speed.

The solution: move computation from CPU numpy arrays to GPU texture operations. Image processing on a GPU is orders of magnitude faster than array traversal on a CPU, and the math is equivalent when the graph has spatial structure.

---

## 4. The New Architecture: Overview

### 4.1 The Core Metaphor

The new system is a **living map**. Not a neural network in the traditional sense. Not a graph database. A spatially-organized, dynamically-scaling, GPU-accelerated computation surface where:

- Clusters are pixels (or groups of pixels) with learned identity vectors
- The forward pass is an image operation — resonance check, then activation propagation
- Hierarchy is spatial subdivision — zoom out to see coarse structure, zoom in to see detail
- The visualization is the computation — the same texture that drives the forward pass is what gets rendered

### 4.2 The Five Components

| Component | What it Replaces | What it Adds |
|-----------|-----------------|--------------|
| Resonance-based activation | Hard cluster cap + layer gating | Self-limiting activation via inhibition |
| Quadtree tile system | Fixed array of clusters | Spatial hierarchy, fixed-size computation units |
| Fractal hierarchy | Flat 8-layer graph | Multi-scale representation, emergent depth |
| GPU texture forward pass | CPU numpy array operations | Orders of magnitude faster, enables massive scale |
| Real-time delta visualization | Full graph rebuild every 10 steps | Smooth real-time rendering of activation paths |

---

## 5. Component 1: Resonance-Based Activation

### 5.1 The Concept

Each cluster has a learned **identity vector** — a single 512-dimensional centroid representing what it has come to encode. When an input arrives, clusters compute their resonance score:

```
resonance(cluster, input) = dot(cluster.identity, input)
```

A cluster activates if its resonance exceeds its **activation threshold**. The threshold is learned, not fixed. Clusters that activate frequently for varied inputs raise their threshold (become more selective). Clusters that rarely activate lower their threshold (stay receptive).

### 5.2 Lateral Inhibition

After initial resonance scoring, activated clusters suppress similar neighbors:

```
for each activated cluster A:
    for each other cluster B:
        if similarity(A.identity, B.identity) > inhibition_radius:
            B.activation *= suppression_factor
```

This is the mechanism that limits simultaneous activation without a hard cap. If ten clusters all encode "dog" (because training was repetitive), they suppress each other. Over time, the weakest nine get pruned through disuse and one strong specialist survives.

### 5.3 What This Enables

- Total clusters: unlimited (grows until inhibition keeps the useful ones)
- Simultaneous active clusters: naturally bounded by inhibition, varies by input complexity
- Specialization: emerges because generalist clusters get outcompeted by specialists that don't suppress each other

### 5.4 Implementation Notes

Identity vector computation:
```python
cluster.identity = F.normalize(
    torch.stack([n.weights for n in cluster.nodes]).mean(dim=0), 
    dim=0
)
```

Resonance check is cheap: one dot product per cluster. Even with 512 clusters, this is 512 × 512 = 262,144 multiplications — numpy does this in microseconds.

The expensive part (FF update, edge traversal) only runs for the ~20-40 clusters that actually activate. Total work per step stays similar to current system even as total clusters grow.

---

## 6. Component 2: Quadtree Tile System

### 6.1 The Problem with Arrays

Current cluster storage: Python list of Cluster objects, each containing a list of Node objects, each with a numpy weight vector. Traversal is sequential Python — every edge lookup, every activation update, every coactivation record is a Python operation.

At 64 clusters this is tolerable. At 512 clusters with 10,000+ edges it becomes the bottleneck.

### 6.2 The Tile Concept

Replace the array of clusters with a **quadtree of 64×64 RGBA tiles**.

Each tile is a fixed-size computation unit:
- 64×64 pixels = 4,096 cells
- Each cell can represent a cluster node (weight encoded as pixel value)
- RGBA = 4 channels = 4 weight dimensions per pixel (full 512-dim weight stored across adjacent pixels)
- Alpha channel = activation strength (0.0 = dormant, 1.0 = fully active)

A single 64×64 tile can represent 8 clusters × 8 nodes = 64 nodes, with weight information encoded in pixel values.

### 6.3 Quadtree Structure

```
Root tile (64×64) — coarse global representation
    ├── Child tile NW (64×64) — finer detail, quadrant 1
    │   ├── Grandchild tile NW (64×64) — even finer
    │   └── Grandchild tile NE (64×64)
    ├── Child tile NE (64×64) — finer detail, quadrant 2
    ├── Child tile SW (64×64) — finer detail, quadrant 3
    └── Child tile SE (64×64) — finer detail, quadrant 4
```

**Key rule: tiles never exceed 64×64.** When a region needs more detail, it spawns 4 child tiles. Each child is still 64×64. Total capacity scales exponentially (1 → 4 → 16 → 64 → 256 tiles) while per-tile computation stays constant.

### 6.4 When Tiles Spawn

A tile spawns children when:
- Its internal activation variance exceeds a threshold (the region contains distinct sub-concepts that need separation)
- A specific sub-region consistently activates while others don't (specialization beginning)
- The tile's error rate (how often its prediction is wrong) exceeds a threshold (it needs finer resolution)

### 6.5 When Tiles Collapse

A tile collapses its children (merges back to parent resolution) when:
- Child tiles are consistently similar in activation (no useful specialization happened)
- Children are dormant for extended periods
- Inhibition at the parent level already handles the discrimination

### 6.6 Spatial Semantics

The quadtree is not just a performance optimization. **Spatial proximity encodes semantic similarity.**

Clusters that learn to respond to related concepts drift toward spatial adjacency through Hebbian edge strengthening. Over time, "dog" cluster and "cat" cluster become neighbors. "Animal" meta-cluster forms as their parent. "Predator" and "prey" form as cousin sub-regions.

This is an emergent property, not engineered. The training dynamics create the spatial organization.

---

## 7. Component 3: Fractal Hierarchy

### 7.1 The Concept

The quadtree naturally creates a multi-scale hierarchy. Each level of the tree operates at a different level of abstraction:

| Tree Level | Abstraction | Example |
|------------|-------------|---------|
| Root | Global | "This input is visual/textual/abstract" |
| Level 1 | Category | "This is a living thing" |
| Level 2 | Subcategory | "This is an animal" |
| Level 3 | Type | "This is a four-legged mammal" |
| Level 4 | Specific | "This is a dog" |
| Level 5+ | Instance-level | "This is a golden retriever outdoors" |

The model doesn't drill to the deepest level for every input. It goes as deep as the input demands:

- Ambiguous or novel input: stops at level 2-3 (uncertain, general)
- Familiar well-trained input: reaches level 4-5 (specific, confident)
- Conceptual/abstract input: may branch differently than visual input

### 7.2 Forward Pass Through Hierarchy

```
1. Input arrives as 512-dim CLIP vector
2. Root tile computes resonance → activates relevant region
3. Active region's child tiles compute resonance → activate sub-regions
4. Recursion continues until:
   a. No child tiles exist (leaf node reached)
   b. Activation drops below propagation threshold
   c. Maximum depth reached (safety cap)
5. Output = weighted average of all activated leaf activations
```

### 7.3 Cross-Level Connections

Not all connections respect the hierarchy. Some clusters at level 3 connect directly to level 1 clusters (skip connections). This enables:
- Fast routing: familiar patterns skip intermediate processing
- Analogy: "this pattern at level 3 resembles this other pattern at level 3 in a different branch"
- Novelty detection: if a level 3 pattern has no level 4 children but activates strongly, it's a known-category unknown-instance

### 7.4 Meta-Clusters

A parent tile is effectively a **meta-cluster** — it doesn't store detailed weight information but it does store a summary identity vector representing the centroid of its children. This enables:
- Fast coarse matching without loading child tiles
- Dormant child tiles that can be re-activated when needed
- Pruning of entire subtrees when a domain is no longer relevant

---

## 8. Component 4: GPU Texture Forward Pass

### 8.1 The Key Insight

A GPU can process a 4096×4096 RGBA texture in milliseconds. The entire current model state (64 clusters × 8 nodes × 512 dimensions) fits in a 64×64 texture. A 512-cluster model fits in a 512×64 texture — still trivial for GPU texture operations.

If the forward pass can be expressed as texture operations (sampling, blending, convolution), it moves from CPU-bound Python to GPU-bound shader operations, gaining 100-1000× speedup.

### 8.2 Encoding Scheme

**Weight encoding:** Each node's 512-dimensional weight vector is stored as 128 adjacent RGBA pixels (128 × 4 channels = 512 values). Node identity determined by x-coordinate in tile. Node weights span 128 pixels to the right.

**Activation encoding:** Alpha channel of the first pixel of each node's block = current activation strength. 0.0 = dormant, 1.0 = fully active.

**Identity vector encoding:** Cluster identity vector (centroid of node weights) stored in the tile's header row.

### 8.3 Forward Pass as Texture Operation

The resonance check becomes a **dot product between input texture and identity texture**:

```glsl
// Simplified GLSL
float resonance = 0.0;
for (int i = 0; i < 128; i++) {
    vec4 identity_pixel = texture(identity_texture, vec2(float(i)/128.0, cluster_y));
    vec4 input_pixel    = texture(input_texture,    vec2(float(i)/128.0, 0.0));
    resonance += dot(identity_pixel, input_pixel);
}
```

This runs in parallel across all clusters simultaneously on the GPU. The resonance check for 512 clusters completes in a single GPU pass.

### 8.4 Update Pass as Texture Operation

Forward-Forward weight update becomes a **texture blend**:

```
positive update: lerp(current_weight_texture, input_texture, learning_rate)
negative update: lerp(current_weight_texture, -input_texture, learning_rate)
```

Again, all clusters update in parallel in a single GPU pass.

### 8.5 Performance Implications

| Operation | Current (CPU) | New (GPU) |
|-----------|--------------|-----------|
| Resonance check (512 clusters) | ~50ms | ~0.1ms |
| Forward pass (64 active clusters) | ~10ms | ~0.5ms |
| Weight update (all active) | ~20ms | ~1ms |
| Total per step | ~80ms | ~2ms |

This means training speed goes from ~12 steps/second to potentially 500+ steps/second on the same hardware. 11,000 steps overnight becomes 500,000+ steps overnight.

### 8.6 Implementation Path

Start with Metal Performance Shaders (MPS) on Apple Silicon — already available through PyTorch's MPS backend. The tile textures are just 2D tensors; GPU texture operations are just 2D tensor operations.

```python
# Current
weights = np.array(...)  # CPU numpy
activation = np.dot(weights, input)  # CPU

# New
weights = torch.tensor(..., device='mps')  # GPU Metal
activation = torch.mm(weights, input.unsqueeze(1))  # GPU parallel
```

The quadtree structure itself stays as Python — only the per-tile computation moves to GPU.

---

## 9. Component 5: Real-Time Delta Visualization

### 9.1 What Stays the Same

The 3D point cloud visualization stays. Clusters are points. Edges are lines. The Three.js renderer stays. The WebSocket connection stays.

What changes is what the visualization shows and how efficiently it updates.

### 9.2 Hierarchical Point Cloud

The quadtree hierarchy maps directly to the 3D view:

- **Root tiles** → large, dim points at the center
- **Level 1 tiles** → medium points, orbit the root
- **Level 2 tiles** → smaller points, cluster around their parent
- **Leaf tiles** → smallest points, densest detail

Camera distance determines what's visible:
- Zoomed out: only root and level 1 points visible (coarse structure)
- Zoomed in: leaf tiles appear, fine structure emerges
- This is exactly how Google Maps works — road network doesn't change, what's visible depends on zoom

### 9.3 Activation Path Visualization

When a forward pass runs, the activation path through the quadtree lights up in real time:

```
Input arrives → root tile resonates (brightens)
    → NW child resonates (brightens)
        → specific leaf cluster resonates (brightens, pulses)
        → neighboring leaf also resonates (brightens slightly)
    → NE child does not resonate (stays dim)
```

The path traces through the 3D space like a highlighted route on a map. Different inputs trace different routes. Over time the viewer can see which paths are well-worn (bright, thick edges) vs exploratory (dim, thin edges).

### 9.4 Delta-Only Updates

Current system: sends full graph state every 10 steps (expensive, causes lag with large graphs).

New system: sends only what changed since last frame:

```python
delta = {
    "activated": [cluster_ids that fired this step],
    "deactivated": [cluster_ids that went dormant],
    "edges_formed": [(source_id, target_id, strength)],
    "edges_pruned": [(source_id, target_id)],
    "tiles_spawned": [new_tile_ids with positions],
    "tiles_collapsed": [removed_tile_ids]
}
```

Even with 10,000 active tiles, a typical step changes <50 of them. The WebSocket message stays small regardless of total model size.

Three.js updates individual points and edges incrementally, never rebuilding the full scene.

### 9.5 Transparency as Activation State

The RGBA alpha channel translates directly to Three.js material opacity:

- `alpha = 0.0` → completely transparent (dormant, not rendered)
- `alpha = 0.3` → dim (cluster exists but low activity)
- `alpha = 0.7` → visible (moderate activation)
- `alpha = 1.0` → fully bright (peak activation)

As the model grows and most clusters are dormant, the visualization naturally shows only the active region — the rest fades out. The scene stays readable regardless of total model size.

### 9.6 Tile Boundary Visualization

Each 64×64 tile's boundary can be shown as a faint cube in 3D space. The cube brightens when any cluster inside it activates. This gives a visual sense of the quadtree structure and which quadrants of the semantic space are being explored.

Child tiles are smaller cubes nested inside parent cubes. Spawning a new child tile animates as a cube subdividing — visually satisfying and informationally meaningful.

---

## 10. Migration Path

### 10.1 What NOT to Do

Don't rebuild everything at once. The current system, despite its architectural limitations, is working. It has 24 fixed bugs, a functioning learning loop, real signal, and a checkpoint at step 11,255 showing genuine learning. Throwing all of that away and rewriting from scratch risks introducing new bugs while losing the lessons from the old ones.

### 10.2 Phased Migration

**Phase 0 — Fix remaining bugs in current system (before any architectural work)**

These bugs should be fixed regardless of migration:
- Projector `offset[2]` index error (one-line fix)
- Text concepts using image question templates (question_gen.py routing fix)
- Update `fixes-log.md` and `current-state.md` to reflect step 11,255 state

**Phase 1 — Decouple cluster cap from cluster count**

Single change: remove the hard 64-cluster cap. Instead, implement inhibition-based activation limiting. The graph can grow freely; at any given step only ~20-64 clusters activate based on resonance + inhibition.

This is the most impactful single change and can be done within the current architecture. Expected result: specialist clusters begin forming because they no longer compete with generalists for the same 64 slots.

**Phase 2 — Add identity vector and resonance scoring**

Each cluster gains an `identity` property (mean of node weights, normalized). The forward pass starts with a resonance pre-screening step before full activation. Clusters below resonance threshold skip the full forward pass.

No change to the FF learning rule. No change to the visualization. Backward compatible with existing checkpoints.

**Phase 3 — Quadtree data structure**

Replace the flat cluster list with a quadtree of tile objects. Each tile contains clusters. The traversal logic changes from layer-by-layer BFS to recursive tile activation.

This is the first breaking change — existing checkpoints won't load directly. Migration script needed: map existing clusters to quadtree positions based on their 3D positions in the last saved projector state.

**Phase 4 — GPU texture computation**

Move per-tile weight operations from CPU numpy to GPU tensors (MPS on Apple Silicon). The quadtree structure stays as Python; only the matrix operations inside tiles move to GPU.

This is purely a performance change. Identical results, much faster.

**Phase 5 — Delta visualization**

Update `viz/emitter.py` to emit deltas instead of full state. Update `viz/projector.py` to support hierarchical layout with zoom-level visibility. Update Three.js frontend to handle incremental updates and tile boundary rendering.

### 10.3 Checkpoints Between Phases

After each phase: run for 1,000+ steps, verify signal stays healthy (50/50 ratio, similarities in expected range), verify no saturation explosion, verify visualization still works. Only proceed to next phase if previous phase is stable.

---

## 11. What Stays the Same

These parts of the current system are correct and should not change:

**Forward-Forward learning rule** — local, gradient-free, biologically plausible. The core learning mechanism is sound.

**CLIP encoding** — 512-dimensional embeddings from CLIP ViT-B/32. Teacher bridge through Ollama/LLaVA. This teacher-student setup is working well.

**Adaptive percentile threshold** — the `_compute_is_positive()` method with 50th percentile over last 25 similarities. This is the correct approach to self-calibrating signal.

**Checkpoint persistence** — SQLite state store, pickle checkpoints every 100 steps. Keep this.

**Growth operations** — BUD (split), INSERT (new layer), EXTEND (grow depth), PRUNE (remove weak edges), DORMANT (deactivate low-use clusters). These map cleanly onto the new quadtree architecture.

**Three.js point cloud + WebSocket** — the visualization paradigm is correct. The renderer, the 3D layout, the dialogue feed, the controls. Keep all of it.

**Question templates** — once the concept/image routing bug is fixed, the question generation is good.

**Stage system** — Stage 0 (pure absorption), Stage 1 (performance-based signal), Stage 2+ (extended growth). The stage transitions are sound. The auto-advancement logic added in this session is correct.

---

## 12. Open Questions

### 12.1 Inhibition Radius

How similar do two cluster identity vectors need to be before one suppresses the other? Too wide and the model can't represent nuance. Too narrow and inhibition never fires and saturation returns.

Starting point: inhibition fires when `cosine_similarity(A.identity, B.identity) > 0.8`. Monitor cluster count growth — if it explodes past 512 within 1000 steps, widen the inhibition radius.

### 12.2 Tile Spawning Threshold

What level of internal variance triggers a tile to spawn children? This controls how fast the quadtree deepens.

Start conservative: spawn only when `activation_variance > 0.5` across the tile's clusters for 100+ consecutive steps. This means tiles only deepen when they're consistently handling two meaningfully distinct things.

### 12.3 Cross-Tile Edges

The current edge system connects clusters within a flat graph. In the quadtree, clusters within a tile are implicitly connected (spatial adjacency). Cross-tile edges (between sibling tiles, between parent and child) need a separate edge table.

Question: should cross-tile edges use the same strength/coactivation mechanism as current edges? Probably yes — the mechanism is correct, just needs to span the tile boundary.

### 12.4 Checkpoint Migration

When Phase 3 ships, existing step-11,255 checkpoint can't load into the quadtree architecture directly. Options:
- Write a migration script that assigns each existing cluster to a quadtree tile based on its last known 3D position
- Start fresh from step 0 with the new architecture (valid — the architecture change is significant enough)
- Keep the old system running in parallel while training the new one

Given how long step 11,255 took to reach, the migration script is worth writing.

### 12.5 When to Advance Stage

The current heuristic (auto-advance at 64 clusters + 100 step buffer) no longer applies in the new architecture (no hard cluster cap).

New heuristics to consider:
- Signal stability: advance when the 50th percentile threshold has been stable within ±0.05 for 500 steps
- Semantic consistency: advance when the same input category produces activations in the same quadtree region >80% of the time
- Manual: developer observes M answers and advances when relevant words appear consistently

### 12.6 Decoder

The current TextDecoder (nearest CLIP vocab word to model output vector) is a bottleneck. At step 11,255 the M outputs are still semantically unrelated to inputs.

This is partly a training quantity problem (more steps needed) but also an architectural one: the model's output vector is the activation-weighted sum of many clusters that may represent contradictory things. The decoder needs a richer target.

Possible improvement: instead of decoding the output vector directly, decode the identity vectors of the top-3 most activated leaf clusters. This gives "what the model's most active specialists think" rather than "what the blended output vector is closest to in CLIP space."

---

## 13. Bugs to Fix Before Migration

These are in the current system and should be fixed now, before any architectural work begins:

### 13.1 Projector Index Error (One-line fix)

**File:** `backend/viz/projector.py` line 96

**Problem:** `offset[2]` crashes because PCA returns 2D not 3D offsets when cluster has too few nodes.

**Fix:**
```python
# Before
float(cz + offset[2])

# After
float(cz + (offset[2] if len(offset) > 2 else 0.0))
```

### 13.2 Text Concept Question Templates

**File:** `backend/loop/question_gen.py`

**Problem:** Items where `item.item_type == "concept"` use image question templates, producing "Name this: [IMAGE: a yesterday]".

**Fix:** Route by `item_type` before selecting template. Concept items use: "Tell me about {label}", "What is {label}?", "Describe {label}."

### 13.3 Update Documentation

**Files:** `current-state.md`, `fixes-log.md`

Reflect:
- Current step: ~11,255
- Current stage: 1
- Clusters: 64 active, 171 total (107 dormant)
- Nodes: 1056
- Edges: 6317
- Layers: 9
- Training items: 500+ images, 342 text concepts
- Bugs fixed in this session (growth cap loop, cleanup/restore ordering, auto-stage-advance, expanded dataset)

---

## Summary

The current system proved the learning loop works — Forward-Forward with adaptive thresholding and a teacher bridge produces genuine signal. Similarities went from 0.39 to 0.65+ over 11,000 steps. That's real.

What it showed doesn't work: a fixed-cap flat graph where all clusters compete for 64 slots. Specialization never emerged because specialists got pruned before they could consolidate. The blob persisted.

The new architecture addresses this at the root: resonance-based activation with inhibition replaces the hard cap. Quadtree tiles replace the flat array. GPU texture operations replace CPU numpy. Hierarchy replaces fixed layers. Delta updates replace full-state snapshots.

The learning rule doesn't change. The teacher doesn't change. The visualization paradigm doesn't change. The philosophy doesn't change.

What changes is the substrate the learning happens on — from a flat fixed-size bucket to an infinitely subdivisible, spatially organized, GPU-accelerated computation surface that grows as complex as the training data demands and no more.

---

*Session context: ~14 hours of debugging and design, steps 1229–11255, March 9-10 2026.*
