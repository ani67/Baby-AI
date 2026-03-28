# Phase C: Self-Referential Learning

## Where We Are (Phases A-F)

```
Phase A: Learn what to see, what to remember          DONE
         ├─ A.1: Adversarial curriculum                ✅ v2.1
         ├─ A.2: Memory buffer                         ✅ v2.3
         └─ Result: spatial 0.012 → 0.375 (31x)

Phase B: Improve internal structure                    DONE
         ├─ B.1: Multi-prototype resonance             ✅ v2.5
         ├─ B.2: Curiosity-driven growth               ✅ v2.6
         ├─ B.3: Multi-head resonance                  ❌ v2.6 (CLIP dims not axis-aligned)
         └─ Result: z_threshold 0.23 → 0.39, dormant clusters reactivated

Phase C: Meaningful specialization                     ← YOU ARE HERE
         ├─ C.1: Per-cluster learning signal (staged)
         ├─ C.2: Content-aware routing
         └─ C.3: Patch-level input

Phase D: Learn to see from scratch (replace CLIP)      —
Phase E: Think in time (prediction, episodes)           —
Phase F: Act in the world (agency, goals)               —
```

## What We Learned from Failed Experiments

20 features attempted across v2.5–v3 + signal enrichment. 12 failed. The failures revealed a principle:

```
WORKS (modify what system SEES)       FAILS (modify update SIGNAL)
──────────────────────────────        ──────────────────────────────
✅ memory buffer (temporal)           ❌ momentum (cancels 50/50)
✅ multi-prototype (selection)        ❌ residual connections (f(x) << x)
✅ activation norm (stability)        ❌ continuous FF signal (dampens)
✅ curiosity growth (structure)       ❌ soft resonance (dampens)
✅ capped adversarial curriculum      ❌ LR warmup (delays learning)
                                      ❌ per-cluster error LR (monopoly)
                                      ❌ adaptive resonance width (concentrates)
                                      ❌ inhibitory edges (monopoly)
                                      ❌ error direction (kills diversity)
                                      ❌ structure reuse (topology ≠ knowledge)
```

**The principle:** FF learning is a minimum-viable-signal system. The binary +/- works because each cluster learns from its own unique input at full magnitude. Anything that dampens magnitude kills learning. Anything that injects shared signals kills diversity.

Every failed experiment tried to make the learning signal smarter. Every successful one changed what the system perceives or how it's structured.

## The Information Bottleneck (Why We're Stuck)

The current pipeline throws away information at three levels:

```
Image (224 × 224 × 3 = 150,528 values)
  │
  ├─ CLIP patches (49 × 512 = 25,088 values)   ← THROWN AWAY (Loss 1)
  │
  ├─ CLIP pooled vector (512 values)            ← we use this
  │
  ├─ Each cluster produces its own output        ← THROWN AWAY (Loss 2)
  │
  ├─ Global output vs teacher (1 float)
  │
  └─ Threshold → is_positive (1 bit)            ← ALL clusters get SAME bit (Loss 3)
```

**Loss 1: Spatial structure erased.** CLIP's vision transformer sees 49 patches (7×7 grid). "Dog running in park" has patches for face, legs, trees, grass. We average them into one vector. A "motion" cluster can't attend to leg patches — it gets the whole image blurred together.

**Loss 2: Per-cluster contribution ignored.** 20 clusters fire. Each produces its own output. We combine them into one global output, compare to teacher, get one bit. Cluster A might be perfectly aligned with the teacher while Cluster C is irrelevant — they both get the same grade. It's grading a group project by giving everyone the same score.

**Loss 3: No routing intelligence.** Clusters don't know what other clusters contributed. They can't decompose the problem, avoid redundancy, or fill gaps.

## The Kitchen Brigade (Not a Farmer's Market)

The system shouldn't be a market of independent stalls. It should be a restaurant kitchen:

```
MARKET (current system)                KITCHEN BRIGADE (what we need)
────────────────────────               ─────────────────────────────
Each stall independent                 Stations communicate
Customers visit stalls randomly        Expeditor routes to right station
Everyone gets same Yelp review         Each station knows if THEIR plate was good
Stalls don't see each other            Stations see what others are plating
```

A kitchen works because:
1. **The expeditor routes** — "fish to station 2, sauce to station 4"
2. **Each station gets specific feedback** — "your sauce was too salty" not "the meal was bad"
3. **Stations know their role relative to others** — saucier doesn't duplicate grill
4. **The order flows in sequence** — prep → cook → plate → garnish

| Kitchen concept | Baby AI equivalent | Status |
|---|---|---|
| Expeditor routing | Content-aware edges | C.2 (planned) |
| Per-station feedback | Per-cluster learning signal | C.1 (planned) |
| Station awareness | Cluster knows neighbors | C.2 side-effect |
| Sequential flow | Layers | ✅ exists, underused |

## The Three Proposals

### C.1: Per-Cluster Learning Signal (Staged)

**Problem:** All 20 active clusters get the same +/- bit. Irrelevant clusters get rewarded for others' work.

**Solution:** Each cluster gets its own +/- based on its own output vs teacher. Phase it in to avoid early noise:

```
Steps 0–5K:     100% global signal (training wheels)
Steps 5K–10K:   70% global + 30% per-cluster (blend)
Steps 10K+:     100% per-cluster signal
```

**Why this isn't Exp 1 (which failed):** Exp 1 switched to per-cluster sign from step 0 when no cluster was specialized. Individual signals were pure noise. Staging gives clusters time to differentiate under global signal first, then switches to individual accountability once they have identity.

**Implementation:**
- Compute per-cluster cosine similarity: `sim_i = cos(cluster_i.output, teacher_answer)`
- Threshold per-cluster: `is_positive_i = sim_i > adaptive_threshold`
- Blend with global: `signal_i = blend * global + (1-blend) * per_cluster_i`
- `blend` decays from 1.0 → 0.0 over first 10K steps

**What changes:** 1 bit shared → 20 independent bits. Information goes up 20x.

**Risk:** Medium. Per-cluster signal was tried and failed, but with no staging. The staging is the hypothesis.

**Validates if:** Spatial score exceeds 0.40 sustained. Communities > 9.

### C.2: Content-Aware Routing

**Problem:** Edges are dumb wires: "if I fire, wake neighbor with strength 0.7." The wine cluster wakes the chicken cluster for EVERY input, even wine-tasting inputs where chicken is irrelevant.

**Solution:** Each edge gets a learned gate direction. The edge fires strongly when the input aligns with that direction:

```
edge_gate = dot(input, edge.direction)
routed_signal = signal * sigmoid(edge_gate) * edge.strength
```

The gate direction learns: when the downstream cluster gets a positive signal through this edge, the direction nudges toward the input that caused it.

**This is the "stalls redirect customers" idea.** A cluster that fires on "dog running" sends a strong signal to the "motion" cluster (gate aligned with action inputs) but a weak signal to the "landscape" cluster (gate not aligned).

**Implementation:**
- Add `direction: Tensor(512)` to Edge (initialized to normalized random)
- In `cluster.forward()`: scale incoming signal by `sigmoid(dot(input, edge.direction))`
- After step: for each edge where downstream got positive signal, `edge.direction += lr * input` (normalize after)
- For negative signal: `edge.direction -= lr * input`

**What changes:** Clusters stop polluting each other. The graph becomes a routing network, not a broadcast network.

**Risk:** Low. This doesn't touch the FF signal at all. It's a graph-level routing mechanism. Worst case: gates don't converge and behavior matches current (sigmoid(random) ≈ 0.5 on average).

**Validates if:** Different inputs activate different subgraphs. Measure by: activation overlap between dissimilar inputs drops below 50% (currently ~80%).

### C.3: Patch-Level Input

**Problem:** We pool 49 CLIP patch embeddings into 1 vector. A cluster can't attend to the "dog" part of "dog running in park" — it gets the blurred average.

**Solution:** Feed all 49 patch embeddings. Each cluster computes resonance against all patches and selects the ones it's most resonant with:

```
Current:  cluster sees pooled_vector              (1 × 512)
Proposed: cluster sees top-k resonant patches     (k × 512)
          "animal" cluster picks face+body patches
          "scene" cluster picks background patches
```

**Implementation:**
1. Re-run CLIP extraction saving patch features (49 × 512 per image), not just pooled
2. Resonance: each cluster computes similarity to all 49 patches, picks top-k (k=3-5)
3. Cluster's input = mean of its selected patches (or weighted by resonance)
4. Different clusters literally see different parts of the same image

**What changes:** Input goes from 512 shared values to 512 × k unique-per-cluster values. Each cluster's input is different because it selects different patches.

**Risk:** Medium-high. Requires re-extracting embeddings (one-time cost). 49x more resonance computation (mitigated by only computing patch resonance for top-20 clusters, not all 500).

**Validates if:** Clusters develop patch-level specialization. Measure by: different clusters consistently select different patch positions for the same image category.

## Execution Order

```
C.1 (per-cluster signal)  ──→  C.2 (content-aware routing)  ──→  C.3 (patches)
        │                              │                              │
   lowest effort                  core insight                   highest payoff
   tests staging hypothesis       makes graph smart              richest information
   no new params                  1 new vec per edge             re-extract embeddings
        │                              │                              │
   validate: spatial > 0.40       validate: activation          validate: patch
   communities > 9                overlap < 50%                  specialization
```

Each builds on the previous:
- C.1 gives clusters individual accountability → they NEED to be different
- C.2 lets the graph route information intelligently → clusters CAN specialize
- C.3 gives clusters different views of the same input → clusters SEE differently

## What We're NOT Doing (and Why)

- **Changing the FF algorithm** — every attempt to modify the learning rule failed. We modify perception and routing instead.
- **Adding more hyperparameters** — C.1 has 1 (blend schedule). C.2 has 0 new (uses existing edge LR). C.3 has 1 (k patches).
- **Building for Phase D yet** — learned encoders require richer signal. C.1-C.3 may provide that foundation, but we validate C first.
- **Self-referential signals (novelty, drift, neighbor divergence)** — interesting but orthogonal. Can layer on top of C.1-C.3 if needed. Don't want to change too many things at once.

## Success Criteria for Phase C

Phase C is complete when:
- Spatial score > 0.50 (sustained over 5K steps)
- Communities > 10 (with at least 3 substantial ones)
- Activation overlap between dissimilar categories < 50%
- Different clusters respond to different aspects of the same input

If C.1-C.3 together don't hit these targets, the next move is self-referential signals (novelty detection, neighbor divergence, drift tracking) before considering Phase D.
