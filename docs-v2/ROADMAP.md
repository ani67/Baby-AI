# Baby AI — Roadmap

*Where we are, where Phase C gets us, and what comes after.*

---

## Completed Phases

### Phase A — Change What Goes IN and What PERSISTS

**A.1 Adversarial Curriculum** (v2.1)
Track per-category performance, oversample weakest categories.
Ensures the brain can't ignore hard cases.

**A.2 Memory Buffer** (v2.3)
Decaying echo of recent cluster activations biases resonance.
Gives the brain temporal context — "dog" primes "frisbee" on the next step.

**Result:** Spatial score went from 0.012 (no buffer, 170K steps) to 0.324 (buffer, 5.7K steps). The blob split from 1 co-firing community to 12. First real structural differentiation.

### Phase B — Change the INTERNAL Structure

**B.1 Multi-Prototype Resonance** (v2.5)
Each cluster's 4-8 nodes are used as individual prototypes for resonance.
`resonance = max(dot(node.weights, input))` instead of `dot(mean, input)`.
Zero new state, zero new params — nodes already ARE multi-directional prototypes.

**Result:** z_threshold jumped from 0.23 to 0.39 immediately. Monitoring spatial score for structural impact (active, ~4K steps in).

### Performance (v2.4)
- Edge adjacency index: O(degree) lookups instead of O(E) scans
- Cached traversal: BFS once per batch, reuse for 32 samples
- Vectorized resonance: single matmul instead of N dot products
- Diff-based proto matrix: only dirty rows rebuilt
- Growth cap at 500 clusters (BUD, INSERT, EXTEND all paused)

**Result:** From frozen at 600+ clusters to ~250 steps/sec.

---

## Current Status (2026-03-25)

```
Step:        ~40K
Clusters:    506 (capped)
Spatial:     0.215
Communities: 7 (sizes: 134, 36, 4, 3, 3)
Buffer:      active (norm ~30)
Prototypes:  active (node-weight multi-prototype)
Speed:       ~3-16 steps/sec at 500+ clusters
```

---

## Remaining Phases

### Phase B.2 — Curiosity-Driven Growth (NOT YET STARTED)

**Problem:** Growth is reactive — BUD splits when confused, INSERT adds when residuals are structured. Nothing builds toward the unknown.

**Proposal:**
```
Every 100 steps:
  1. Find inputs where no cluster had high resonance
  2. Spawn new cluster seeded at that direction
  3. Cap at 2 curiosity clusters per window

"I don't know what this is → build a detector for it"
```

### Phase C.1 — Cluster Roles (NOT YET STARTED)

**Problem:** All clusters are identical agents playing the same game.

**Proposal:** Three structurally different roles:
```
DETECTOR    — responds to specific inputs, narrow activation
              FF update: move toward positive inputs only

INTEGRATOR  — combines signals from multiple detectors
              "animal" = dog + cat + bird
              FF update: move toward weighted combo of edge signals

PREDICTOR   — anticipates next input based on context
              Activates BEFORE the input, using the buffer
              FF update: minimize prediction error
```

New clusters start as Detectors. INSERT creates Integrators. EXTEND creates Predictors.

### Phase C.2 — Typed Edges (NOT YET STARTED)

**Problem:** Edges are undifferentiated scalars. "dog→cat" can't mean both "same category" and "competing concept."

**Proposal:**
```
Excitatory   — pass signal (current behavior)
Inhibitory   — suppress target when source fires
Modulatory   — multiply target activation by factor
```

---

## The Fundamental Question Phase C Answers

```
Does FF learning + growing graph produce genuine abstraction?

                    Results at Phase C
                    ┌────────────────────┐
                    │                    │
              YES: structure             NO: plateau
              emerges                    persists
                    │                    │
    ┌───────────────┤            ┌──────┤
    │               │            │      │
 Semantic      Prediction    Still a   The learning
 islands       chains        blob at   rule itself
 form          appear        scale     is the ceiling
    │               │            │      │
    ▼               ▼            ▼      ▼
 Phase D         Phase E      Pivot    Pivot
                              arch.    learning rule
```

---

## If YES: The Road After Phase C

### Phase D — Learn Your Own Eyes

Right now the brain processes CLIP embeddings — it sorts using borrowed features, never learns to see.

**D.1 Raw Pixel Encoder**
Replace CLIP with a learned encoder inside the brain. Small CNN or patch-based encoder trained via FF. The brain learns its own visual features.

**D.2 Multi-Modal Grounding**
```
Audio  (spectrograms → FF encoder) ──┐
                                      ├──→ growing graph
Video  (frames → FF encoder)     ────┤
                                      │
Text   (tokens → FF encoder)     ────┘

"Dog" = visual cluster + bark sound + word "dog"
connected by typed edges
```

**D.3 Self-Supervised Curriculum**
No more teacher. The brain generates its own questions. Curiosity growth (B.2) becomes the primary driver — "I've never seen this region of input space, look there."

### Phase E — Think In Time

The buffer gives ~10 step memory. Each input is still mostly independent.

**E.1 Prediction As Learning Signal**
Predictor clusters (from Phase C) guess the next input. Prediction error replaces the teacher signal. The brain learns by trying to predict, not by being told.

**E.2 Episode Memory**
Buffer becomes a proper episodic store.
```
"Last time I saw a dog, a ball appeared next"
→ temporal edge: dog_cluster → ball_cluster (means "tends to follow")
```

**E.3 Video / Sequential Input**
Feed video frames in order. The brain learns motion, object permanence, physics. Prediction chains become the core learning mechanism.

### Phase F — Act In The World

Currently a passive observer. Takes input, produces output. No agency.

**F.1 Environment Loop**
```
                    ┌──────────┐
  observation ────→ │  brain   │ ────→ action
                    │          │
  reward     ────→ │ (growth  │
                    │  signal) │
                    └──────────┘
                         ↑
                    environment
```
Simple grid world or robot sim. Brain outputs action, receives next observation. Reward modulates growth (dopamine analog).

**F.2 Goal-Directed Behavior**
Predictor clusters predict outcomes of actions. The brain plans by simulating action chains:
```
"If I go left, I predict food. If I go right, wall."
→ choose action that maximizes predicted reward
```

**F.3 Communication**
The brain generates language, not just decodes it. Compositional output from integrator chains:
```
activated path: [dog_detector] → [running_integrator] → [outside_predictor]
output: "dog running outside"
```

---

## The Big Picture

```
Where we are          Where Phase C gets us       Where it could go
──────────────────── ──────────────────────────── ────────────────────────

  Pattern matcher       Self-organizing              Developmental AI
  in CLIP space         semantic graph               that learns to see,
                                                     predict, and act
  ● ● ● ● ●           ●───●───●
  (blob)                │       │                    eye → brain → hand
                       ●───●   ●───●                see    think   act
                       (islands) (roles)
                                                     Builds its own
  Borrowed eyes        Own structure                 world model from
  (CLIP)               (emergent)                    raw experience

  Teacher tells        Teacher guides                No teacher —
  everything           weaknesses                    curiosity drives
```

---

## Honest Assessment

Phase C answers whether this architecture **works at all**. If spatial score breaks 0.5, communities exceed 20, and prediction chains form — then Phases D-F are a genuine research program toward developmental AI.

If it plateaus — the path forward is replacing FF with a more powerful local learning rule (predictive coding, contrastive learning, or energy-based models), keeping the growing graph architecture but giving it sharper tools to learn with.

Either way, what's been built is the **substrate** — a self-growing, self-pruning, visualizable neural graph. The learning rule is a plug-in. The architecture is the contribution.

---

## Metrics to Track

| Metric | Current | Phase C Target | Phase D+ Target |
|--------|---------|---------------|-----------------|
| Spatial score | 0.215 | > 0.50 | > 0.70 |
| Co-firing communities | 7 | > 20 | > 50 |
| Category similarity (best) | 0.20 | > 0.50 | > 0.80 |
| Unique activation patterns | ~5 | 50+ | 200+ |
| Cluster roles | 1 (all same) | 3 types | 3 types + subtypes |
| Edge types | 1 (scalar) | 3 (exc/inh/mod) | 3 + gated |
| Working memory | ~10 steps | ~10 steps | episodic |
| Input modalities | 1 (CLIP) | 1 (CLIP) | 3+ (vision/audio/text) |
