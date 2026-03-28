# Where We Are — Grand Scheme Assessment

*Updated 2026-03-28 after multi-round attention, sequential curriculum, memory, and projection experiments.*

---

## The Journey in Numbers

```
v2.2 baseline:   spatial 0.012, 1 community    (the blob)
v3 confirmed:    spatial 0.375, 7 communities   (31x improvement)
Exp 4 peak:      spatial 0.157, 12 communities  (most structural diversity)
+distributed err: spatial 0.539, 4 communities  (3.5x v3)
+multi-round attn: spatial 0.579, 4 communities (convergence-based forward)
+sequential curr:  spatial 0.650, 28 communities (episode-based feeding)
current run @50K:  spatial 0.591, 2 communities  (fresh graph, maturing)
```

---

## What's Proven

```
✅ Memory buffer breaks the blob (temporal context → differentiation)
✅ Multi-prototype resonance improves selectivity (2x spatial)
✅ Activation normalization helps (9 communities vs 3)
✅ Curiosity growth works (reactivates dormant clusters)
✅ Growth cap prevents collapse (500 cluster limit)
✅ Capped adversarial curriculum prevents starvation
✅ The architecture CAN produce structural differentiation
--- 2026-03-28 session ---
✅ Multi-round attention (convergence) improves spatial (0.54→0.58)
✅ Sequential curriculum is the biggest lever (communities 4→28, avg_sim 2.4x)
✅ Temporal co-firing enriches wiring signal
✅ Per-sample buffer update gives coherent within-episode priming
✅ Palate cleanser (buffer zero at episode boundary) fixes cross-category bleed
✅ Episodic memory (store high-error, replay worst categories) — active, TBD
✅ Negatives are early-training thrashing, resolve by 40-60K on mature graphs
✅ The FF signal ceiling is NOT 0.375 — distributed error + sequential broke it
```

## What's Disproven

```
❌ Momentum helps FF learning (cancels in 50/50 alternation)
❌ Residual connections work at small f(x) magnitude
❌ Inhibitory edges create competition (creates monopoly)
❌ Multi-head resonance on pre-trained embeddings (CLIP not axis-aligned)
❌ Any multiplicative dampening of FF signal
❌ Per-cluster sign from step 0 (noisy before specialization)
❌ Teacher direction as update target (kills per-cluster uniqueness)
❌ Topology alone encodes knowledge (weights are the knowledge)
--- 2026-03-28 session ---
❌ Global 512x512 projection (cross-category interference, whipsaw with momentum)
❌ Per-cluster lens from step 0 (disrupts spatial formation on young graphs)
❌ Momentum on projection updates (amplifies sequential episode oscillation)
❌ "Frosted window" hypothesis — graph learns fine in CLIP space, projection not needed
```

## The Fundamental Discovery

```
FF learning is a MINIMUM-VIABLE-SIGNAL system.

The binary +/- signal works BECAUSE:
  1. Each cluster learns from its OWN unique input
  2. Updates are at FULL magnitude every step
  3. Per-cluster input uniqueness IS the differentiation engine

Anything that:
  - Dampens magnitude → kills learning
  - Injects shared signals → kills diversity
  - Replaces per-cluster input with global target → kills differentiation

The signal cannot be made "smarter" by conventional means.
It can only be made richer by preserving per-cluster uniqueness.
```

---

## Roadmap Status

```
Phase A: Change what goes IN and what PERSISTS
  A.1  Adversarial curriculum (capped) ........ ✅ DONE + FIXED
  A.2  Memory buffer .......................... ✅ DONE (0.012 → 0.375)

Phase B: Change the INTERNAL structure
  B.1  Multi-prototype resonance .............. ✅ DONE (2x spatial)
  B.2  Curiosity-driven growth ................ ✅ DONE
  B.3  Multi-head resonance ................... ❌ FAILED (needs learned projections)

Phase C: Change the RULES of the game
  C.1  Cluster roles .......................... ✅ DONE (labels, detector/integrator/predictor)
  C.2  Typed edges (inhibitory) ............... ❌ FAILED (winner-take-all)
  C.2' Typed edges (excitatory only) .......... ✅ DONE (edge_type field exists)

Signal Enrichment: Make FF signal richer
  Exp 1  Per-cluster sign .................... ❌ FAILED (noisy early)
  Exp 2  Error direction ..................... ❌ FAILED (kills uniqueness)
  Exp 3  Contrastive pairs ................... ~ MARGINAL (fixable bug)
  Exp 4  Multi-target (0.5x) ................ ✓ CONDITIONAL (12 communities!)
  Exp 5  Structure reuse .................... ❌ FAILED (topology ≠ knowledge)

Transformer-Inspired:
  Residual connections ....................... ❌ FAILED (f(x) << x)
  Momentum .................................. ❌ FAILED (cancels in +/- alternation)
  Activation normalization ................... ✅ DONE (stabilizes signal)

Phase D: Input Adaptation (2026-03-28)
  D.1 Global projection (3 variants) ........ ❌ FAILED (cross-category)
  D.2 Per-cluster lens (2 variants) ......... ❌ FAILED (hurts spatial early)
  D.3 Conclusion: CLIP space is fine, no projection needed

Phase D': Sequential Learning (2026-03-28) — THE BREAKTHROUGH
  D'.1 Sequential curriculum (16-item episodes) ✅ DONE (communities 4→28)
  D'.2 Temporal co-firing ................... ✅ DONE (cross-step pairs)
  D'.3 Palate cleanser ...................... ✅ DONE (buffer zero at boundaries)
  D'.4 Episodic memory ...................... ✅ DONE (store/replay, active)
  D'.5 Per-sample buffer update ............. ✅ DONE (coherent priming)

Phase E: Language Grounding — PLANNED
  E.1 Grounded word embeddings (CLIP-bootstrapped) ... ○ planned
  E.2 Developmental staging (1→2→4 words) ............ ○ planned
  E.3 Decoder training (currently never called!) ...... ○ planned

Phase F-G: NOT YET STARTED
  F  Agency (curiosity-driven exploration) ... ○
  G  Environment interaction ................. ○
```

---

## The Honest Assessment

### What we achieved
- 49x improvement in spatial score (0.012 → 0.591 and climbing)
- 28 communities on mature graph (was 1 blob for 170K steps)
- 49 categories tracked (all COCO categories)
- avg_sim 0.18 for best categories (was 0.076)
- Deep understanding of FF learning dynamics
- Comprehensive RCAs for every failure (including 5 projection variants)

### What we didn't achieve
- Zero negatives from fresh start (5 mild negatives at 50K, shrinking)
- Community formation from fresh start is slow (2 at 50K vs 28 on mature graph)
- Language output (decoder was never trained — discovered this session)

### The ceiling (REVISED)
The FF binary signal ceiling of 0.375 was broken by distributed error +
sequential curriculum. Spatial 0.59+ with room to grow. The ceiling was not
the signal richness — it was the curriculum (random vs sequential) and the
error distribution (global vs per-cluster). Richer curriculum > richer signal.

---

## What's Next

### Current run (step 50K, no resets planned)
- Let it mature to 80-100K
- Watch negatives resolve (expected by 60K based on prior data)
- Watch communities split (need more co-firing history)
- If negatives persist at 80K: add per-cluster lens to mature graph (not from step 0)

### Language grounding (Phase E, planned)
- Replace broken TextDecoder with CLIP-bootstrapped grounded word embeddings
- Nearest-neighbor retrieval: "what words describe what I'm seeing?"
- Developmental staging: 1 word → 2 words → 4 words
- ACTUALLY TRAIN the decoder (train_step was never called — found this session)

### After language
- Agency: baby chooses what to look at (curiosity-driven curriculum)
- Interaction: coherent multi-turn dialogue

---

## Key Documents

```
docs/
  architecture-genealogy.md            — Visual evolution from blob to brain
  session-2026-03-28.md                — Full session log
  where-we-are.md                      — This document
  design/                              — Design docs
  experiments/                         — RCAs and experiment results
  archive/v1/, archive/v2/             — Historical (superseded)
```
