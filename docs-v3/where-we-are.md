# Where We Are — Grand Scheme Assessment

*Updated 2026-03-26 after completing all Phase A-C experiments and 5 signal enrichment tests.*

---

## The Journey in Numbers

```
v2.2 baseline:   spatial 0.012, 1 community    (the blob)
v3 confirmed:    spatial 0.375, 7 communities   (31x improvement)
Exp 4 peak:      spatial 0.157, 12 communities  (most structural diversity)
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

Phase D-F: NOT YET STARTED
  D  Learned encoder (replace CLIP) .......... ○
  E  Temporal reasoning + prediction ......... ○
  F  Environment interaction ................. ○
```

---

## The Honest Assessment

### What we achieved
- 31x improvement in spatial score (0.012 → 0.375)
- 7 stable communities (was 1 blob for 170K steps)
- 32 categories tracked with diverse curriculum
- Deep understanding of FF learning dynamics
- Comprehensive RCAs for every failure

### What we didn't achieve
- Spatial target of 0.50 (stuck at 0.375)
- Community target of 20 (stuck at 7, Exp 4 hit 12 but traded spatial)
- Category similarity above 0.20

### The ceiling
The FF binary signal has a hard ceiling around spatial 0.375, 7 communities.
This is not an architecture problem — it's a signal richness problem.
The binary +/- signal carries ~1 bit per step. More structure requires more
information per step, which requires a richer learning signal.

---

## What This Means for Phase D+

### The fork in the road

```
Option 1: Stay with FF, enrich the signal carefully
  ─────────────────────────────────────────────────
  - Tune Exp 4 (multi-target at 0.05x) for communities without spatial loss
  - Fix Exp 3 (contrastive pairs, stateless)
  - NEW: self-referential signals (per-cluster novelty, neighbor contrast)
  - Estimated ceiling: spatial ~0.45, communities ~15

  Pro: preserves the FF philosophy (no backprop, local learning)
  Con: may never reach 0.50+ spatial with binary signal

Option 2: Hybrid approach — keep architecture, upgrade signal
  ─────────────────────────────────────────────────
  - Replace binary FF with continuous contrastive signal
  - Each node gets a real-valued error, not just +/-
  - Still local (no backprop through graph), but richer per-node
  - Essentially: local contrastive learning on the FF architecture

  Pro: preserves self-growing graph, interpretability, sparsity
  Con: moves away from pure FF, needs careful design

Option 3: Phase D — learned encoder
  ─────────────────────────────────────────────────
  - Replace CLIP with a small learned encoder
  - The brain learns to see, not just sort
  - Requires richer signal (can't learn an encoder on binary +/-)
  - Implies Option 2 is a prerequisite

  Pro: true developmental AI — learns representations from scratch
  Con: massive undertaking, requires solving Option 2 first
```

### Recommendation

**Option 1 first** — spend one more session tuning the enrichment experiments.
If spatial doesn't break 0.45 with the refined experiments, **move to Option 2**
(hybrid contrastive signal). Option 3 depends on Option 2 succeeding.

The architecture is proven. The learning rule is the bottleneck. The question
is whether the rule can be improved within the FF framework or needs to be
replaced with something stronger.

---

## Key Documents

```
docs-v3/
  reference-modern-ml.md              — Transformer/LLM architecture reference
  evaluation-against-modern-ml.md      — Baby AI vs transformer comparison
  RCA-failed-optimizations.md          — 6 detailed root cause analyses
  experiment-plan-signal-enrichment.md — Test protocol for 5 experiments
  experiment-results-signal-enrichment.md — Results + analysis
  where-we-are.md                      — This document

docs-v2/
  ROADMAP.md                           — Original phase plan (A-F)
  ARCHITECTURE-CURRENT.md              — Technical architecture
  ARCHITECTURAL-LIMITS-AND-SOLUTIONS.md — Game-theoretic analysis
```
