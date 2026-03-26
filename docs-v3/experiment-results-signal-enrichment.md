# FF Signal Enrichment — Experiment Results

*Five experiments to enrich the binary FF signal. Three failed, one marginal, one conditional pass. Each produced insights about how FF learning actually works.*

---

## Scorecard

| Exp | Feature | Spatial | Communities | vs Baseline | Verdict |
|-----|---------|---------|-------------|-------------|---------|
| — | Baseline (none) | 0.375 | 7 | — | CONTROL |
| 1 | Per-cluster sign | 0.137 | 6 | -63% / -14% | ❌ |
| 2 | Error direction | None | 4 | — / -43% | ❌ |
| 3 | Contrastive pairs | 0.180 | 7 | -52% / = | Marginal |
| 4 | Multi-target | 0.157 | 12 | -58% / +71% | Conditional ✓ |
| 5 | Structure reuse | 0.023 | 1 | -94% / -86% | ❌ |

---

## Detailed Analysis

### Exp 1: Per-Cluster Sign ❌

**What:** Each cluster gets own +/- from its output vs teacher.

**Positive learnings:**
- The concept is sound — targeted feedback per cluster is better than global binary
- Communities still formed (6), proving per-cluster sign isn't fundamentally broken

**Negative learnings:**
- When clusters haven't specialized, their output vs teacher is random noise
- Per-cluster sign adds entropy early, signal late

**Root cause:** Implementation wrong for early training. Per-cluster sign should activate LATE (after 5K+ steps when clusters have identities), not from step 0.

**Future fix:** Staged approach — global sign early → per-cluster late.

---

### Exp 2: Error Direction ❌

**What:** Push toward teacher answer direction, not just input.

**Positive learnings:**
- Teacher vector IS more informative than the input
- The idea of providing directional guidance is sound

**Negative learnings:**
- ALL clusters push toward the SAME teacher answer → collapses diversity
- In baseline, each cluster gets unique combined input (x + edge signals), making updates naturally per-cluster-unique

**Root cause:** Replaces per-cluster-unique input direction with shared target direction. Kills the differentiation mechanism.

**Future fix:** Blend input and teacher: `update_dir = normalize(0.8 * input + 0.2 * teacher)`. Preserves per-cluster uniqueness while nudging toward teacher.

---

### Exp 3: Contrastive Pairs — Marginal

**What:** Rank pairs within batch instead of threshold.

**Positive learnings:**
- Matched community count (7)! Good community distribution (120, 21, 13, 8, 5)
- Relative ranking produces cleaner +/- decisions than adaptive threshold
- Communities formed at baseline rate

**Negative learnings:**
- Extra forward passes for ranking corrupted model state (_last_visited, _last_outputs overwritten)
- Spatial suffered (0.180 vs 0.375) due to implementation bug, not concept

**Root cause:** Implementation bug — ranking forward passes overwrote model state before training forward pass.

**Future fix:** Run ranking forward passes WITHOUT storing state (stateless evaluation), then do the real forward pass. Deserves a clean re-test.

---

### Exp 4: Multi-Target — Conditional ✓

**What:** Additive bonus update toward teacher direction (0.5x LR).

**Positive learnings:**
- **BEST COMMUNITY COUNT EVER: 12** (vs baseline 7, +71%)
- Three groups with 20+ members — real structural diversity
- Extra update energy drives churn that fragments the blob more aggressively
- PROVES more communities are achievable

**Negative learnings:**
- Spatial drops to 0.157 (-58%) because bonus pushes all clusters toward shared teacher answer
- 0.5x bonus is too strong — overwhelms per-cluster uniqueness

**Root cause:** Same shared-target problem as Exp 2, but weaker (additive not replacement). The strength matters — at 0.5x the teacher signal is too dominant.

**Future fix:** Reduce bonus to 0.05x-0.1x. Just a whisper of teacher direction on top of the full input-based update. Find the sweet spot: maximum communities without tanking spatial.

---

### Exp 5: Structure Reuse ❌

**What:** Load topology from trained model, restart with fresh weights.

**Positive learnings:**
- Definitive answer: **topology alone is NOT knowledge.** Weights are the knowledge, structure is scaffolding.
- This is a fundamental insight about the architecture.

**Negative learnings:**
- 507 random clusters competing from step 0 is worse than 4 growing organically
- Gradual growth lets each new cluster find its niche
- Pre-built structure fitted to DIFFERENT weights is useless

**Root cause:** Growth is part of learning. Topology and weights co-evolve — BUD splits a confused cluster at exactly the right moment, INSERT adds a bridge where one is needed. The structure is fitted to the weight state that produced it.

**Future fix:** Test topology + PARTIAL weights — keep high-level cluster weights, randomize only bottom layers. Tests if high-level structure transfers while low-level features relearn.

---

## The Meta-Pattern

```
WHAT HELPS FF LEARNING                WHAT HURTS FF LEARNING
──────────────────────────────────── ────────────────────────────────
Modify what system SEES               Modify update DIRECTION
  ✅ buffer (temporal input)             ❌ error direction (shared target)
  ✅ multi-prototype (selection)         ❌ per-cluster sign (noisy early)
  ✅ activation norm (stability)

Modify STRUCTURE                      Modify update MAGNITUDE
  ✅ curiosity growth                    ❌ momentum (cancels 50/50)
  ✅ growth cap                          ❌ all dampeners
  ✅ capped adversarial curriculum       ❌ continuous signal

Add ENERGY (carefully)                Replace UNIQUENESS
  ✓ multi-target at 0.5x (12 comms)    ❌ teacher replaces input direction
  but spatial trades off               ❌ pre-built topology (no fitted weights)
```

## The Core Insight

The FF binary signal has a specific strength: **it lets each cluster learn from its OWN unique input.** This per-cluster uniqueness IS the differentiation engine.

Anything that injects shared information (teacher direction, global ranking) reduces per-cluster uniqueness and hurts spatial organization.

The path to richer signal is NOT "give each cluster better information about the teacher."

The path IS "give each cluster better information about ITSELF":
- How well did MY specific output match?
- How different am I from my neighbors?
- What inputs do I uniquely respond to that others don't?

These are **self-referential signals**, not teacher-referential. They preserve per-cluster uniqueness while adding information.

---

## Proposed Next Experiments

### Fix 3: Contrastive Pairs (stateless)
Ranking is sound. Fix the state corruption by running evaluation forward passes without storing _last_visited/_last_outputs.

### Tune 4: Multi-Target at 0.05x
12 communities is real. Find the sweet spot where communities stay high but spatial recovers.

### NEW: Self-Referential Novelty
Per-cluster "novelty" — how different is this input from what this cluster usually sees? Update more for novel inputs, less for familiar. Per-cluster, not global.

### NEW: Neighbor Contrast
Each cluster gets a bonus update AWAY from its most similar neighbor's direction. Drives differentiation without shared teacher signal. Local, per-cluster, preserves uniqueness.
