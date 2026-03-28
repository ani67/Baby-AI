# Root Cause Analyses: Failed Optimizations in Baby AI

*Six features proposed, tested, and rejected. Each seemed reasonable based on
transformer/backprop intuitions. Each failed because Forward-Forward learning
has fundamentally different signal dynamics.*

---

## Summary Table

| Feature | PR | Assumption | Result | Root Cause |
|---------|-----|-----------|--------|-----------|
| Momentum (0.9) | #13 | Smooth noisy updates | spatial -16%, communities -22% | 50/50 +/- signals cancel in EMA |
| Momentum (0.5) | #13 | Lower coefficient helps | same degradation | Any coefficient cancels alternating signals |
| Residual connections | #13 | Enable deep graph learning | spatial →0.05, 1 community | output ≈ input (node output too small vs x) |
| Inhibitory edges | #12 | Competition drives differentiation | spatial -66%, mega-blob | Winner-take-all monopoly, not competition |
| Multi-head resonance | #12 | Compositional matching | coverage 0.0 (dead) | CLIP dims not axis-aligned, scores diluted |
| Continuous FF signal | #14 | Scale by confidence | communities delayed | Dampens already-weak binary signal |
| Soft resonance | #14 | Weight by relevance | coverage collapse | Compounds with other dampeners |
| LR warmup | #14 | Prevent wild early swings | delayed learning | FF needs full signal from step 0 |
| Adaptive resonance width | #15 | Narrow when coverage high | same clusters every step → 1 community | Concentration prevents diversity |
| Per-cluster error LR | #15 | Bad clusters learn faster | amplified dominant clusters | Monopoly dynamics |
| Adversarial curriculum (uncapped) | v2.1 | Target weaknesses | 87% training on one category | Starvation spiral, no diversity |

---

## The Meta-Pattern

Every failure shares one of two root causes:

```
1. SIGNAL DAMPENING
   FF has a binary signal (+/-). It's already the minimum viable signal.
   Anything that reduces it — multiplicatively or through narrowing —
   pushes effective updates below the learning threshold.

   Failed: momentum, continuous signal, soft resonance, LR warmup,
           adaptive resonance width, per-cluster LR

2. WINNER-TAKE-ALL
   Mechanisms intended to create competition instead create monopolies.
   Without homeostatic guardrails, the strongest entity captures all
   resources and prevents competitors from emerging.

   Failed: inhibitory edges, adversarial curriculum (uncapped)
```

---

## RCA 1: Momentum

### Assumption
Smooth noisy weight updates. Standard in every optimizer (Adam β₁=0.9).

### Why it fails in FF
FF signal alternates ~50% positive, ~50% negative by design (adaptive threshold).
Momentum averages these opposite-direction updates toward zero.

```
Steady-state amplitude with coefficient β and alternating +/- signal:
  effective = (1 - β) / (1 + β) × raw_update

  β = 0.9: effective = 0.053 × raw  → 95% reduction
  β = 0.5: effective = 0.333 × raw  → 67% reduction
```

### Why it works in transformers
Backprop gradients point consistently toward the loss minimum. Consecutive
gradients are correlated. Momentum accelerates the consistent direction.
In FF, the first moment (mean direction) is ~zero by construction.

### Principle
**Any technique assuming temporal coherence of update direction fails in FF.**
The information lives in which-inputs-get-which-signs, not in the mean direction.

---

## RCA 2: Residual Connections

### Assumption
`output = normalize(x + f(x))` enables deep graph learning (proven at 96+ layers).

### Why it fails in Baby AI
Node-weighted output f(x) has magnitude ~0.01-0.1. Input x has magnitude ~1.0.
After addition: `x + f(x) ≈ x`. After normalization: output ≈ input direction.
Every cluster outputs the same thing (the input). Differentiation dies.

### Why it works in transformers
Transformer blocks have ~1M parameters per layer. f(x) has magnitude comparable
to x. The transformation is 15-45° rotation — large and meaningful.
Baby AI clusters have ~2K-4K parameters. f(x) is a whisper next to x.

### Principle
**Residual connections require f(x) ≈ ||x|| in magnitude.** In FF with small
local updates, f(x) << x, making the residual a passthrough, not a highway.

### Alternative
Scaled residual with small α: `normalize(α*x + f(x))` where α ≈ ||f(x)||/||x||.
Or: accept the shallow architecture — differentiation works with 1-2 hops.

---

## RCA 3: Inhibitory Edges

### Assumption
Similar clusters should compete. Subtracting signal forces differentiation.

### Why it fails
Creates a positive feedback loop:
1. Tiny asymmetry between two similar clusters
2. Stronger cluster A suppresses weaker B more effectively
3. B gets less activation → less Hebbian reinforcement → weaker edges
4. A gets more activation → more reinforcement → stronger
5. B is functionally silenced. A becomes a monopolist.

### Paradox
MORE communities (8 vs 4) but WORSE spatial (0.098 vs 0.289). The extra
communities are debris — orphaned clusters ejected by suppression, not
meaningful functional groups. One mega-blob of 117, rest are fragments.

### Principle
**Competition without homeostatic regulation is monopoly.** Biology prevents
this with synaptic scaling, metabolic constraints, bounded receptive fields.
Baby AI lacks all three.

### Alternative
Identity repulsion (push similar identities apart without suppressing activation)
or winner-take-all within normalized local groups (k-WTA per input).

---

## RCA 4: Multi-Head Resonance

### Assumption
CLIP's 512 dims encode different semantic aspects (subject, action, scene).
Splitting into 4×128 heads enables compositional matching.

### Why it fails
CLIP embeddings are rotationally symmetric — no privileged basis. Dims 0-127
don't encode "subject." Every dimension participates in every concept.
Splitting along coordinate axes cuts an arbitrary cake along arbitrary lines.

Coverage dropped to 0.0 because partial dot products have:
- Lower signal magnitude (1/4 each)
- Higher relative noise (√(128) vs √(512) concentration)
- No guarantee of semantic alignment

### Why it works in transformers
Transformer attention uses LEARNED projections (W_Q, W_K) that rotate the
space before splitting. Each head discovers its own meaningful subspace
through gradient descent. Fixed axis-aligned splits assume structure
that doesn't exist.

### Principle
**Don't decompose a space you didn't construct.** Meaningful subspace
decomposition requires learned projections, not coordinate slicing.

---

## RCA 5: Signal Dampeners (Compound Effect)

### The five dampeners
1. Continuous FF signal: magnitude × signal_strength [0, 0.5]
2. Soft resonance: activation × resonance_score [0, 1]
3. LR warmup: LR × warmup_ratio [0, 1]
4. Adaptive resonance width: narrows K from 20 to 15
5. Per-cluster error LR: × error_boost [0.5, 2.0]

### The multiplication problem
```
effective_update = LR × warmup × signal_strength × soft_resonance × cluster_lr
                 = 0.001 × 0.2  × 0.15          × 0.4            × 0.7
                 = 0.0000084

Baseline (no dampeners): 0.001
Reduction: 119x
```

Each dampener reduces magnitude by 0.2-0.7x. Four dampeners compose to
0.008x. The signal drops below the threshold of functional learning.

### Why FF is uniquely sensitive
- Binary signal (no gradient magnitude)
- No error backflow (each node learns alone)
- No optimizer state to compensate (no Adam denominator)
- Already has intrinsic dampener: (1 - activation²) ≈ 0.3-0.6
- Sparse activation means fewer learning opportunities per step

### Principle
**In FF, the update signal is the scarcest resource. Never multiply it by
a [0,1] factor.** Prefer additive bonuses (×[1.0, 1.5]) over multiplicative
dampening (×[0, 1]).

---

## RCA 6: Adversarial Curriculum Starvation

### Assumption
Oversample worst categories to improve them.

### Why it fails
```
weight = max(0.1, 1.0 - avg_sim / max_sim)
limit = int(32 * weight)

Worst category: limit = int(32 * 0.9) = 28 slots out of 32
```

"Tennis" got 87% of all training (6,794 presentations). Only 13 of 80+
categories ever tracked. The worst category stays worst because:
- No contrastive diversity (can't learn what tennis IS NOT)
- Other categories starved (never appear, never get tracked)
- Positive feedback: worst → more exposure → stays worst

### Connection to inhibitory edges
Both are winner-take-all dynamics:
- Inhibitory edges: strongest cluster monopolizes activation
- Adversarial curriculum: worst category monopolizes training

Both need the same fix: caps + guaranteed diversity.

### Fix applied
- Cap per-category weight at 25%
- 50% of batch always random (diversity guarantee)
- Result: 32 categories tracked (was 13), spatial computable

### Principle
**Adversarial/corrective feedback without guardrails is indistinguishable
from a positive feedback loop.** Always cap and always guarantee a baseline.

---

## What Survived

| Feature | PR | Why it works |
|---------|-----|------------|
| Memory buffer | #6 | Modifies INPUT, not update magnitude. Adds temporal context. |
| Multi-prototype resonance | #8 | Modifies SELECTION, not update. Max over nodes = more selective matching. |
| Activation normalization | #13 | Stabilizes INPUT to nodes. Preserves magnitude information. |
| Curiosity growth | #12 | Modifies STRUCTURE, not signal. Reactivates dormant clusters. |
| Cluster roles | #12 | Labels only, no behavioral change. |
| Growth cap (500) | #12 | Prevents performance collapse, no signal change. |
| Adversarial curriculum (capped) | fix | 50% adversarial + 50% random. Targets weakness without monopoly. |

### The pattern
Everything that survived modifies **what the system sees** (input, selection,
structure) without touching **how strongly it learns** (update magnitude).
The FF learning signal is sacred — route it, select it, structure around it,
but never scale it down.

---

## Implications for Next Steps

### What's confirmed working (spatial 0.375, 7 communities, 32 categories)
The current stack achieves 31x improvement over the original v2.2 baseline
(spatial 0.012 → 0.375). The blob problem is solved. Multiple communities
form and grow. Categories are tracked diversely.

### What the failures tell us about Phase D+
1. **Multi-head needs learned projections** — can't decompose CLIP without them
2. **Deep graph learning needs a different approach** — not residual connections
3. **Competition needs homeostasis** — not raw inhibition
4. **Optimizer improvements need FF-native design** — not borrowed from backprop
5. **Curriculum design is a control problem** — needs bounds and diversity guarantees

### The fundamental FF insight
FF learning is a **minimum-viable-signal** system. The binary +/- signal carries
just enough information to drive differentiation when transmitted at full strength.
Every optimization must be evaluated against: "Does this preserve full signal
strength?" If the answer is "it reduces it, even slightly" — it will compound
with other reductions and kill learning.

This is the opposite of the backprop world, where the problem is usually too
MUCH gradient and the design philosophy is regularization. FF's design philosophy
must be **signal preservation and amplification**.
