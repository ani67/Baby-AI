# Baby AI: Evaluation Against Modern ML Principles

*What the transformer/LLM world got right that we can learn from, what we got right that they don't have, and concrete improvements.*

---

## Side-by-Side Architecture Comparison

```
                    Transformer/GPT              Baby AI
                    ─────────────────            ─────────────────
Learning rule       Backprop + Adam              Forward-Forward (local)
Communication       Attention (soft, learned)    Resonance (hard, cosine)
Computation         MLP per position             Node activation per cluster
Depth               Residual blocks × N          BFS through layer graph
Normalization       RMSNorm/LayerNorm            L2 weight normalization
Position encoding   RoPE / learned               Layer index (implicit)
Memory              KV-cache (exact)             Buffer (decaying echo)
Sparsity            MoE (top-K experts)          Resonance (top-20 clusters)
Growth              Fixed architecture            BUD/INSERT/EXTEND/PRUNE
Tokenization        BPE subwords                 CLIP 512-d vectors
Output              Next-token distribution       Nearest cluster identity
```

---

## What Transformers Got Right That Baby AI Is Missing

### 1. RESIDUAL CONNECTIONS — Critical Gap

```
Transformer: x = x + attention(x)     ← gradient highway
             x = x + mlp(x)

Baby AI:     output = cluster.forward(x, edges)   ← no skip connection
             next_cluster = forward(output, ...)    ← signal degrades per layer
```

**Impact**: Baby AI has 14 layers but signal degrades through each. Transformers train 96+ layers because residuals let gradients flow unimpeded. Without residuals, deeper clusters learn slower than shallow ones.

**Fix**: Add residual connection in cluster.forward:
```python
# Before:
output = F.normalize(weighted_sum_of_nodes, dim=0)

# After:
output = F.normalize(x + weighted_sum_of_nodes, dim=0)
```

**Effort**: 1 line. **Impact**: Potentially large — enables deeper graph learning.

### 2. SEPARATION OF COMMUNICATION AND COMPUTATION

```
Transformer: attention = communication (tokens talk)
             MLP       = computation (tokens think)
             Two distinct mechanisms, both needed.

Baby AI:     cluster.forward = does BOTH at once
             incoming edges = communication
             node activations = computation
             But they're fused into one step, not composed.
```

**Impact**: In transformers, attention decides WHAT information to gather, then MLP decides WHAT TO DO with it. Baby AI mixes these — the combined signal goes into nodes that both receive and process simultaneously.

**Fix**: This maps to the cluster roles design (C.1). Detectors ARE the communication layer, Integrators ARE the computation layer. The fix is already in v2.6 structurally — but the learning rules don't yet differentiate. Future: integrator clusters should weight edge signals more heavily than raw input.

### 3. LEARNED ATTENTION WEIGHTS (vs Fixed Cosine)

```
Transformer: attention weights = softmax(Q·K^T/√d)
             Q, K are LEARNED projections — the model decides what matters
             Attention is soft — all tokens contribute, weighted

Baby AI:     resonance = cos(node_weights, input)
             No learned projection — raw cosine similarity
             Resonance is hard — top-20 only, rest ignored
```

**Impact**: Transformers learn WHAT to attend to. Baby AI uses fixed cosine similarity — it can't learn that "for this input, attend to color features" vs "for that input, attend to shape features." The multi-head split (B.3) helps but heads are fixed subspaces, not learned.

**Fix (future Phase D+)**: Add a small learned projection before resonance:
```python
# Before: resonance = dot(node.weights, input)
# After:  resonance = dot(node.weights, projection(input))
# Where projection is a (512, 512) matrix trained via FF
```

This would let the model learn what aspects of the input are relevant for each cluster.

### 4. PROPER NORMALIZATION

```
Transformer: RMSNorm before every block — keeps activations stable
             Applied BEFORE attention and BEFORE MLP

Baby AI:     L2 normalization on weights (permanent)
             No activation normalization between layers
             No normalization of edge signals
```

**Impact**: As signal passes through layers, activation magnitudes can drift. This contributes to the saturation issues seen in early runs.

**Fix**: Add RMSNorm (or simple L2 norm) to the combined signal in cluster.forward before node activation:
```python
combined = F.normalize(combined, dim=0)  # stabilize before activation
```

**Effort**: 1 line. **Impact**: Medium — prevents activation drift in deep graphs.

### 5. PROPER OPTIMIZATION (Adam vs Raw FF)

```
Transformer: Adam with momentum + adaptive LR per parameter
             LR warmup + cosine decay
             Gradient clipping

Baby AI:     FF update = sign * plasticity * lr * activation * input * (1-act²)
             No momentum, no per-parameter adaptation
             LR decays but no warmup
```

**Impact**: Adam converges faster and more reliably because it adapts the learning rate per parameter. FF's fixed-rate update can oscillate or get stuck.

**Fix**: Add momentum to FF update:
```python
# In node.ff_update:
self._momentum = 0.9 * self._momentum + 0.1 * update
self.weights = self.weights + self._momentum
```

**Effort**: 3 lines + 1 field. **Impact**: Medium — smoother weight trajectories.

---

## What Baby AI Got Right That Transformers Don't Have

### 1. SELF-GROWING ARCHITECTURE
Transformers have fixed depth, width, attention heads. Baby AI grows its own structure. This is a genuine advantage — no architecture search needed.

### 2. LOCAL LEARNING (No Backprop)
FF updates are biologically plausible and don't require storing full computation graphs. Scales differently — no memory-proportional-to-depth requirement.

### 3. INTERPRETABLE STRUCTURE
You can watch the graph grow, see which clusters fire, trace activation paths. Transformers are opaque — attention maps are post-hoc explanations, not the structure itself.

### 4. SPARSE ACTIVATION
Only 20 out of 500 clusters fire per step. This is natural MoE without the routing complexity. Transformers needed MoE as an add-on; Baby AI has it by design.

### 5. INHIBITORY EDGES
Transformers have no inhibition — all attention weights are positive (softmax). Baby AI's inhibitory edges enable competitive dynamics that transformers lack.

---

## Concrete Improvements (Priority Order)

### HIGH IMPACT, LOW EFFORT

| # | Change | Lines | Expected Impact |
|---|--------|-------|-----------------|
| 1 | **Residual connections** in cluster.forward | 1 | Deep layers actually learn |
| 2 | **Activation normalization** before node activation | 1 | Prevents drift in deep graphs |
| 3 | **Momentum in FF update** | 4 | Smoother convergence |

### MEDIUM IMPACT, MEDIUM EFFORT

| # | Change | Lines | Expected Impact |
|---|--------|-------|-----------------|
| 4 | **Separate communication and computation** phases in forward | ~20 | Cleaner signal flow |
| 5 | **Soft resonance** (weighted, not hard top-20) | ~15 | Gradual activation, less information loss |
| 6 | **LR warmup** (ramp up first 500 steps) | 5 | Stable early training |

### HIGH IMPACT, HIGH EFFORT (Future Phases)

| # | Change | Lines | Expected Impact |
|---|--------|-------|-----------------|
| 7 | **Learned resonance projection** | ~50 | Model learns what to attend to |
| 8 | **KV-cache analog** (store cluster outputs across steps) | ~80 | True working memory |
| 9 | **BPE-style tokenization** of CLIP features | ~100 | Compositional input representation |

---

## The Deepest Insight

### From microGPT:

> "The model has thousands of adjustable parameters. The optimizer nudges them incrementally to reduce loss."

Baby AI has the adjustable parameters (node weights) and the nudging (FF updates). What it's missing is the **loss landscape quality**. In a transformer:

```
Loss = -log(probability of correct next token)

This is a SHARP, SPECIFIC signal:
  "You said 'cat' but the answer was 'dog'"
  → every parameter gets a precise gradient toward 'dog'
```

In Baby AI:

```
is_positive = cosine_similarity > adaptive_threshold

This is a BINARY, VAGUE signal:
  "You were kinda close" or "You were kinda far"
  → every visited node gets the same direction (toward or away)
  → no per-node specificity
```

**This is the fundamental limit.** Not the architecture, not the resonance, not the growth — the learning signal itself. Transformers succeed because cross-entropy loss provides a rich, specific gradient per parameter. FF provides a uniform push/pull.

### What This Means for Phase D+

Before building learned encoders (Phase D) or temporal reasoning (Phase E), the highest-leverage change might be:

**Improve the learning signal granularity.**

Options:
1. **Contrastive FF**: instead of binary positive/negative, use the similarity score as a continuous signal
2. **Per-node targeting**: nodes that contributed most to the wrong output get stronger updates
3. **Prediction error as signal**: instead of teacher comparison, predict the next input (self-supervised)

These don't require backprop. They work within the FF framework but give each node a more informative update direction.

---

## Summary

```
What to steal from transformers:     What to keep from Baby AI:
─────────────────────────────────    ──────────────────────────────
✓ Residual connections (1 line)      ✓ Self-growing architecture
✓ Activation normalization (1 line)  ✓ Local learning (no backprop)
✓ Momentum in updates (4 lines)     ✓ Interpretable structure
○ Learned attention (Phase D+)       ✓ Natural sparse activation
○ Rich loss signal (Phase D+)        ✓ Inhibitory dynamics
○ KV-cache memory (Phase E)          ✓ Curiosity-driven growth
```

The 3 one-line fixes (residual, normalization, momentum) should be implemented immediately. They're proven at scale in every modern architecture and cost nothing.
