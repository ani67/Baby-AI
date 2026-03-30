# Distributed Brain with REFLECT — Design Doc

## Status: EXPERIMENTAL (feature branch, not merged)

Compare against current architecture. Merge if metrics improve; delete if not.

---

## Problem Statement

The current brain architecture has a fundamental scaling problem:

```
Current: 1 neuron ≈ 1 concept (localist representation)
  - 44K training items → needs ~44K neurons → OOM on 16GB M1 Pro
  - Growth is the primary learning mechanism (bud neurons for new concepts)
  - Forward-Forward (FF) learning is local — no coordination between neurons
  - 15 edges/neuron — massively under-connected
  - Growth check at 8K neurons takes 20 seconds
  - Brain OOM-killed at ~20K total neurons
```

## Proposed Architecture

Shift from **localist** (one neuron = one concept) to **distributed** (one concept = pattern across many neurons):

```
Proposed: 1 concept = activation pattern across many neurons
  - Fixed neuron budget (10K max, developmental growth only)
  - Dense edge network (200+ edges/neuron) — connections ARE the intelligence
  - Bidirectional edge flow (REFLECT) — backward error propagation
  - Same memory budget (~44MB), vastly more representational capacity
```

### Memory Comparison

```
CURRENT (localist):
  20,000 neurons × 512 dims × 4 bytes  = 40MB (weights)
  300K edges × 12 bytes                 =  3.6MB (edges)
  Total: ~44MB — and growing, will OOM

PROPOSED (distributed):
  10,000 neurons × 512 dims × 4 bytes  = 20MB (weights, FIXED)
  2M edges × 12 bytes                   = 24MB (edges, bounded)
  Total: ~44MB — bounded forever
```

## Architecture Components

### 1. Fixed Neuron Budget with Developmental Phases

```
Phase 1 — Foundation (steps 0-50K):
  Rapid neuron budding up to MAX_NEURONS (10,000)
  Build the raw substrate. Establish initial specializations.

Phase 2 — Refinement (steps 50K+):
  Zero neuron budding. Fixed population.
  All growth = edge formation.
  Rare neuron recycling (overwrite truly dead neurons).
```

### 2. Dense Edge Network

Edges are the primary learning substrate, not neurons.

```
Current:  15 edges/neuron  →  "filing cabinet" (one drawer per concept)
Proposed: 200+ edges/neuron →  "city street network" (concepts = routes)

Edge formation triggers:
  - Co-firing (existing mechanism, keep)
  - Backward error correlation (NEW: REFLECT-guided)
  - Random exploration (small chance of connecting distant neurons)

Edge pruning:
  - Unused edges (low traffic in both directions) decay and die
  - Prevents unbounded edge growth
  - Target: ~200 edges/neuron steady state
```

### 3. REFLECT — Backward Error Flow

After the forward pass produces a prediction, error flows backward through the same edges:

```
FORWARD (existing):
  Input → SENSE → FIRE → THINK (message passing) → OUTPUT → prediction

REFLECT (new):
  error = target - prediction
  For each layer, top to bottom:
    For each edge (i → j) where j is in current layer:
      error[i] += edge_strength(i,j) × error[j]

  Same edges. Same strengths. Reversed direction.
```

REFLECT provides:
- **Directional correction**: each neuron knows which way to shift
- **Magnitude**: how much to shift
- **Cross-layer coordination**: output errors reach input layers

Limitations:
- Not exact gradients (no activation derivatives)
- Signal attenuates through sparse hops (why dense edges matter)
- Approximation improves as edge density increases

### 4. Updated Learning Rule

```
Current FF update:
  Δw = local_goodness_signal

Proposed update:
  Δw = α × local_goodness_signal + β × reflect_error_signal

  α, β tunable. Start with α=0.7, β=0.3
  (keep FF dominant, REFLECT as guidance)
```

### 5. Edge-Guided Growth

```
Current:  "High error on input X → bud new neuron near X"
Proposed: "High error on input X → form new edges between neurons
           that could COLLECTIVELY represent X"

Edge formation during REFLECT:
  If neuron A has high backward error AND neuron B has high activation
  AND no edge exists between them → create edge(A, B)

  "Neurons that SHOULD coordinate get wired together"
```

## Implementation Plan

### Phase 1: BrainV2 class (new file, no existing code changes)

```
backend/model/brain_v2.py — new BrainState subclass or parallel class
  - Same forward() interface
  - Added reflect(error) method
  - Modified update() using both FF + REFLECT signals
  - New edge formation logic (density-aware)
  - Developmental growth phases
  - Neuron recycling
```

### Phase 2: Benchmark harness

```
backend/tests/test_brain_v2.py — comparison tests
  - Same training data, same curriculum
  - Run both BrainState (current) and BrainV2 (proposed)
  - Compare after N steps:
    - Cosine similarity on held-out items
    - Category accuracy (per-category)
    - Memory usage
    - Steps per second
    - Neuron utilization (% of neurons that fire regularly)
    - Edge efficiency (useful edges / total edges)
```

### Phase 3: Integration (only if Phase 2 passes)

- Swap BrainState → BrainV2 in train_worker.py
- Full training run with metrics comparison

## Success Criteria

The new architecture MUST beat the current one on:

```
1. Memory bounded:     Never exceeds 50MB brain state (current: OOM at ~44MB)
2. Learning quality:   Category avg_sim ≥ current after same step count
3. Throughput:         Steps/sec ≥ current (dense edges vs more neurons)
4. Stability:          No OOM, no crashes over 100K steps
5. Neuron efficiency:  Higher % of neurons actively contributing
```

Nice to have:
- Vision categories improve (currently near-zero)
- Reasoning accuracy ≥ current
- Text distill unaffected

## Failure Modes

```
1. REFLECT signal too noisy     → β too high, learning destabilizes
   Mitigation: start β=0.1, tune up gradually

2. Dense edges = slow forward   → 2M edges in message passing
   Mitigation: sparse matrix ops, limit THINK rounds

3. Fixed neurons insufficient   → 10K not enough for 44K concepts
   Mitigation: that's the whole hypothesis — distributed reps
   should handle this. If not, increase to 15K.

4. Edge memory exceeds budget   → 200 edges/neuron × 10K = 2M × 12B = 24MB
   Mitigation: edge pruning keeps density at target

5. FF + REFLECT interfere       → two learning signals fight
   Mitigation: α/β tuning, or alternate (FF on even steps, REFLECT on odd)
```

## References

- Hinton (2022): "The Forward-Forward Algorithm"
- Lillicrap et al. (2016): "Random synaptic feedback weights support error backpropagation for deep learning" (feedback alignment)
- Predictive coding: Rao & Ballard (1999)
- Equilibrium propagation: Scellier & Bengio (2017)

## Decision

After benchmark results:
- **Metrics better**: merge PR, migrate training to BrainV2
- **Metrics worse or equal**: close PR, keep current architecture, revisit approach
