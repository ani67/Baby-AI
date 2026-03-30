# Baseline Metrics — Current BrainState (step 232K)

Captured 2026-03-30 for comparison against BrainV2.

## Brain Structure
- Active neurons: 8,141
- Dormant neurons: 10,977
- Total: 19,118
- Edges: 118,578 (~14.6 edges/active neuron)
- Layers: 20

## Distillation
- Text cos_sim: 0.802 (trend +0.033)
- Vision cos_sim: 0.740 (trend +0.037)

## Reasoning
- overall_accuracy: 1.0 (1 task — too few to be meaningful)

## Generation
- All null (probe not yet firing)

## Performance
- Scoring round (every 5th batch): ~6.7s prepare + 2.6s compute = ~9.4s
- Reuse batch: ~0.8-2.0s compute
- Growth check spike: up to 20s at 8K neurons
- Memory: ~44MB brain state, OOM-killed at ~20K total neurons

## Known Issues
- Growth unbounded → OOM
- 15 edges/neuron → under-connected
- No backward error flow → FF can't coordinate neurons
- Growth compensates for weak learning rule
