# FF Signal Enrichment — Experiment Plan

## Baseline
Spatial 0.375, 7 communities, 32 categories at ~12K steps.
All experiments OFF. This is the control.

## How to toggle experiments

```bash
# Check current state
curl -s localhost:8000/experiments | python3 -m json.tool

# Turn ON one experiment
curl -X POST localhost:8000/experiments -H 'Content-Type: application/json' \
  -d '{"exp_per_cluster_sign": true}'

# Turn OFF
curl -X POST localhost:8000/experiments -H 'Content-Type: application/json' \
  -d '{"exp_per_cluster_sign": false}'

# Save/load topology (Exp 5)
curl -X POST localhost:8000/topology/save
curl -X POST localhost:8000/topology/load
```

## Test Protocol

For each experiment:
1. Reset from frontend (clean start)
2. Toggle ONE experiment ON via API
3. Run to 10K steps
4. Check: spatial > 0.35? communities >= 7?
5. If YES → candidate for inclusion
6. If NO → experiment failed, document why
7. Toggle OFF, move to next

## Test Sequence

```
Test 1: Exp 1 — Per-Cluster Sign
──────────────────────────────────────────────────
What: Each cluster gets its own +/- based on whether
      ITS output matches the teacher, not the global signal.
Toggle: curl -X POST localhost:8000/experiments \
  -d '{"exp_per_cluster_sign": true}'
Expect: Better spatial (targeted learning), same or more communities.
Risk: Low — same magnitude, different direction.

Test 2: Exp 2 — Error Direction
──────────────────────────────────────────────────
What: For positive examples, push weights toward TEACHER
      ANSWER direction instead of input direction.
Toggle: curl -X POST localhost:8000/experiments \
  -d '{"exp_error_direction": true}'
Expect: Higher spatial (512-d direction vs 1-bit sign).
Risk: Medium — changes what the model optimizes toward.

Test 3: Exp 3 — Contrastive Pairs
──────────────────────────────────────────────────
What: Pair up batch samples, better one gets +, worse gets -.
      Eliminates threshold noise.
Toggle: curl -X POST localhost:8000/experiments \
  -d '{"exp_contrastive_pairs": true}'
Expect: More stable signal, possibly faster community formation.
Risk: Low — relative ranking is cleaner than absolute threshold.

Test 4: Exp 4 — Multi-Target
──────────────────────────────────────────────────
What: Additive bonus update toward teacher direction (0.5x LR).
      Two updates per step: input + teacher.
Toggle: curl -X POST localhost:8000/experiments \
  -d '{"exp_multi_target": true}'
Expect: Faster learning (more signal per step).
Risk: Medium — effectively 1.5x update magnitude per positive step.

Test 5: Exp 5 — Structure Reuse
──────────────────────────────────────────────────
What: Save current topology, reset weights, restart.
      Tests if structure itself encodes knowledge.
Steps:
  1. Let model run to 10K+ with good metrics
  2. curl -X POST localhost:8000/topology/save
  3. Reset from frontend
  4. curl -X POST localhost:8000/topology/load
  5. Start training on old topology with fresh weights
Expect: Faster community formation if topology is valuable.
Risk: None — topology is just graph structure.

## Success Criteria

| Metric | Baseline | Pass | Fail |
|--------|----------|------|------|
| Spatial | 0.375 | > 0.35 | < 0.25 |
| Communities | 7 | >= 7 | < 5 |
| Categories | 32 | >= 25 | < 15 |

## Combination Phase

After individual tests, combine ALL passing experiments:
1. Toggle all winners ON simultaneously
2. Reset + run to 10K
3. If combined result > any individual → synergy
4. If combined result < best individual → interference (pick best single)
```
