# Design: Multi-Round Attention via Graph Convergence

## The Problem

The model organizes well (spatial 0.543, 21 communities) but can't predict
(avg_sim 0.05 = random). Internal structure is rich but output is mush.

## Current Architecture — Analogy: A Focus Group

Imagine a company runs a focus group to understand a product photo:

1. **Recruitment** (resonance): 20 panelists are selected based on their expertise
   matching the photo. A cat photo recruits animal experts, texture experts, etc.

2. **Individual assessment** (forward pass): Each panelist looks at the photo
   independently in a soundproof booth. They write their opinion on a card.

3. **Averaging** (output): A facilitator collects all 20 cards and averages them
   into one summary. "Cat" + "chair" + "fur" + "brown" -> "some brown animal thing."

4. **Feedback** (learning): The facilitator compares the averaged summary to the
   correct answer and tells each panelist "you were X% responsible for the error."

**What's wrong:**
- Panelists never talk to each other
- The animal expert doesn't know the texture expert also saw fur -> can't combine into "furry cat"
- The averaging destroys every individual's insight
- The facilitator blames everyone equally even though the cat expert was right

## Proposed Architecture — Analogy: A Deliberation

Same focus group, but now they **deliberate**:

1. **Recruitment** (resonance): Same — select 20 relevant panelists.

2. **Round 1 — First impressions** (current forward pass):
   Each panelist processes the photo independently. Produces an opinion AND a
   confidence score (their resonance similarity).

3. **Round 2 — Share and revise** (NEW — convergence round):
   Panelists seated next to each other (connected by edges) share their opinions.
   - The cat expert (confidence 0.9) tells the texture expert "it's a cat"
   - The texture expert revises: "oh, then this is specifically cat fur" (now 0.85)
   - The chair expert (confidence 0.2) hears "cat" and goes quiet
   - Opinions CONVERGE toward the strongest, most confident signal

4. **Output** (confidence-weighted, not flat average):
   Instead of averaging all 20, weight by post-deliberation confidence.
   Cat expert (0.9) dominates. Chair expert (0.2) barely contributes.
   Output: clearly "cat", not "brown animal thing."

5. **Feedback** (learning): Same distributed error, but now the error is smaller
   because the output is better. Clusters that contributed correctly get reinforced.

## The Diff: Focus Group -> Deliberation

```
                    FOCUS GROUP              DELIBERATION
                    (current)                (proposed)
----------------------------------------------------------------------
Processing rounds   1                        2-3
Cluster interaction None                     Via edges, weighted by confidence
Edge role           Static strength          Carries confidence + output direction
Output method       mean(top_layer)          confidence-weighted across all layers
Information flow    Forward only (BFS)       Forward + lateral (neighbors inform)
What edges learn    Co-activation strength   "When I'm confident, my neighbor benefits"
```

## Minimal Implementation

### What changes in forward():

```python
# CURRENT: single-pass BFS
for cluster in layer_order:
    output = cluster.forward(x, incoming_edges)

# PROPOSED: add convergence round after initial pass
# Round 1: same as current (unchanged)
for cluster in layer_order:
    output = cluster.forward(x, incoming_edges)
    confidence[cluster.id] = resonant_ids.get(cluster.id, 0.0)

# Round 2: clusters update based on confident neighbors
for cluster in visited:
    neighbor_signals = []
    for edge in graph.edges_for(cluster.id):
        neighbor_id = edge.other(cluster.id)
        if neighbor_id in outputs and confidence[neighbor_id] > confidence[cluster.id]:
            # Listen to more-confident neighbors
            weight = edge.strength * confidence[neighbor_id]
            neighbor_signals.append((outputs[neighbor_id], weight))
    if neighbor_signals:
        # Blend neighbor signal into own output
        total_w = sum(w for _, w in neighbor_signals)
        neighbor_blend = sum(s * w for s, w in neighbor_signals) / total_w
        outputs[cluster.id] = F.normalize(
            outputs[cluster.id] + 0.3 * neighbor_blend, dim=0
        )
        # Update confidence (boosted if neighbors agree)
        agreement = F.cosine_similarity(
            outputs[cluster.id].unsqueeze(0),
            neighbor_blend.unsqueeze(0)
        ).item()
        confidence[cluster.id] *= (1 + 0.2 * agreement)
```

### What changes in output aggregation:

```python
# CURRENT:
top_layer = [c for c in visited if c.layer_index == max_layer]
result = mean(top_layer_outputs)

# PROPOSED:
# Use ALL visited clusters, weighted by post-convergence confidence
weighted_sum = sum(outputs[c] * confidence[c] for c in visited)
result = F.normalize(weighted_sum, dim=0)
```

### What changes in edges:

Nothing structurally. Edges already connect clusters bidirectionally with
strength values. The convergence round reads edge.strength and neighbor
confidence — both already exist. No new fields needed.

## What This Does NOT Change

- Resonance screening (still z-score + absolute floor)
- Growth (BUD, INSERT, dormancy — unchanged)
- Learning signal (distributed error — unchanged, but should work BETTER
  because output is better -> error is more informative)
- Patch system (C.3 — unchanged, patches still bias learning targets)
- Health monitor — unchanged

## Risks

1. **Convergence to single voice**: If one cluster is always most confident,
   it dominates every output. Mitigation: the 0.3 blend weight limits
   influence. Clusters keep 70% of their own opinion.

2. **Computational cost**: Extra pass over visited clusters (~20). At 20
   clusters with ~5 edges each, that's ~100 operations. Negligible vs the
   matmul in resonance screening (500+ clusters x 512 dims).

3. **Echo chambers**: Tightly connected clusters could reinforce each other's
   errors. Mitigation: confidence is rooted in resonance (input similarity),
   not peer agreement. A wrong-but-confident cluster can mislead neighbors,
   but learning will correct it over time.

## Success Criteria

- avg_sim for actively-seen categories should improve from 0.05 -> 0.15+
- Spatial score should hold or improve (0.543+)
- Output should be less "mush" — more aligned with the dominant category
