# Baby AI — Engine Architecture (Exploded View)

## The Complete Training Cycle

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                        ONE TRAINING STEP (128 items)                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

   CURRICULUM                    BRAIN                         GROWTH
   (what to learn)               (how it thinks + learns)      (how it evolves)
   ─────────────                 ─────                         ──────

   ┌──────────────┐
   │ Embedding DB │  5000 COCO images, pre-encoded to 512-dim CLIP vectors
   │ (SQLite)     │  Each row: image_emb(512), caption_emb(512), caption_text,
   │              │  category ("dog", "bus", etc), image_url
   └──────┬───────┘
          │
          ▼
   ┌──────────────┐
   │  EPISODES    │  Round-robin through 49 categories
   │              │  Each episode: 16 items from same category
   │  "dog" x16   │  With batch_size=128: 8 episodes per step
   │  "cat" x16   │
   │  "bus" x16   │  item.label = category name (clean, no regex)
   │  ...         │  item.expected_vector = caption CLIP embedding
   │  (8 episodes)│  item.patches = 49×512 spatial features (optional)
   └──────┬───────┘
          │
          │  128 CurriculumItems
          ▼
   ┌──────────────┐
   │ DIFF-SKIP    │  For each item: cosine_sim(anchor_prediction, item_vector)
   │              │  If sim > 0.85 → skip (model already knows this)
   │              │  Preserves at least 4 items per batch
   │              │  Reduces redundant training by ~30-60% at maturity
   └──────┬───────┘
          │
          │  ~80-128 filtered samples
          ▼
   ┌──────────────┐
   │ REPLAY MIX   │  8 episodic replay samples mixed in
   │              │  Drawn from high-error past experiences
   │              │  Weighted toward weak categories
   └──────┬───────┘
          │
          │  ~90-136 total samples
          │
══════════│════════════════════════════════════════════════════════════════════
          │
          │  FOR EACH SAMPLE IN BATCH:
          ▼

   ╔══════════════════════════════════════════════════════════════════════╗
   ║                     FORWARD PASS (brain.forward)                   ║
   ║                                                                    ║
   ║  Input: x (512-dim CLIP vector)                                    ║
   ║                                                                    ║
   ║  ┌─────────────────────────────────────────────────────────────┐   ║
   ║  │ BUFFER BLEND                                                │   ║
   ║  │                                                             │   ║
   ║  │  activation_buffer: running memory of recent top-5 neurons  │   ║
   ║  │  Decays 0.9x per step. Primes the brain for related input. │   ║
   ║  │                                                             │   ║
   ║  │  effective_x = normalize(x + 0.15 × buffer_direction)      │   ║
   ║  └─────────────────────────┬───────────────────────────────────┘   ║
   ║                            │                                       ║
   ║                            ▼                                       ║
   ║  ┌─────────────────────────────────────────────────────────────┐   ║
   ║  │ PROJECTION (learned, Phase D)                               │   ║
   ║  │                                                             │   ║
   ║  │  512×512 residual transform, trained by error signal        │   ║
   ║  │  α ramps 0→1 over first 10K update steps                   │   ║
   ║  │                                                             │   ║
   ║  │  effective_x = normalize(x + α × Projection @ x)           │   ║
   ║  │                                                             │   ║
   ║  │  WHY: Adapts the input space so neurons can match better.   │   ║
   ║  │  The raw CLIP space may not align with what the brain       │   ║
   ║  │  needs — projection learns the mapping.                     │   ║
   ║  └─────────────────────────┬───────────────────────────────────┘   ║
   ║                            │                                       ║
   ║                            ▼                                       ║
   ║  ┌─────────────────────────────────────────────────────────────┐   ║
   ║  │ 1. SENSE — all N neurons evaluate simultaneously            │   ║
   ║  │                                                             │   ║
   ║  │  identities = normalize(weights)           (N × 512)       │   ║
   ║  │  scores = identities @ effective_x         (N,)            │   ║
   ║  │                                                             │   ║
   ║  │  One matmul. Every neuron gets a similarity score to input. │   ║
   ║  │  This is the "does this input look like me?" check.         │   ║
   ║  └─────────────────────────┬───────────────────────────────────┘   ║
   ║                            │                                       ║
   ║                            ▼                                       ║
   ║  ┌─────────────────────────────────────────────────────────────┐   ║
   ║  │ 2. FIRE — self-activation thresholds                        │   ║
   ║  │                                                             │   ║
   ║  │  fired = (scores > thresholds) & not_dormant               │   ║
   ║  │                                                             │   ║
   ║  │  Each neuron has its OWN threshold (homeostatic).           │   ║
   ║  │  Guarantees at least 4 fire (fallback to top-4 by score).  │   ║
   ║  │  Typically ~5% of neurons fire (~N/20).                    │   ║
   ║  │                                                             │   ║
   ║  │  confidence[i] = (score[i] - threshold[i]) / threshold[i]  │   ║
   ║  │  ↑ How FAR above threshold — measures conviction.           │   ║
   ║  └─────────────────────────┬───────────────────────────────────┘   ║
   ║                            │                                       ║
   ║                            ▼                                       ║
   ║  ┌─────────────────────────────────────────────────────────────┐   ║
   ║  │ 3. THINK — surprise-weighted message passing (every 5 steps)│   ║
   ║  │                                                             │   ║
   ║  │  Sparse edge matrix (COO format, ~2MB at 10K neurons)      │   ║
   ║  │                                                             │   ║
   ║  │  For each round (max 3):                                    │   ║
   ║  │    surprise = z_score(fired_scores)                         │   ║
   ║  │    send_weight = score × clamp(surprise, min=0)             │   ║
   ║  │    messages = sparse_edges @ send_weight                    │   ║
   ║  │    scores += 0.05 × messages                                │   ║
   ║  │    newly_fired = (new_scores > thresholds) & !already_fired │   ║
   ║  │                                                             │   ║
   ║  │  KEY INSIGHT: Only SURPRISED neurons send messages.          │   ║
   ║  │  A generalist fires on everything → low surprise → quiet.  │   ║
   ║  │  A specialist fires rarely → high surprise → loud.         │   ║
   ║  │  This prevents generalists from drowning out experts.       │   ║
   ║  └─────────────────────────┬───────────────────────────────────┘   ║
   ║                            │                                       ║
   ║                            ▼                                       ║
   ║  ┌─────────────────────────────────────────────────────────────┐   ║
   ║  │ 4. OUTPUT — confidence-weighted aggregate                   │   ║
   ║  │                                                             │   ║
   ║  │  attn = softmax(confidence × 2.0)                          │   ║
   ║  │  prediction = attn @ fired_weights     → (512,)            │   ║
   ║  │  prediction = normalize(prediction)                         │   ║
   ║  │                                                             │   ║
   ║  │  NOT raw scores — confidence (margin above threshold).      │   ║
   ║  │  Post-deliberation: neurons reinforced by neighbors         │   ║
   ║  │  contribute more. Soft attention, not winner-take-all.      │   ║
   ║  └─────────────────────────┬───────────────────────────────────┘   ║
   ║                            │                                       ║
   ║  Output: prediction (512,), activations {cluster_id: score}        ║
   ╚════════════════════════════╪════════════════════════════════════════╝
                                │
                                ▼
   ╔══════════════════════════════════════════════════════════════════════╗
   ║                     LEARNING (brain.update)                        ║
   ║                                                                    ║
   ║  teacher_vec = CLIP embedding of correct answer (512,)             ║
   ║  error = teacher_vec - prediction                                  ║
   ║                                                                    ║
   ║  ┌─────────────────────────────────────────────────────────────┐   ║
   ║  │ ATTRACT / REPEL (per-neuron signal)                         │   ║
   ║  │                                                             │   ║
   ║  │  For each fired neuron i:                                   │   ║
   ║  │    sim = dot(weight[i], teacher_vec)                        │   ║
   ║  │    if sim > 0:  sign = +1.0   (I match teacher → attract)  │   ║
   ║  │    if sim < 0:  sign = -0.5   (I don't match → repel soft) │   ║
   ║  │                                                             │   ║
   ║  │  THIS IS THE REPULSIVE FORCE.                               │   ║
   ║  │  Without it: all neurons chase same target → one blob.      │   ║
   ║  │  With it: misaligned neurons push away → specialization.    │   ║
   ║  └─────────────────────────┬───────────────────────────────────┘   ║
   ║                            │                                       ║
   ║                            ▼                                       ║
   ║  ┌─────────────────────────────────────────────────────────────┐   ║
   ║  │ DISTRIBUTED ERROR (each neuron fixes its share)             │   ║
   ║  │                                                             │   ║
   ║  │  total_score = sum(fired_scores)                            │   ║
   ║  │  share[i] = score[i] / total_score                         │   ║
   ║  │  local_target[i] = normalize(weight[i] + share[i] × error) │   ║
   ║  │  delta[i] = local_target[i] - weight[i]                    │   ║
   ║  │  update[i] = sign[i] × delta[i]                            │   ║
   ║  │  weight[i] += lr × update[i]                                │   ║
   ║  │  weight[i] = normalize(weight[i])                           │   ║
   ║  │                                                             │   ║
   ║  │  WHY NOT just: weight += lr × score × error (same for all)?│   ║
   ║  │  Because that makes every neuron converge to the same spot. │   ║
   ║  │  Distributed: neuron contributing 10% fixes 10% of gap.     │   ║
   ║  │  Result: each neuron moves toward its OWN local optimum.    │   ║
   ║  └─────────────────────────┬───────────────────────────────────┘   ║
   ║                            │                                       ║
   ║                            ▼                                       ║
   ║  ┌─────────────────────────────────────────────────────────────┐   ║
   ║  │ PROJECTION TRAINING                                         │   ║
   ║  │                                                             │   ║
   ║  │  Projection += 0.0001 × outer(error, input)                │   ║
   ║  │  α = min(1.0, update_count / 10000)                        │   ║
   ║  │                                                             │   ║
   ║  │  The projection learns to transform inputs so neurons can   │   ║
   ║  │  match them better. Trained by the same error signal.       │   ║
   ║  └─────────────────────────┬───────────────────────────────────┘   ║
   ║                            │                                       ║
   ║                            ▼                                       ║
   ║  ┌─────────────────────────────────────────────────────────────┐   ║
   ║  │ SYNAPTIC PLASTICITY (Hebbian + decay)                       │   ║
   ║  │                                                             │   ║
   ║  │  Every 100 steps: ALL edges decay × 0.99                   │   ║
   ║  │    Edges below 0.005 → die (synapse pruned)                │   ║
   ║  │    Half-life: ~7000 steps                                   │   ║
   ║  │                                                             │   ║
   ║  │  Top-5 co-fired pairs: edge += 0.001 × co_activation      │   ║
   ║  │    Edges capped at 1.0                                      │   ║
   ║  │                                                             │   ║
   ║  │  NET EFFECT: Use it or lose it. Only edges between neurons  │   ║
   ║  │  that genuinely fire together survive. No manual pruning.   │   ║
   ║  └────────────────────────────────────────────────────────────┘    ║
   ╚════════════════════════════════════════════════════════════════════╝

                                │
   ═════════════════════════════│═══════════════════════════════════════
          AFTER EACH SAMPLE     │     AFTER FULL BATCH
                                │
                                ▼
   ┌─────────────────────────────────────────────────────────────┐
   │ THRESHOLD ADAPTATION (homeostasis)                          │
   │                                                             │
   │  fire_rate[i] = 0.999 × fire_rate[i] + 0.001 × fired[i]  │
   │  deviation = fire_rate - target (0.05)                      │
   │  threshold[i] += 0.001 × deviation                         │
   │  clamp(0.01, 0.95)                                          │
   │                                                             │
   │  Fires too much → threshold rises → fires less.             │
   │  Fires too little → threshold drops → fires more.           │
   │  Self-regulating: every neuron converges to ~5% fire rate.  │
   └─────────────────────────────────────────────────────────────┘


══════════════════════════════════════════════════════════════════════════
                     AFTER BATCH: MAIN THREAD WORK
══════════════════════════════════════════════════════════════════════════

   ┌──────────────────────────────────────────────────────────────┐
   │ CO-FIRING (z-score filtered)                                 │
   │                                                              │
   │  For the last sample's activations:                          │
   │    z_threshold = mean + 1σ                                   │
   │    significant = {clusters with activation > z_threshold}    │
   │    Record all pairs as co-fired                              │
   │                                                              │
   │  Temporal co-firing: z-score top-5 from sample[t-1]          │
   │    paired with z-score top-5 from sample[t]                  │
   │    (consecutive items in episode → same-category co-firing)  │
   │                                                              │
   │  Flushed to DB every 50 steps or 50K pairs                  │
   └──────────────────────────────────────────────────────────────┘

   ┌──────────────────────────────────────────────────────────────┐
   │ LOGGING (z-score filtered)                                   │
   │                                                              │
   │  clusters_active = only z-score significant clusters         │
   │  (NOT all fired clusters — prevents label contamination)     │
   │                                                              │
   │  Labels derived by TF-IDF:                                   │
   │    word_score = tf(word, cluster) × log(N / df(word))       │
   │    High freq in THIS cluster + rare overall = strong label   │
   └──────────────────────────────────────────────────────────────┘

   ┌──────────────────────────────────────────────────────────────┐
   │ CATEGORY TRACKING                                            │
   │                                                              │
   │  Sample 4 items per batch                                    │
   │  For each: forward(expected_vector) → prediction             │
   │  sim = dot(prediction, expected_vector)                      │
   │  Store per-category avg_sim, positive_rate                   │
   │  Uses item.label from curriculum (clean, not regex)          │
   └──────────────────────────────────────────────────────────────┘

   ┌──────────────────────────────────────────────────────────────┐
   │ EPISODIC MEMORY                                              │
   │                                                              │
   │  For sampled items with high error → store for replay        │
   │  Top-25% error items saved                                   │
   │  Replayed as 8 extra samples in next batch                   │
   │  Weighted toward weak categories                             │
   └──────────────────────────────────────────────────────────────┘


══════════════════════════════════════════════════════════════════════════
                    GROWTH CHECK (every 200-500 steps)
══════════════════════════════════════════════════════════════════════════

   ┌──────────────────────────────────────────────────────────────┐
   │ BUD — neuron overworked → split into two                    │
   │                                                              │
   │  Trigger: fire_rate > 1.5 × target (0.075) AND age > 200   │
   │  Rate limit: max(4, active_count / 25) per check            │
   │                                                              │
   │  Parent weight + random noise → child_a                      │
   │  Parent weight - random noise → child_b                      │
   │  child_b gets +0.5 layer depth (hierarchy over time)        │
   │  Parent → dormant. Edges transferred. Sibling edge = 0.5   │
   │                                                              │
   │  ┌─────┐         ┌────┐  ┌────┐                             │
   │  │  P  │   →     │ Pa │──│ Pb │   (P goes dormant)          │
   │  └─────┘         └────┘  └────┘                             │
   └──────────────────────────────────────────────────────────────┘

   ┌──────────────────────────────────────────────────────────────┐
   │ DORMANCY — neuron useless → sleep                           │
   │                                                              │
   │  Trigger: fire_rate < adaptive_threshold AND age > 1000     │
   │  Threshold: target × min(0.1, 100/active_count)             │
   │    At 100 active: 0.005                                      │
   │    At 10K active: 0.0005                                     │
   │  Gentler at scale so growth isn't choked.                    │
   │  Dormant neurons stop firing, stop learning, stop counting.  │
   └──────────────────────────────────────────────────────────────┘

   ┌──────────────────────────────────────────────────────────────┐
   │ CONNECT — co-firing neurons get edges                       │
   │                                                              │
   │  Top-5 fired pairs that lack edges → create edge (0.1)      │
   │  Edges enable message passing (THINK phase)                  │
   │  Edges decay without reinforcement (synaptic plasticity)     │
   └──────────────────────────────────────────────────────────────┘

   ┌──────────────────────────────────────────────────────────────┐
   │ MPS MIGRATION — CPU → GPU when large enough                 │
   │                                                              │
   │  At 2000+ active neurons: move all tensors to Metal GPU     │
   │  Benchmarked: MPS wins 1.9× at 10K, loses at <2K           │
   │  One-time migration, checkpoints always saved as CPU        │
   └──────────────────────────────────────────────────────────────┘


══════════════════════════════════════════════════════════════════════════
                         THE FORCES THAT SHAPE THE BRAIN
══════════════════════════════════════════════════════════════════════════

   ATTRACTIVE FORCES (pull together)         REPULSIVE FORCES (push apart)
   ────────────────────────────              ─────────────────────────────
   ● Error correction:                       ● Attract/repel sign:
     weight moves toward teacher               misaligned neurons pushed AWAY
                                               from teacher (sign = -0.5)
   ● Hebbian co-firing:
     edges strengthen between                ● Distributed error:
     neurons that fire together                each neuron gets its own local
                                               target, not the same for all
   ● Message passing:
     fired neurons activate                  ● Homeostatic thresholds:
     connected neighbors                       over-active neurons raise bar
                                               → fire less → specialize more
   ● Projection learning:
     input space adapts to                   ● Synaptic decay:
     match neuron structure                    unused edges die (half-life 7K)
                                               → stale connections pruned
   ● Activation buffer:
     recent context primes                   ● Dormancy:
     related neurons                           neurons that never contribute
                                               go to sleep → resources freed

                                             ● Surprise-weighted messaging:
                                               generalists are quiet,
                                               specialists are loud

   EQUILIBRIUM: Attraction pulls neurons toward useful representations.
   Repulsion prevents them from collapsing into the same representation.
   The balance creates SPECIALIZATION: many distinct neurons, each expert
   in a different aspect of the input space.


══════════════════════════════════════════════════════════════════════════
                              METRICS
══════════════════════════════════════════════════════════════════════════

   SPATIAL SCORE (silhouette-like)
   ────────────────────────────────
   For each category with 2+ clusters:
     intra = avg cosine_sim between clusters with SAME TF-IDF label
     inter = avg cosine_sim between clusters with DIFFERENT labels
     spatial = avg(intra) - avg(inter)

   > 0 = clusters specialize by category (good)
   = 0 = no specialization (bad)
   < 0 = same-category clusters are LESS similar than random (very bad)

   COMMUNITIES (co-firing graph structure)
   ────────────────────────────────────────
   Union-find on strong co-firing pairs (top z-score)
   More communities = more distinct functional groups
   1 blob = everything fires together (no structure)

   CATEGORY SIMILARITY (learning quality)
   ────────────────────────────────────────
   For each COCO category: avg cosine_sim(prediction, expected)
   Higher = baby predicts that category better
   "man" at 0.56 means the brain has a decent model of "man" inputs
```

## Data Flow Summary

```
COCO images (5K)
  │
  ├─ CLIP encode → 512-dim vectors (pre-computed, stored in SQLite)
  │
  ├─ Episodes (16 same-category items) × 8 per batch = 128 items
  │
  ├─ Diff-skip (remove items brain already knows)
  │
  ├─ + 8 replay samples from episodic memory
  │
  └─ FOR EACH SAMPLE:
       │
       ├─ Buffer blend → Projection → SENSE → FIRE → THINK → OUTPUT
       │                                                      │
       │                              prediction (512) ◄──────┘
       │
       ├─ error = teacher - prediction
       │
       ├─ Per-neuron: attract (aligned) or repel (misaligned)
       │
       ├─ Distributed: each neuron fixes its share of the error
       │
       ├─ Projection trains on error ⊗ input
       │
       ├─ Hebbian: co-fired edges strengthen, others decay
       │
       └─ Thresholds adapt: over-active → harder to fire
```
