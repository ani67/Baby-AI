# Baby AI — Architecture Documentation (Current State)

## What This Is

A curiosity-driven developmental AI that grows its own neural architecture.
No backpropagation — uses Forward-Forward learning (gradient-free).
A teacher LLM (Ollama) provides training signal. The model builds itself
by splitting, connecting, and pruning clusters of neurons.

```
┌─────────────────────────────────────────────────────────────┐
│                     BABY AI SYSTEM                          │
│                                                             │
│   ┌──────────┐    ┌──────────┐    ┌──────────────────────┐  │
│   │  Ollama   │───▶│ Learning │───▶│     Baby Model       │  │
│   │ (Teacher) │    │   Loop   │    │  (Growing Neural     │  │
│   └──────────┘    └────┬─────┘    │   Graph)             │  │
│                        │          └──────────────────────┘  │
│                        │                                    │
│                   ┌────▼─────┐    ┌──────────────────────┐  │
│                   │  SQLite  │    │    Frontend (React    │  │
│                   │  Store   │    │    + Three.js)        │  │
│                   └──────────┘    └──────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## 1. THE NEURAL GRAPH

The core data structure. Everything else exists to feed and grow this.

```
                    Graph
                      │
          ┌───────────┼───────────┐
          │           │           │
       Clusters     Edges     Quadtree
       (list)      (list)     (tiles)
          │
     ┌────┼────┐
     │    │    │
   Nodes Nodes Nodes
   (8ea) (8ea) (8ea)
```

### Node (neuron)
- 512-dimensional weight vector + bias
- Activates via dot product: `tanh(weights · input + bias)`
- Learns via Forward-Forward: positive examples push weights toward input,
  negative examples push away. No gradients, no backprop.
- Tracks activation history (last 64 steps)

### Cluster (concept unit)
- Group of 8 nodes that fire together
- Has an "identity" = normalized mean of all node weights
- Types determined by internal wiring density:
  - Integration (dense internal connections)
  - Transformation (moderate density + external)
  - Routing (sparse internal, many external)
  - Arbitration (everything else)
- Can be dormant (inactive but not deleted)

### Edge (connection)
- Links two clusters
- Has strength (0-1), updated by Hebbian rule:
  `Δstrength = 0.01 × from_activation × to_activation - decay`
- Bidirectional by default
- Pruned when strength < 0.01 and unused for 150+ steps

### Quadtree (spatial index)
- Each cluster mapped to a tile in [0,1)×[0,1) space
- Tiles hold 64×64 float32 identity textures
- Split when variance > 0.1, collapse after 500 steps of similarity
- Used for O(log n) nearest-neighbor lookup
- Max depth: 32

---

## 2. FORWARD PASS

How a single input vector flows through the graph.

```
Input (512-d vector)
        │
        ▼
┌───────────────┐
│   Resonance   │  Cosine similarity to each cluster identity
│   Screening   │  Threshold: 0.05 (min 12 clusters pass)
└───────┬───────┘
        │  (only resonant clusters participate)
        ▼
┌───────────────┐
│  Entry Layer  │  Layer 0 clusters that passed resonance
│  (seeds BFS)  │
└───────┬───────┘
        │
        ▼
┌───────────────┐
│   BFS by      │  Process clusters in layer order
│   Layer Index │  Each cluster: combine input + incoming edge signals
│               │  Activate all nodes, produce weighted output
└───────┬───────┘
        │
        ▼
┌───────────────┐
│  MPS Forward  │  Batched dot-product on identity textures
│  (benchmark)  │  CPU or MPS, whichever is faster (benchmarked on first call)
└───────┬───────┘
        │
        ▼
┌───────────────┐
│  Inhibition   │  Strongly activated clusters suppress similar neighbors
│  (lateral)    │  Sorted by activation, winners suppress losers
└───────┬───────┘
        │
        ▼
┌───────────────┐
│  Saturation   │  If activation > 0.8: gentle continuous decay (0.97-0.99×)
│  Decay        │  Prevents weight explosion, no harsh 0.7x cliff
└───────┬───────┘
        │
        ▼
   Output (512-d)  = mean of highest-layer visited cluster outputs
```

### Edge cases
- Zero clusters pass resonance → all clusters let through
- Zero-vector output → fallback to pre-inhibition top-4 mean, then best resonant identity, then random
- No entry clusters → orphaned subgraphs seeded into BFS queue

---

## 3. LEARNING: FORWARD-FORWARD RULE

No backprop. Each node updates itself from its own activation.

```
                    ┌─────────────────────┐
                    │  For each node:     │
                    │                     │
Positive example:   │  weights += lr ×    │
(good signal)       │    plasticity ×     │
                    │    |activation| ×   │
                    │    input ×          │
                    │    (1 - act²)       │
                    │                     │
Negative example:   │  Same but sign      │
(random noise)      │  flipped (−)        │
                    └─────────────────────┘
```

### Learning Rate
```
base_lr = 0.01 × exp(-0.0003 × step)    # decays over time
         │
         ├── Stage 0: × 1.0   (full speed, building structure)
         ├── Stage 1: × 0.5   (half speed, refining)
         └── Stage 2+: × 0.1  (slow, fine-tuning)
         │
         └── Batch mode: ÷ batch_size  (so 32 samples = same magnitude as 1)

At step 5000, Stage 2:  effective_lr = 0.000223
```

### Hebbian Edge Update
After each forward pass, edges between co-activated clusters strengthen.
Edges between inactive pairs decay. This is how the graph learns its wiring.

---

## 4. GROWTH OPERATIONS

The graph grows itself. Checked every 50 + (cluster_count / 10) steps.

```
┌──────────┬────────────────────────────────┬──────────────────────┐
│ Operation│ Trigger                        │ What happens         │
├──────────┼────────────────────────────────┼──────────────────────┤
│ BUD      │ High bimodality + low          │ Split cluster into   │
│          │ coherence + age > 200          │ two children (k=2    │
│          │                                │ means on weights)    │
│          │ Max per check: clusters // 50  │ c_00 → c_00a, c_00b │
├──────────┼────────────────────────────────┼──────────────────────┤
│ CONNECT  │ Two clusters co-fire           │ Add edge between     │
│          │ frequently (>30% of recent     │ them (strength 0.1)  │
│          │ steps)                         │                      │
├──────────┼────────────────────────────────┼──────────────────────┤
│ PRUNE    │ Edge strength < 0.01 AND       │ Remove edge          │
│          │ unused for 150+ steps          │ Min edges: 2× active │
│          │ (200-step cooldown after       │ clusters             │
│          │ restore)                       │                      │
├──────────┼────────────────────────────────┼──────────────────────┤
│ INSERT   │ High structured residual       │ New cluster between  │
│          │ between connected pair         │ two existing ones    │
│          │ (PCA explains >40%)            │ Weights from PCA     │
├──────────┼────────────────────────────────┼──────────────────────┤
│ EXTEND   │ Top layer coherence < 0.2      │ New cluster at top   │
│          │ (Stage 2+ only)               │ of graph             │
├──────────┼────────────────────────────────┼──────────────────────┤
│ DORMANT  │ Mean activation < 0.05         │ Mark cluster dormant │
│          │ for 500+ steps                │ (excluded from       │
│          │                                │ forward pass)        │
└──────────┴────────────────────────────────┴──────────────────────┘
```

### BUD naming convention
```
c_00  (original)
  ├── c_00a  (first split)
  │     ├── c_00aa  (second split of 'a' child)
  │     └── c_00ab
  └── c_00b
        ├── c_00ba
        └── c_00bb

Depth = number of suffix letters (c_00=0, c_00a=1, c_00ab=2)
Parent = ID with last letter removed (c_00ab → c_00a)
Note: BUD removes the parent from the graph (c_00 is gone after splitting)
```

---

## 5. LEARNING LOOP (ORCHESTRATOR)

One training cycle. Called repeatedly while running.

```
┌─────────────────────────────────────────────┐
│              step_once()                     │
│                                             │
│  1. OBSERVE   → graph summary               │
│  2. SCORE     → pick curriculum item         │
│  3. QUESTION  → generate question text       │
│  4. ASK       → send to Ollama teacher       │
│       OR                                     │
│     PRECOMPUTED → use cached CLIP embedding  │
│  5. ENCODE    → text → 512-d vector          │
│  6. PREDICT   → forward pass through graph   │
│  7. UPDATE    → FF weight update             │
│  8. GROW      → check BUD/CONNECT/PRUNE/etc  │
│  9. HEALTH    → auto-tune parameters         │
│ 10. CO-FIRING → record cluster pair counts   │
│ 11. LOG       → save to SQLite               │
│ 12. EMIT      → push delta to frontend (WS)  │
│ 13. AUTO-ADV  → check stage transitions      │
│                                             │
│  Returns: StepResult                         │
└─────────────────────────────────────────────┘
```

### Batch Mode (precomputed curriculum)
When `CURRICULUM_SOURCE=precomputed`:
- Fetches 32 items at once from embedding cache
- Runs forward + FF update for each, with lr ÷ 32
- Growth check, health, emit once per batch
- Step counter advances by 32
- Heavy compute runs in `run_in_executor` (thread pool)
  so FastAPI endpoints stay responsive

### Positive/Negative Signal
```
Stage 0:  Every 3rd step is negative (random vector)
Stage 1+: Cosine similarity between prediction and answer
          Adaptive threshold (median of recent similarities)
          If positive_rate < 40%, threshold drops to 40th percentile
```

---

## 6. DEVELOPMENTAL STAGES

The model progresses through stages as it grows.

```
Stage 0 (Sensory)     → Basic pattern exposure
  │                     Hard cap: 80 clusters (BUD disabled above)
  │                     Auto-advance: step ≥ 800 AND ≥ 60 active clusters
  ▼
Stage 1 (Association)  → Positive/negative contrast learning
  │                     Auto-advance: step ≥ 3000 AND > 120 clusters
  │                     AND positive_rate > 55%
  ▼
Stage 2 (Concept)      → Fine-tuning, EXTEND unlocked
  │                     Learning rate drops to 0.1×
  ▼
Stage 3 (Reasoning)    → Complex question templates
  ▼
Stage 4 (Language)     → Vocabulary concepts
```

---

## 7. DATA FLOW: BACKEND → FRONTEND

```
Backend                              Frontend
───────                              ────────

FastAPI ──WebSocket──▶ useWebSocket hook
  │                        │
  │  Messages:             ├──▶ graphStore (nodes, clusters, edges)
  │  • snapshot            ├──▶ dialogueStore (Q&A history)
  │  • delta               └──▶ loopStore (state, step, stage)
  │  • status                       │
  │                                 ▼
  │                         ┌───────────────┐
  │  REST endpoints:        │  Three.js      │
  │  /clusters/labels  ───▶ │  Visualization │
  │  /clusters/tree    ───▶ │  (3 modes)     │
  │  /clusters/cofiring───▶ │               │
  │  /debug/cluster/X  ───▶ └───────────────┘
  │
  │  /status ───▶ StatusBar
  │  /chat   ───▶ HumanChat
```

### WebSocket Delta Message (sent every step)
```json
{
  "type": "delta",
  "step": 1234,
  "stage": 1,
  "activated": ["c_00", "c_03"],
  "deactivated": ["c_02"],
  "activation_values": {"c_00": 0.87, "c_03": 0.42},
  "edges_formed": [["c_00", "c_03", 0.1]],
  "edges_pruned": [],
  "clusters_added": [],
  "clusters_dormanted": [],
  "positions": {"n_001": [0.5, 1.2, -0.3]},
  "dialogue": {
    "question": "What is this?",
    "answer": "A dog sitting on grass",
    "model_answer": "dog grass sit",
    "is_positive": true
  },
  "growth_events": [{"event_type": "CONNECT", ...}]
}
```

---

## 8. VISUALIZATION (Three.js)

Three modes, toggled via UMAP / LIVE / TREE buttons.

### UMAP Mode
```
Standard point cloud.
Positions from backend projector (force-directed layout).
Edge culling: top 500 strongest only (from 11K+).
Shared sphere geometry (6-segment, single GPU upload).
Hover: tooltip with cluster ID + emergent labels.
```

### LIVE Mode (Force-Directed Simulation)
```
Full physics simulation running at 60fps:

  Attraction ←── co-firing strength from /clusters/cofiring
  Repulsion  ←── inverse square, 2.0-unit cutoff
  Damping    ←── velocity × 0.85 each frame
  Boundary   ←── soft push when beyond R = sqrt(n) × 0.5

Edge rendering: top 300 strongest co-firing pairs only.
Co-firing data fetched every 60s from backend.
New clusters spawn near their strongest co-firing partner.

What you see evolve:
  Early:  uniform ball (everything co-fires equally)
  Mid:    islands forming (concept regions developing)
  Late:   tight semantic clusters with sparse bridges
```

### TREE Mode (Stacked Planes)
```
                    Depth 0 (roots)
                    ═══════════════
                    c_00  c_01  c_03
                      │         │
                    Depth 1
                    ═══════════════
                    c_00a  c_00b  c_03a  c_03b
                      │
                    Depth 2
                    ═══════════════
                    c_00aa  c_00ab

  Each depth level = horizontal plane at z = depth × 2.0
  Within plane: repulsion-only force layout (no overlap)
  Vertical lines = parent-child edges
  Sphere size = number of BUD children
  Phantom nodes (removed parents): small + transparent
  Camera auto-positions at elevated angle
```

---

## 9. PERSISTENCE

### SQLite Tables
```
dialogues          → Q&A history with cluster activations
graph_events       → BUD/CONNECT/PRUNE/INSERT/EXTEND/DORMANT
latent_snapshots   → Full graph JSON at intervals
model_checkpoints  → Pickled weights + graph structure
human_chat         → User conversation history
cluster_cofiring   → Cluster pair co-activation counts
embedding_cache    → Pre-computed CLIP embeddings (COCO)
```

### Checkpoint Flow
```
Every 100 steps:
  1. Collect all node weights/biases → state_dict
  2. Serialize graph structure → graph_json
  3. Pickle both to checkpoints/step_N.pt
  4. Mark as 'complete' in SQLite

On startup:
  1. Find latest 'complete' checkpoint
  2. Rebuild Graph() from graph_json
  3. Load weights into nodes from state_dict
  4. Fix ID counters so new nodes don't collide
  5. 200-step prune cooldown after restore
```

---

## 10. PRECOMPUTED CURRICULUM

Alternative to live Ollama. Uses pre-embedded COCO dataset.

```
┌──────────────────┐     ┌──────────────────┐
│  download_coco   │     │  embedding_cache  │
│  (one-time)      │────▶│  (SQLite table)   │
│                  │     │                   │
│  COCO val2017    │     │  image_id         │
│  or train2017    │     │  image_emb (BLOB) │
│  + CLIP ViT-B/32 │     │  caption_emb      │
│                  │     │  caption_text     │
│  Best caption    │     │  image_url        │
│  per image       │     └──────────────────┘
└──────────────────┘              │
                                  ▼
                         ┌──────────────────┐
                         │  Curriculum       │
                         │  next_batch(32)   │
                         │  returns items    │
                         │  with pre-encoded │
                         │  vectors          │
                         └──────────────────┘

Switch: CURRICULUM_SOURCE=precomputed bash start.sh
```

---

## 11. HEALTH MONITOR

Auto-tunes parameters every 50 steps to keep learning stable.

```
Metric              Healthy Range    If Out of Range
─────────────────── ──────────────── ─────────────────────────
positive_rate        35-65%          Adjust resonance_threshold
growth_rate          0-15/50 steps   Increase bud_cooldown
active_per_step      4-40 clusters   Adjust resonance_threshold
edge_ratio           1.5-6.0×       Adjust resonance_min_pass
similarity_trend     -0.05 to 0.5   Lower resonance_threshold

Emergency: Stage 0 + >100 active clusters → freeze all growth
```

---

## 12. FILE MAP

```
Baby/
├── start.sh                    # Launch everything
├── stop.sh                     # Kill everything
├── docs-v1/                    # All documentation (you are here)
│
├── backend/
│   ├── main.py                 # FastAPI server + all endpoints
│   ├── config.py               # Configuration dataclass
│   ├── models.py               # Pydantic request/response schemas
│   │
│   ├── model/
│   │   ├── node.py             # Neuron: weights, activation, FF update
│   │   ├── cluster.py          # Group of nodes: identity, type
│   │   ├── graph.py            # Quadtree graph: clusters + edges + MPS
│   │   ├── baby_model.py       # Assembled model: forward, update, grow
│   │   ├── growth.py           # BUD, INSERT, EXTEND, GrowthMonitor
│   │   ├── forward_forward.py  # Learning rate schedule (stage-aware)
│   │   └── serializer.py       # Save/load to .pt + .json
│   │
│   ├── loop/
│   │   ├── orchestrator.py     # Learning loop: step_once / step_batch
│   │   ├── curriculum.py       # Training items (live or precomputed)
│   │   ├── curiosity.py        # Novelty scoring
│   │   ├── question_gen.py     # Question templates per stage
│   │   ├── health_monitor.py   # Auto-parameter tuning
│   │   └── sentence_splitter.py
│   │
│   ├── state/
│   │   ├── store.py            # SQLite persistence
│   │   └── schema.sql          # Table definitions
│   │
│   ├── viz/
│   │   ├── emitter.py          # WebSocket streaming (non-blocking)
│   │   ├── projector.py        # 3D force-directed layout
│   │   └── diff.py             # Graph delta computation
│   │
│   ├── encoder/
│   │   ├── clip_mlx.py         # CLIP ViT-B/32 wrapper
│   │   ├── encoder.py          # Image/Text/Video encoders
│   │   ├── decoder.py          # Text decoder (vocab-based)
│   │   └── vocab.py            # Word ↔ ID mapping
│   │
│   ├── teacher/
│   │   ├── bridge.py           # Ollama HTTP client
│   │   └── prompts.py          # Stage-aware system prompts
│   │
│   └── scripts/
│       └── download_coco.py    # COCO dataset → embedding cache
│
└── frontend/
    └── src/
        ├── App.tsx             # 3-panel layout
        ├── components/
        │   ├── LatentSpace.tsx  # 3D viz (UMAP/LIVE/TREE modes)
        │   ├── Controls.tsx    # Playback, speed, stage, reset
        │   ├── DialogueFeed.tsx # Q&A history log
        │   ├── HumanChat.tsx   # Direct chat interface
        │   └── StatusBar.tsx   # State/step/stage display
        ├── store/
        │   ├── graphStore.ts   # Nodes, clusters, edges, activations
        │   ├── dialogueStore.ts # Dialogue history
        │   └── loopStore.ts    # Loop state
        ├── hooks/
        │   ├── useWebSocket.ts # Real-time backend connection
        │   ├── useClusterLabels.ts  # Emergent labels (30s cache)
        │   ├── useClusterTree.ts    # BUD hierarchy (30s cache)
        │   ├── useGraphState.ts     # Cluster centroid helpers
        │   └── useLoopControl.ts    # API method wrappers
        ├── lib/
        │   ├── api.ts          # HTTP client for all endpoints
        │   ├── colors.ts       # Cluster type → color mapping
        │   └── constants.ts    # URLs, animation timings
        └── styles/
            └── global.css      # Dark theme, all component styles
```

---

## 13. KEY NUMBERS (Current)

```
Clusters:        ~334 active, 0 dormant
Nodes:           ~2500 (8 per cluster)
Edges:           ~5000+ connections
Layers:          13 depth levels
Step:            ~7000+
Stage:           2 (Concept)
Batch size:      32 (precomputed mode)
Learning rate:   ~0.0002 (stage 2, step 7000)
Resonance:       0.05 threshold, min 12 pass
Split threshold: 0.1 (quadtree variance)
```
