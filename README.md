# Baby AI

A developmental AI that grows its own neural architecture from scratch. No backpropagation — uses Forward-Forward learning. A teacher provides training signal, the model builds itself by splitting, connecting, and pruning clusters of neurons. You watch it happen in real-time 3D.

Everything runs locally on Apple Silicon. One command starts everything.

## What it does

```
  image/text ──→ CLIP encode ──→ memory buffer ──→ multi-prototype resonance
                  (512-d)         (temporal context)  (top-20 clusters)
                                                          │
                  ┌───────────────────────────────────────┘
                  ▼
             BFS graph walk ──→ cluster forward ──→ lateral inhibition
                                  (node activations)   (suppress similar)
                                                          │
                  ┌───────────────────────────────────────┘
                  ▼
             FF weight update ──→ Hebbian edges ──→ growth check
               (local, no backprop)  (strengthen co-firing)  (BUD/PRUNE/INSERT)
                                                          │
                  ┌───────────────────────────────────────┘
                  ▼
             3D visualization ──→ metrics panel ──→ dialogue feed
               (WebSocket delta)    (spatial score,    (Q/T/M per step)
                                     communities)
```

The model doesn't use gradient descent. Each node learns independently using the Forward-Forward algorithm — positive examples strengthen activations, negative examples suppress them. The architecture itself grows and prunes based on activation patterns.

## Current Results

Trained on 123K COCO images (precomputed CLIP embeddings):

| Metric | Without buffer | With buffer + multi-prototype |
|--------|---------------|-------------------------------|
| Spatial score | 0.012 | 0.324 (peak) |
| Co-firing communities | 1 (one blob) | 7-12 (distinct groups) |
| Best category similarity | 0.51 | 0.20 (improving) |
| Growth rate | 5 clusters/1K steps | 50/1K (capped at 500) |

The blob problem — where all clusters activate identically for all inputs — has been broken. Clusters now organize by category and form distinct co-firing communities.

## Architecture

```
frontend (React + Three.js)     backend (FastAPI + Python)
┌─────────────────────────┐     ┌────────────────────────────────┐
│  3D Graph (UMAP/Live/   │     │  Learning Loop (orchestrator)  │
│           Tree modes)   │     │  Baby Model                    │
│  Metrics Panel          │◄───►│    ├─ Memory Buffer             │
│  Dialogue Feed          │ WS  │    ├─ Multi-Prototype Resonance │
│  Controls & Chat        │     │    ├─ Cached BFS Traversal      │
└─────────────────────────┘     │    ├─ Edge Adjacency Index      │
                                │    └─ Growth Monitor             │
                                │  CLIP Encoder/Decoder            │
                                │  Adversarial Curriculum          │
                                │  Viz Emitter (delta-based)       │
                                └────────────────────────────────┘
```

### Baby Model

- **Nodes**: 512-dim weight vector + bias. Activation = `tanh(w·x + b)`. Each learns independently via FF.
- **Clusters**: Groups of 4-8 nodes. Output = weighted sum of node activations.
- **Graph**: Clusters connected by directed edges with Hebbian-learned strengths. Organized in layers.
- **Memory Buffer**: Decaying echo of recent cluster activations. Biases input toward recently active directions, giving the model temporal context.
- **Multi-Prototype Resonance**: Each node's weight vector serves as a prototype. A cluster resonates if ANY of its nodes match the input well — not just the blurred mean.
- **Lateral Inhibition**: Strongly activated clusters suppress similar neighbors (cosine > 0.92), preventing the blob problem.

### Growth Operations

| Operation | What it does | Trigger |
|-----------|-------------|---------|
| **BUD** | Splits a bimodal cluster into two children | Activation bimodality > 0.05 |
| **CONNECT** | Adds edge between co-firing clusters | Co-activation correlation > 0.6 |
| **PRUNE** | Removes bottom 5% weakest edges | Every growth check |
| **INSERT** | Adds cluster between two with structured residual | PCA explained variance > 0.4 |
| **EXTEND** | Adds new top layer | Top layer bimodality > 0.3 |
| **DORMANT** | Deactivates low-activity clusters | Mean activation < 0.05 for 500+ steps |

All growth operations pause above 500 active clusters to prevent performance collapse.

### Performance

At 500 clusters, the system uses several optimizations to stay responsive:

- **Edge adjacency index**: O(degree) edge lookups instead of O(E) scans
- **Cached traversal**: BFS computed once per batch, reused for all 32 samples
- **Diff-based proto matrix**: only ~20 changed rows rebuilt per step (not all 3000)
- **Vectorized resonance**: single matrix-vector multiply for all prototypes
- **Targeted updates**: FF and Hebbian updates only touch visited clusters (~20)

### Training Data

Two modes:

- **Precomputed** (recommended): 123K COCO train+val images as pre-encoded CLIP embeddings. ~250+ steps/sec.
- **Live**: Ollama + LLaVA answers questions about images. ~1 step/sec but more varied signal.

## Running

```bash
# Fast mode: precomputed COCO embeddings (recommended)
CURRICULUM_SOURCE=precomputed bash start.sh

# Live mode: Ollama teacher (slower but richer)
bash start.sh

# This starts:
#   - Ollama (teacher LLM, live mode only)
#   - Backend (FastAPI on port 8000)
#   - Frontend (Vite on port 5180)
#   - Opens browser automatically
```

### Manual control

```bash
curl -X POST localhost:8000/start          # start training
curl -X POST localhost:8000/pause          # pause
curl -X POST localhost:8000/step           # single step
curl -s localhost:8000/status              # check status
curl -s localhost:8000/dashboard           # full metrics
```

## What You See

- **3D graph**: Three visualization modes — UMAP (point cloud), LIVE (force-directed physics), TREE (BUD lineage planes). Clusters as colored spheres. Edges show connections. Hover for labels.
- **Metrics panel**: Live spatial score, co-firing communities, memory buffer stats, category performance, growth rate. Bottom-right overlay, collapsible.
- **Dialogue feed**: Each training step shows Q (question), T (teacher answer), M (model answer). Green = positive, red = negative.
- **Chat**: Send text to the model and see its decoded response.
- **Controls**: Start/pause/step, speed slider, stage buttons, image upload, reset with experiment notes.

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.11+
- Node.js 18+
- Ollama with LLaVA model (live mode only)

## Stack

- **Backend**: Python, FastAPI, PyTorch (MPS), CLIP ViT-B/32, SQLite
- **Frontend**: React, TypeScript, Three.js (React Three Fiber), Zustand, Vite
- **Teacher**: Ollama + LLaVA (live mode) or precomputed COCO embeddings
- **Encoding**: CLIP ViT-B/32 (512-dim vectors)

## Roadmap

See [docs-v2/ROADMAP.md](docs-v2/ROADMAP.md) for the full arc: Phases A-F, decision framework, and metrics targets.

```
Phase A ✅  Memory buffer + adversarial curriculum
Phase B 🔄  Multi-prototype resonance + curiosity growth
Phase C ○   Cluster roles + typed edges
Phase D ○   Learned encoders (replace CLIP)
Phase E ○   Temporal reasoning + prediction
Phase F ○   Environment interaction + agency
```
