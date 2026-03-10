# Baby AI

A developmental AI that grows a neural architecture from scratch. It starts with no knowledge, asks questions of a local teacher LLM, learns from the answers, and evolves its own internal structure over time. You watch it happen in real time through a 3D graph visualization.

Everything runs locally on an M1 Mac. One script starts everything.

## What it does

1. A **curriculum** picks a training item (image or text concept)
2. A **teacher LLM** (LLaVA via Ollama) answers a question about it
3. **CLIP** encodes the answer into a 512-dimensional vector
4. The **baby model** processes it through a graph of clusters and nodes
5. **Forward-Forward learning** updates weights locally (no backpropagation)
6. A **growth system** evolves the architecture: splitting clusters, adding layers, pruning weak connections
7. The **3D visualizer** shows the living graph in real time via WebSocket

The model doesn't use gradient descent. Each node learns independently using the Forward-Forward algorithm — positive examples strengthen activations, negative examples suppress them. The architecture itself grows and prunes based on activation patterns.

## Architecture

```
frontend (React + Three.js)     backend (FastAPI + Python)
┌─────────────────────────┐     ┌────────────────────────────────┐
│  3D Graph Visualizer    │     │  Learning Loop (orchestrator)  │
│  Dialogue Feed (Q/T/M)  │◄───►│  Baby Model (graph of clusters)│
│  Controls & Chat        │ WS  │  Teacher Bridge (Ollama/LLaVA) │
└─────────────────────────┘     │  CLIP Encoder/Decoder          │
                                │  Growth Monitor                │
                                │  Viz Emitter                   │
                                └────────────────────────────────┘
```

### Baby Model

- **Nodes**: Each has a 512-dim weight vector and bias. Activation = `tanh(w·x + b)`.
- **Clusters**: Groups of 8 nodes. A cluster's output is the weighted sum of its nodes' activations.
- **Graph**: Clusters connected by edges with learned strengths. Organized in layers.
- **Forward pass**: Input routes through the graph layer by layer, following edges.
- **Learning**: Forward-Forward algorithm — no backprop, no global loss function. Each node adjusts its own weights based on whether the current example is positive or negative.

### Growth Operations

| Operation | What it does |
|-----------|-------------|
| **BUD** | Splits a cluster with bimodal activations into two |
| **CONNECT** | Adds an edge between clusters that frequently co-activate |
| **PRUNE** | Removes edges that are weak and unused |
| **INSERT** | Adds a new cluster between two existing ones with high residual |
| **EXTEND** | Adds a new layer on top when the top layer collapses |
| **DORMANT** | Deactivates clusters that haven't fired in a long time |

Growth is capped at 64 active clusters. PRUNE and DORMANT still run to reclaim space.

### Learning Stages

- **Stage 0**: Pure exposure. The model sees everything as positive (except every 3rd step = noise). Growth is aggressive. Auto-advances to stage 1 after 100 steps at the cluster cap.
- **Stage 1**: Real learning. Model predictions are compared to teacher answers via cosine similarity. An adaptive threshold (median of last 25 scores) determines positive vs negative — roughly 50/50 split for maximum contrast signal.
- **Stage 2+**: Same as 1, with EXTEND enabled. Not meaningfully different yet.

### Training Data

~998 items across two modalities:
- **656 images** across 93 categories (animals, vehicles, food, nature, objects, scenes)
- **342 text concepts** (emotions, opposites, actions, abstract concepts, sensory words, spatial, temporal)

Images are downloaded from loremflickr.com via `seed_data.py`. Text concepts are CLIP-encoded at training time.

## Running

```bash
# Full system (recommended)
bash start.sh

# This starts:
#   - Ollama (teacher LLM)
#   - Backend (FastAPI on port 8000)
#   - Frontend (Vite on port 5180)
#   - Opens browser automatically

# Seed more training data
cd backend && python3 seed_data.py

# Manual control
curl -X POST localhost:8000/start    # start training
curl -X POST localhost:8000/pause    # pause
curl -X POST localhost:8000/step     # single step
curl -s localhost:8000/status         # check status
```

## Requirements

- macOS with Apple Silicon (M1/M2/M3)
- Python 3.11+
- Node.js 18+
- Ollama with LLaVA model

## What you see

- **3D graph**: Clusters as colored spheres (size = activation, pulse = active, dim = dead). Edges show connections (opacity = strength). Force-directed layout where structure emerges from the data.
- **Dialogue feed**: Each training step shows Q (question), T (teacher answer), M (model answer). Green check = positive example, red cross = negative.
- **"Talk to it"**: Send text to the model and see its decoded response.
- **Controls**: Start/pause/step, speed slider, stage buttons, image upload.

## Stack

- **Backend**: Python, FastAPI, PyTorch, MLX CLIP, SQLite
- **Frontend**: React, TypeScript, Three.js (React Three Fiber), Zustand, Vite
- **Teacher**: Ollama + LLaVA
- **Encoding**: CLIP ViT-B/32 (512-dim vectors)
