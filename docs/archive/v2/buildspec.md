# BUILDSPEC: Developmental AI System
*A growing neural architecture with curiosity-driven learning, rhizomic latent space, and local teacher model*

---

## Overview

A system that grows a neural model from scratch — starting with no knowledge — by having it ask questions of a local teacher LLM, receive answers, update its own architecture, and repeat. The model's internal structure (latent space) is visible in real time as a living graph. A human can observe, intervene, and converse with the growing model at any point.

Everything runs locally on M1 Mac. One script starts everything.

```
start.sh
  ├── starts teacher model   (Ollama + Phi-4-mini or Mistral)
  ├── starts backend         (Python, model engine + API)
  └── starts frontend        (React dev server, UI)
```

---

## System Map

```
┌─────────────────────────────────────────────────────────────────┐
│  FRONTEND (React)                                               │
│                                                                 │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────┐  │
│  │  Latent Space    │  │  Dialogue Feed   │  │  Human Chat  │  │
│  │  Visualizer      │  │  (model↔teacher) │  │  Interface   │  │
│  └──────────────────┘  └──────────────────┘  └──────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Controls Panel                                          │   │
│  │  [Start] [Pause] [Step] [Reset] [Speed] [Stage]         │   │
│  └──────────────────────────────────────────────────────────┘   │
└───────────────────────────┬─────────────────────────────────────┘
                            │ WebSocket + REST
┌───────────────────────────▼─────────────────────────────────────┐
│  BACKEND (Python / FastAPI)                                     │
│                                                                 │
│  ┌────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │  Learning Loop │  │  Model Engine   │  │  Teacher Bridge │  │
│  │  Orchestrator  │  │  (Baby Model)   │  │  (Ollama API)   │  │
│  └────────────────┘  └─────────────────┘  └─────────────────┘  │
│  ┌────────────────┐  ┌─────────────────┐                        │
│  │  State Store   │  │  Viz Emitter    │                        │
│  │  (SQLite)      │  │  (graph deltas) │                        │
│  └────────────────┘  └─────────────────┘                        │
└───────────────────────────┬─────────────────────────────────────┘
                            │ HTTP (localhost:11434)
┌───────────────────────────▼─────────────────────────────────────┐
│  TEACHER MODEL (Ollama)                                         │
│  Phi-4-mini or Mistral 7B — answers the baby model's questions  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Components

---

### 1. Baby Model (Rhizomic Neural Architecture)

**What it is:**
The core thing being built. A neural network that starts near-empty and grows its own structure through experience. Unlike a transformer, its architecture is not fixed — new nodes, clusters, layers, and connections form as the model learns.

**Boundaries:**
- Lives entirely in Python/PyTorch (or MLX for M1 acceleration)
- Receives inputs as embedding vectors (from the Encoder)
- Produces outputs as embedding vectors (decoded by the Decoder)
- Does NOT speak to the teacher directly — the Learning Loop does that
- Does NOT know about the UI — the Viz Emitter reads its state

**Internal structure:**

```
INPUT EMBEDDING
      ↓
RHIZOMIC LATENT SPACE
  ├── Node Clusters (functional groups, not fixed layers)
  │     each cluster: a group of nodes with shared activation patterns
  │     internal connectivity: dense early clusters, sparse late clusters
  │     cluster type emerges from use: integration / transformation /
  │                                    arbitration / routing
  │
  ├── Dynamic Edges (connections between clusters)
  │     form when clusters co-activate frequently
  │     strengthen via Hebbian update (fire together → wire together)
  │     prune when unused below threshold
  │
  ├── Layer Structure (depth axis, also dynamic)
  │     a "layer" = a detected cluster boundary, not a geometric slice
  │     new layers insert when residual between adjacent clusters
  │     contains structured (non-random) patterns
  │     layers also extend upward as abstraction demands increase
  │
  └── Growth Operations
        BUD:        one cluster splits into two (overloaded module)
        CONNECT:    new edge forms between co-activating clusters
        INSERT:     new layer inserts between two existing layers
        EXTEND:     new layer appends at the top
        PRUNE:      weak edge removed (below disuse threshold)
        DORMANT:    whole cluster suspended (low activation, long period)
```

**Learning rule:**
Forward-Forward algorithm (no backpropagation). Each cluster updates locally from its own activation signal. No global backward pass. No freeze cycle. The model can infer and update simultaneously.

**Key properties:**
- Architecture is not fixed at init — it grows
- Size (parameter count) is not predetermined — it reflects knowledge accumulated
- Computation per inference scales with subgraph activated, not total model size
- Simple inputs traverse short paths; complex inputs traverse longer ones

---

### 2. Encoder / Decoder

**What it is:**
Translates between raw input modalities (text, images, video frames) and the embedding vectors the Baby Model consumes. Also translates Baby Model outputs back into human-readable text.

**Boundaries:**
- Encoder: raw input → fixed-size embedding vector
- Decoder: Baby Model output vector → text (via small generative head)
- Does not learn during this phase — uses pretrained CLIP (images) and a tokenizer (text)
- Can be swapped out later for a jointly-trained encoder

**Connections:**
- Input comes from the Learning Loop (what to process next)
- Output goes to the Baby Model
- Baby Model output comes back here for decoding before going to the Dialogue Feed

**Supported modalities (v1):**
- Text (tokenized, embedded via small frozen embedding layer)
- Images (CLIP ViT-B/32 via MLX — single forward pass → 512-dim vector)
- Video frames (sequence of CLIP embeddings with temporal position encoding)

---

### 3. Learning Loop Orchestrator

**What it is:**
The "nervous system" of the whole process. Runs the curiosity-driven learning cycle continuously (or step-by-step when paused). Decides what to ask, sends it to the teacher, receives the answer, feeds it to the Baby Model, measures the update, and repeats.

**Boundaries:**
- Owns the learning cycle timing and sequencing
- Does NOT implement the model update itself (that's the Baby Model)
- Does NOT implement the teacher API call (that's the Teacher Bridge)
- Receives control signals from the Controls Panel via the backend API

**The loop (one step):**

```
1. OBSERVE
   current state of Baby Model latent space
   what clusters exist, what's uncertain, what's novel

2. GENERATE QUESTION
   curiosity score = uncertainty × novelty
   pick the highest-scoring gap in the model's current knowledge
   formulate it as a natural language question
   (early stage: simple — "what is this?" from an image embedding)
   (later stage: relational — "why does X cause Y?")

3. ASK TEACHER
   send question + context to Teacher Bridge
   receive answer

4. ENCODE ANSWER
   pass answer through Encoder → embedding vector

5. UPDATE BABY MODEL
   feed encoded answer into Baby Model
   run Forward-Forward local update
   record which clusters activated and how strongly

6. MEASURE DELTA
   compare latent space before and after
   which clusters changed? which edges strengthened?
   did any growth operations trigger?

7. LOG
   save full step to State Store
   (question, answer, model state before/after, delta)

8. EMIT
   send graph delta to Viz Emitter → frontend
```

**Developmental stages (control the curriculum):**

```
STAGE 0 — Sensory primitives
  input: images only
  questions: "what is this?" (image → label)
  goal: build first perceptual clusters

STAGE 1 — First words
  input: image + label pairs
  questions: "what do X and Y have in common?"
  goal: ground language tokens to perceptual clusters

STAGE 2 — Concepts
  input: image pairs, short video clips
  questions: "why are these similar/different?"
  goal: form categorical and relational clusters

STAGE 3 — Causal structure
  input: short video sequences (before/after)
  questions: "what caused this change?"
  goal: temporal and causal cluster formation

STAGE 4 — Language generalization
  input: text descriptions + images
  questions: freely generated from model's own uncertainty
  goal: abstract clusters, cross-modal bridges
```

**Controls (from frontend):**
- START / PAUSE / RESUME — run or halt the loop
- STEP — run exactly one loop iteration then pause
- SPEED — set loop delay (0ms = max speed, 2000ms = slow/observable)
- STAGE — manually set current developmental stage
- RESET — wipe Baby Model back to initial state, clear logs

---

### 4. Teacher Bridge

**What it is:**
A thin wrapper around the Ollama API. Takes a question string + optional context, calls the local teacher LLM, returns the answer string.

**Boundaries:**
- Stateless — no memory between calls
- Does not know about the Baby Model or its state
- Only speaks to Ollama (localhost:11434)
- If Ollama is not running, returns an error that the Learning Loop handles gracefully

**Interface:**
```python
async def ask(question: str, context: str = "") -> str:
    # calls ollama /api/generate
    # returns plain text answer
    # max_tokens: 200 (keep answers short and dense)
```

**Teacher model choice:**
- Default: `phi4-mini` (fast, good reasoning, fits in 4GB)
- Fallback: `mistral:7b-instruct` (slower, richer answers)
- Configurable via environment variable `TEACHER_MODEL`

---

### 5. State Store

**What it is:**
Persistent storage for everything that happens. Two purposes: (1) the UI can replay history, (2) if the process is killed and restarted, the Baby Model can be restored from the last checkpoint.

**Boundaries:**
- SQLite database (single file, no external dependencies)
- Written by the Learning Loop and Baby Model
- Read by the API endpoints that serve the frontend
- Does NOT store raw model weights inline (those are separate files)

**Schema (high level):**

```
dialogues
  id, timestamp, stage, question, answer, curiosity_score

model_snapshots
  id, timestamp, step_number, weights_path, graph_json

graph_events
  id, timestamp, event_type (BUD/CONNECT/INSERT/PRUNE/DORMANT),
  cluster_a, cluster_b, metadata_json

latent_snapshots
  id, timestamp, step_number, node_positions_json, edge_list_json
```

---

### 6. Viz Emitter

**What it is:**
Reads the Baby Model's current graph state (nodes, edges, cluster memberships, activation levels) and emits structured diffs over WebSocket to the frontend. The frontend uses these diffs to animate the latent space visualization.

**Boundaries:**
- Read-only access to Baby Model state
- Emits on every loop step (or on explicit request)
- Does NOT compute anything — just serializes and streams
- If no frontend is connected, still runs silently (no bottleneck)

**Emitted payload per step:**
```json
{
  "step": 1042,
  "stage": 2,
  "nodes": [
    {"id": "n_14", "cluster": "visual_shape", "activation": 0.73,
     "x": 0.42, "y": 0.18, "z": 0.61}
  ],
  "edges": [
    {"from": "n_14", "to": "n_31", "strength": 0.88, "age": 204}
  ],
  "events": [
    {"type": "CONNECT", "from": "visual_shape", "to": "word_noun"}
  ],
  "last_question": "what is the difference between a dog and a cat?",
  "last_answer": "Dogs and cats are both mammals but differ in..."
}
```

**Node positions:**
Computed by dimensionality reduction (UMAP or t-SNE) on the node activation vectors. Runs every N steps (not every step — expensive). Between reduction runs, nodes animate smoothly to new positions using interpolation.

---

### 7. Backend API (FastAPI)

**What it is:**
The HTTP + WebSocket server that connects the frontend to everything else.

**Boundaries:**
- Owns no logic itself — routes to the relevant component
- Single process that holds the Learning Loop, Baby Model, State Store all in memory
- WebSocket endpoint for live viz streaming
- REST endpoints for control + history queries

**Endpoints:**

```
WebSocket
  /ws/latent          live graph stream (Viz Emitter output)
  /ws/dialogue        live dialogue stream (new Q&A as they happen)

REST
  POST /control       { action: start|pause|step|reset, params: {...} }
  GET  /history       paginated dialogue history from State Store
  GET  /snapshot      current model graph as JSON (for initial load)
  POST /chat          { message: string } → human sends message to Baby Model
  GET  /chat/history  conversation history between human and Baby Model
  GET  /status        current stage, step count, loop state, model size
```

---

### 8. Frontend (React)

**What it is:**
The UI. Three main panels plus a controls bar. Connects to the backend via REST and WebSocket.

**Boundaries:**
- No ML logic — pure display and control
- Stateless between page loads (all state lives in backend/State Store)
- Designed for a single user (no auth, no multi-user)

**Panels:**

```
┌─────────────────────────────────────────────────────────┐
│  LATENT SPACE VISUALIZER                                │
│                                                         │
│  3D scatter plot of nodes (Three.js or react-three-fiber│
│  nodes colored by cluster                               │
│  edges shown as lines, opacity = strength               │
│  recently activated nodes pulse                         │
│  growth events animate (new edge draws in, pruned edge  │
│  fades out, new cluster buds as splitting animation)    │
│                                                         │
│  controls: rotate, zoom, filter by cluster, highlight   │
│  tooltip on hover: node id, cluster, activation history │
└─────────────────────────────────────────────────────────┘
┌────────────────────┐  ┌──────────────────────────────────┐
│  DIALOGUE FEED     │  │  HUMAN CHAT                      │
│                    │  │                                  │
│  live scrolling    │  │  text input → POST /chat         │
│  stream of:        │  │  Baby Model responds via         │
│  [step 1042]       │  │  its current generative head     │
│  Q: what is...     │  │                                  │
│  A: It is a...     │  │  response includes which         │
│  ↑ confidence      │  │  clusters were activated         │
│  ↑ clusters used   │  │  (small tag list under answer)   │
│                    │  │                                  │
│  filterable by     │  │  message history scrollable      │
│  stage / cluster   │  │                                  │
└────────────────────┘  └──────────────────────────────────┘
┌─────────────────────────────────────────────────────────┐
│  CONTROLS                                               │
│                                                         │
│  [▶ Start] [⏸ Pause] [⏭ Step] [↺ Reset]               │
│  Speed:  [━━━━●━━━━━━] Slow ← → Fast                   │
│  Stage:  [0] [1] [2] [3] [4]  (manual override)        │
│  Model:  14 clusters · 312 nodes · 48 edges · step 1042 │
│  Teacher: phi4-mini ● connected                        │
└─────────────────────────────────────────────────────────┘
```

---

### 9. start.sh

**What it is:**
Single entry point. Run `./start.sh` from the project root and everything starts.

**What it does:**
```bash
1. Check dependencies (ollama, python3, node, uv or pip)
2. Pull teacher model if not already present
   ollama pull phi4-mini
3. Start Ollama in background (if not already running)
4. Create/activate Python venv, install backend deps
5. Start backend (uvicorn, port 8000) in background
6. Install frontend deps if node_modules absent
7. Start frontend dev server (port 3000) in background
8. Wait for both servers to be healthy (curl checks)
9. Open browser to localhost:3000
10. Print process table so user knows what's running
11. Trap SIGINT → clean shutdown of all three processes
```

**What it does NOT do:**
- Train anything automatically on start (user must press Start in UI)
- Require any external services (everything local)
- Require GPU (runs on M1 CPU+Neural Engine, though GPU helps)

---

## Connections Map

```
Component               Talks to                   Protocol
─────────────────────── ────────────────────────── ──────────────
Frontend                Backend API                REST + WebSocket
Backend API             Learning Loop              in-process call
Backend API             Baby Model                 in-process call
Backend API             State Store                in-process (SQLite)
Learning Loop           Baby Model                 in-process call
Learning Loop           Teacher Bridge             async function call
Learning Loop           State Store                in-process write
Learning Loop           Viz Emitter                in-process call
Teacher Bridge          Ollama                     HTTP localhost:11434
Viz Emitter             Baby Model                 in-process read
Viz Emitter             Frontend (via WS handler)  WebSocket push
Encoder/Decoder         Baby Model                 in-process call
Encoder/Decoder         CLIP model (MLX)           in-process call
```

---

## Component Build Order

The order matters. Each component depends on the ones before it.

```
PHASE 1 — Foundation
  1a. State Store (SQLite schema + read/write helpers)
  1b. Teacher Bridge (Ollama wrapper + test against running model)
  1c. Encoder (CLIP via MLX + text tokenizer)

PHASE 2 — Core Model
  2a. Node + Cluster primitives (basic data structures)
  2b. Forward-Forward update rule (local learning, no backprop)
  2c. Growth operations (BUD, CONNECT, INSERT, PRUNE, DORMANT)
  2d. Baby Model (assembles 2a-2c into one object)

PHASE 3 — Loop
  3a. Curiosity scorer (uncertainty × novelty from model state)
  3b. Question generator (gap → natural language question)
  3c. Learning Loop Orchestrator (assembles full cycle)

PHASE 4 — API + Streaming
  4a. Viz Emitter (model state → JSON delta)
  4b. Backend API (FastAPI, all endpoints, WebSocket)

PHASE 5 — Frontend
  5a. Latent Space Visualizer (Three.js graph, WebSocket consumer)
  5b. Dialogue Feed (REST poll + WebSocket)
  5c. Human Chat interface
  5d. Controls Panel
  5e. Layout + wiring

PHASE 6 — Launcher
  6a. start.sh (process management, health checks, browser open)
```

---

## Technology Choices

```
Backend
  Python 3.11+
  FastAPI          HTTP + WebSocket server
  PyTorch (MPS)    Baby Model (Metal GPU backend on M1)
  MLX              CLIP encoder (Apple Silicon optimized)
  SQLite           State Store (via standard library sqlite3)
  uv               Fast dependency management

Frontend
  React 18         UI framework
  Three.js /       3D latent space visualization
  react-three-fiber
  Tailwind CSS     Styling
  Vite             Dev server + bundler

Teacher
  Ollama           Local LLM runtime
  phi4-mini        Default teacher model (~4GB)

Launcher
  Bash             start.sh
  curl             Health checks
```

---

## What Each Future Component Doc Should Contain

Each component gets its own doc with:
- Precise interface (inputs, outputs, types)
- Internal structure (data flow within the component)
- Growth/change behavior (if applicable)
- Test cases (how to verify it works in isolation)
- Known hard parts (where the implementation gets tricky)
- M1-specific notes (memory, MPS backend, MLX vs PyTorch tradeoffs)

**Component docs to write (in build order):**
1. `spec-state-store.md`
2. `spec-teacher-bridge.md`
3. `spec-encoder-decoder.md`
4. `spec-baby-model.md`  ← largest, split into sub-docs if needed
5. `spec-learning-loop.md`
6. `spec-viz-emitter.md`
7. `spec-backend-api.md`
8. `spec-frontend.md`
9. `spec-start-sh.md`
```
