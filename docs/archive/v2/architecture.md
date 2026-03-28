# Architecture & File Map

## System Overview
```
User uploads images → Curriculum pool → Learning loop picks item →
Teacher (LLaVA/Ollama) answers → CLIP encodes → Baby model forward pass →
Forward-Forward update → Growth check → Viz emitter → WebSocket → Frontend 3D view
```

## Backend (FastAPI, Python)

### Core Model
- **`model/node.py`** — Node dataclass. weights(512,), bias(1,). `activate()` = tanh(w·x+b). `ff_update()` = Forward-Forward local update.
- **`model/cluster.py`** — Cluster dataclass. Groups of nodes. `forward()` activates all nodes, returns weighted sum. Uses incoming edge signals with actual edge strength.
- **`model/graph.py`** — Graph registry. Clusters, edges, traversal methods. `to_json()` serializes full state. Sends `cluster_type` (not `type`) to match frontend.
- **`model/baby_model.py`** — Main model. Forward pass routes through graph layer by layer. `update()` runs FF learning. `growth_check()` triggers BUD/CONNECT/PRUNE/INSERT/EXTEND/DORMANT. Max 64 clusters cap. `restore_from_checkpoint()` rebuilds from saved state.
- **`model/growth.py`** — GrowthMonitor tracks coactivation, residuals, activation history. Growth operations: `bud()` splits clusters, `insert_layer()`, `extend_top()`.
- **`model/forward_forward.py`** — PlasticitySchedule for learning rate decay.

### Learning Loop
- **`loop/orchestrator.py`** — 12-step learning cycle: observe → score → select → question → ask teacher → encode → predict → update → measure → growth → checkpoint → emit. Checkpoint every 100 steps. Decodes model prediction for display.
- **`loop/curriculum.py`** — Manages training item pools. Loads images from `data/stage0/` dirs + text concepts from `concepts.txt`. `next_item()` picks random from pool.
- **`loop/question_gen.py`** — Generates questions. IMAGE_TEMPLATES for image items, concept templates for text items.
- **`loop/curiosity.py`** — CuriosityScorer (basic, scores item novelty).

### Encoders
- **`encoder/clip_mlx.py`** — CLIPWrapper for CLIP-ViT-B/32.
- **`encoder/encoder.py`** — ImageEncoder, TextEncoder, VideoEncoder wrappers.
- **`encoder/decoder.py`** — TextDecoder: maps 512-dim vector → nearest words in CLIP vocab space.

### Teacher
- **`teacher/bridge.py`** — TeacherBridge talks to Ollama (LLaVA model). Sends image bytes for vision questions.

### Visualization
- **`viz/emitter.py`** — VizEmitter streams graph diffs over WebSocket. Runs projector every `projection_interval` steps. Sends node_positions, activations, dialogue (with model_answer), growth events.
- **`viz/projector.py`** — Force-directed 3D layout. Clusters attract via edges, repel each other. Layer gravity for vertical structure. PCA offsets within clusters. Warm-starts from previous positions.
- **`viz/diff.py`** — Computes minimal graph diffs (add/remove/update nodes/clusters/edges).

### State
- **`state/store.py`** — SQLite persistence. Checkpoints (pickle .pt files), dialogues, graph events, snapshots, human chat.
- **`state/schema.sql`** — 5 tables: dialogues, graph_events, latent_snapshots, model_checkpoints, human_chat.

### Config & Entry
- **`config.py`** — projection_interval=10, snapshot_interval=50, initial_clusters=4, nodes_per_cluster=8.
- **`main.py`** — FastAPI app. Mounts static files for images. Endpoints: /start, /pause, /resume, /step, /reset, /status, /stage, /speed, /chat, /image, /image-url, /images-bulk, /snapshot, /health. Restores from checkpoint on startup.
- **`seed_data.py`** — Downloads ~141 images across 20 categories + 106 text concepts.

## Frontend (React + Vite + Three.js + Zustand)

### Store
- **`store/graphStore.ts`** — Zustand store for graph state: nodes, clusters, edges, activations, growthQueue. Actions: setSnapshot, applyDiff, setActivations, updatePositions, triggerGrowthEvents.
- **`store/dialogueStore.ts`** — Dialogue entries with question, answer, model_answer, curiosity_score, is_positive, image_url.
- **`store/loopStore.ts`** — Loop status: state, step, stage, delay_ms, graph_summary.

### Components
- **`components/LatentSpace.tsx`** — Three.js 3D visualization. NodeSphere (size=activation, pulse=active, dim=dead), ClusterEdge (opacity=strength), GridPlane, Legend overlay with dismiss/show toggle.
- **`components/DialogueFeed.tsx`** — Shows Q/T/M (question, teacher answer, model answer). Image thumbnails. Positive/negative indicators.
- **`components/Controls.tsx`** — Status bar, pause/step/reset, speed slider, stage buttons, image URL input (multiline for bulk).
- **`components/HumanChat.tsx`** — "Talk to it" panel. Sends text to /chat endpoint, gets model's decoded response.

### Hooks
- **`hooks/useWebSocket.ts`** — Connects to backend WS. Handles snapshot, step (diff + activations + positions + growth + dialogue), status messages.
- **`hooks/useGraphState.ts`** — Helper: `clusterCentroid()` with null-safe pos handling.

### Lib
- **`lib/colors.ts`** — CLUSTER_COLORS: integration=#3b7fff, transformation=#9b59b6, arbitration=#e67e22, routing=#1abc9c, dormant=#2a2a2a.
- **`lib/constants.ts`** — ANIMATION constants, MAX_DIALOGUE_ENTRIES, BASE_URL.
- **`lib/api.ts`** — HTTP helpers with error handling.

### Styles
- **`styles/global.css`** — Dark theme. Graph legend styles, model-answer green text, dialogue-thumb images.
