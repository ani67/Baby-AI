# Fixes Log — All Bugs Found and Fixed

## CRITICAL Fixes

### 1. Weight Normalization Killing Learning
**File:** `model/node.py`
**Problem:** `F.normalize(self.weights, dim=0)` after every weight update constrained weights to unit sphere, erasing all magnitude changes from learning.
**Fix:** Removed the normalize call after weight updates. Weights now grow freely.

### 2. Forward Pass Skipped at Stage 0
**File:** `loop/orchestrator.py`
**Problem:** Forward pass only ran at stage >= 1. At stage 0, activations were always empty `{}`, so nothing learned.
**Fix:** Always run forward pass regardless of stage. Use answer_vectors[0] as input_vec.

### 3. Layer 1 Clusters Unreachable
**File:** `model/baby_model.py`
**Problem:** No initial edges between layer 0 and layer 1 clusters. Only 2 of 4 clusters were visited during forward pass.
**Fix:** Added initial edges from all layer 0 → layer 1 clusters with strength=0.2 in `_init_clusters()`.

### 4. Edge Strengths Ignored in Forward Pass
**File:** `model/cluster.py`
**Problem:** Incoming edge signals were always multiplied by hardcoded 0.3 regardless of actual edge strength.
**Fix:** Changed to pass `(signal, strength)` tuples. Cluster forward uses actual edge.strength.

### 5. is_positive Always True at Stage 0
**File:** `loop/orchestrator.py`
**Problem:** No negative examples at stage 0. Model never learned contrast.
**Fix:** Every 3rd step at stage 0 is negative (feeds random vector instead of answer).

## HIGH Priority Fixes

### 6. 3D Positions Never Sent to Frontend
**Files:** `viz/emitter.py`, `config.py`, `model/graph.py`
**Problem (multi-part):**
- `config.py` had `projection_interval=200` (too rare). Changed to 10.
- `to_json()` hardcoded `pos: None`. Changed to `getattr(n, 'pos', None)`.
- Emitter ran `to_json()` BEFORE `reproject()`. Positions were captured before being computed.
- Frontend only updated positions for newly-added nodes, not existing ones.
**Fix:** Reordered to reproject→toJson. Added `node_positions` dict in step messages. Added `updatePositions` action in graphStore. Config projection_interval=10.

### 7. cluster_type Field Mismatch
**File:** `model/graph.py`
**Problem:** Backend sent `"type"` but frontend expected `"cluster_type"`.
**Fix:** Changed `to_json()` to send `"cluster_type": c.cluster_type`.

### 8. 3D View Crash — Null Positions
**File:** `hooks/useGraphState.ts`
**Problem:** `clusterCentroid()` accessed `n.pos[0]` when pos was null. TypeError crash.
**Fix:** Null-safe checks: filter nodes with `n.pos`, use `n.pos[0] ?? 0`.

### 9. Coactivation Threshold Too High
**File:** `model/growth.py`
**Problem:** Activation threshold was 0.1 but actual activations were ~0.03. Nothing counted as "active."
**Fix:** Lowered threshold from 0.1 → 0.01.

### 10. Only 2 Curriculum Items Cycling
**File:** `loop/curriculum.py`
**Problem:** `top_n = len(scored) // 5` with 14 items = 2. Only top 2 items ever selected.
**Fix:** Changed to `random.choice(pool)` — uniform random from entire pool.

### 11. Residuals Not Recorded for INSERT
**File:** `model/growth.py`
**Problem:** `record_step()` didn't record residuals between adjacent cluster pairs. INSERT operation could never fire.
**Fix:** Added residual recording loop over all edges in `record_step()`.

## MEDIUM Priority Fixes

### 12. Image URL Fetching Failures
**File:** `main.py`
**Problem:** Wikipedia URLs returned 403/429. No User-Agent header, no redirect following.
**Fix:** Added `follow_redirects=True`, browser User-Agent header, Referer header. Error message suggests downloading and uploading directly.

### 13. Label Sanitization
**File:** `main.py`
**Problem:** Labels from URLs contained special chars, creating weird folder names ("400", "test").
**Fix:** `re.sub(r'[^a-zA-Z0-9 ]', '', label).strip()`

### 14. Backend Stale Process on Port 8000
**File:** `start.sh`
**Problem:** Old backend process sometimes lingered on port 8000, blocking new starts.
**Fix:** Added `lsof -ti:$BACKEND_PORT | head -1` check and kill in start.sh.

### 15. Step Counter Stuck at 0 in UI
**File:** `hooks/useWebSocket.ts`
**Problem:** Status bar only updated from `type: 'status'` messages, not step messages.
**Fix:** Added loopStore update from step messages (reads step, stage, graph_summary).

### 16. Question Generation — "Tell me about test"
**File:** `loop/question_gen.py`
**Problem:** Image items with actual files were using label-based templates ("Tell me about test").
**Fix:** Added IMAGE_TEMPLATES list ("What is this?", "Name what you see.", etc.) for image items with files.

## Feature Additions

### 17. Checkpoint Persistence Across Restarts
**Files:** `model/baby_model.py`, `main.py`, `loop/orchestrator.py`
**What:** Added `restore_from_checkpoint()` to BabyModel — rebuilds graph from saved pickle (clusters, nodes with weights, edges with metadata, ID counters). On startup, main.py checks for latest checkpoint and restores. Checkpoint interval reduced from 500 → 100 steps.

### 18. Growth Cap
**File:** `model/baby_model.py`
**What:** `max_clusters = 64`. When active clusters >= 64, BUD/INSERT/EXTEND are skipped. PRUNE/DORMANT still work to reclaim space. Growth check interval: 50 steps.

### 19. Force-Directed 3D Layout
**File:** `viz/projector.py`
**What:** Replaced geometric ring layout with force-directed simulation. Clusters repel (inverse-square), connected clusters attract (spring × edge strength), gentle layer gravity for vertical structure. Warm-starts from previous positions. Produces organic t-SNE-like layouts.

### 20. Graph Legend
**File:** `components/LatentSpace.tsx`, `styles/global.css`
**What:** Overlay legend explaining cluster types (color swatches), layout axes, node size/pulse/dim meaning, edge opacity. Dismissible with `×`, restorable with `?` button.

### 21. Model Answer Display
**Files:** `loop/orchestrator.py`, `viz/emitter.py`, `store/dialogueStore.ts`, `components/DialogueFeed.tsx`, `styles/global.css`
**What:** Decodes model's prediction vector into words (via TextDecoder) and sends as `model_answer` in step messages. Frontend shows as green "M" line alongside teacher's "T" answer.

### 22. Training Data Seeding
**Files:** `seed_data.py`, `loop/curriculum.py`
**What:** Script downloads ~141 images across 20 categories (dog, cat, bird, fish, horse, cow, elephant, lion, car, truck, bicycle, airplane, boat, tree, flower, apple, banana, house, ball, baby) from Lorem Flickr. Also writes 106 text concepts (colors, shapes, emotions, actions, body parts, nature, food, family, objects, animals). Curriculum loads concepts from `data/stage0/concepts.txt`.

### 23. Image Display in Dialogue
**Files:** `components/DialogueFeed.tsx`, `store/dialogueStore.ts`, `styles/global.css`
**What:** Dialogue entries show image thumbnails (64x64) when available. Static file serving via FastAPI mount at `/images/data`.

### 24. Bulk Image URL Upload
**Files:** `main.py`, `components/Controls.tsx`
**What:** Multiline textarea for pasting multiple image URLs. Backend `/images-bulk` endpoint. Enter submits, Shift+Enter for newlines.
