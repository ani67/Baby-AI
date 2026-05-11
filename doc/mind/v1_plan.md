# v1.0 — Wave Field Mind Architecture (Implementation Plan)

The mind transitions from a sequential pipeline into a continuously running
wave field. **Keep everything that exists; extend; don't replace yet.**
The existing concept graph (60K nodes, 320K edges), affect engine, identity
spine, native head, and curriculum data are preserved. New components are
added alongside; sequential pipeline remains as fallback throughout.

---

## API ADAPTATIONS (CRITICAL — APPLY THROUGHOUT)

The original spec used some attribute names that don't match this codebase.
Every subagent applies these substitutions in their code:

| Spec code uses | Actual API (this codebase) |
|---|---|
| `loop.graph._nodes` | `loop.graph.nodes` (public attribute) |
| `paths.mind_db` | `paths.db` |
| `g._index.search_k(q, k=N)` | `sims, ids = g._index.search(q, k=N)` (returns tuple) |
| `g._index.to_numpy_matrix()` | does not exist — build via `np.stack([loop.graph.nodes[cid].embedding for cid in loop.graph._index._id_map]).astype(np.float32)` |
| `g._index.__class__(d_rep=512)` | `from backend.graph import MPSConceptIndex; MPSConceptIndex()` (no kwargs) |
| `loop.graph.last_written_id` | does not exist — capture `cid, _edge = graph.write_on_surprise(...)` |
| `loop.input_pipeline.encode_text(text)` | `from backend.encoders import encode_text_glove; encode_text_glove(text, dim=config.D_REP)` |
| `active_inference.run_inference_cycle(now, n_pairs=5)` | `run_inference_cycle(now)` (no kwarg) |
| `gap.gap_signal` | not a field — user's code uses `hasattr` check, leave as-is |

**ConceptNode** has these fields (verified): `concept_id`, `name`, `embedding`,
`affect_trace`, `activation_count`, `last_activated`, `created_at`,
`surprise_at_birth`. `AffectTrace.running_state` exists.

**PredictionGap** has: `magnitude`, `surprise_score`, `is_surprise`,
`was_new_write`, `concept_id`.

**EdgeType** has: `IS_A`, `CAUSES`, `SIMILAR_TO`, `OPPOSITE_OF`, `REFERS_TO`,
`CO_OCCUR` (all referenced in spec exist).

`AffectStack.W` is shape `(N_AFF=12, D_REP)` — migration script must expand
to `(12, 512)`.

---

## SUBAGENT 1 — 512-dim migration + wave field substrate

**Owns:** `backend/wave_field.py` (new), `scripts/migrate_to_512.py` (new),
`backend/config.py` (single change: `D_REP = 512`)

**Do NOT touch:** anything else.

### Part A — config change

`backend/config.py`: change `D_REP = 256` to `D_REP = 512`. Add a comment
explaining the migration.

### Part B — migration script

`scripts/migrate_to_512.py`: load mind, project all node embeddings 256 →
512 via top-half-identity matrix (top 256 rows preserve, bottom 256 rows
random ×0.01). Migrate W matrix `(12, 256) → (12, 512)` same way. Rebuild
MPS index at 512-dim. Backup `mind.db` to `mind.db.pre-512-migration`
before save. Save migrated state.

### Part C — wave_field.py

Full code in section ARCH-1 below. Implements `WaveField` class — continuous
wave propagation over graph, GPU tensor state, typed adjacency matrices per
edge type, `inject` / `inject_representation` / `update_affect_gate` /
`step` / `step_n` / `run_until_settled` / `get_top_concepts` /
`get_field_centroid` / `get_bridge_concepts` / `add_node` / `rebuild_if_dirty`.

### Test

`test_wave_field.py`: load mind 'first', wrap in WaveField, inject 3
justice-named concepts, run until settled, verify > 200 steps/s and
top concepts include philosophy/justice-adjacent.

### Commit message

`feat: wave field engine + 512-dim migration`

---

## SUBAGENT 2 — Predictive Coding Hierarchy

**Owns:** `backend/predictive_coding.py` (new)

**Do NOT touch:** wave_field.py, graph.py, affect.py, predict.py.

Implements 5-level PC hierarchy: PCLevel (encoder + predictor), then
PredictiveCodingHierarchy with `.process(input_rep, learn=True)` returning
dict with states / errors / surprise / top_down / should_crystallize.
Online Adam optimizer, MPS device, error-threshold gated learning.

`get_top_down_for_wave_field(graph_index, n_concepts=50)` converts L5
state into expected concept activations for the wave field.

Full code in section ARCH-2. Tests verify error decreases with repeated
exposure.

Commit: `feat: predictive coding hierarchy — 5-level online learning`

---

## SUBAGENT 3 — Self Model

**Owns:** `backend/self_model.py` (new)

**Do NOT touch:** any existing files.

Implements:
- `SelfModel` class — self-echo detection (30s window), self-prediction via
  `SelfPredictor` neural net (affect+centroid → next affect), theory of mind
  via `OtherModel` (per-person interaction stats), `register_output`,
  `is_self_echo`, `self_echo_resonance`, `update`, `update_other`,
  `what_does_person_want`.

Full code in section ARCH-3. Tests verify echo detection within window,
self-prediction loss decreases.

Commit: `feat: self model — self-echo + theory of mind + self-prediction`

---

## SUBAGENT 4 — Multimodal Fusion

**Owns:** `backend/fusion.py` (new)

**Do NOT touch:** any existing files.

Implements thalamus-layer fusion: `ModalityProjector` (512→512 linear),
`FusionTransformer` (3 tokens × attn × ff), `MultimodalFusion` with
`.fuse(text_rep, vision_rep=None, audio_rep=None)` → 512-dim, and
contrastive learning. Vision/audio are zero-stubs until CLIP/Whisper
are wired.

Full code in section ARCH-4. Tests verify text-only fusion outputs 512-dim
unit vector.

Commit: `feat: multimodal fusion — thalamus layer with text/vision/audio stubs`

---

## SUBAGENT 5 — Contradiction Detection

**Owns:** `backend/contradiction.py` (new), ~5-line addition to
`backend/graph.py` (inside `write_on_surprise` after writing).

**Do NOT touch:** wave_field.py, affect.py, predict.py, api.py.

`ContradictionDetector` finds new concepts in similar semantic
neighborhood (cosine > 0.65) but opposing direction (dot < -0.3),
records `Contradiction`, writes a tension edge via `EdgeType.OPPOSITE_OF`,
seeds wave field if available, exposes `get_active_contradictions`
and `attempt_resolution`.

`graph.py` add: in `write_on_surprise` after `cid` is allocated, call
`self._contradiction_detector.check_new_concept(cid, now)` if attribute is
set. Guard with `hasattr`.

Full code in section ARCH-5. Tests verify opposing-direction concepts are
flagged.

Commit: `feat: contradiction detection — ACC equivalent`

---

## SUBAGENT 6 — Graph Traversal Expression

**Owns:** `backend/expression_graph.py` (new)

**Do NOT touch:** expression.py (existing), native_head files.

`GraphTraversalExpression`: identify word-concept nodes (single-word
names from graph.nodes), then generate by spreading-activation walk
through word-nodes (`inject` chosen word → step wave field → pick next
word from new activation), with repetition penalty, stop-word penalty,
EOS detection.

`generate(wave_activation, affect, max_words)` and `generate_and_score`
methods. Runs alongside native head as alternative path.

Full code in section ARCH-6. Tests verify it produces > 0 words from a
settled wave field.

Commit: `feat: graph traversal expression — unified brain-mouth`

---

## SUBAGENT 7 — Continuous Runtime

**Owns:** `backend/mind_runtime.py` (new). Also makes additive lifespan
+ endpoint changes to `backend/api.py` (lifespan startup + /ingest and
/status using the runtime).

**Do NOT touch:** any other existing files.

`MindRuntime`: background thread, dt=0.05s, integrates wave field + PC
hierarchy + self model + fusion + contradiction + graph expression +
existing affect/graph/persistence. `send(text)` / `receive(timeout)`
queue-based API. Periodic save. Sleep consolidation. Active inference
during low arousal. Self-echo loop (own output piped back as input
with `person_id='self'`).

`MindRuntime.load(mind_name)` class method loads everything.

`backend/api.py`: in `lifespan`, after the existing init, start a
`MindRuntime` instance and store in `state["runtime"]`. New endpoints
`/ingest_runtime` and `/runtime_status` use it. **DO NOT remove existing
endpoints** — additive only; the v0.9 ingestion path stays as fallback.

Full code in section ARCH-7. Tests verify runtime starts, accepts input,
produces output within 10s.

Commit: `feat: continuous mind runtime — wave field + all layers integrated`

---

## RECONCILIATION (parent agent, AFTER all 7 subagents commit)

The parent (not a subagent) handles these in sequence, with user
confirmation gates on destructive steps:

1. Run `python3 scripts/migrate_to_512.py` ← **destructive; needs user OK**
2. Run all 7 component tests + `test_runtime.py`
3. Run `phase1.py` + `phase3.py` regression (must still pass)
4. Benchmark wave field (>200 steps/s on MPS) + end-to-end latency (<3s)
5. Update SYNTHESIS.md with v1.0 phase log
6. Tag `v1.0` and push ← **shared state; needs user OK**

---

## ARCH-1 through ARCH-7 — Full Code

Spawned subagents receive their assigned ARCH-N section as inline code
in their prompt; the code is the user's original prompt with the API
adaptations applied. See the conversation prompt for the canonical
spec listings.
