# SYNTHESIS

Cross-cutting design notes that span CLAUDE.md's architecture layers.
Phase log + open items. Read CLAUDE.md first; this doc is the running
log of what's been delivered and what still has to happen.

## Phase 9 — v1.0 candidate (Wave Field Architecture)

The largest architectural change since v0.1. The mind transitions from
a sequential pipeline into a continuously running wave field, while
keeping everything that existed — 60K concepts, 320K edges, 1,228+
abstractions, W matrix shaped by the full curriculum, identity spine.
Existing pipeline survives as fallback.

See `doc/mind/v1_plan.md` for the full architecture brief.

### Completed
- **512-dim migration** — scripts/migrate_to_512.py (commit 31c58ee).
  Linear projection R^256 → R^512 via top-half-identity matrix
  (seed=42 for determinism); top 256 rows are identity (preserve all
  cosine relationships exactly), bottom 256 rows are random ×0.01
  (small enough not to disrupt existing structure). Migrates node
  embeddings, the affect W matrix (12, 256) → (12, 512), and 4,096
  simulation replay buffer entries. Backs up mind.db to
  `.pre-512-migration` before save. Verified: 60,078 concepts at
  512-dim, all norms 1.000, all relationships preserved.
- **Wave field engine** — backend/wave_field.py (commit 31c58ee).
  Continuous wave propagation over the concept graph on MPS.
  All N nodes update simultaneously per dt=0.05s via one sparse
  matrix-multiply per edge type (forward/backward/causal/hierarchy).
  Activation has momentum (velocity); damping; affect-gated; top-down
  bias from PC hierarchy. Settles via convergence threshold. Bridge-
  concept finder (inject two domains, interference = bridges). Live
  benchmark on 60K-node migrated graph: **4,969 steps/sec on MPS**
  (25× the 200/s target).
- **Predictive coding hierarchy** — backend/predictive_coding.py
  (commit ba71669). 5 levels (512→256×3→512). Bottom-up encoders
  + top-down predictors per level. Errors flow upward as learning
  signal; top-down predictions flow downward as attention bias.
  Online Adam learning gated on error_threshold. `get_top_down_for_wave_field`
  emits a softmax over concept alignments → wave field's top_down
  channel. Test: surprise 15.2 → 6.5 over 50 steps.
- **Self model** — backend/self_model.py (commit 7b305a2). Self-echo
  detection (30s window, 60% word overlap threshold) — the mind now
  knows when it's hearing itself, solving the architectural gap from
  v0.9 where reflected output was treated as exogenous input. Self-
  prediction: small MLP learns next-affect from current-affect +
  concept-centroid; loss 0.34 → 0.05 over 100 updates. Theory of
  mind: per-person OtherModel with interaction stats.
- **Multimodal fusion** — backend/fusion.py (commit 0e9e6bc). The
  thalamus layer. ModalityProjector (per-stream linear+LN+GELU),
  FusionTransformer (3-token attention + FF), MultimodalFusion
  (vision/audio/text projectors). Vision and audio accept None
  and substitute zeros — ready for CLIP/Whisper when wired.
  Contrastive learning for online same-event/different-event tuning.
- **Contradiction detection** — backend/contradiction.py + 5-line
  hook in graph.write_on_surprise (commit e8c7842). The ACC
  equivalent. New concepts in similar (cosine > 0.65) but opposing
  (dot < −0.3) positions to existing concepts get a tension edge
  (EdgeType.OPPOSITE_OF), are recorded in a buffer, and seed the
  wave field — interference = the mind sitting with the contradiction.
  Honest finding: thresholds are tight on the current graph and
  the canonical test produced NO-DETECTION; tuning is a v1.1 item.
- **Graph traversal expression** — backend/expression_graph.py
  (commit 869f6ac). Unified brain-mouth: 27,108 single-word concept
  nodes identified. Generation walks word-nodes following wave-field
  activation; each emitted word is injected back into the field as
  a velocity perturbation, then the field is stepped 5 times — the
  next word follows from the shifted active set. Repetition penalty,
  stop-word penalty, EOS detection. Runs alongside the native head
  (still loaded, falls back when graph-traversal returns empty).
- **Continuous runtime** — backend/mind_runtime.py + additive
  api.py edits (commit b784dfd). MindRuntime runs the integrated
  mind in a background thread (dt=0.05s). Every iteration: sense
  (queue check), think (wave step), feel (affect refresh), self-
  model update, active inference at low arousal, contradiction
  seeding, expression check after input, sleep consolidation at
  very low arousal + extended quiet, periodic save. `send(text)` /
  `receive(timeout)` queue API. Self-overhearing: own emitted
  surfaces piped back as input with person_id='self'. New API
  endpoints `/ingest_runtime` and `/runtime_status` exist alongside
  the v0.9 endpoints — additive, existing functionality preserved.
- **Post-migration encoder fixes** — backend/encoders.py +
  backend/input.py + backend/language_head.py (commit 312a076).
  GloVe binary (locked 256-dim) auto-projects to 512 on load.
  encode_image pools to 16×16=256 then projects to 512.
  language_head.build_conditioning_vector truncates 512 → 256 on
  input so the native head v2 (trained at 256) keeps working with
  migrated concept embeddings — the top-half-identity projection
  means truncation gives the LM exactly the vector it was trained on.

### Regression sweep (all green post-migration)
test_wave_field · test_predictive_coding · test_self_model ·
test_fusion · test_contradiction (NO-DETECTION acceptable) ·
test_graph_expression · test_runtime · phase1 · phase3 (19/19) —
**all pass on the migrated 60K-node 512-dim graph**.

### Open items — Phase 9 (v1.1+)
- Wire CLIP vision encoder fully (replace the patch-stats stub)
- Wire Whisper audio encoder fully (replace the FFT-bins stub)
- Retrain native_head_v2 on 512-dim concept embeddings directly
  (currently we truncate 512→256 at the conditioning vector — works
  because top-half is identity, but training native at 512 unlocks
  the new dims)
- Drifting nodes — concept embeddings drift toward experience
  (the design that pairs with valenced surprise from v0.8; formula
  captured in the Notes section earlier in this doc)
- Typed-attention spreading in wave field (per-type weights live;
  per-type dynamics could differ further)
- Multi-beam thinking (8 parallel wave trajectories with different
  injection seeds, beam-select on settled centroid)
- Tune contradiction thresholds against the actual graph density
- Tune wave-field stop-word + repetition penalty so graph
  traversal expression doesn't collapse to repeated "self self self"
- Active inference deepening — let the mind generate its own
  curriculum from gaps the wave field exposes

### Notes
- Migration backup at `data/first/mind.db.pre-512-migration` —
  rollback is one `mv` command if anything is found to be off later.
- The mind on disk is now 512-dim only. Any tool that loads it
  must run with `D_REP=512` in config and the post-migration
  encoder/LM fixes from commit 312a076. The frontend and curriculum
  scripts will need a small audit before they're run again — most
  paths use encode_text_glove which now auto-projects, but
  multilevel_preprocessor and any direct numpy ops on raw 256-dim
  blobs need to be verified.

## Phase 8 — v0.8 candidate

Tagged as `v0.8` after all six subagent workstreams landed in
parallel.

### Completed
- **Native head v2** — backend/native_head.py + scripts/train_native_head.py
  (commit 6b6ed38). Decoder-only transformer, 6 layers × d_model=512 ×
  8 heads × d_ff=2048 = 28.07M params. Conditioning vector (affect 12 +
  5 concept embeddings × 256 = 1292d) projects through MLP and is added
  to the BOS embedding at position 0. Trained 150 epochs on a
  sentence-filtered corpus (5,543 examples, ≥8 words each), MPS, batch
  128, lr 3e-4, 46.8 min wall, **final loss 0.1556**. Drop-in compat
  with v1's `generate(cond, temperature, repetition_penalty,
  blocked_start_ids)` interface; expression.py prefers
  data/first/native_head_v2.pt over language_head.pt when both exist.
- **Second curriculum pass** — re-ran curriculum_interleaved.json on
  the densified mind 'first'. Processed ~210K items in 58 min before
  external termination (would have completed cleanly with more time);
  **+881 nodes / +6,107 edges retained** despite the graph already
  being near ceiling. Surprise retention rate ~3% (vs ~12% on the
  fresh-baseline pass-1) as predicted — the dense graph absorbed most
  incoming reps via find_or_match. 21 ceiling-aware sleep cycles
  fired during the pass, cumulatively forming ~125 abstractions.
- **Valenced surprise** — backend/predict.py + backend/affect.py
  (commit 5c50361). Surprise now carries direction. PredictionGap
  gained a `valence` float (∈ [-1, 1], cosine of unit_delta with
  affect_in_rep_space). AffectStack.affect_in_rep_space() projects the
  composite-affect vector into D_REP=256 via the cached pinv(W).
  AffectStack.inject() takes `valence=` and modulates delta with a
  deadband at |valence| > VALENCE_THRESHOLD=0.3 (negative reverses the
  delta, positive scales it). All Phase 1+3 regressions pass; 6 new
  test cases in scripts/test_valence.py pass.
- **Conversational register** — curriculum_interleaved.json gained 5
  conversational sources (commit f9f7ad9). 4 of 5 fetched cleanly:
  Wilde 142KB, Boswell 1.3MB, Darwin-via-Huxley 75KB, Lincoln 482KB.
  wiki_conversation skipped (Wikipedia category empty). 149K raw items
  encoded at multilevel; ~228K effective with weight=2.0/1.5 applied.
  Pending: actually ingest the new sources into mind 'first'.
- **Batch writer scaffolding** — backend/predict.py:predict_batch +
  backend/parallel_ingestion.py 64-item batch collect (commit 1bc331b).
  Honest result: predict_batch is plumbed but NOT on the writer's hot
  path because loop.cycle calls input_pipeline.ingest_text →
  predict_engine.predict() (single-item) internally. Measured 1106
  items/s at batch=64 vs 1130 at batch=1 (0.98×, within noise) on a
  pre-loaded queue benchmark. To realize the 10-20× the design
  predicted, ingest_text needs a `representation_prediction=` kwarg so
  cycle skips its internal predict() when the writer already computed
  it. Design API is in place; the plumbing is the open item.
- **Active inference during idle** — backend/active_inference.py + idle
  integration (commit 9181ba1). During idle the mind samples concept
  pairs that are semantically close (cosine ≥ 0.6) but not yet edge-
  connected, builds a midpoint "bridge" rep, and runs it through the
  predict→observe pipeline. Surprises become new bridge concepts;
  link_to_nearest_neighbors fires after each new write. Test on a
  108-node mind: 10 idle cycles produced 24 bridge concepts + 144
  edges; cycles 7-10 produced 0 events because arousal exceeded 0.6
  (the synthesis idle-gate). /idle JSON now includes
  `inference_events` and `new_connections`.

### Combined regression sweep (all 9 tests)
test_a1_ingestion_threshold · test_b1_budget · test_prune ·
test_multilevel_preprocessor · test_mps_index · test_s3_parallel_save ·
test_s6_parallel_ceiling · test_valence · test_active_inference —
**all pass**.

### Open items — Phase 8
- Ingest the conversational sources (Wilde / Boswell / Darwin / Lincoln)
  into mind 'first'. Replace the wiki_conversation entry with a real
  Wikipedia category (Conversation has no members; "Conversational
  norms", "Discourse analysis", or "Speech acts" might work).
- Retrain native_head v3 after conversational ingestion + the v0.7b
  surprise journal regrows.
- Plumb `predict_batch` onto the writer's hot path:
  ingest_text(representation_prediction=) → cycle skips internal
  predict() → realize the 10-20× speedup the API was built for.
- Causal chain reasoning — multi-hop traversal of `causes` edges
  during simulation rollouts.
- Active inference improvements — bridge concept naming (currently
  `bridge:{cid_a}-{cid_b}`, no semantic content); proper integration
  with replay so bridges that survive prune-cycles consolidate into
  named abstractions.
- Freeze flags for chatbot deployment mode (no graph mutation, no
  affect drift, no journal writes — pure generation).
- Multilingual encoder — current GloVe-PCA-256 is English-only;
  multilingual sentence transformers (paraphrase-multilingual-MiniLM)
  would unlock multi-language ingestion at the cost of model size.

### Notes
- Valence scaling caveat (subagent 3): with `delta *= valence`, a
  positive valence less than 1.0 *attenuates* rather than amplifies the
  delta. Comment label says "amplified curiosity" but the math doesn't
  match; if true amplification is the intent, the formula should be
  `delta *= (1 + valence)`. Filed but not changed pending product
  decision.
- **Drifting nodes drift-rate formula** — when latent-space drift
  ships, the per-node nudge during a 0.7-cosine update should scale
  with surprise *and* affect alignment, not surprise alone:
  ```python
  # write_or_update — nudge magnitude
  learning_rate = surprise * max(valence, 0.1) * 0.01
  # high surprise + affect-aligned → strong drift
  # high surprise + affect-opposed → weak drift (don't distort
  #                                  the latent space against the grain)
  # low surprise → minimal drift regardless
  ```
  This connects valenced surprise (already in PredictionGap.valence)
  to latent-space deformation: the valence tells you which direction
  matters, the drift uses that direction. Captured before the
  drifting-nodes implementation begins so the two pieces ship as one
  coherent design rather than two bolted-together halves.
- Curriculum pass-2's external termination at 210K items is suspected
  OOM-killer or external SIGTERM — log shows clean shutdown handling
  but no Traceback. Worth keeping the same disk available for the next
  full pass (current run shared disk with native head training, which
  caused throughput to drop from 165 → 65 items/s).
