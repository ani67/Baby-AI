# SYNTHESIS

Cross-cutting design notes that span CLAUDE.md's architecture layers.
Phase log + open items. Read CLAUDE.md first; this doc is the running
log of what's been delivered and what still has to happen.

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
- Curriculum pass-2's external termination at 210K items is suspected
  OOM-killer or external SIGTERM — log shows clean shutdown handling
  but no Traceback. Worth keeping the same disk available for the next
  full pass (current run shared disk with native head training, which
  caused throughput to drop from 165 → 65 items/s).
