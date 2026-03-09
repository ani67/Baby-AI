# Build Decisions Log

---

## Component 1 — Base Inference (complete)

- mlx_lm 0.31.0: temperature no longer passed as temp= kwarg
  use make_sampler(temp=temperature) → sampler= instead
- model loads at import, both inference calls equally fast
- test verifies both calls complete in <5s
- time taken: ~15 minutes

---

Component 2 complete. Time: ~15 min.
- LoRA B matrix zero-initialized by default in mlx_lm
- freeze/unfreeze: _model.freeze() then
  _model.unfreeze(keys="lora_a/lora_b", strict=False)
- 112 trainable parameters total
- scale in mlx_lm config = LoRA alpha (16.0)
- adapter saves as .safetensors + adapter_config.json
- loss before: 3.171875, loss after: 3.16796875 on one step

---

Component 3 complete.
- Metal is NOT thread-safe: single _model_lock serialises all MLX ops
- training step: ~0.23s, inference: ~0.6s, worst case: ~0.83s
- chat template applied in query() — critical for response quality
  without it: 6s for 512 tokens, verbose, with it: ~0.6s concise
- atomic swap: rename B→A, recreate B from A
- start()/stop() lifecycle required before using query/submit_correction

---

Component 4 complete.
- ChromaDB built-in all-MiniLM-L6-v2 (ONNX) — no separate sentence-transformers
- init() required before any operations, re-callable safely
- sync on init: JSON → ChromaDB re-indexing automatic
- src/ needs __init__.py to work as a package
- get_episode_by_id(), update_episode(), get_episode_count() added
- times_referenced increments on every similarity retrieval

---

Component 5 complete.
- Hook pattern: component4._score_fn = None, set by component5 at import
- No circular dependency — component4 unaware of component5
- "Same prompt type corrected before": exact prompt match, tune later
- round(score, 4) to avoid floating-point precision artifacts
- learning_rate linear interpolation: score 0.1->1e-5, 0.5->1e-4, 1.0->5e-4

---

Component 6 complete.
- Held-out set: top 5 selected episodes (practical for small datasets)
- Per-episode Adam optimizer — fresh instance per episode
- Pruning outside _model_lock — pure Python, no MLX
- Safety: snapshot before consolidation, revert if loss >10% increase
- Triggers: 100 episodes, 24hrs, or store >500
- start()/stop() lifecycle, checks every 60s
- Logs to data/consolidation_log.jsonl
- Slight loss increase on synthetic data normal — real corrections behave better

---

Component 7 complete.
- DefaultEmbeddingFunction() from chromadb.utils — same ONNX MiniLM model ChromaDB uses
- Ring buffers via collections.deque(maxlen=N) for all four metrics
- State recomputed every 10 inferences (_STATE_INTERVAL)
- uncertainty_flag: bool read by Component 8, True when uncertainty > 0.7
- Importance boost: replaces component4._score_fn with 1.5x wrapper when uncertain
- Novelty: cosine distance from prompt embedding to centroid of stored episodes
- Coherence: proportion of similar-prompt pairs with consistent responses
- State logged to data/state_log.jsonl as JSONL

---

Component 8 complete.
- Ollama phi4-mini: checked via GET /api/tags, queried via POST /api/generate (stream=False)
- Gemini Flash: uses generativelanguage.googleapis.com/v1beta REST API, key from GEMINI_API_KEY env var
- Consensus: word overlap ratio >0.3 on ANSWER: sections = agreement
- Rate limiter: deque of timestamps, prune entries older than 1 hour
- Background worker: queue.Queue + single daemon thread, waits for rate limit
- Training-worthy results stored to component4 episodic memory automatically
- All 14 tests use unittest.mock — no live Ollama or API key required
- Graceful degradation: returns None when no teachers available or rate-limited

---

Component 8 complete and live-verified.
- phi4-mini confirmed working at localhost:11434
- Single source confidence: 0.5, training_worthy requires 2 sources
- Availability check: GET /api/tags timeout=2, POST timeout=30s
- Reasoning trace format: REASONING: [steps] ANSWER: [conclusion]
- start()/stop() lifecycle for background thread
- Rate limit: 10/hour, excess queued
- Logs to data/teacher_log.jsonl

---

Component 9 complete.
- ChromaDB collection "knowledge_store" — separate from "episodes", same chroma_db/ directory
- Fact dataclass: id, fact, source, confidence, times_retrieved, timestamp
- Persistence: facts.json + ChromaDB (same pattern as component4)
- init() required before operations, re-callable safely
- Hooks: monkey-patch component3.query and submit_correction at import time
- Hooks are no-ops when component9 not yet initialised — zero overhead on component3 tests
- Prompt augmentation: retrieve top-5 facts, prepend confidence > 0.7 as "Relevant facts: ... Now answer: ..."
- Auto-population: submit_correction stores fact at confidence 0.95, source "user_correction"
- 13 tests, all using temp directories for isolation

---

Component 9 complete and verified.
- ChromaDB collection: "knowledge_store", same instance as component4
- Monkey-patch pattern: patches component3 at import time
- Corrections -> facts at confidence 0.95 automatically
- Teacher training_worthy -> facts at confidence 0.8
- Confidence > 0.7 threshold for prompt augmentation
- Prompt format: "Relevant facts: [f1]. [f2]. Now answer: [prompt]"
- Persistence: facts.json + ChromaDB, survives restart
- 83 facts accumulated across all test sessions (expected, correct)

---

Component 10 complete.
- chat() retrieves facts + episodes, builds augmented prompt, runs inference
- Bypasses component9's fact hook (calls _original_query) to avoid double augmentation
- Episode stored synchronously (next chat() needs it); state update + teacher in background
- Augmented prompt format: past interactions context + relevant facts + "Now answer: [prompt]"
- start(data_dir) initialises component4, component9, then starts component3/6/8
- stop() stops component8/6/3 in reverse order
- Self-narrative: computed from component7 state + component4 episode history
- End-to-end test passes: Canberra correction survives simulated restart
- Test teardown resets component9._initialised to avoid stale ChromaDB in later test modules
- 82 total tests, all passing
