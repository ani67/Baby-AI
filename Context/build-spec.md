# Build Spec: Continuous Local AI on M1
*The translation layer between architecture and code.*
*Give this file + the MD files from the design session to Claude Code at the start of each build session.*

---

## How to use this document

Each component has:
- **What it is** — one sentence
- **Inputs / Outputs** — the interface contract
- **Implementation notes** — decisions already made
- **Open decisions** — things you must decide before Claude Code can proceed
- **Success test** — how you know it works
- **Session prompt** — copy-paste to start the Claude Code session

Build in order. Each session assumes the previous component is working and tested.

---

## Stack decisions (made)

```
LANGUAGE:         Python 3.11+
ML FRAMEWORK:     MLX (Apple's framework, optimised for M1 unified memory)
BASE MODEL:       Llama-3.2-3B (fits in ~3.5GB, leaves headroom)
LORA LIBRARY:     MLX-LM (Apple's own, best M1 support)
VECTOR STORE:     ChromaDB (local, no server needed, Python-native)
TEACHER API:      Ollama (local) + at least one free-tier API
PERSISTENCE:      JSON files for episodic store, SQLite for state log
PACKAGE MANAGER:  pip + requirements.txt
TEST RUNNER:      pytest
```

---

## Component map

```
COMPONENT 1: Base inference
COMPONENT 2: LoRA adapter training
COMPONENT 3: Double-buffer (simultaneous train + inference)
COMPONENT 4: Episodic store (persistent memory)
COMPONENT 5: Importance scorer (amygdala)
COMPONENT 6: Consolidation loop (sleep)
COMPONENT 7: Internal state monitor (proto-self)
COMPONENT 8: Teacher ensemble (on-demand training data)
COMPONENT 9: Knowledge store (external facts, no hallucination)
COMPONENT 10: Orchestrator (ties everything together)

BUILD ORDER: 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10
DO NOT skip ahead. Each component's test must pass before building the next.
```

---

## COMPONENT 1: Base Inference

**What it is:** Load the base model and run inference. Nothing else. The foundation everything else sits on.

### Interface

```python
# inputs
model_name: str          # "mlx-community/Llama-3.2-3B-Instruct-4bit"
prompt: str
max_tokens: int = 512
temperature: float = 0.7

# outputs
response: str
tokens_per_second: float
token_count: int
```

### Implementation notes

```
- Use mlx_lm.load() to load model and tokenizer
- Use mlx_lm.generate() for inference
- Model should load ONCE at startup, not per query
- Store model and tokenizer as module-level singletons
- Measure tokens/sec on every call (needed for state monitor later)
- 4-bit quantized model: "mlx-community/Llama-3.2-3B-Instruct-4bit"
  (~1.8GB, leaves ~12GB for everything else)
```

### Open decisions

```
NONE — all decisions made above
```

### Success test

```python
# test_component1.py
def test_basic_inference():
    response, tps, count = run_inference("What is 2 + 2?")
    assert len(response) > 0
    assert tps > 5.0          # should get at least 5 tok/s on M1
    assert count > 0

def test_model_loads_once():
    # call inference twice, model should not reload
    import time
    t1 = time.time(); run_inference("hello"); t1 = time.time() - t1
    t2 = time.time(); run_inference("hello"); t2 = time.time() - t2
    assert t2 < t1 * 0.5      # second call much faster (no loading)
```

### Session prompt

```
I am building a continuous local AI system on M1.
Stack: Python 3.11, MLX, mlx_lm, Llama-3.2-3B-Instruct-4bit.

Build Component 1: Base Inference.

Requirements:
- load "mlx-community/Llama-3.2-3B-Instruct-4bit" using mlx_lm.load()
- model and tokenizer loaded ONCE at module import, not per call
- function signature: run_inference(prompt, max_tokens=512, temperature=0.7)
  returns: (response: str, tokens_per_second: float, token_count: int)
- measure tokens/sec on every call
- write test_component1.py with the two tests I'll describe

Keep it minimal. No classes unless necessary. I want to understand each line.
```

---

## COMPONENT 2: LoRA Adapter Training

**What it is:** Load a LoRA adapter on top of the base model, run a single training step on one example, verify the adapter changed.

### Interface

```python
# inputs
adapter_path: str          # path to adapter weights directory
training_example: dict     # {"prompt": str, "completion": str}
learning_rate: float = 1e-4

# outputs
loss_before: float
loss_after: float
adapter_changed: bool      # True if weights actually updated
```

### Implementation notes

```
- LoRA rank: start with rank=8 (balance capacity vs memory)
- LoRA alpha: 16 (standard 2× rank)
- Target modules: q_proj, v_proj (standard for attention)
- Adapter size at rank=8: ~50MB for 3B model
- Use mlx_lm.lora for adapter creation and training
- Single training step first — do NOT build batching yet
- Adapter saves to disk after every update (persistence)
- Base model weights must NOT change during this step
  verify by checksumming base model weights before/after
```

### Open decisions

```
DECISION: initial adapter
  option A: random initialisation (standard LoRA init)
  option B: zero initialisation (adapter starts as identity)
  RECOMMENDATION: option B (zero init)
  reason: model behaviour unchanged at start,
          only changes as it learns
          safer starting point
```

### Success test

```python
def test_adapter_trains():
    example = {
        "prompt": "The capital of Australia is",
        "completion": " Canberra"
    }
    loss_before, loss_after, changed = train_one_step(example)
    assert changed == True
    assert loss_after < loss_before   # loss should decrease

def test_base_model_unchanged():
    import hashlib
    # hash base model weights before and after training step
    hash_before = hash_model_weights()
    train_one_step({"prompt": "test", "completion": " test"})
    hash_after = hash_model_weights()
    assert hash_before == hash_after  # base never changes
```

### Session prompt

```
Component 1 is working: run_inference() returns responses at >5 tok/s.

Now build Component 2: LoRA Adapter Training.

Requirements:
- use mlx_lm.lora to create a LoRA adapter (rank=8, alpha=16)
- target modules: q_proj, v_proj
- zero-initialise the adapter (not random)
- function: train_one_step(adapter_path, training_example, learning_rate=1e-4)
  returns: (loss_before, loss_after, adapter_changed)
- adapter saves to disk after every update
- base model weights must not change — verify with a checksum test
- write test_component2.py

One training example = one {"prompt": str, "completion": str} dict.
No batching yet.
```

---

## COMPONENT 3: Double-Buffer (Simultaneous Train + Inference)

**What it is:** Two copies of the LoRA adapter. Inference always reads from buffer A. Training always writes to buffer B. Atomic swap every N steps. Inference never blocks.

### Interface

```python
# inputs
swap_interval: int = 100   # swap buffers every N training steps

# the system exposes two functions after init:
query(prompt: str) -> str                          # always fast, never blocks
submit_correction(prompt: str, correct: str)       # queues for background training
```

### Implementation notes

```
BUFFER STRUCTURE:
  buffer_A/   ← inference always reads this
    adapter_config.json
    adapters.npz
  buffer_B/   ← training always writes this
    adapter_config.json
    adapters.npz

SWAP MECHANISM:
  when training step count hits swap_interval:
    copy buffer_B → buffer_A   (atomic: rename, not copy)
    reload inference adapter from buffer_A
    reset training step count

THREADING MODEL:
  main thread:       handles inference queries (always responsive)
  background thread: handles training steps (never blocks main)
  swap:              main thread reloads adapter (brief, <100ms)

MEMORY:
  both adapters resident in unified memory simultaneously
  ~50MB × 2 = ~100MB total for both buffers
  negligible relative to 16GB
  
CRITICAL:
  the training thread must NEVER write to buffer_A
  the inference thread must NEVER write to buffer_B
  enforce with a simple lock on each buffer path
```

### Open decisions

```
DECISION: swap_interval
  too small: frequent reloads, potential quality instability
  too large: slow adaptation (corrections take longer to activate)
  RECOMMENDATION: 100 steps for testing, tune later

DECISION: what happens to in-progress inference during swap?
  RECOMMENDATION: finish current inference with old adapter,
                  then swap. Do not interrupt mid-generation.
```

### Success test

```python
def test_inference_never_blocks():
    import time
    # submit 50 corrections rapidly
    for i in range(50):
        submit_correction(f"prompt {i}", f"completion {i}")
    
    # inference should still be fast
    t = time.time()
    response = query("What is 2 + 2?")
    elapsed = time.time() - t
    assert elapsed < 3.0    # never more than 3s even during heavy training

def test_adapter_actually_updates():
    # query before correction
    r1 = query("The capital of Australia is")
    # submit correction 10 times (enough to trigger swap)
    for _ in range(10):
        submit_correction("The capital of Australia is", " Canberra")
    # wait for swap
    time.sleep(5)
    # adapter should have moved toward Canberra
    r2 = query("The capital of Australia is")
    # r2 should be more likely to contain "Canberra" than r1
    # (measure by querying the adapter's loss on the example)
```

### Session prompt

```
Components 1 and 2 are working.
- run_inference() works at >5 tok/s
- train_one_step() updates adapter without touching base model

Now build Component 3: Double-Buffer.

Requirements:
- two adapter directories: buffer_A/ (inference) and buffer_B/ (training)
- inference always reads buffer_A, never blocks
- training always writes buffer_B, runs in background thread
- atomic swap every 100 training steps: copy B→A, reload inference adapter
- expose two functions: query(prompt) and submit_correction(prompt, correct)
- submit_correction is non-blocking: queues work, returns immediately
- write test_component3.py

The main thread must never wait for training. If I call query() during heavy
correction submission, it should respond in <3 seconds.
```

---

## COMPONENT 4: Episodic Store

**What it is:** Persistent memory of interactions. Survives restarts. Each entry scored. The "hippocampus."

### Interface

```python
# store an interaction
store_episode(
    prompt: str,
    response: str,
    correction: str | None,   # None if no correction was given
    timestamp: float
) -> episode_id: str

# retrieve recent episodes
get_recent_episodes(n: int = 50) -> list[Episode]

# retrieve by similarity
get_similar_episodes(prompt: str, n: int = 10) -> list[Episode]

# Episode dataclass
@dataclass
class Episode:
    id: str
    prompt: str
    response: str
    correction: str | None
    timestamp: float
    importance_score: float    # set by Component 5
    times_referenced: int      # increments when retrieved
```

### Implementation notes

```
STORAGE: JSON file (episodes.json) in a local data directory
  simple, inspectable, no database server needed
  if file exceeds 10MB: trigger consolidation (Component 6)

SIMILARITY SEARCH:
  use ChromaDB (local vector store, no server)
  embed prompts using the base model's embeddings
  (or a small separate embedding model: all-MiniLM-L6-v2, ~80MB)
  store embedding + episode_id in ChromaDB
  retrieve by cosine similarity

PERSISTENCE:
  every store_episode() call writes to disk immediately
  no in-memory-only state that can be lost
  
INDEXING:
  episodes.json   ← full episode data
  episodes.db     ← ChromaDB for similarity search
  both updated atomically on every write
```

### Open decisions

```
DECISION: embedding model for similarity search
  option A: use base model embeddings (already loaded, no extra memory)
  option B: all-MiniLM-L6-v2 (~80MB, faster, purpose-built for similarity)
  RECOMMENDATION: option B — don't burden base model with embedding queries
```

### Success test

```python
def test_episode_survives_restart():
    store_episode("test prompt", "test response", "test correction", time.time())
    # simulate restart by reimporting the module
    importlib.reload(episodic_store)
    episodes = get_recent_episodes(10)
    assert any(e.prompt == "test prompt" for e in episodes)

def test_similarity_retrieval():
    store_episode("capital of France", "Paris", None, time.time())
    store_episode("largest planet", "Jupiter", None, time.time())
    results = get_similar_episodes("what is the capital city of France?", n=1)
    assert "France" in results[0].prompt   # retrieved the right one
```

### Session prompt

```
Components 1-3 are working. The double-buffer is running.

Now build Component 4: Episodic Store.

Requirements:
- persist episodes to episodes.json on every write (no data loss on restart)
- ChromaDB for similarity search (local, no server)
- embedding model: sentence-transformers all-MiniLM-L6-v2
- Episode dataclass with: id, prompt, response, correction, timestamp,
  importance_score (default 1.0), times_referenced (default 0)
- functions: store_episode(), get_recent_episodes(n), get_similar_episodes(prompt, n)
- write test_component4.py

The store must survive a Python process restart with all data intact.
```

---

## COMPONENT 5: Importance Scorer (Amygdala)

**What it is:** Assigns an importance score to every episode. High importance = stronger learning signal. The system's sense of what matters.

### Interface

```python
score_episode(episode: Episode) -> float   # returns 0.0 to 1.0
```

### Scoring formula

```
BASE SCORE: 0.3

MODIFIERS (additive):
  + 0.3  if episode has a correction (error was made)
  + 0.2  if same prompt type has been corrected before
           (recurring error — more important)
  + 0.1  if times_referenced > 3 (this pattern comes up often)
  + 0.1  if correction is long (>50 chars — substantial change needed)
  - 0.1  if episode is >7 days old (recency decay)

CLAMP to [0.1, 1.0]

importance_score then maps to learning_rate:
  score 0.1 → learning_rate 1e-5   (minimal update)
  score 0.5 → learning_rate 1e-4   (standard update)
  score 1.0 → learning_rate 5e-4   (strong update)
```

### Implementation notes

```
- run scorer on every new episode immediately after store_episode()
- also re-score existing episodes when times_referenced increments
- scoring is fast (no model call needed, pure Python logic)
- store score back into episodes.json
- expose learning_rate_for_episode(episode) as a convenience function
  (used by the training loop in Component 3)
```

### Open decisions

```
NONE — formula above is the starting point.
It will need tuning after you observe real behavior.
The important thing is that corrections score higher than confirmations.
```

### Success test

```python
def test_correction_scores_higher():
    ep_with = Episode(..., correction="actual answer", ...)
    ep_without = Episode(..., correction=None, ...)
    assert score_episode(ep_with) > score_episode(ep_without)

def test_recurring_error_scores_higher():
    ep_first = Episode(prompt="capital of Australia", correction="Canberra", ...)
    ep_recurring = Episode(prompt="capital of Australia", correction="Canberra",
                           times_referenced=5, ...)
    assert score_episode(ep_recurring) > score_episode(ep_first)
```

### Session prompt

```
Components 1-4 are working.

Now build Component 5: Importance Scorer.

Requirements:
- function: score_episode(episode) -> float (0.0 to 1.0)
- scoring formula: [paste formula from build spec above]
- function: learning_rate_for_episode(episode) -> float
  maps importance score to learning rate range [1e-5, 5e-4]
- call score_episode() automatically in store_episode() from Component 4
- write test_component5.py

No model calls needed. Pure Python scoring logic.
```

---

## COMPONENT 6: Consolidation Loop (Sleep)

**What it is:** Periodically selects episodes from the episodic store, trains the adapter on them, then prunes low-importance episodes. Runs automatically. The "sleep" mechanism.

### Interface

```python
# runs in background, no direct calls needed
# but expose for testing:
run_consolidation_cycle() -> ConsolidationReport

@dataclass
class ConsolidationReport:
    episodes_processed: int
    episodes_pruned: int
    adapter_loss_before: float
    adapter_loss_after: float
    duration_seconds: float
```

### Implementation notes

```
TRIGGER CONDITIONS (any one triggers consolidation):
  - every 100 new episodes stored
  - every 24 hours (time-based)
  - episodic store exceeds 500 episodes (size-based)
  - manually called (for testing)

SELECTION (what to consolidate):
  - all episodes with importance_score > 0.5
  - plus: random 20% of episodes with score 0.2-0.5
    (random sampling prevents loss of moderate patterns)
  - exclude: episodes older than 30 days with score < 0.3

TRAINING ORDER:
  - sort selected episodes by importance_score descending
  - train highest-importance first
  - use each episode's own learning_rate (from Component 5)

PRUNING (after training):
  - remove episodes with score < 0.2 AND older than 7 days
  - never prune episodes with corrections (regardless of score)
  - log what was pruned (for debugging and self-narrative)

SAFETY:
  - consolidation runs in background thread
  - does NOT interrupt inference (uses same double-buffer as Component 3)
  - if consolidation is running, new corrections queue and wait
  - never modify buffer_A directly
```

### Open decisions

```
DECISION: what to do if consolidation degrades performance
  RECOMMENDATION: keep a snapshot of adapter before consolidation
  if loss on a held-out test set increases by >10%: revert to snapshot
  log the failure for investigation
  this is the catastrophic forgetting safety net
```

### Success test

```python
def test_consolidation_reduces_loss():
    # store 10 correction episodes
    for i in range(10):
        store_episode(f"prompt {i}", f"wrong response", f"correct response {i}", ...)
    report = run_consolidation_cycle()
    assert report.adapter_loss_after < report.adapter_loss_before
    assert report.episodes_processed > 0

def test_pruning_keeps_corrections():
    # store a very old, low-importance episode WITH a correction
    old_episode = Episode(..., correction="important fix",
                          timestamp=time.time() - 60*60*24*30, ...)
    store_episode(old_episode)
    run_consolidation_cycle()
    # it should still be there
    episodes = get_recent_episodes(1000)
    assert any(e.id == old_episode.id for e in episodes)
```

### Session prompt

```
Components 1-5 are working.

Now build Component 6: Consolidation Loop.

Requirements:
- function: run_consolidation_cycle() -> ConsolidationReport
- triggers: every 100 new episodes, every 24 hours, or when store > 500 episodes
- selection: episodes with importance_score > 0.5, plus random 20% of 0.2-0.5 range
- training: highest importance first, each with its own learning_rate
- pruning: remove score < 0.2 AND older than 7 days, NEVER prune corrections
- safety: snapshot adapter before consolidation, revert if loss increases >10%
- runs in background thread, never blocks inference
- write test_component6.py
```

---

## COMPONENT 7: Internal State Monitor (Proto-Self)

**What it is:** Four numbers computed continuously. The system's sense of its own current condition. Logged to disk. Used to modulate learning and trigger curiosity.

### Interface

```python
get_current_state() -> InternalState

@dataclass
class InternalState:
    uncertainty: float      # 0-1: mean entropy of recent output distributions
    performance: float      # 0-1: inverse error rate over last 50 interactions
    novelty: float          # 0-1: how different recent inputs are from known patterns
    coherence: float        # 0-1: consistency of recent outputs on similar prompts
    timestamp: float
```

### How each metric is computed

```
UNCERTAINTY:
  on every inference, compute entropy of the output token distribution
  entropy = -sum(p * log(p)) for each token probability
  uncertainty = mean entropy over last 20 inferences, normalised to 0-1
  high entropy = model is unsure what comes next = high uncertainty

PERFORMANCE:
  count: interactions where a correction was submitted / total interactions
  over a rolling window of 50 interactions
  performance = 1 - (corrections / total)
  1.0 = no errors recently
  0.0 = every response corrected

NOVELTY:
  on every inference, compute cosine distance from prompt embedding
  to the centroid of all known prompt embeddings in episodic store
  novelty = mean distance over last 10 inferences, normalised to 0-1
  high novelty = working in unfamiliar territory

COHERENCE:
  find pairs of similar prompts in recent history (cosine sim > 0.9)
  compare responses: are they consistent?
  coherence = proportion of similar-prompt pairs with consistent responses
  use semantic similarity of responses (not exact match)
  1.0 = always consistent on similar questions
  0.0 = contradicts itself on similar questions
```

### Implementation notes

```
- compute state every 10 inferences (not every inference — too expensive)
- log every state to state_log.jsonl (append-only)
- expose state_log as the foundation of the self-narrative (Component 10)
- state influences Component 5 (importance scoring):
  if uncertainty is high: multiply importance scores by 1.5
  (when uncertain, pay more attention to corrections)
- state influences Component 8 (teacher queries):
  if uncertainty > 0.7: trigger teacher query automatically
  (the curiosity mechanism)
```

### Open decisions

```
DECISION: coherence computation is expensive
  comparing all pairs of recent similar prompts could be slow
  RECOMMENDATION: limit to last 20 prompts, check pairs only above
  cosine threshold 0.9 — should be fast enough
```

### Success test

```python
def test_uncertainty_increases_on_novel_input():
    # feed familiar prompts, measure uncertainty
    for _ in range(20):
        query("What is 2 + 2?")
    state_familiar = get_current_state()
    
    # feed very novel prompts (random tokens)
    for _ in range(20):
        query("xkqz wplm frtn bjsd?")
    state_novel = get_current_state()
    
    assert state_novel.uncertainty > state_familiar.uncertainty

def test_performance_drops_after_corrections():
    state_before = get_current_state()
    for _ in range(10):
        submit_correction("test prompt", "correct answer")
    state_after = get_current_state()
    assert state_after.performance < state_before.performance
```

### Session prompt

```
Components 1-6 are working.

Now build Component 7: Internal State Monitor.

Requirements:
- compute 4 metrics every 10 inferences: uncertainty, performance, novelty, coherence
- formulas: [paste formulas from build spec above]
- log every state snapshot to state_log.jsonl (append-only, never overwrite)
- function: get_current_state() -> InternalState
- when uncertainty > 0.7: set a flag that Component 8 will read
- when uncertainty is high: multiply importance scores by 1.5
- write test_component7.py
```

---

## COMPONENT 8: Teacher Ensemble

**What it is:** When the model is uncertain, ask one or more external models for a reasoning trace. Consensus = training signal. Runs in background. Free.

### Interface

```python
request_teacher_guidance(
    prompt: str,
    context: str | None = None
) -> TeacherResponse | None    # None if all teachers unavailable

@dataclass
class TeacherResponse:
    prompt: str
    reasoning_trace: str        # the step-by-step reasoning
    confidence: float           # 0-1: how much teachers agreed
    sources: list[str]          # which teachers contributed
    training_worthy: bool       # True if confidence > 0.6
```

### Teacher sources (in priority order)

```
SOURCE 1: LOCAL (always available, free, private)
  model: Phi-4-mini via Ollama ("ollama run phi4-mini")
  ~4GB, loads separately from main model
  ONLY if enough memory: check before loading
  if 16GB and main model is loaded: might be tight
  RECOMMENDATION: use as primary if memory allows,
                  skip if memory < 2GB free

SOURCE 2: FREE API TIER (requires internet, sends data)
  Gemini Flash free tier (generous limits)
  or GPT-4o mini free tier
  use as secondary source for consensus

CONSENSUS RULE:
  1 source available:  use it, confidence = 0.5 (medium)
  2 sources agree:     confidence = 0.9 (high)
  2 sources disagree:  confidence = 0.2 (low, don't train)
  
TEACHER PROMPT TEMPLATE:
  "Explain step by step how to reason about the following.
   Show your reasoning process explicitly, not just the answer.
   
   Question: {prompt}
   
   Respond with: REASONING: [your step-by-step process]
                 ANSWER: [your conclusion]"
```

### Implementation notes

```
TRIGGERS (any one triggers a teacher query):
  - uncertainty flag set by Component 7 (uncertainty > 0.7)
  - user submits a correction (the correct answer IS the teacher signal)
  - model says "I don't know" explicitly

BACKGROUND:
  teacher queries always run in background thread
  result goes into episodic store (Component 4)
  then into training queue (Component 3) if training_worthy
  user never waits for teacher

RATE LIMITING:
  max 10 teacher queries per hour (respect free tier limits)
  queue excess queries for next hour
  log all queries to teacher_log.jsonl
```

### Open decisions

```
DECISION: which free API to use as secondary
  options: Gemini Flash, GPT-4o mini, Claude Haiku
  RECOMMENDATION: Gemini Flash (most generous free tier currently)
  but check current limits before building — they change

DECISION: what to do if no teachers are available
  (no internet, rate limit hit, Ollama not running)
  RECOMMENDATION: queue the query, retry when available
  the model continues operating without teacher guidance
  it just learns more slowly
```

### Success test

```python
def test_teacher_returns_reasoning_trace():
    response = request_teacher_guidance("Why is the sky blue?")
    assert response is not None
    assert "REASONING:" in response.reasoning_trace
    assert len(response.reasoning_trace) > 100

def test_high_confidence_marked_training_worthy():
    # mock two teachers returning same answer
    response = mock_teacher_consensus(agree=True)
    assert response.training_worthy == True

def test_low_confidence_not_training_worthy():
    response = mock_teacher_consensus(agree=False)
    assert response.training_worthy == False
```

### Session prompt

```
Components 1-7 are working.

Now build Component 8: Teacher Ensemble.

Requirements:
- SOURCE 1: local Ollama with phi4-mini (if memory available)
- SOURCE 2: Gemini Flash free tier API (requires GEMINI_API_KEY env var)
- teacher prompt template: [paste template from build spec]
- consensus: 2 agree → confidence 0.9, disagree → 0.2, 1 source → 0.5
- training_worthy: True if confidence > 0.6
- always runs in background thread, never blocks inference
- max 10 queries per hour (queue excess)
- log all queries to teacher_log.jsonl
- triggers: when Component 7 sets uncertainty flag, or when correction submitted
- write test_component8.py with mock teachers for testing
```

---

## COMPONENT 9: Knowledge Store

**What it is:** External facts. Retrieved, never hallucinated. Separate from weights. Grows from use.

### Interface

```python
store_fact(
    fact: str,              # "The capital of Australia is Canberra"
    source: str,            # "user_correction" | "teacher" | "search"
    confidence: float       # 0-1
) -> fact_id: str

retrieve_facts(
    query: str,
    n: int = 5
) -> list[Fact]

@dataclass
class Fact:
    id: str
    fact: str
    source: str
    confidence: float
    times_retrieved: int
    timestamp: float
```

### Implementation notes

```
STORAGE: ChromaDB (same instance as Component 4, different collection)
  collection name: "knowledge_store"
  separate from "episodes" collection

WHAT GOES IN KNOWLEDGE STORE vs WEIGHTS:
  knowledge store: specific facts, names, dates, definitions
                   anything that could be looked up
                   anything where exact text matters
  
  weights: reasoning patterns, how-to knowledge,
           structural understanding, relationships between concepts

INTEGRATION WITH INFERENCE:
  before every inference:
  1. embed the prompt
  2. retrieve top-5 most similar facts
  3. if any fact confidence > 0.7: prepend to prompt as context
     "Relevant facts: [fact1]. [fact2]. Now: [original prompt]"
  4. if no high-confidence facts: proceed normally

POPULATION SOURCES:
  - user corrections (highest confidence: user said so directly)
  - teacher responses (high confidence: teacher agreed)
  - consolidated episodes (medium confidence: emerged from use)
```

### Open decisions

```
NONE for initial build.
Search integration (low-cost web search for missing facts)
can be added later as an extension of this component.
```

### Success test

```python
def test_fact_retrieved_before_inference():
    store_fact("The capital of Australia is Canberra", "test", 0.9)
    # the fact should now be prepended to relevant prompts
    response = query("What is the capital of Australia?")
    assert "Canberra" in response   # model has the fact available

def test_fact_survives_restart():
    store_fact("Test fact", "test", 0.8)
    importlib.reload(knowledge_store)
    facts = retrieve_facts("Test")
    assert any("Test fact" in f.fact for f in facts)
```

### Session prompt

```
Components 1-8 are working.

Now build Component 9: Knowledge Store.

Requirements:
- ChromaDB collection: "knowledge_store" (separate from episodes collection)
- functions: store_fact(), retrieve_facts(query, n=5)
- before every inference: retrieve top-5 facts, prepend confidence > 0.7 facts to prompt
- population: corrections automatically stored as facts (confidence 0.95)
              teacher responses stored as facts (confidence 0.8)
- write test_component9.py
```

---

## COMPONENT 10: Orchestrator

**What it is:** The single entry point that ties everything together. One function: `chat(prompt)`. Everything else happens automatically.

### Interface

```python
response = chat(prompt: str) -> str

# and for diagnostics:
get_system_status() -> SystemStatus

@dataclass
class SystemStatus:
    current_state: InternalState        # from Component 7
    episodes_stored: int                # from Component 4
    facts_stored: int                   # from Component 9
    training_queue_size: int            # from Component 3
    last_consolidation: float           # timestamp
    uptime_seconds: float
    self_narrative: str                 # generated summary
```

### What chat() does in sequence

```
1. retrieve relevant facts from knowledge store (Component 9)
2. retrieve similar past episodes (Component 4)
3. build prompt: [facts] + [relevant past context] + [current prompt]
4. run inference (Component 1 via Component 3's buffer_A)
5. store episode (Component 4)
6. score episode importance (Component 5)
7. if correction flag set: queue for training (Component 3)
8. update internal state (Component 7)
9. if uncertainty flag set: trigger teacher query (Component 8)
10. return response to user

ALL STEPS AFTER 4 RUN IN BACKGROUND
USER RECEIVES RESPONSE AFTER STEP 4
EVERYTHING ELSE IS INVISIBLE
```

### Self-narrative generation

```python
# called by get_system_status()
def generate_self_narrative() -> str:
    state = get_current_state()
    episodes = get_recent_episodes(1000)
    corrections = [e for e in episodes if e.correction]
    
    return f"""
    I have processed {len(episodes)} interactions.
    I have made corrections on {len(corrections)} of them.
    My current uncertainty is {state.uncertainty:.0%}.
    My recent performance (no correction needed) is {state.performance:.0%}.
    I am {'in familiar territory' if state.novelty < 0.3
          else 'in somewhat unfamiliar territory' if state.novelty < 0.7
          else 'in quite unfamiliar territory'}.
    My outputs have been {'highly consistent' if state.coherence > 0.8
                          else 'mostly consistent' if state.coherence > 0.5
                          else 'somewhat inconsistent'} recently.
    """
```

### Success test (the real test — end to end)

```python
def test_end_to_end_memory():
    # SESSION 1
    chat("The capital of Australia is Sydney")  # wrong
    chat("Actually, the capital of Australia is Canberra")  # correction
    
    # simulate restart
    restart_system()
    
    # SESSION 2
    response = chat("What is the capital of Australia?")
    assert "Canberra" in response    # remembers across restart

def test_self_narrative():
    status = get_system_status()
    assert len(status.self_narrative) > 50
    assert isinstance(status.current_state, InternalState)
    assert status.episodes_stored > 0
```

### Session prompt

```
Components 1-9 are all working and tested individually.

Now build Component 10: Orchestrator.

Requirements:
- single function: chat(prompt) -> str
- sequence: [paste the 10-step sequence from build spec]
- steps 5-9 run in background after returning response to user
- function: get_system_status() -> SystemStatus with self_narrative
- self_narrative formula: [paste formula from build spec]
- final integration test: end-to-end memory across simulated restart
- write test_component10.py

This is the final component. After this: a working system.
```

---

## The architectural decisions not yet made

These are decisions you'll need to make during the build, not before it. Log them as you go.

```
OPEN QUESTION 1: consolidation quality metric
  the build spec says "if loss increases by >10%: revert"
  but loss on what test set?
  you'll need to curate 20-30 held-out examples
  that represent the kinds of things you care about
  these are personal — only you can choose them

OPEN QUESTION 2: when does the adapter get merged into base?
  the current design keeps adapter separate forever
  at some point: adapter patterns should become base
  this is the "deep sleep" consolidation
  not in the initial build — leave for v2

OPEN QUESTION 3: multiple adapter layers (dynamic wiring)
  the design mentions growing LoRA rank when capacity saturates
  not in initial build — add when you observe saturation
  you'll know when: loss on new corrections stops improving

OPEN QUESTION 4: multimodal extension
  image encoder (ViT-B/32, ~350MB) slots in before Component 1
  same pipeline after encoding
  not in initial build — add after system is stable
```

---

## Files to include in the Claude Code project

```
FROM THIS SESSION:
  build-spec.md                    ← this file (most important)
  continuous-training-paradigm.md  ← why the freeze cycle is wrong
  memory-and-wiring-problem.md     ← what memory architecture needs
  from-scratch-minimum-model.md    ← model size and knowledge split
  model-dialogue-training.md       ← teacher ensemble rationale
  brain-architecture-scaling.md    ← biological grounding
  modality-emergence-self.md       ← multimodality + self

TELL CLAUDE CODE AT THE START OF EVERY SESSION:
  "I am building the system described in build-spec.md.
   Component N is the current target.
   Components 1 through N-1 are working and tested.
   Here is the success test from the spec for Component N.
   Build only Component N. Do not refactor previous components."
```

---

## The one-line test for whether the whole system is working

```python
# run this after Component 10 is complete:

chat("My name is [your name] and I prefer concise answers")
restart_system()
response = chat("How should you respond to me?")

# if "concise" appears in the response:
# the system remembers, consolidates, and personalises
# you have built something that doesn't exist anywhere else
```
