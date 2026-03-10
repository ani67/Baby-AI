# SPEC: Learning Loop Orchestrator
*Component 5 of 9 — The autonomous learning cycle*

---

## What it is

The engine that runs continuously once you press Start.
It owns the full cycle: observe → generate question → ask teacher →
encode answer → update model → check growth → log → repeat.

Nothing else drives the model's learning. This is the only place
where all other components meet.

It also responds to control signals from the frontend:
start, pause, step, reset, stage change.

---

## Location in the project

```
project/
  backend/
    loop/
      orchestrator.py     ← LearningLoop class
      curiosity.py        ← CuriosityScorer
      question_gen.py     ← QuestionGenerator
      curriculum.py       ← stage definitions + advancement logic
      sentence_splitter.py ← splits teacher answers before encoding
```

---

## The cycle (one step)

```
┌─────────────────────────────────────────────────────┐
│                                                     │
│  1. OBSERVE          read current model state       │
│          ↓                                          │
│  2. SCORE            compute curiosity per concept  │
│          ↓                                          │
│  3. SELECT           pick highest curiosity item    │
│          ↓                                          │
│  4. QUESTION         turn selected item into text   │
│          ↓                                          │
│  5. ASK              send to TeacherBridge          │
│          ↓                                          │
│  6. ENCODE           turn answer into vectors       │
│          ↓                                          │
│  7. PREDICT          model guesses before updating  │
│          ↓                                          │
│  8. UPDATE           Forward-Forward on Baby Model  │
│          ↓                                          │
│  9. MEASURE          compute delta, is_positive     │
│          ↓                                          │
│  10. GROW            check growth triggers          │
│          ↓                                          │
│  11. LOG             write to State Store           │
│          ↓                                          │
│  12. EMIT            send graph delta to frontend   │
│          ↓                                          │
│  13. WAIT            respect speed setting          │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## Interface

```python
class LearningLoop:
    def __init__(
        self,
        model: BabyModel,
        teacher: TeacherBridge,
        encoder: tuple,           # (ImageEncoder, TextEncoder, VideoEncoder)
        decoder: TextDecoder,
        store: StateStore,
        viz_emitter: VizEmitter,
        curriculum: Curriculum
    ):

    # Control
    async def start(self) -> None
    async def pause(self) -> None
    async def resume(self) -> None
    async def step_once(self) -> StepResult
    async def reset(self) -> None
    def set_speed(self, delay_ms: int) -> None     # 0 = max speed
    def set_stage(self, stage: int) -> None

    # Status
    def get_status(self) -> LoopStatus

    # Human interaction (called by API handler)
    async def human_message(self, text: str) -> str
```

---

## Control states

```
IDLE        not started, or after reset
            model exists but loop is not running

RUNNING     loop is executing continuously
            one step after another with delay_ms between them

PAUSED      loop is suspended mid-run
            model state is intact
            resumes from next step on resume()

STEPPING    executing exactly one step then returning to PAUSED
            triggered by step_once()

ERROR       teacher unreachable, or unrecoverable internal error
            loop halted, error surfaced to UI
            human must intervene (check Ollama, press Reset)
```

State machine:

```
IDLE ──start()──► RUNNING
RUNNING ──pause()──► PAUSED
PAUSED ──resume()──► RUNNING
PAUSED ──step_once()──► STEPPING ──► PAUSED
RUNNING ──error──► ERROR
ERROR ──reset()──► IDLE
any ──reset()──► IDLE
```

---

## Step implementation

```python
async def step_once(self) -> StepResult:
    """
    Executes one full learning cycle.
    Returns StepResult with everything that happened.
    """

    # ── 1. OBSERVE ──────────────────────────────────────────
    graph_summary = self.model.graph.summary()
    active_clusters = [
        c for c in self.model.graph.clusters if not c.dormant
    ]

    # ── 2. SCORE ─────────────────────────────────────────────
    curriculum_item = self.curriculum.next_item(
        stage=self._stage,
        model_state=graph_summary
    )
    curiosity_score = self.curiosity.score(
        item=curriculum_item,
        model=self.model
    )

    # ── 3. SELECT ────────────────────────────────────────────
    # curriculum.next_item already selected based on curiosity
    # this step is implicit in step 2

    # ── 4. QUESTION ──────────────────────────────────────────
    question = self.question_gen.generate(
        item=curriculum_item,
        stage=self._stage,
        recent_questions=self._recent_questions
    )
    self._recent_questions.append(question)

    # ── 5. ASK ───────────────────────────────────────────────
    try:
        teacher_response = await self.teacher.ask(
            question=question,
            stage=self._stage,
            context=curriculum_item.context
        )
    except TeacherUnavailableError:
        self._state = LoopState.ERROR
        raise
    except TeacherTimeoutError:
        # Skip this step — timeout is not fatal
        return StepResult(skipped=True, reason="teacher_timeout")

    # ── 6. ENCODE ────────────────────────────────────────────
    answer_vectors = self._encode_answer(
        teacher_response.answer,
        curriculum_item
    )
    # answer_vectors: list of (512,) tensors
    # may be 1 (short answer) or N (multi-sentence answer)

    # ── 7. PREDICT ───────────────────────────────────────────
    # Model predicts before seeing the answer
    # Used to compute is_positive signal
    if self._stage >= 1:
        input_vector = curriculum_item.input_vector
        prediction, activations = self.model.forward(
            input_vector, return_activations=True
        )
        is_positive = self._compute_is_positive(
            prediction=prediction,
            answer_vectors=answer_vectors
        )
    else:
        # Stage 0: always positive — just absorbing
        activations = {}
        is_positive = True

    # ── 8. UPDATE ────────────────────────────────────────────
    changes = {}
    for vec in answer_vectors:
        step_changes = self.model.update(
            x=vec,
            is_positive=is_positive
        )
        for k, v in step_changes.items():
            changes[k] = changes.get(k, 0.0) + v

    # ── 9. MEASURE ───────────────────────────────────────────
    delta_summary = {
        "weight_change_magnitude": sum(changes.values()),
        "edges_formed": [],
        "edges_pruned": [],
        "clusters_budded": [],
        "layers_inserted": [],
        "is_positive": is_positive,
        "curiosity_score": curiosity_score
    }

    # ── 10. GROW ─────────────────────────────────────────────
    growth_events = self.model.growth_check(self.store)
    for event in growth_events:
        etype = event["event_type"]
        if etype == "CONNECT":
            delta_summary["edges_formed"].append(
                f"{event['cluster_a']}->{event['cluster_b']}"
            )
        elif etype == "PRUNE":
            delta_summary["edges_pruned"].append(
                f"{event['cluster_a']}->{event['cluster_b']}"
            )
        elif etype == "BUD":
            delta_summary["clusters_budded"].append(event["cluster_a"])
        elif etype == "INSERT":
            delta_summary["layers_inserted"].append(event["metadata"]["new_cluster"])

    # ── 11. LOG ──────────────────────────────────────────────
    self.store.log_dialogue(
        step=self.model.step,
        stage=self._stage,
        question=question,
        answer=teacher_response.answer,
        curiosity_score=curiosity_score,
        clusters_active=list(activations.keys()),
        delta_summary=delta_summary
    )

    # Periodic snapshot
    if self.model.step % self.model.snapshot_interval == 0:
        graph_json = self.model.graph.to_json()
        self.store.log_latent_snapshot(
            step=self.model.step,
            graph_json=graph_json
        )

    # Periodic checkpoint
    if self.model.step % 500 == 0:
        self.store.save_checkpoint(
            step=self.model.step,
            stage=self._stage,
            model_state_dict=self.model.get_state_dict(),
            graph_json=self.model.graph.to_json()
        )
        self.store.prune_old_snapshots()

    # ── 12. EMIT ─────────────────────────────────────────────
    await self.viz_emitter.emit_step(
        step=self.model.step,
        stage=self._stage,
        graph=self.model.graph,
        activations=activations,
        last_question=question,
        last_answer=teacher_response.answer,
        growth_events=growth_events
    )

    return StepResult(
        step=self.model.step,
        question=question,
        answer=teacher_response.answer,
        curiosity_score=curiosity_score,
        is_positive=is_positive,
        delta_summary=delta_summary,
        growth_events=growth_events,
        duration_ms=teacher_response.duration_ms,
        skipped=False
    )
```

---

## The running loop

```python
async def start(self) -> None:
    if self._state not in (LoopState.IDLE, LoopState.PAUSED):
        return
    self._state = LoopState.RUNNING

    # Warm up the teacher model
    await self.teacher.ask("Hello", stage=0)

    while self._state == LoopState.RUNNING:
        try:
            await self.step_once()
        except TeacherUnavailableError:
            self._state = LoopState.ERROR
            break
        except Exception as e:
            # Log unexpected errors but don't halt
            # Give the loop a chance to recover on next step
            logging.error(f"Step error: {e}", exc_info=True)

        # Respect speed setting
        if self._delay_ms > 0:
            await asyncio.sleep(self._delay_ms / 1000)
```

---

## CuriosityScorer

The model's internal measure of what it needs to learn next.
Drives step 2 of the cycle.

```python
class CuriosityScorer:
    """
    Computes a curiosity score for a curriculum item.

    curiosity = uncertainty × novelty × stage_relevance

    uncertainty:      how far off was the model's last prediction
                      on this type of input?
                      (1 - cosine_similarity(prediction, answer))

    novelty:          how different is this item from recent inputs?
                      (1 - max_cosine_similarity(item, recent_inputs))

    stage_relevance:  how well does this item match the
                      current developmental stage?
                      (1.0 for perfect match, 0.5 for off-stage)
    """

    def __init__(self, window: int = 50):
        self._recent_inputs: deque = deque(maxlen=window)
        self._prediction_errors: dict = {}   # item_type → recent errors

    def score(
        self,
        item: CurriculumItem,
        model: BabyModel
    ) -> float:
        uncertainty = self._uncertainty(item, model)
        novelty = self._novelty(item)
        stage_relevance = item.stage_relevance

        score = uncertainty * novelty * stage_relevance
        self._recent_inputs.append(item.input_vector)
        return float(score)

    def _uncertainty(
        self,
        item: CurriculumItem,
        model: BabyModel
    ) -> float:
        if item.input_vector is None:
            return 0.8   # default medium-high for items without vectors yet
        output, _ = model.forward(item.input_vector)
        if item.expected_vector is None:
            # No ground truth yet — use activation variance as proxy
            variance = sum(
                c.activation_variance
                for c in model.graph.clusters
                if not c.dormant
            ) / max(len(model.graph.clusters), 1)
            return min(1.0, variance * 2)
        similarity = torch.dot(output, item.expected_vector).item()
        return 1.0 - max(0.0, similarity)   # 0 = certain, 1 = maximally uncertain

    def _novelty(self, item: CurriculumItem) -> float:
        if not self._recent_inputs or item.input_vector is None:
            return 1.0   # everything is novel at the start
        similarities = [
            torch.dot(item.input_vector, prev).item()
            for prev in self._recent_inputs
        ]
        max_similarity = max(similarities)
        return 1.0 - max(0.0, max_similarity)
```

---

## QuestionGenerator

Turns a CurriculumItem into a natural language question.

```python
class QuestionGenerator:
    """
    Generates a question string from a curriculum item.

    The question has to be:
    - Short (one sentence)
    - Answerable by the teacher in one to five sentences
    - Appropriate to the current developmental stage
    - Not a repeat of a recent question

    At early stages: rigid templates with slot-filling
    At later stages: more flexible phrasing
    """

    TEMPLATES = {
        0: [
            "What is this? [IMAGE: {description}]",
            "What do you call this? [IMAGE: {description}]",
            "Name this: [IMAGE: {description}]",
        ],
        1: [
            "What kind of thing is a {label}?",
            "What category does a {label} belong to?",
            "Is a {label} an animal, a plant, or an object?",
        ],
        2: [
            "What do {label_a} and {label_b} have in common?",
            "How is a {label_a} different from a {label_b}?",
            "Why would you group {label_a} with {label_b}?",
        ],
        3: [
            "What caused {event_b} to happen after {event_a}?",
            "Why did {event_b} follow from {event_a}?",
            "What is the relationship between {event_a} and {event_b}?",
        ],
        4: [
            "What does {concept} mean?",
            "How would you explain {concept} to someone who had never heard of it?",
            "What is {concept} an example of?",
        ]
    }

    def generate(
        self,
        item: CurriculumItem,
        stage: int,
        recent_questions: list[str]
    ) -> str:
        templates = self.TEMPLATES.get(stage, self.TEMPLATES[4])
        for attempt in range(10):
            template = random.choice(templates)
            question = template.format(**item.template_slots)
            if not self._too_similar(question, recent_questions):
                return question
        # Fallback if all attempts are too similar to recent
        return f"Tell me about {item.label or 'this'}."

    def _too_similar(
        self,
        question: str,
        recent: list[str],
        threshold: float = 0.8
    ) -> bool:
        # Simple word-overlap similarity (no embedding needed here)
        words = set(question.lower().split())
        for prev in recent[-10:]:
            prev_words = set(prev.lower().split())
            if not words or not prev_words:
                continue
            overlap = len(words & prev_words) / len(words | prev_words)
            if overlap > threshold:
                return True
        return False
```

---

## Curriculum

Manages what the model sees and when.
Each developmental stage has a pool of items.
The CuriosityScorer picks from the pool.

```python
@dataclass
class CurriculumItem:
    id: str
    stage: int                        # which stage this item belongs to
    item_type: str                    # "image" | "image_pair" | "video" | "concept"
    input_vector: torch.Tensor | None # pre-encoded input (None = encode on demand)
    expected_vector: torch.Tensor | None  # pre-encoded expected answer (None = unknown)
    label: str | None                 # human-readable label
    description: str | None          # human-readable description of input
    context: str | None              # additional context for teacher
    template_slots: dict             # filled into QuestionGenerator templates
    stage_relevance: float           # 1.0 = perfectly matches current stage


class Curriculum:
    """
    The pool of experiences available at each stage.

    Stage 0: images from a small curated set (~100 images in 10 categories)
    Stage 1: same images + their labels, paired for comparison
    Stage 2: image pairs from same/different categories + short video clips
    Stage 3: video clips with before/after events
    Stage 4: abstract concept strings (from accumulated teacher vocabulary)

    The curriculum is not fixed — it grows as the model learns.
    New items are added when:
      - Human drops a new image into the UI
      - Teacher uses a new word the model hasn't seen (auto-added to Stage 4)
      - Human manually advances the stage (pool expands)
    """

    def __init__(self, data_dir: str = "backend/data"):
        self._pools: dict[int, list[CurriculumItem]] = {
            0: [], 1: [], 2: [], 3: [], 4: []
        }
        self._data_dir = data_dir
        self._load_stage_0()    # loads images from data/stage0/

    def next_item(
        self,
        stage: int,
        model_state: dict
    ) -> CurriculumItem:
        """
        Returns the next item to learn about.

        Selection:
        1. Gather pool for current stage + one stage below (review)
        2. Score each item by curiosity (uncertainty × novelty)
        3. Sample from top 20% (not always top-1 — prevents fixation)
        """
        pool = self._pools.get(stage, []) + self._pools.get(stage - 1, [])
        if not pool:
            # Fallback: anything in any pool
            pool = [item for items in self._pools.values() for item in items]
        if not pool:
            raise EmptyPoolError(f"No curriculum items available for stage {stage}")

        # Score all items
        scored = sorted(pool, key=lambda x: x.stage_relevance, reverse=True)

        # Sample from top 20%
        top_n = max(1, len(scored) // 5)
        return random.choice(scored[:top_n])

    def add_item(self, item: CurriculumItem) -> None:
        self._pools[item.stage].append(item)

    def add_image(
        self,
        image: PIL.Image.Image,
        label: str | None = None
    ) -> CurriculumItem:
        """
        Called when human drops an image in the UI.
        Creates a Stage 0 item. Encodes lazily (on first use).
        """
        item = CurriculumItem(
            id=f"img_{uuid4().hex[:8]}",
            stage=0,
            item_type="image",
            input_vector=None,    # encoded on first use
            expected_vector=None,
            label=label,
            description=f"image{' of ' + label if label else ''}",
            context=None,
            template_slots={"description": label or "this image"},
            stage_relevance=1.0
        )
        self._pools[0].append(item)
        return item

    def add_teacher_vocabulary(self, word: str) -> None:
        """
        Called when teacher uses a new word.
        Adds a Stage 4 concept item automatically.
        """
        item = CurriculumItem(
            id=f"word_{word}",
            stage=4,
            item_type="concept",
            input_vector=None,
            expected_vector=None,
            label=word,
            description=word,
            context=None,
            template_slots={"concept": word},
            stage_relevance=1.0
        )
        if not any(i.label == word for i in self._pools[4]):
            self._pools[4].append(item)

    def _load_stage_0(self) -> None:
        """
        Loads images from backend/data/stage0/
        Expected structure:
          stage0/
            dog/
              dog_01.jpg
              dog_02.jpg
            cat/
              cat_01.jpg
            ...
        Category name = folder name = label.
        """
        stage0_dir = Path(self._data_dir) / "stage0"
        if not stage0_dir.exists():
            return
        for category_dir in stage0_dir.iterdir():
            if not category_dir.is_dir():
                continue
            label = category_dir.name
            for img_path in category_dir.glob("*.jpg"):
                item = CurriculumItem(
                    id=f"img_{img_path.stem}",
                    stage=0,
                    item_type="image",
                    input_vector=None,
                    expected_vector=None,
                    label=label,
                    description=f"a {label}",
                    context=None,
                    template_slots={"description": f"a {label}"},
                    stage_relevance=1.0
                )
                self._pools[0].append(item)
```

---

## Encoding answer vectors

```python
def _encode_answer(
    self,
    answer: str,
    item: CurriculumItem
) -> list[torch.Tensor]:
    """
    Encodes teacher answer into one or more 512-dim vectors.

    Short answer (< 30 words): one vector for the whole thing
    Long answer (>= 30 words): split into sentences, one vector each

    Also: extract new words and add to curriculum Stage 4
    """
    _, text_encoder, _ = self.encoders
    words = answer.split()

    if len(words) < 30:
        return [text_encoder.encode(answer)]

    sentences = split_sentences(answer)
    vectors = [text_encoder.encode(s) for s in sentences if s.strip()]

    # Add new vocabulary words to Stage 4 pool
    for word in words:
        clean = word.strip(".,!?;:").lower()
        if (len(clean) > 4
                and clean not in self._known_words
                and clean.isalpha()):
            self._known_words.add(clean)
            self.curriculum.add_teacher_vocabulary(clean)

    return vectors
```

---

## Computing is_positive

```python
def _compute_is_positive(
    self,
    prediction: torch.Tensor,
    answer_vectors: list[torch.Tensor]
) -> bool:
    """
    True if the model's prediction was reasonably close to the answer.

    For multi-vector answers: use the mean of the answer vectors.
    Threshold: cosine similarity > 0.5

    At Stage 0: always True (no predictions, just absorbing)
    At Stage 1+: computed from prediction vs answer similarity
    """
    if not answer_vectors:
        return True
    mean_answer = torch.stack(answer_vectors).mean(dim=0)
    mean_answer = F.normalize(mean_answer, dim=0)
    similarity = torch.dot(prediction, mean_answer).item()
    return similarity > 0.5
```

---

## Human message handler

```python
async def human_message(self, text: str) -> str:
    """
    Called when the human sends a chat message to the Baby Model.
    Does NOT pause the learning loop.
    The model responds based on its current state.

    Process:
    1. Encode human message → input vector
    2. Run Baby Model forward pass
    3. Decode output vector → text response
    4. Log to human_chat in State Store
    5. Return response string
    """
    _, text_encoder, _ = self.encoders
    input_vector = text_encoder.encode(text)
    output_vector, activations = self.model.forward(
        input_vector, return_activations=True
    )
    response = self.decoder.decode(output_vector, max_words=30)

    active_clusters = list(activations.keys())

    self.store.log_human_message(
        step=self.model.step,
        role="human",
        message=text
    )
    self.store.log_human_message(
        step=self.model.step,
        role="model",
        message=response,
        clusters_active=active_clusters
    )

    return response
```

---

## LoopStatus and StepResult

```python
@dataclass
class LoopStatus:
    state: str              # "idle"|"running"|"paused"|"stepping"|"error"
    step: int
    stage: int
    delay_ms: int
    error_message: str | None
    graph_summary: dict     # from model.graph.summary()
    teacher_healthy: bool

@dataclass
class StepResult:
    step: int = 0
    question: str = ""
    answer: str = ""
    curiosity_score: float = 0.0
    is_positive: bool = True
    delta_summary: dict = field(default_factory=dict)
    growth_events: list = field(default_factory=list)
    duration_ms: int = 0
    skipped: bool = False
    reason: str = ""        # if skipped, why
```

---

## Reset

```python
async def reset(self) -> None:
    """
    Pauses the loop if running.
    Resets the Baby Model to initial state.
    Clears all in-memory state (curiosity history, recent questions, etc.)
    Does NOT clear the State Store database.
    (History is preserved — you can still view what happened before.)
    Sets state to IDLE.
    """
    await self.pause()
    self.model = BabyModel()    # fresh model
    self.curiosity = CuriosityScorer()
    self._recent_questions.clear()
    self._known_words.clear()
    self._stage = 0
    self._state = LoopState.IDLE
```

---

## sentence_splitter.py

```python
import nltk
nltk.download("punkt_tab", quiet=True)

def split_sentences(text: str) -> list[str]:
    """
    Splits text into sentences using NLTK punkt tokenizer.
    Falls back to period-splitting if NLTK fails.
    Filters out empty strings and very short fragments (< 3 words).
    """
    try:
        from nltk.tokenize import sent_tokenize
        sentences = sent_tokenize(text)
    except Exception:
        sentences = [s.strip() for s in text.split(".") if s.strip()]

    return [s for s in sentences if len(s.split()) >= 3]
```

---

## Tests

```python
# test_learning_loop.py

async def test_single_step_produces_step_result():
    loop = make_test_loop()   # helper that wires mock components
    await loop.start()
    result = await loop.step_once()
    assert not result.skipped
    assert result.question != ""
    assert result.answer != ""
    assert result.curiosity_score > 0

async def test_state_transitions():
    loop = make_test_loop()
    assert loop.get_status().state == "idle"
    asyncio.create_task(loop.start())
    await asyncio.sleep(0.1)
    assert loop.get_status().state == "running"
    await loop.pause()
    assert loop.get_status().state == "paused"
    await loop.resume()
    assert loop.get_status().state == "running"

async def test_step_once_pauses_after():
    loop = make_test_loop()
    await loop.pause()
    result = await loop.step_once()
    assert loop.get_status().state == "paused"

async def test_teacher_timeout_skips_step():
    loop = make_test_loop(teacher=TimeoutTeacher())
    result = await loop.step_once()
    assert result.skipped
    assert result.reason == "teacher_timeout"

async def test_teacher_unavailable_sets_error():
    loop = make_test_loop(teacher=UnavailableTeacher())
    with pytest.raises(TeacherUnavailableError):
        await loop.step_once()
    assert loop.get_status().state == "error"

async def test_model_updates_after_step():
    loop = make_test_loop()
    before = loop.model.graph.summary()["node_count"]
    for _ in range(5):
        await loop.step_once()
    # At minimum, model has been updated (weights changed)
    # Node count may or may not have changed this early
    assert loop.model.step == 5

async def test_dialogue_logged_after_step():
    loop = make_test_loop()
    await loop.step_once()
    dialogues = loop.store.get_dialogues()
    assert len(dialogues) == 1

async def test_curiosity_drives_variety():
    """Model should not ask the same question repeatedly."""
    loop = make_test_loop()
    questions = []
    for _ in range(10):
        result = await loop.step_once()
        questions.append(result.question)
    unique = len(set(questions))
    assert unique >= 6   # at least 60% unique

async def test_human_message_returns_string():
    loop = make_test_loop()
    response = await loop.human_message("What is a dog?")
    assert isinstance(response, str)
    assert len(response) > 0

async def test_reset_clears_model():
    loop = make_test_loop()
    for _ in range(20):
        await loop.step_once()
    await loop.reset()
    assert loop.model.step == 0
    assert loop.get_status().state == "idle"
```

---

## Hard parts

**Asyncio and the running loop.**
`start()` runs as a coroutine that loops indefinitely.
It must be launched as a background task (`asyncio.create_task`)
by the FastAPI startup handler — not awaited directly.
`pause()` sets a flag that the loop checks between steps.
There is no mid-step interruption — a step always runs to completion
before the pause takes effect. This is correct behavior.

**Curriculum pool exhaustion.**
If the human provides only 10 images and the loop runs at max speed,
the same 10 items will cycle through the curiosity scorer rapidly.
Novelty scores will drop toward zero for all items, making
curiosity scores very low. The loop will still run but learning
will slow — the model is going over ground it already knows.
Fix: add more images, or advance to Stage 1 where new item types appear.
The UI should surface low average curiosity as a signal to the human.

**The `is_positive` threshold is the most fragile value.**
At 0.5 cosine similarity, most Stage 0 outputs will be negative
(the model starts near-random, similarity will be low).
This means the model will get almost all negative signals early on —
which is correct in Forward-Forward (negative = suppress this response
to this input) but requires that negative signals actually teach
something useful, not just suppress everything.
Consider: at Stage 0, always use `is_positive=True` regardless.
At Stage 1, threshold = 0.3. At Stage 2+, threshold = 0.5.
The spec uses a fixed 0.5 — treat this as the first thing to tune.

**Vocabulary extraction from teacher answers is noisy.**
Extracting "new words" from teacher answers and adding them to
Stage 4 curriculum will add noise words (stop words, partial words,
proper nouns). The `len(clean) > 4` and `clean.isalpha()` guards
reduce this but don't eliminate it. A word like "their" or "about"
will pass these guards. Add a simple stopword list:
```python
STOPWORDS = {"about", "their", "there", "these", "those",
             "would", "could", "should", "which", "where", "after"}
if clean in STOPWORDS: continue
```

**Speed at 0ms delay will saturate Ollama.**
At max speed with phi4-mini responding in 300ms, the loop runs at
~3 steps/second. This is fine. But if the model is very simple
(early stages) the forward pass takes microseconds while Ollama
takes 300ms — so Ollama is always the bottleneck.
This is expected behavior. The learning loop's speed is bounded
by the teacher, not the model.

---

## M1-specific notes

The loop itself is pure asyncio — no ML compute in the orchestrator.
ML happens in BabyModel (MPS tensors) and Encoder (MLX).
The asyncio event loop and the MPS/MLX operations coexist fine
on M1 because they run on different hardware (CPU vs GPU/Neural Engine).

`asyncio.sleep(delay_ms / 1000)` yields control to the event loop
between steps. This allows FastAPI to handle WebSocket/REST requests
during the sleep window. At 0ms delay, `asyncio.sleep(0)` still
yields (it's a checkpoint, not a no-op) — enough for FastAPI to
handle one pending request between steps.
