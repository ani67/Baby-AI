# SPEC: Teacher Bridge
*Component 2 of 9 — Wraps the local teacher LLM*

---

## What it is

A thin async wrapper around the Ollama HTTP API.
Takes a question string, calls the local teacher LLM, returns the answer string.
That's the entire job.

No memory. No state. No knowledge of the Baby Model.
Every call is independent.

---

## Location in the project

```
project/
  backend/
    teacher/
      bridge.py       ← the class
      prompts.py      ← prompt templates per developmental stage
```

---

## Dependencies

```
Ollama              running locally on localhost:11434
                    (started by start.sh before the backend)

httpx               async HTTP client (pip install httpx)
                    chosen over aiohttp for simpler API
                    and over requests because we need async
```

No PyTorch. No MLX. No ML at all in this component.

---

## Configuration

Set via environment variables, with defaults:

```bash
TEACHER_MODEL=phi4-mini          # which model Ollama serves
TEACHER_HOST=http://localhost:11434
TEACHER_MAX_TOKENS=200           # keep answers short and dense
TEACHER_TEMPERATURE=0.3          # low — we want consistent factual answers
TEACHER_TIMEOUT=30               # seconds before giving up on a response
```

`phi4-mini` is the default because it:
- Fits in ~4GB RAM (leaves room for the Baby Model)
- Has strong reasoning for its size
- Responds fast enough for a tight learning loop

Fallback if `phi4-mini` is not available: `mistral:7b-instruct`

---

## Interface

One class, three public methods.

```python
class TeacherBridge:
    def __init__(self, host: str, model: str, max_tokens: int,
                 temperature: float, timeout: float): ...

    async def ask(
        self,
        question: str,
        stage: int,
        context: str | None = None
    ) -> TeacherResponse: ...

    async def health_check(self) -> bool: ...

    async def list_available_models(self) -> list[str]: ...
```

### `TeacherResponse`

```python
@dataclass
class TeacherResponse:
    answer: str           # the text the model returned
    model: str            # which model actually answered
    duration_ms: int      # how long the call took
    tokens_used: int      # prompt + completion tokens
    truncated: bool       # True if answer hit max_tokens limit
```

---

## The `ask` method in detail

```python
async def ask(
    self,
    question: str,
    stage: int,
    context: str | None = None
) -> TeacherResponse:
    """
    Sends question to the teacher LLM.

    stage:    controls which system prompt is used (see prompts.py)
    context:  optional — additional context prepended to the question
              e.g. an image description, a previous answer being refined

    Returns TeacherResponse.
    Raises TeacherUnavailableError if Ollama is not reachable.
    Raises TeacherTimeoutError if response takes longer than TEACHER_TIMEOUT.
    Never raises on bad answers — bad answers are valid answers.
    """
```

### What it sends to Ollama

```python
payload = {
    "model": self.model,
    "prompt": build_prompt(question, context, stage),  # from prompts.py
    "stream": False,
    "options": {
        "num_predict": self.max_tokens,
        "temperature": self.temperature,
        "stop": ["\n\n", "###"]   # stop at double newline or separator
    }
}
```

Uses `/api/generate` (not `/api/chat`) — simpler, single-turn, no message history.
The teacher has no memory of previous questions. Each call is fresh.
Conversation history is not the teacher's job — that's the Learning Loop's job
if it decides to include prior context.

---

## Prompt templates (`prompts.py`)

The system prompt changes per developmental stage.
Early stages need short concrete answers. Later stages can handle more nuance.

```python
SYSTEM_PROMPTS = {
    0: """You are teaching a very young AI its first concepts.
Answer in one short sentence. Use simple words only.
Name the object or describe what you see. Nothing else.""",

    1: """You are teaching a young AI basic words and categories.
Answer in one or two short sentences.
Name the thing, then say what category it belongs to.""",

    2: """You are teaching an AI about concepts and relationships.
Answer in two to three sentences.
Explain what things have in common or how they differ.""",

    3: """You are teaching an AI about cause and effect.
Answer in two to four sentences.
Explain what caused something to happen and why.""",

    4: """You are teaching an AI that is developing abstract reasoning.
Answer clearly and precisely in three to five sentences.
You can use more complex ideas but stay grounded and specific."""
}

def build_prompt(question: str, context: str | None, stage: int) -> str:
    system = SYSTEM_PROMPTS.get(stage, SYSTEM_PROMPTS[4])
    if context:
        return f"{system}\n\nContext: {context}\n\nQuestion: {question}"
    return f"{system}\n\nQuestion: {question}"
```

The system prompt is prepended to the prompt string (not sent as a separate
`system` field) because `/api/generate` does not have a native system role.
This works fine for instruction-tuned models like phi4-mini and mistral-instruct.

---

## `health_check`

```python
async def health_check(self) -> bool:
    """
    Calls GET /api/tags on Ollama.
    Returns True if Ollama is running and the configured model is available.
    Returns False otherwise (does not raise).
    Used by start.sh health check loop and by the backend status endpoint.
    """
```

Also checks that `self.model` appears in the tags response.
If Ollama is running but the model hasn't been pulled yet,
`health_check` returns False and the backend logs a clear message:
`"Teacher model 'phi4-mini' not found. Run: ollama pull phi4-mini"`

---

## `list_available_models`

```python
async def list_available_models(self) -> list[str]:
    """
    Returns list of model name strings currently pulled in Ollama.
    Used by the frontend status panel to show available fallbacks.
    Example return: ["phi4-mini", "mistral:7b-instruct", "llama3.2:3b"]
    """
```

---

## Error types

```python
class TeacherError(Exception):
    """Base class for all teacher bridge errors."""

class TeacherUnavailableError(TeacherError):
    """Ollama is not running or not reachable at the configured host."""

class TeacherTimeoutError(TeacherError):
    """The model took longer than TEACHER_TIMEOUT seconds to respond."""

class TeacherModelNotFoundError(TeacherError):
    """The configured model is not pulled in Ollama."""
```

The Learning Loop catches these and decides what to do:
- `TeacherUnavailableError` → pause the loop, surface error to UI
- `TeacherTimeoutError` → skip this step, log it, continue
- `TeacherModelNotFoundError` → halt entirely, surface to UI

---

## Full implementation shape

```python
import httpx
import time
from dataclasses import dataclass
from .prompts import build_prompt

@dataclass
class TeacherResponse:
    answer: str
    model: str
    duration_ms: int
    tokens_used: int
    truncated: bool

class TeacherBridge:
    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "phi4-mini",
        max_tokens: int = 200,
        temperature: float = 0.3,
        timeout: float = 30.0
    ):
        self.host = host
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self._client = httpx.AsyncClient(timeout=timeout)

    async def ask(
        self,
        question: str,
        stage: int,
        context: str | None = None
    ) -> TeacherResponse:
        prompt = build_prompt(question, context, stage)
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": self.max_tokens,
                "temperature": self.temperature,
                "stop": ["\n\n", "###"]
            }
        }
        t0 = time.monotonic()
        try:
            response = await self._client.post(
                f"{self.host}/api/generate",
                json=payload
            )
            response.raise_for_status()
        except httpx.ConnectError:
            raise TeacherUnavailableError(
                f"Cannot reach Ollama at {self.host}"
            )
        except httpx.TimeoutException:
            raise TeacherTimeoutError(
                f"Model '{self.model}' timed out after {self.timeout}s"
            )

        duration_ms = int((time.monotonic() - t0) * 1000)
        data = response.json()

        return TeacherResponse(
            answer=data["response"].strip(),
            model=data.get("model", self.model),
            duration_ms=duration_ms,
            tokens_used=data.get("eval_count", 0) +
                        data.get("prompt_eval_count", 0),
            truncated=data.get("done_reason") == "length"
        )

    async def health_check(self) -> bool:
        try:
            response = await self._client.get(
                f"{self.host}/api/tags",
                timeout=5.0
            )
            response.raise_for_status()
            models = [m["name"] for m in response.json().get("models", [])]
            return any(self.model in m for m in models)
        except Exception:
            return False

    async def list_available_models(self) -> list[str]:
        try:
            response = await self._client.get(
                f"{self.host}/api/tags",
                timeout=5.0
            )
            response.raise_for_status()
            return [m["name"] for m in response.json().get("models", [])]
        except Exception:
            return []

    async def close(self):
        """Call on backend shutdown to release the HTTP client."""
        await self._client.aclose()
```

---

## Ollama API reference

The two endpoints this component uses:

```
GET  /api/tags
     → { "models": [{ "name": "phi4-mini", ... }, ...] }
     Used by health_check and list_available_models.

POST /api/generate
     Body: { "model", "prompt", "stream": false, "options": {...} }
     → {
         "response": "It is a dog.",
         "model": "phi4-mini",
         "done": true,
         "done_reason": "stop",      # or "length" if truncated
         "eval_count": 12,           # completion tokens
         "prompt_eval_count": 48,    # prompt tokens
         "total_duration": 843000000 # nanoseconds
       }
```

`stream: false` means Ollama buffers the full response before returning.
This is simpler than streaming for our use case — the learning loop
waits for the full answer anyway before proceeding.

---

## Tests

```python
# test_teacher_bridge.py
# These tests require Ollama to be running with phi4-mini pulled.
# Mark as integration tests — skip in CI, run locally only.

import pytest
import asyncio

@pytest.mark.integration
async def test_health_check_when_running():
    bridge = TeacherBridge()
    assert await bridge.health_check() is True

@pytest.mark.integration
async def test_basic_ask():
    bridge = TeacherBridge()
    response = await bridge.ask(
        question="What is a dog?",
        stage=0
    )
    assert len(response.answer) > 0
    assert response.duration_ms > 0
    assert response.tokens_used > 0

@pytest.mark.integration
async def test_stage_0_answer_is_short():
    bridge = TeacherBridge()
    response = await bridge.ask(
        question="What is a tree?",
        stage=0
    )
    # Stage 0 prompt enforces one short sentence
    word_count = len(response.answer.split())
    assert word_count < 25, f"Stage 0 answer too long: {word_count} words"

@pytest.mark.integration
async def test_ask_with_context():
    bridge = TeacherBridge()
    response = await bridge.ask(
        question="What is the difference between these two things?",
        stage=2,
        context="Thing 1: a dog. Thing 2: a cat."
    )
    assert len(response.answer) > 0

# Unit tests — no Ollama required

async def test_unavailable_raises():
    bridge = TeacherBridge(host="http://localhost:9999")
    with pytest.raises(TeacherUnavailableError):
        await bridge.ask("test", stage=0)

async def test_health_check_returns_false_when_down():
    bridge = TeacherBridge(host="http://localhost:9999")
    result = await bridge.health_check()
    assert result is False

async def test_list_models_returns_empty_when_down():
    bridge = TeacherBridge(host="http://localhost:9999")
    result = await bridge.list_available_models()
    assert result == []
```

---

## Hard parts

**Answer quality at Stage 0 is fragile.**
Instruction-tuned models don't always obey "one sentence only."
phi4-mini is better at this than most, but you will get occasional
multi-paragraph answers at Stage 0. The Learning Loop should check
`response.truncated` and also measure word count — if it's too long,
trim to first sentence before feeding to the Baby Model.
Don't send long answers to the Baby Model at early stages.
The signal-to-noise ratio matters for learning.

**`done_reason: "length"` is a real problem at low `max_tokens`.**
200 tokens is enough for most answers but some questions at Stage 3-4
genuinely need more. If `truncated` is True, consider re-asking with
`max_tokens * 2` once before logging it as truncated.
Don't do this by default — it doubles latency for that step.

**Model warm-up latency.**
First call after Ollama loads the model takes 2-10 seconds while the
model loads into memory. Subsequent calls are fast (200-800ms for phi4-mini).
The learning loop should issue a dummy warm-up call during startup,
before the loop begins, so the first real learning step isn't slow.

```python
# In Learning Loop startup:
await teacher_bridge.ask("Hello", stage=0)   # warm-up, discard response
```

**phi4-mini model name variants.**
Ollama uses `phi4-mini` but some installs have it as `phi4-mini:latest`
or `phi4:mini`. The `health_check` uses `any(self.model in m for m in models)`
which handles these variants. If the user pulls a different variant,
they should set `TEACHER_MODEL` explicitly in their environment.

---

## M1-specific notes

Ollama on M1 uses Metal automatically for GPU inference.
phi4-mini at 4-bit quantization runs at ~200-400ms per response
with the GPU active. If Ollama falls back to CPU (check `ollama logs`),
responses will be 2-5x slower and the learning loop will feel sluggish.

Ensure Ollama is using Metal:
```bash
ollama run phi4-mini "hello"
# Should show metal=1 in the server logs
```

The `httpx.AsyncClient` is shared across all calls (created once in `__init__`).
This reuses the TCP connection to Ollama rather than opening a new connection
per request. On a tight learning loop running at maximum speed,
this makes a measurable difference.
Call `await bridge.close()` on backend shutdown to clean up gracefully.
