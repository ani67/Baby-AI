"""Component 8: Teacher Ensemble

When the model is uncertain, ask external teacher models for reasoning
traces. Consensus among teachers determines confidence and whether the
response is training-worthy. Runs in a background thread; never blocks
inference. Respects free-tier rate limits.

Teacher sources:
  1. Local Ollama phi4-mini (if running)
  2. Gemini Flash free tier (if GEMINI_API_KEY set)
"""

import collections
import json
import logging
import os
import queue
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import requests

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import component4
import component7

logger = logging.getLogger(__name__)

# ---------- dataclass ----------


@dataclass
class TeacherResponse:
    prompt: str
    reasoning_trace: str
    confidence: float        # 0-1
    sources: list[str]       # which teachers contributed
    training_worthy: bool    # True if confidence > 0.6


# ---------- configuration ----------

_OLLAMA_URL = "http://localhost:11434"
_OLLAMA_MODEL = "phi4-mini"
_GEMINI_MODEL = "gemini-2.0-flash"

_MAX_QUERIES_PER_HOUR = 10
_HOUR_SECONDS = 3600

_TEACHER_PROMPT_TEMPLATE = (
    "Explain step by step how to reason about the following.\n"
    "Show your reasoning process explicitly, not just the answer.\n\n"
    "Question: {prompt}\n\n"
    "Respond with: REASONING: [your step-by-step process]\n"
    "              ANSWER: [your conclusion]"
)

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_LOG_FILE = _DATA_DIR / "teacher_log.jsonl"

# ---------- rate limiter ----------

_query_timestamps: collections.deque = collections.deque()
_rate_lock = threading.Lock()


def _can_query() -> bool:
    """Check if we're under the rate limit."""
    now = time.time()
    with _rate_lock:
        # Remove timestamps older than 1 hour
        while _query_timestamps and _query_timestamps[0] < now - _HOUR_SECONDS:
            _query_timestamps.popleft()
        return len(_query_timestamps) < _MAX_QUERIES_PER_HOUR


def _record_query():
    """Record that a query was made."""
    with _rate_lock:
        _query_timestamps.append(time.time())


# ---------- teacher source: Ollama ----------


def _ollama_available() -> bool:
    """Check if Ollama is running and has phi4-mini."""
    try:
        resp = requests.get(f"{_OLLAMA_URL}/api/tags", timeout=3)
        if resp.status_code != 200:
            return False
        models = resp.json().get("models", [])
        return any(_OLLAMA_MODEL in m.get("name", "") for m in models)
    except Exception:
        return False


def _query_ollama(prompt: str) -> str | None:
    """Query Ollama phi4-mini. Returns response text or None on failure."""
    try:
        # Quick availability check before committing to the POST
        check = requests.get(f"{_OLLAMA_URL}/api/tags", timeout=2)
        if check.status_code != 200:
            return None
    except Exception:
        return None
    try:
        teacher_prompt = _TEACHER_PROMPT_TEMPLATE.format(prompt=prompt)
        resp = requests.post(
            f"{_OLLAMA_URL}/api/generate",
            json={"model": _OLLAMA_MODEL, "prompt": teacher_prompt, "stream": False},
            timeout=30,
        )
        if resp.status_code == 200:
            return resp.json().get("response", "")
        return None
    except Exception as e:
        logger.warning("Ollama query failed: %s", e)
        return None


# ---------- teacher source: Gemini Flash ----------


def _gemini_available() -> bool:
    """Check if GEMINI_API_KEY is set."""
    return bool(os.getenv("GEMINI_API_KEY"))


def _query_gemini(prompt: str) -> str | None:
    """Query Gemini Flash free tier. Returns response text or None on failure."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None

    try:
        teacher_prompt = _TEACHER_PROMPT_TEMPLATE.format(prompt=prompt)
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{_GEMINI_MODEL}:generateContent?key={api_key}"
        payload = {
            "contents": [{"parts": [{"text": teacher_prompt}]}],
        }
        resp = requests.post(url, json=payload, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            candidates = data.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                if parts:
                    return parts[0].get("text", "")
        else:
            logger.warning("Gemini query failed with status %d", resp.status_code)
        return None
    except Exception as e:
        logger.warning("Gemini query failed: %s", e)
        return None


# ---------- consensus ----------


def _responses_agree(resp1: str, resp2: str) -> bool:
    """Check if two teacher responses roughly agree.

    Extracts ANSWER: sections and compares them. If both contain an ANSWER:
    line, checks word overlap. Otherwise falls back to overall similarity.
    """
    def extract_answer(text: str) -> str:
        for line in text.split("\n"):
            stripped = line.strip()
            if stripped.upper().startswith("ANSWER:"):
                return stripped[7:].strip().lower()
        # Fallback: last non-empty line
        lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
        return lines[-1].lower() if lines else ""

    ans1 = extract_answer(resp1)
    ans2 = extract_answer(resp2)

    if not ans1 or not ans2:
        return False

    # Word overlap ratio
    words1 = set(ans1.split())
    words2 = set(ans2.split())
    if not words1 or not words2:
        return False

    overlap = len(words1 & words2)
    total = len(words1 | words2)
    return (overlap / total) > 0.3 if total > 0 else False


def _build_consensus(responses: dict[str, str]) -> TeacherResponse:
    """Build a TeacherResponse from one or more teacher responses."""
    sources = list(responses.keys())

    if len(responses) == 1:
        source_name = sources[0]
        trace = responses[source_name]
        confidence = 0.5
    elif len(responses) == 2:
        resp_list = list(responses.values())
        if _responses_agree(resp_list[0], resp_list[1]):
            confidence = 0.9
            # Use the longer response as the trace
            trace = max(resp_list, key=len)
        else:
            confidence = 0.2
            trace = resp_list[0]  # use first source
    else:
        # Should not happen, but handle gracefully
        trace = list(responses.values())[0]
        confidence = 0.5

    training_worthy = confidence > 0.6

    return TeacherResponse(
        prompt="",  # filled by caller
        reasoning_trace=trace,
        confidence=round(confidence, 4),
        sources=sources,
        training_worthy=training_worthy,
    )


# ---------- logging ----------


def _log_query(prompt: str, response: TeacherResponse | None):
    """Log a teacher query to disk."""
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": time.time(),
        "prompt": prompt,
    }
    if response is not None:
        entry.update(asdict(response))
    else:
        entry["result"] = "no_teachers_available"

    with open(_LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ---------- main query function ----------


def request_teacher_guidance(
    prompt: str, context: str | None = None
) -> TeacherResponse | None:
    """Query available teachers and return consensus response.

    Returns None if no teachers are available or rate limit hit.
    """
    if not _can_query():
        _log_query(prompt, None)
        return None

    full_prompt = prompt
    if context:
        full_prompt = f"{context}\n\n{prompt}"

    responses: dict[str, str] = {}

    # Source 1: Ollama phi4-mini
    if _ollama_available():
        result = _query_ollama(full_prompt)
        if result:
            responses["ollama_phi4mini"] = result

    # Source 2: Gemini Flash
    if _gemini_available():
        result = _query_gemini(full_prompt)
        if result:
            responses["gemini_flash"] = result

    if not responses:
        _log_query(prompt, None)
        return None

    _record_query()

    teacher_response = _build_consensus(responses)
    teacher_response.prompt = prompt

    _log_query(prompt, teacher_response)
    return teacher_response


# ---------- background queue ----------

_query_queue: queue.Queue = queue.Queue()
_worker_thread: threading.Thread | None = None
_stop_event = threading.Event()


def _worker_loop():
    """Background worker: processes queued teacher queries."""
    while not _stop_event.is_set():
        try:
            prompt, context = _query_queue.get(timeout=5)
        except queue.Empty:
            continue

        # Wait for rate limit if needed
        while not _can_query() and not _stop_event.is_set():
            time.sleep(10)

        if _stop_event.is_set():
            break

        try:
            result = request_teacher_guidance(prompt, context)
            if result and result.training_worthy:
                # Store in episodic memory for training
                component4._ensure_init()
                component4.store_episode(
                    prompt=result.prompt,
                    response=result.reasoning_trace,
                    correction=result.reasoning_trace if result.training_worthy else None,
                    timestamp=time.time(),
                )
        except Exception as e:
            logger.error("Teacher query failed: %s", e)


def enqueue_query(prompt: str, context: str | None = None):
    """Add a query to the background queue. Never blocks."""
    _query_queue.put((prompt, context))


def start():
    """Start the background teacher worker thread."""
    global _worker_thread
    _stop_event.clear()
    _worker_thread = threading.Thread(target=_worker_loop, daemon=True)
    _worker_thread.start()


def stop():
    """Stop the background teacher worker thread."""
    _stop_event.set()
    if _worker_thread is not None:
        _worker_thread.join(timeout=10)


def reset():
    """Reset rate limiter and queue. Useful for testing."""
    with _rate_lock:
        _query_timestamps.clear()
    # Drain the queue
    while not _query_queue.empty():
        try:
            _query_queue.get_nowait()
        except queue.Empty:
            break
