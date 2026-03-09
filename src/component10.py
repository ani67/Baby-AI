"""Component 10: Orchestrator

Single entry point: chat(prompt) -> str. Ties all components together.
Steps 1-4 run synchronously (retrieve context, inference, return).
Steps 5-9 run after returning: store episode, update state, trigger teacher.

Maintains within-session conversation history (last 6 exchanges).
Auto-learns from user statements, AI self-statements, and repeated topics.
Sleep/wake cycle: after 10 min inactivity, runs consolidation + heavy training.
"""

import json
import logging
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import component3
import component4
import component5  # register scoring hook
import component6
import component7
import component8
import component9
from component7 import InternalState

logger = logging.getLogger(__name__)

# ---------- dataclass ----------


@dataclass
class SystemStatus:
    current_state: InternalState
    episodes_stored: int
    facts_stored: int
    training_queue_size: int
    last_consolidation: float
    uptime_seconds: float
    self_narrative: str


# ---------- state ----------

_start_time: float | None = None
_conversation_history: list[tuple[str, str]] = []  # current session messages
_MAX_HISTORY = 6  # exchanges to keep in prompt

# ---------- session management ----------

_sessions_file: Path | None = None
_sessions: list[dict] = []  # [{id, title, created_at, messages: [{prompt, response}]}]
_current_session_id: str | None = None


def _save_sessions():
    """Save all sessions to disk."""
    if _sessions_file is None:
        return
    try:
        _sessions_file.parent.mkdir(parents=True, exist_ok=True)
        with open(_sessions_file, "w") as f:
            json.dump({"sessions": _sessions, "current": _current_session_id}, f)
    except Exception:
        pass


def _load_sessions():
    """Load sessions from disk."""
    global _current_session_id
    if _sessions_file is None or not _sessions_file.exists():
        return
    try:
        with open(_sessions_file, "r") as f:
            data = json.load(f)
        _sessions.clear()
        _sessions.extend(data.get("sessions", []))
        _current_session_id = data.get("current")
    except Exception:
        pass


def _sync_history_to_session():
    """Write current _conversation_history back into the active session."""
    for s in _sessions:
        if s["id"] == _current_session_id:
            s["messages"] = [
                {"prompt": p, "response": r} for p, r in _conversation_history
            ]
            break
    _save_sessions()


def _load_session_into_history(session_id: str):
    """Load a session's messages into _conversation_history."""
    global _current_session_id
    _conversation_history.clear()
    for s in _sessions:
        if s["id"] == session_id:
            for m in s.get("messages", []):
                _conversation_history.append((m["prompt"], m["response"]))
            _current_session_id = session_id
            break
    _save_sessions()


def new_session() -> str:
    """Create a new empty session and switch to it."""
    global _current_session_id
    import uuid
    # Save current session first
    if _current_session_id:
        _sync_history_to_session()

    session_id = uuid.uuid4().hex[:12]
    session = {
        "id": session_id,
        "title": "New conversation",
        "created_at": time.time(),
        "messages": [],
    }
    _sessions.insert(0, session)
    _conversation_history.clear()
    _current_session_id = session_id
    _save_sessions()
    return session_id


def switch_session(session_id: str) -> bool:
    """Switch to an existing session. Returns True if found."""
    if _current_session_id:
        _sync_history_to_session()
    for s in _sessions:
        if s["id"] == session_id:
            _load_session_into_history(session_id)
            return True
    return False


def get_sessions() -> list[dict]:
    """Return list of sessions (without full messages)."""
    return [
        {
            "id": s["id"],
            "title": s["title"],
            "created_at": s["created_at"],
            "message_count": len(s.get("messages", [])),
        }
        for s in _sessions
    ]


def delete_session(session_id: str) -> bool:
    """Delete a session. If it's the current one, switch to another or create new."""
    global _current_session_id
    for i, s in enumerate(_sessions):
        if s["id"] == session_id:
            _sessions.pop(i)
            if _current_session_id == session_id:
                _conversation_history.clear()
                if _sessions:
                    _load_session_into_history(_sessions[0]["id"])
                else:
                    new_session()
            _save_sessions()
            return True
    return False


# ---------- sleep/wake state ----------

_SLEEP_INACTIVITY = 600  # 10 minutes of no chat() calls triggers sleep
_sleep_state: str = "awake"  # "awake" | "sleeping" | "waking"
_sleep_started_at: float | None = None
_last_sleep_duration: float | None = None
_last_activity: float = 0.0
_awake_event = threading.Event()
_awake_event.set()  # starts awake
_inactivity_thread: threading.Thread | None = None
_inactivity_stop = threading.Event()
_sleep_lock = threading.Lock()  # protects sleep state transitions

_SLEEP_LOG_DIR = Path(__file__).resolve().parent.parent / "data"
_SLEEP_LOG_FILE = _SLEEP_LOG_DIR / "sleep_log.jsonl"

# ---------- auto-learning patterns ----------

_USER_DECLARATIVES = [
    "i am", "i work", "i use", "i like", "i prefer",
    "my name", "i built", "i have", "i create",
]

_AI_SELF_PATTERNS = [
    "i am", "my name is", "i prefer", "i will",
]

_STOP_WORDS = {
    "about", "after", "again", "along", "before", "being", "below",
    "could", "doing", "every", "going", "great", "having", "hello",
    "might", "never", "other", "place", "point", "quite", "really",
    "right", "shall", "should", "since", "still", "their", "there",
    "these", "thing", "think", "those", "under", "until", "using",
    "where", "which", "while", "whole", "would", "write", "wrong",
    "always", "answer", "because", "between", "called", "capital",
    "change", "different", "doesn", "during", "enough", "example",
    "first", "following", "found", "given", "happened", "however",
    "including", "instead", "known", "large", "later", "leave",
    "little", "looking", "making", "means", "might", "needs",
    "number", "often", "order", "original", "people", "perhaps",
    "please", "possible", "provide", "rather", "reason", "result",
    "return", "second", "several", "short", "simply", "since",
    "small", "something", "sometimes", "start", "state", "still",
    "system", "taken", "thank", "thanks", "that's", "these",
    "things", "thought", "through", "today", "together", "truly",
    "understand", "various", "wasn't", "without", "world", "years",
    "actually", "already", "another", "anything", "around", "asked",
    "based", "being", "better", "certain", "clearly", "comes",
    "consider", "correct", "current", "exactly", "general", "going",
    "happen", "important", "information", "interesting", "issue",
    "just", "know", "like", "long", "look", "make", "many", "more",
    "most", "much", "must", "need", "next", "only", "over", "part",
    "same", "seem", "show", "some", "such", "sure", "take", "tell",
    "test", "text", "than", "that", "them", "then", "they", "this",
    "time", "very", "want", "well", "what", "when", "will", "with",
    "work", "your",
}

# ---------- prompt building ----------

_SYSTEM_PROMPT = (
    "You are Kaida Reed, an AI assistant built by Ani Dalal. "
    "The person talking to you is Ani Dalal. "
    "You are not Ani. You are not human. "
    "Never introduce yourself as Ani Dalal. "
    "Never describe yourself as a product designer or generative artist. "
    "Those describe Ani, not you. "
    "Match response length to message length. "
    "Short input gets short output. "
    "Never say 'here is the concise answer' or any variation. "
    "Never add commentary to facts Ani tells you. "
    "When Ani tells you something about themselves, just acknowledge briefly. "
    "When uncertain, say so. Have opinions when asked."
)

_MAX_PROMPT_TOKENS = 2000

# ---------- ablation flags ----------

ABLATION_NO_STORE = False  # skip knowledge/episodic retrieval in prompts
ABLATION_NO_LORA = False   # use base model weights only for inference

# ---------- meta-question detection ----------

_META_PATTERNS = [
    "what do you know about me",
    "what have i told you",
    "what do you remember",
    "do you know who i am",
    "what do you know about yourself",
]


def _is_meta_question(prompt: str) -> bool:
    lower = prompt.lower()
    return any(p in lower for p in _META_PATTERNS)


def _handle_meta_question(prompt: str) -> str:
    """Handle meta-questions by listing all stored facts."""
    all_facts = component9.get_all_facts(n=20)
    if not all_facts:
        return "I don't have any stored facts yet."
    lines = [f"- {f.fact}" for f in all_facts]
    return "Here's what I know:\n" + "\n".join(lines)


def _count_tokens(text: str) -> int:
    """Estimate token count. ~4 chars per token for English text."""
    from component1 import _tokenizer
    try:
        return len(_tokenizer.encode(text))
    except Exception:
        return len(text) // 4


def _truncate(text: str, max_len: int = 80) -> str:
    """Truncate text to max_len characters."""
    if len(text) <= max_len:
        return text
    return text[:max_len].rstrip() + "..."


def _is_garbled(text: str) -> bool:
    """Detect garbled output: high non-ASCII ratio or random capitalisation."""
    if not text or len(text) < 5:
        return False
    # Check non-ASCII ratio
    non_ascii = sum(1 for c in text if ord(c) > 127)
    if non_ascii / len(text) > 0.2:
        return True
    # Check for random capitalisation sequences (3+ consecutive caps in a
    # word that isn't a known acronym pattern like "AI", "API", "URL")
    words = text.split()
    garbled_words = 0
    for w in words:
        alpha = "".join(c for c in w if c.isalpha())
        if len(alpha) >= 4 and alpha.isupper():
            garbled_words += 1
    if len(words) > 0 and garbled_words / len(words) > 0.15:
        return True
    return False


def _build_augmented_prompt(
    prompt: str,
    facts: list[component9.Fact],
    episodes: list[component4.Episode],
    history: list[tuple[str, str]] | None = None,
) -> str:
    """Build augmented prompt with facts, history, and past episode context.

    Enforces a token budget: system prompt and facts are never truncated.
    If the prompt exceeds _MAX_PROMPT_TOKENS, history is trimmed first,
    then episodes.
    """
    # 0. System prompt — ALWAYS first, cannot be overridden
    system_part = _SYSTEM_PROMPT

    # 0b. Date injection — always right after system prompt
    import datetime
    date_part = f"Today's date is {datetime.date.today().strftime('%B %d %Y')}."

    # 1. Past episode context (long-term memory) — corrections only
    corrections = [ep for ep in (episodes or [])[:5] if ep.correction is not None]
    correction_lines = []
    if corrections:
        for ep in corrections:
            correction_lines.append(
                f"- \"{_truncate(ep.prompt)}\" → Correct answer: \"{_truncate(ep.correction)}\""
            )

    # 2. Relevant facts — framed as facts about the USER, not about you
    facts_part = ""
    if facts:
        fact_texts = ". ".join(f.fact for f in facts[:3])
        facts_part = f"Known facts about the user: {fact_texts}."

    # 3. Recent conversation history — only user messages, truncated
    hist_lines = []
    if history:
        for h_prompt, h_response in history[-3:]:
            hist_lines.append(
                f"User: {_truncate(h_prompt)} → You said: {_truncate(h_response, 60)}"
            )

    # 4. Current prompt — clear instruction
    user_part = f"Answer ONLY the following. Be concise.\nUser: {prompt}"

    # --- token budget enforcement ---
    # Fixed parts that are never truncated: system_part, date_part, facts_part, user_part
    fixed = "\n\n".join(p for p in [system_part, date_part, facts_part, user_part] if p)
    budget_remaining = _MAX_PROMPT_TOKENS - _count_tokens(fixed)

    # Trim history first (oldest first)
    while hist_lines and budget_remaining < _count_tokens(
        "Recent conversation (context only):\n" + "\n".join(hist_lines)
    ):
        hist_lines.pop(0)

    # Then trim corrections (oldest first)
    while correction_lines and budget_remaining < _count_tokens(
        "Recent conversation (context only):\n" + "\n".join(hist_lines)
    ) + _count_tokens(
        "Corrections to remember:\n" + "\n".join(correction_lines)
    ):
        correction_lines.pop(0)

    # Assemble final prompt
    parts = [system_part, date_part]
    if correction_lines:
        parts.append("Corrections to remember:\n" + "\n".join(correction_lines))
    if facts_part:
        parts.append(facts_part)
    if hist_lines:
        parts.append("Recent conversation (context only):\n" + "\n".join(hist_lines))
    parts.append(user_part)

    result = "\n\n".join(parts)
    logger.info("[PROMPT] %d tokens | %d chars:\n%s", _count_tokens(result), len(result), result)
    return result


# ---------- auto-learning ----------


def _extract_user_facts(prompt: str):
    """Store first-person declarative statements as facts and queue for LoRA training."""
    lower = prompt.lower()
    for pattern in _USER_DECLARATIVES:
        if pattern in lower:
            component9.store_fact(prompt, "conversation", 0.85)
            component3.submit_correction(prompt, prompt)
            return


def _extract_ai_facts(response: str):
    """Store AI self-statements as facts."""
    lower = response.lower()
    for pattern in _AI_SELF_PATTERNS:
        if pattern in lower:
            # Extract the sentence containing the pattern
            sentences = re.split(r'[.!?\n]', response)
            for sentence in sentences:
                if pattern in sentence.lower() and len(sentence.strip()) > 5:
                    component9.store_fact(sentence.strip(), "self", 0.7)
                    return


def _check_repeated_topics():
    """Promote frequently-discussed topics to facts."""
    episodes = component4.get_recent_episodes(20)
    if len(episodes) < 3:
        return

    # Count significant words across episode prompts
    word_episode_count: dict[str, int] = {}
    for ep in episodes:
        words = set(
            w.lower() for w in re.findall(r'[A-Za-z]+', ep.prompt)
            if len(w) > 4 and w.lower() not in _STOP_WORDS
        )
        for w in words:
            word_episode_count[w] = word_episode_count.get(w, 0) + 1

    for word, count in word_episode_count.items():
        if count >= 3:
            # Check if already stored as a frequency fact
            existing = component9.retrieve_facts(
                f"frequently discusses {word}", n=5
            )
            already = any(
                f.source == "frequency" and word in f.fact.lower()
                for f in existing
            )
            if not already:
                component9.store_fact(
                    f"User frequently discusses: {word}",
                    "frequency", 0.6,
                )


# ---------- sleep/wake cycle ----------


def _log_sleep(event: str, details: dict | None = None):
    """Append an entry to the sleep log."""
    _SLEEP_LOG_DIR.mkdir(parents=True, exist_ok=True)
    entry = {"timestamp": time.time(), "event": event}
    if details:
        entry.update(details)
    try:
        with open(_SLEEP_LOG_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass


def _heavy_lora_training():
    """Train on all episodes with importance > 0.7. Caller holds model lock context."""
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from component2 import _build_training_tokens, _compute_loss
    from component1 import _model

    component4._ensure_init()
    episodes = [
        ep for ep in component4._episodes.values()
        if ep.importance_score > 0.7
    ]
    if not episodes:
        return 0

    loss_and_grad = nn.value_and_grad(_model, _compute_loss)
    trained = 0
    for ep in episodes:
        lr = component5.learning_rate_for_episode(ep)
        optimizer = optim.Adam(learning_rate=lr)
        if ep.correction is not None:
            tokens = _build_training_tokens(ep.prompt, ep.correction)
        else:
            tokens = _build_training_tokens(ep.prompt, ep.response)
        (loss, _), grads = loss_and_grad(_model, tokens)
        optimizer.update(_model, grads)
        mx.eval(_model.parameters(), optimizer.state)
        trained += 1

    # Save to buffer_b
    from component3 import _save_adapter_to, BUFFER_B
    _save_adapter_to(BUFFER_B)

    return trained


def _run_sleep_cycle():
    """Execute the full sleep cycle. Blocks until complete."""
    global _sleep_state, _sleep_started_at, _last_sleep_duration

    with _sleep_lock:
        if _sleep_state != "awake":
            return  # already sleeping
        _sleep_state = "sleeping"
        _sleep_started_at = time.time()
        _awake_event.clear()

    _log_sleep("sleep_start")
    logger.info("Sleep cycle starting")

    try:
        # Step 1: Block inference (already done by clearing _awake_event)

        # Step 2: Run full consolidation cycle
        try:
            report = component6.run_consolidation_cycle()
            _log_sleep("consolidation_done", {
                "episodes_processed": report.episodes_processed,
                "episodes_pruned": report.episodes_pruned,
            })
        except Exception as e:
            logger.error("Sleep consolidation failed: %s", e)
            _log_sleep("consolidation_error", {"error": str(e)})

        # Step 3: Batch process queued teacher queries
        try:
            processed = 0
            while not component8._query_queue.empty():
                try:
                    prompt, context = component8._query_queue.get_nowait()
                    component8.request_teacher_guidance(prompt, context)
                    processed += 1
                except Exception:
                    break
            _log_sleep("teacher_batch_done", {"queries_processed": processed})
        except Exception as e:
            logger.error("Sleep teacher batch failed: %s", e)

        # Step 4: Run fact frequency promotion
        try:
            _check_repeated_topics()
            _log_sleep("fact_promotion_done")
        except Exception as e:
            logger.error("Sleep fact promotion failed: %s", e)

        # Step 5: Heavy LoRA training on high-importance episodes
        try:
            from component3 import _model_lock
            with _model_lock:
                trained = _heavy_lora_training()
            _log_sleep("heavy_training_done", {"episodes_trained": trained})
        except Exception as e:
            logger.error("Sleep heavy training failed: %s", e)
            _log_sleep("heavy_training_error", {"error": str(e)})

    finally:
        # Step 6: Wake — unblock inference
        duration = time.time() - _sleep_started_at if _sleep_started_at else 0.0
        with _sleep_lock:
            _last_sleep_duration = duration
            _sleep_state = "awake"
            _sleep_started_at = None
            _awake_event.set()

        _log_sleep("sleep_end", {"duration_seconds": round(duration, 2)})
        logger.info("Sleep cycle complete (%.1fs)", duration)


def _inactivity_monitor():
    """Background thread: check for inactivity and trigger sleep."""
    while not _inactivity_stop.is_set():
        if (
            _sleep_state == "awake"
            and _last_activity > 0
            and time.time() - _last_activity >= _SLEEP_INACTIVITY
        ):
            _run_sleep_cycle()
        _inactivity_stop.wait(30)  # check every 30 seconds


def get_sleep_status() -> dict:
    """Return current sleep state info."""
    result = {
        "state": _sleep_state,
        "sleeping_since": None,
        "elapsed_seconds": None,
        "last_sleep_duration": _last_sleep_duration,
    }
    if _sleep_state == "sleeping" and _sleep_started_at is not None:
        from datetime import datetime, timezone
        result["sleeping_since"] = datetime.fromtimestamp(
            _sleep_started_at, tz=timezone.utc
        ).isoformat()
        result["elapsed_seconds"] = round(time.time() - _sleep_started_at, 1)
    return result


# ---------- background post-inference ----------


def _post_inference_tasks(prompt: str, response: str):
    """Run post-inference tasks: state update, auto-learning, teacher trigger."""
    # Update internal state (no token entropies available through component3)
    component7.notify_inference(prompt, response, [])

    # Notify consolidation of new episode
    component6.notify_new_episode()

    # Auto-learn from conversation
    _extract_user_facts(prompt)
    _extract_ai_facts(response)
    _check_repeated_topics()

    # If uncertainty flag set: trigger teacher query
    if component7.uncertainty_flag:
        component8.enqueue_query(prompt)


# ---------- public API ----------


def chat(prompt: str) -> str:
    """Process a user message. Returns the model's response.

    Steps 1-4 run synchronously; steps 5-9 run in background.
    If the system is sleeping, waits for the sleep cycle to complete first.
    """
    global _last_activity

    # Wait for sleep cycle to complete if sleeping (blocks, does not interrupt)
    _awake_event.wait()

    _last_activity = time.time()

    # 0. Meta-question shortcut — list all facts, skip inference
    if _is_meta_question(prompt):
        response = _handle_meta_question(prompt)
        _conversation_history.append((prompt, response))
        for s in _sessions:
            if s["id"] == _current_session_id:
                if s["title"] == "New conversation":
                    s["title"] = prompt[:50] + ("..." if len(prompt) > 50 else "")
                break
        _sync_history_to_session()
        component4.store_episode(prompt, response, None, time.time())
        return response

    # 1. Retrieve relevant facts (skip if ablation)
    if ABLATION_NO_STORE:
        high_conf_facts = []
        episodes = []
    else:
        facts = component9.retrieve_facts(prompt, n=5)
        high_conf_facts = [f for f in facts if f.confidence > 0.7]
        episodes = component4.get_similar_episodes(prompt, n=5)

    # 2. Build augmented prompt (facts + history + current prompt)
    augmented = _build_augmented_prompt(
        prompt, high_conf_facts, episodes,
        history=_conversation_history[-_MAX_HISTORY:],
    )

    # 3. Run inference
    if ABLATION_NO_LORA:
        # Use base model weights — zero out LoRA params, infer, restore
        from component3 import _model_lock
        from component1 import _model, _tokenizer
        from mlx_lm.sample_utils import make_sampler
        from mlx.utils import tree_flatten
        import mlx.core as mx
        import mlx_lm

        with _model_lock:
            snapshot = {name: mx.array(param) for name, param in tree_flatten(_model.trainable_parameters())}
            _model.load_weights(
                [(name, mx.zeros_like(param)) for name, param in snapshot.items()],
                strict=False,
            )
            mx.eval(_model.parameters())
            formatted = _tokenizer.apply_chat_template(
                [{"role": "user", "content": augmented}],
                add_generation_prompt=True, tokenize=False,
            )
            response = mlx_lm.generate(
                _model, _tokenizer, prompt=formatted,
                max_tokens=256, sampler=make_sampler(temp=0.7),
            )
            _model.load_weights(list(snapshot.items()), strict=False)
            mx.eval(_model.parameters())
    else:
        response = component9._original_query(augmented)

    # 3b. Sanity check — retry once with lower temperature if garbled
    if _is_garbled(response):
        logger.warning("[GARBLED] Response failed sanity check, retrying with temp=0.3: %s", response[:200])
        from component3 import _model_lock
        with _model_lock:
            from component1 import _model, _tokenizer
            from mlx_lm.sample_utils import make_sampler
            import mlx_lm
            formatted = _tokenizer.apply_chat_template(
                [{"role": "user", "content": augmented}],
                add_generation_prompt=True, tokenize=False,
            )
            response = mlx_lm.generate(
                _model, _tokenizer, prompt=formatted,
                max_tokens=256, sampler=make_sampler(temp=0.3),
            )
        if _is_garbled(response):
            logger.error("[GARBLED] Retry also garbled: %s", response[:200])
            response = "I encountered an error, please try again."

    # Record in conversation history
    _conversation_history.append((prompt, response))

    # Update session title from first message
    for s in _sessions:
        if s["id"] == _current_session_id:
            if s["title"] == "New conversation":
                s["title"] = prompt[:50] + ("..." if len(prompt) > 50 else "")
            break

    _sync_history_to_session()

    # 4. Store episode (synchronous — next chat() needs to see it)
    component4.store_episode(prompt, response, None, time.time())

    # 5. Scoring is automatic via component5 hook in store_episode

    # 6-8. Background: state update, auto-learning, teacher trigger
    threading.Thread(
        target=_post_inference_tasks,
        args=(prompt, response),
        daemon=True,
    ).start()

    return response


def get_system_status() -> SystemStatus:
    """Return current system diagnostics including self-narrative."""
    state = component7.get_current_state()
    uptime = time.time() - _start_time if _start_time else 0.0

    return SystemStatus(
        current_state=state,
        episodes_stored=component4.get_episode_count(),
        facts_stored=component9.get_fact_count(),
        training_queue_size=component3.get_queue_size(),
        last_consolidation=component6._last_consolidation_time,
        uptime_seconds=uptime,
        self_narrative=_generate_self_narrative(),
    )


# ---------- consolidation-only ----------


def run_consolidation_only() -> dict:
    """Run consolidation immediately without full sleep cycle."""
    report = component6.run_consolidation_cycle()
    return {
        "episodes_processed": report.episodes_processed,
        "episodes_pruned": report.episodes_pruned,
        "loss_before": round(report.adapter_loss_before, 6),
        "loss_after": round(report.adapter_loss_after, 6),
        "duration_seconds": round(report.duration_seconds, 2),
    }


# ---------- ablation ----------


def get_ablation_status() -> dict:
    return {"no_store": ABLATION_NO_STORE, "no_lora": ABLATION_NO_LORA}


def set_ablation(no_store: bool | None = None, no_lora: bool | None = None):
    global ABLATION_NO_STORE, ABLATION_NO_LORA
    if no_store is not None:
        ABLATION_NO_STORE = no_store
    if no_lora is not None:
        ABLATION_NO_LORA = no_lora


# ---------- self-narrative ----------


def _generate_self_narrative() -> str:
    """Generate a self-report grounded in actual internal state and history."""
    state = component7.get_current_state()
    episodes = component4.get_recent_episodes(1000)
    corrections = [e for e in episodes if e.correction is not None]

    if state.novelty < 0.3:
        territory = "in familiar territory"
    elif state.novelty < 0.7:
        territory = "in somewhat unfamiliar territory"
    else:
        territory = "in quite unfamiliar territory"

    if state.coherence > 0.8:
        consistency = "highly consistent"
    elif state.coherence > 0.5:
        consistency = "mostly consistent"
    else:
        consistency = "somewhat inconsistent"

    return (
        f"I have processed {len(episodes)} interactions. "
        f"I have made corrections on {len(corrections)} of them. "
        f"My current uncertainty is {state.uncertainty:.0%}. "
        f"My recent performance is {state.performance:.0%}. "
        f"I am {territory}. "
        f"My outputs have been {consistency} recently."
    )


# ---------- lifecycle ----------


def start(data_dir=None):
    """Initialise all components and start background services."""
    global _start_time, _last_activity, _sleep_state, _sleep_started_at
    global _inactivity_thread
    _start_time = time.time()
    _last_activity = time.time()

    # Reset sleep state
    _sleep_state = "awake"
    _sleep_started_at = None
    _awake_event.set()
    _inactivity_stop.clear()

    # Load sessions from disk (persists across restarts)
    global _sessions_file, _current_session_id
    data_path = Path(data_dir) if data_dir else Path(__file__).resolve().parent.parent / "data"
    _sessions_file = data_path / "sessions.json"
    _sessions.clear()
    _conversation_history.clear()
    _current_session_id = None
    _load_sessions()

    # Always start a fresh session on restart — old sessions stay in sidebar
    # This prevents garbled history from feeding back into the next prompt
    # Null out current so new_session() doesn't sync empty history into old session
    _current_session_id = None
    new_session()

    # Initialise stores
    component4.init(data_dir=data_dir)
    component9.init(data_dir=data_dir)

    # Start background services
    component3.start()
    component6.start()
    component8.start()

    # Start inactivity monitor
    _inactivity_thread = threading.Thread(
        target=_inactivity_monitor, daemon=True
    )
    _inactivity_thread.start()


def stop():
    """Stop all background services."""
    # Stop inactivity monitor
    _inactivity_stop.set()
    if _inactivity_thread is not None:
        _inactivity_thread.join(timeout=5)

    # Ensure awake (in case stop() called during sleep)
    _awake_event.set()

    component8.stop()
    component6.stop()
    component3.stop()
