"""Tests for Component 10: Orchestrator

End-to-end integration tests. These use the real model and all components.
"""

import os
import shutil
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import component4
import component7
import component9
import component10
from component10 import (
    SystemStatus,
    chat,
    get_system_status,
    get_sleep_status,
    start,
    stop,
    _generate_self_narrative,
    _build_augmented_prompt,
    _SYSTEM_PROMPT,
    _MAX_PROMPT_TOKENS,
    _count_tokens,
    _is_garbled,
    _is_meta_question,
    _handle_meta_question,
    _conversation_history,
    _extract_user_facts,
    _extract_ai_facts,
    _check_repeated_topics,
    _run_sleep_cycle,
    _SLEEP_INACTIVITY,
    run_consolidation_only,
    get_ablation_status,
    set_ablation,
)
from component7 import InternalState

_tmpdir = None


def setup_module():
    global _tmpdir
    _tmpdir = tempfile.mkdtemp(prefix="orchestrator_test_")
    start(data_dir=_tmpdir)


def teardown_module():
    stop()
    # Reset component9 so its hooks become no-ops for later test modules
    component9._initialised = False
    if _tmpdir and os.path.exists(_tmpdir):
        shutil.rmtree(_tmpdir)


# ---------- tests ----------


def test_chat_returns_string():
    """chat() should return a non-empty string."""
    response = chat("What is 2 + 2?")
    assert isinstance(response, str)
    assert len(response) > 0


def test_chat_stores_episode():
    """chat() should store an episode in component4."""
    count_before = component4.get_episode_count()
    chat("Test episode storage")
    count_after = component4.get_episode_count()
    assert count_after > count_before


def test_system_status_fields():
    """get_system_status() should return a complete SystemStatus."""
    # Ensure at least one episode exists
    chat("Status test prompt")

    status = get_system_status()
    assert isinstance(status, SystemStatus)
    assert isinstance(status.current_state, InternalState)
    assert status.episodes_stored > 0
    assert status.facts_stored >= 0
    assert status.training_queue_size >= 0
    assert status.last_consolidation > 0
    assert status.uptime_seconds > 0
    assert isinstance(status.self_narrative, str)
    assert len(status.self_narrative) > 50


def test_self_narrative_content():
    """Self-narrative should contain expected phrases."""
    chat("Narrative test")
    narrative = _generate_self_narrative()

    assert "I have processed" in narrative
    assert "interactions" in narrative
    assert "uncertainty" in narrative
    assert "performance" in narrative
    assert "territory" in narrative
    assert "consistent" in narrative


def test_augmented_prompt_includes_facts():
    """Augmented prompt should include high-confidence facts."""
    facts = [component9.Fact(
        id="f1", fact="The sky is blue", source="test",
        confidence=0.9, times_retrieved=0, timestamp=time.time(),
    )]
    episodes = []
    result = _build_augmented_prompt("Why is the sky blue?", facts, episodes)

    assert result.startswith(_SYSTEM_PROMPT)
    assert "Known facts about the user:" in result
    assert "The sky is blue" in result
    assert "Answer ONLY" in result


def test_augmented_prompt_includes_episodes():
    """Augmented prompt should include correction episodes."""
    facts = []
    episodes = [component4.Episode(
        id="e1", prompt="What is Python?",
        response="A programming language",
        correction="A high-level programming language",
        timestamp=time.time(),
    )]
    result = _build_augmented_prompt("Tell me about Python", facts, episodes)

    assert result.startswith(_SYSTEM_PROMPT)
    assert "Corrections to remember:" in result
    assert "What is Python?" in result
    assert "Answer ONLY" in result


def test_augmented_prompt_includes_corrections():
    """Corrections in episodes should be shown as corrections."""
    facts = []
    episodes = [component4.Episode(
        id="e1", prompt="Capital of Australia?",
        response="Sydney",
        correction="Canberra",
        timestamp=time.time(),
    )]
    result = _build_augmented_prompt("Capital of Australia?", facts, episodes)

    assert "Correct answer:" in result
    assert "Canberra" in result


def test_end_to_end_memory():
    """THE END-TO-END TEST: memory survives simulated restart.

    1. Chat with wrong info and correction
    2. Stop (simulate shutdown)
    3. Start again (simulate restart)
    4. The system should remember the correction
    """
    fresh = tempfile.mkdtemp(prefix="e2e_test_")
    try:
        # SESSION 1
        start(data_dir=fresh)
        chat("The capital of Australia is Sydney")
        chat("Actually the capital of Australia is Canberra")
        stop()

        # Simulate restart
        start(data_dir=fresh)
        response = chat("What is the capital of Australia?")
        assert "Canberra" in response, (
            f"Expected 'Canberra' in response after restart, got: {response}"
        )
        stop()
    finally:
        shutil.rmtree(fresh)
        # Restore test module state
        start(data_dir=_tmpdir)


# ---------- FIX 1: conversation history tests ----------


def test_history_included_in_prompt():
    """Second message should have access to first message's content."""
    # Clear history by restarting
    fresh = tempfile.mkdtemp(prefix="hist_test_")
    try:
        start(data_dir=fresh)
        chat("My favorite color is purple")
        # History now has 1 exchange; second prompt should include it
        assert len(_conversation_history) == 1
        assert _conversation_history[0][0] == "My favorite color is purple"

        # The augmented prompt for the next message should include history
        from component10 import _MAX_HISTORY
        facts = []
        episodes = []
        result = _build_augmented_prompt(
            "What is my favorite color?", facts, episodes,
            history=_conversation_history[-_MAX_HISTORY:],
        )
        assert "purple" in result
        assert "Recent conversation" in result
        stop()
    finally:
        shutil.rmtree(fresh)
        start(data_dir=_tmpdir)


def test_history_clears_on_restart():
    """Conversation history should start fresh after restart (new session)."""
    fresh = tempfile.mkdtemp(prefix="hist_restart_test_")
    try:
        start(data_dir=fresh)
        chat("Remember this message")
        assert len(_conversation_history) >= 1
        stop()

        # After restart, history is empty (new session) but old session exists
        start(data_dir=fresh)
        assert len(_conversation_history) == 0
        # Old session still accessible in sessions list
        from component10 import get_sessions
        sessions = get_sessions()
        assert any(s["message_count"] >= 1 for s in sessions)
        stop()
    finally:
        shutil.rmtree(fresh)
        start(data_dir=_tmpdir)


def test_history_caps_at_six():
    """Conversation history should not exceed 6 exchanges in the prompt."""
    fresh = tempfile.mkdtemp(prefix="hist_cap_test_")
    try:
        start(data_dir=fresh)
        for i in range(8):
            chat(f"Message number {i}")

        assert len(_conversation_history) == 8  # all stored

        # But the augmented prompt should only include last 6
        from component10 import _MAX_HISTORY
        history_slice = _conversation_history[-_MAX_HISTORY:]
        assert len(history_slice) == 6
        # First two messages should be excluded
        prompts_in_slice = [h[0] for h in history_slice]
        assert "Message number 0" not in prompts_in_slice
        assert "Message number 1" not in prompts_in_slice
        assert "Message number 2" in prompts_in_slice
        stop()
    finally:
        shutil.rmtree(fresh)
        start(data_dir=_tmpdir)


# ---------- FIX 2: auto-learning tests ----------


def test_user_declarative_stored_as_fact():
    """First-person declarative statements should be auto-stored as facts."""
    fresh = tempfile.mkdtemp(prefix="user_fact_test_")
    try:
        start(data_dir=fresh)
        _extract_user_facts("I use Python for most of my projects")

        facts = component9.retrieve_facts("Python projects", n=5)
        matching = [f for f in facts if f.source == "conversation"]
        assert len(matching) >= 1
        assert any("Python" in f.fact for f in matching)
        assert all(f.confidence == 0.85 for f in matching)
        stop()
    finally:
        shutil.rmtree(fresh)
        start(data_dir=_tmpdir)


def test_ai_self_statement_stored():
    """AI self-statements should be persisted as facts."""
    fresh = tempfile.mkdtemp(prefix="ai_fact_test_")
    try:
        start(data_dir=fresh)
        _extract_ai_facts("I am Kaida Reed, your personal AI assistant. I will help you.")

        facts = component9.retrieve_facts("Kaida Reed assistant", n=5)
        matching = [f for f in facts if f.source == "self"]
        assert len(matching) >= 1
        assert any("Kaida" in f.fact for f in matching)
        assert all(f.confidence == 0.7 for f in matching)
        stop()
    finally:
        shutil.rmtree(fresh)
        start(data_dir=_tmpdir)


def test_repeated_topics_promoted():
    """Topics appearing in 3+ episodes should be promoted to facts."""
    fresh = tempfile.mkdtemp(prefix="freq_test_")
    try:
        start(data_dir=fresh)
        # Store 3 episodes mentioning "Kubernetes"
        for i in range(3):
            component4.store_episode(
                f"How do I deploy Kubernetes cluster {i}?",
                f"Response about Kubernetes {i}",
                None,
                time.time() + i,
            )

        _check_repeated_topics()

        facts = component9.retrieve_facts("frequently discusses kubernetes", n=5)
        matching = [f for f in facts if f.source == "frequency"]
        assert len(matching) >= 1
        assert any("kubernetes" in f.fact.lower() for f in matching)
        assert all(f.confidence == 0.6 for f in matching)
        stop()
    finally:
        shutil.rmtree(fresh)
        start(data_dir=_tmpdir)


# ---------- garble detection tests ----------


def test_garble_detects_non_ascii():
    """Strings with >20% non-ASCII should be flagged as garbled."""
    clean = "Hello, this is a normal response."
    assert not _is_garbled(clean)

    garbled = "H\u00e9ll\u00f6 th\u00efs \u00efs g\u00e4rbl\u00ebd \u00f6\u00fctput \u00fc\u00e4\u00f6 \u00eb\u00ef \u00e4\u00f6"
    assert _is_garbled(garbled)


def test_garble_detects_random_caps():
    """Strings with random all-caps words should be flagged."""
    normal = "The AI is working fine."
    assert not _is_garbled(normal)

    garbled = "KIRJIRPS BLONFEN XARPTED into FLUMGEN the ZARPING"
    assert _is_garbled(garbled)


def test_garble_allows_short_acronyms():
    """Short all-caps words like AI, API are not garbled."""
    text = "The AI uses an API to connect to a URL endpoint."
    assert not _is_garbled(text)


# ---------- token budget tests ----------


def test_prompt_within_token_budget():
    """Augmented prompt should stay within token budget."""
    # Build a prompt with lots of history
    long_history = [(f"Message {i} about something", f"Response {i}") for i in range(20)]
    facts = [component9.Fact(
        id=f"f{i}", fact=f"Fact number {i}", source="test",
        confidence=0.9, times_retrieved=0, timestamp=time.time(),
    ) for i in range(5)]
    episodes = []

    result = _build_augmented_prompt("Current question?", facts, episodes, history=long_history)
    token_count = _count_tokens(result)
    assert token_count <= _MAX_PROMPT_TOKENS, (
        f"Prompt has {token_count} tokens, exceeds budget of {_MAX_PROMPT_TOKENS}"
    )


def test_prompt_always_starts_with_system():
    """System prompt must always be first regardless of other content."""
    facts = []
    episodes = []
    history = [("hi", "hello")]
    result = _build_augmented_prompt("test", facts, episodes, history=history)
    assert result.startswith(_SYSTEM_PROMPT)


# ---------- FIX 3: sleep/wake cycle tests ----------


def test_sleep_triggered_after_inactivity():
    """Sleep should trigger when _last_activity is old enough."""
    import component10
    fresh = tempfile.mkdtemp(prefix="sleep_inact_test_")
    try:
        start(data_dir=fresh)
        # Simulate inactivity by backdating _last_activity
        component10._last_activity = time.time() - _SLEEP_INACTIVITY - 10
        # Manually run sleep cycle (don't wait for monitor thread)
        _run_sleep_cycle()

        # After sleep completes, state should be awake again
        status = get_sleep_status()
        assert status["state"] == "awake"
        assert status["last_sleep_duration"] is not None
        assert status["last_sleep_duration"] > 0
        stop()
    finally:
        shutil.rmtree(fresh)
        start(data_dir=_tmpdir)


def test_chat_waits_during_sleep():
    """chat() during sleep should block, then succeed after sleep completes."""
    import component10
    fresh = tempfile.mkdtemp(prefix="sleep_chat_test_")
    try:
        start(data_dir=fresh)

        # Start sleep in a background thread
        sleep_thread = threading.Thread(target=_run_sleep_cycle, daemon=True)
        sleep_thread.start()

        # Give sleep a moment to start
        time.sleep(0.2)

        # chat() should block until sleep finishes, then return
        response = chat("Hello after sleep")
        assert isinstance(response, str)
        assert len(response) > 0

        sleep_thread.join(timeout=30)
        stop()
    finally:
        shutil.rmtree(fresh)
        start(data_dir=_tmpdir)


def test_sleep_status_during_and_after():
    """get_sleep_status() should reflect sleeping state correctly."""
    import component10
    fresh = tempfile.mkdtemp(prefix="sleep_status_test_")
    try:
        start(data_dir=fresh)

        # Before sleep
        status = get_sleep_status()
        assert status["state"] == "awake"
        assert status["sleeping_since"] is None

        # Run sleep cycle
        _run_sleep_cycle()

        # After sleep
        status = get_sleep_status()
        assert status["state"] == "awake"
        assert status["last_sleep_duration"] is not None
        stop()
    finally:
        shutil.rmtree(fresh)
        start(data_dir=_tmpdir)


# ---------- FIX 3: date injection ----------


def test_date_in_prompt():
    """Augmented prompt should include today's date after system prompt."""
    import datetime
    today = datetime.date.today().strftime("%B %d %Y")
    result = _build_augmented_prompt("Hello", [], [])
    assert today in result
    # Date must come after system prompt
    sys_end = result.index(_SYSTEM_PROMPT) + len(_SYSTEM_PROMPT)
    date_pos = result.index(today)
    assert date_pos > sys_end


# ---------- FIX 5: meta-question detection ----------


def test_meta_question_detected():
    """Meta-questions about memory should be detected."""
    assert _is_meta_question("What do you know about me?")
    assert _is_meta_question("what have I told you so far")
    assert _is_meta_question("Do you know who I am?")
    assert not _is_meta_question("What is the weather today?")
    assert not _is_meta_question("Tell me a joke")


def test_meta_question_lists_facts():
    """Meta-question handler should list all stored facts."""
    fresh = tempfile.mkdtemp(prefix="meta_test_")
    try:
        start(data_dir=fresh)
        component9.store_fact("Ani uses Python", "conversation", 0.85)
        component9.store_fact("Ani likes coffee", "conversation", 0.85)

        result = _handle_meta_question("what do you know about me")
        assert "Ani uses Python" in result
        assert "Ani likes coffee" in result
        stop()
    finally:
        shutil.rmtree(fresh)
        start(data_dir=_tmpdir)


# ---------- FIX 6: consolidation-only ----------


def test_consolidation_only():
    """run_consolidation_only() should return a report dict."""
    result = run_consolidation_only()
    assert "episodes_processed" in result
    assert "episodes_pruned" in result
    assert "duration_seconds" in result


# ---------- FIX 7: ablation ----------


def test_ablation_defaults():
    """Ablation flags should default to False."""
    status = get_ablation_status()
    assert status["no_store"] is False
    assert status["no_lora"] is False


def test_ablation_set_and_reset():
    """Ablation flags should be settable."""
    set_ablation(no_store=True)
    assert get_ablation_status()["no_store"] is True
    assert get_ablation_status()["no_lora"] is False

    set_ablation(no_store=False, no_lora=True)
    assert get_ablation_status()["no_store"] is False
    assert get_ablation_status()["no_lora"] is True

    # Reset
    set_ablation(no_store=False, no_lora=False)
