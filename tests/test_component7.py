"""Tests for Component 7: Internal State Monitor (Proto-Self)"""

import math
import os
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import component4
import component5  # register scoring hook
import component7
from component7 import (
    InternalState,
    get_current_state,
    notify_correction,
    notify_inference,
    reset,
)

_tmpdir = None


def setup_module():
    global _tmpdir
    _tmpdir = tempfile.mkdtemp(prefix="state_monitor_test_")
    component4.init(data_dir=_tmpdir)


def teardown_module():
    if _tmpdir and os.path.exists(_tmpdir):
        shutil.rmtree(_tmpdir)


def setup_function():
    """Reset state between tests."""
    reset()


# ---------- tests ----------


def test_get_current_state_returns_dataclass():
    """get_current_state should return an InternalState."""
    state = get_current_state()
    assert isinstance(state, InternalState)
    assert 0.0 <= state.uncertainty <= 1.0
    assert 0.0 <= state.performance <= 1.0
    assert 0.0 <= state.novelty <= 1.0
    assert 0.0 <= state.coherence <= 1.0


def test_uncertainty_low_on_confident_tokens():
    """Low entropy tokens should give low uncertainty."""
    # Simulate 20 inferences with very low entropy
    for _ in range(20):
        notify_inference("hello", "world", [0.01, 0.01, 0.01])

    state = get_current_state()
    assert state.uncertainty < 0.3, f"Expected low uncertainty, got {state.uncertainty}"


def test_uncertainty_high_on_uncertain_tokens():
    """High entropy tokens should give high uncertainty."""
    max_entropy = math.log(128256)
    # Simulate inferences with high entropy (close to max)
    for _ in range(20):
        notify_inference("random gibberish", "stuff", [max_entropy * 0.8] * 5)

    state = get_current_state()
    assert state.uncertainty > 0.5, f"Expected high uncertainty, got {state.uncertainty}"


def test_performance_starts_perfect():
    """With no corrections, performance should be 1.0."""
    for _ in range(10):
        notify_inference("prompt", "response", [0.1])

    state = get_current_state()
    assert state.performance == 1.0


def test_performance_drops_after_corrections():
    """Performance should decrease when corrections are submitted."""
    # 10 inferences, then correct 5 of them
    for i in range(10):
        notify_inference(f"prompt {i}", f"response {i}", [0.1])
        if i >= 5:
            notify_correction(f"prompt {i}", f"correction {i}")

    state = get_current_state()
    assert state.performance < 1.0, f"Expected perf < 1.0, got {state.performance}"


def test_uncertainty_flag_set():
    """uncertainty_flag should be True when uncertainty > 0.7."""
    max_entropy = math.log(128256)
    # Push uncertainty high
    for _ in range(20):
        notify_inference("gibberish", "stuff", [max_entropy * 0.9] * 5)

    assert component7.uncertainty_flag is True


def test_uncertainty_flag_unset():
    """uncertainty_flag should be False when uncertainty <= 0.7."""
    for _ in range(20):
        notify_inference("hello", "world", [0.01])

    assert component7.uncertainty_flag is False


def test_importance_boost_when_uncertain():
    """When uncertainty is high, component4._score_fn should boost scores."""
    max_entropy = math.log(128256)
    # Push uncertainty high to trigger boost
    for _ in range(20):
        notify_inference("gibberish", "stuff", [max_entropy * 0.9] * 5)

    # Now store an episode — its score should be boosted
    eid = component4.store_episode("boost test", "resp", "fix", time.time())
    ep = component4.get_episode_by_id(eid)

    # Normal score for correction = 0.6. With 1.5x boost = 0.9
    assert ep.importance_score > 0.6, (
        f"Expected boosted score > 0.6, got {ep.importance_score}"
    )


def test_state_logged_to_disk():
    """State snapshots should be appended to state_log.jsonl."""
    log_file = component7._STATE_LOG

    for i in range(10):
        notify_inference(f"prompt {i}", f"response {i}", [0.1])

    assert log_file.exists(), "state_log.jsonl should exist"
    with open(log_file) as f:
        lines = f.readlines()
    assert len(lines) >= 1, "Should have at least one log entry"


def test_novelty_returns_valid_range():
    """Novelty should be in [0, 1]."""
    # Store some episodes so there's a centroid to compare against
    component4.store_episode("familiar topic", "response", None, time.time())

    for _ in range(10):
        notify_inference("familiar topic", "response", [0.1])

    state = get_current_state()
    assert 0.0 <= state.novelty <= 1.0


def test_coherence_perfect_on_identical():
    """Identical prompt/response pairs should give high coherence."""
    for _ in range(10):
        notify_inference("What is 2+2?", "4", [0.1])

    state = get_current_state()
    assert state.coherence >= 0.8, f"Expected high coherence, got {state.coherence}"
