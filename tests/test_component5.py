"""Tests for Component 5: Importance Scorer (Amygdala)"""

import os
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import component4
import component5
from component4 import Episode
from component5 import learning_rate_for_episode, score_episode

_tmpdir = None


def setup_module():
    global _tmpdir
    _tmpdir = tempfile.mkdtemp(prefix="scorer_test_")
    component4.init(data_dir=_tmpdir)


def teardown_module():
    if _tmpdir and os.path.exists(_tmpdir):
        shutil.rmtree(_tmpdir)


# ---------- scoring tests ----------


def test_correction_scores_higher():
    """Episode with a correction should score higher than one without."""
    t = time.time()
    ep_with = Episode("a", "prompt", "resp", "actual answer", t)
    ep_without = Episode("b", "prompt", "resp", None, t)
    assert score_episode(ep_with) > score_episode(ep_without)


def test_base_score_no_correction():
    """Episode without correction should get base score of 0.3."""
    ep = Episode("c", "prompt", "resp", None, time.time())
    assert score_episode(ep) == 0.3


def test_correction_adds_0_3():
    """Correction should add 0.3 to base (0.3 + 0.3 = 0.6)."""
    ep = Episode("d", "unique_prompt_xyz", "resp", "fix", time.time())
    # No prior correction for this prompt, times_referenced=0, correction < 50 chars
    assert score_episode(ep) == 0.6


def test_recurring_error_scores_higher():
    """Same prompt corrected before should score +0.2 more."""
    t = time.time()
    # First: store a prior correction for this prompt
    component4.store_episode("capital of Australia", "Sydney", "Canberra", t - 10)

    # Now score a NEW episode with the same prompt and a correction
    ep_recurring = Episode(
        "e", "capital of Australia", "Sydney", "Canberra", t,
        times_referenced=0,
    )
    # Should get: 0.3 (base) + 0.3 (correction) + 0.2 (recurring) = 0.8
    assert score_episode(ep_recurring) == 0.8


def test_times_referenced_bonus():
    """times_referenced > 3 should add 0.1."""
    ep = Episode("f", "unique_ref_prompt", "resp", None, time.time(),
                 times_referenced=5)
    # 0.3 (base) + 0.1 (referenced) = 0.4
    assert score_episode(ep) == 0.4


def test_long_correction_bonus():
    """Correction > 50 chars should add 0.1."""
    long_correction = "x" * 51
    ep = Episode("g", "unique_long_prompt", "resp", long_correction, time.time())
    # 0.3 (base) + 0.3 (correction) + 0.1 (long correction) = 0.7
    assert score_episode(ep) == 0.7


def test_recency_decay():
    """Episode older than 7 days should lose 0.1."""
    old_timestamp = time.time() - (8 * 24 * 60 * 60)  # 8 days ago
    ep = Episode("h", "unique_old_prompt", "resp", None, old_timestamp)
    # 0.3 (base) - 0.1 (old) = 0.2
    assert score_episode(ep) == 0.2


def test_clamp_minimum():
    """Score should never go below 0.1."""
    very_old = time.time() - (365 * 24 * 60 * 60)  # 1 year ago
    ep = Episode("i", "unique_clamp_prompt", "resp", None, very_old)
    assert score_episode(ep) >= 0.1


def test_clamp_maximum():
    """Score should never exceed 1.0 even with all bonuses."""
    # Store a prior correction for recurring error
    component4.store_episode("max_score_prompt", "wrong", "right " * 20, time.time() - 5)

    ep = Episode("j", "max_score_prompt", "wrong", "right " * 20, time.time(),
                 times_referenced=10)
    # 0.3 + 0.3 + 0.2 + 0.1 + 0.1 = 1.0, should not exceed
    assert score_episode(ep) == 1.0


# ---------- learning rate tests ----------


def test_learning_rate_low_score():
    """Score 0.1 should map to ~1e-5."""
    ep = Episode("k", "p", "r", None, time.time(), importance_score=0.1)
    lr = learning_rate_for_episode(ep)
    assert abs(lr - 1e-5) < 1e-6


def test_learning_rate_high_score():
    """Score 1.0 should map to ~5e-4."""
    ep = Episode("l", "p", "r", None, time.time(), importance_score=1.0)
    lr = learning_rate_for_episode(ep)
    assert abs(lr - 5e-4) < 1e-6


def test_learning_rate_mid_score():
    """Score 0.5 should give a learning rate between 1e-5 and 5e-4."""
    ep = Episode("m", "p", "r", None, time.time(), importance_score=0.5)
    lr = learning_rate_for_episode(ep)
    assert 1e-5 < lr < 5e-4


# ---------- integration: auto-scoring in store_episode ----------


def test_store_episode_auto_scores():
    """store_episode should automatically set importance_score via the hook."""
    eid = component4.store_episode(
        "auto_score_prompt", "resp", "a correction", time.time()
    )
    ep = component4.get_episode_by_id(eid)
    # Has a correction → should be scored > base 0.3
    assert ep.importance_score > 0.3
    # Should be 0.6 (base 0.3 + correction 0.3)
    assert ep.importance_score == 0.6
