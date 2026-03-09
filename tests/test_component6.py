"""Tests for Component 6: Consolidation Loop (Sleep)"""

import os
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import component4
import component5  # register scoring hook
import component6
from component6 import (
    ConsolidationReport,
    run_consolidation_cycle,
    _prune_episodes,
    _select_episodes,
    should_consolidate,
    notify_new_episode,
)

_tmpdir = None


def setup_module():
    global _tmpdir
    _tmpdir = tempfile.mkdtemp(prefix="consolidation_test_")
    component4.init(data_dir=_tmpdir)
    # Ensure component3 buffers exist
    from component3 import _init_buffers, _model_lock, BUFFER_B
    from pathlib import Path
    Path(BUFFER_B).parent.mkdir(parents=True, exist_ok=True)
    with _model_lock:
        from component3 import _save_adapter_to
        _save_adapter_to(BUFFER_B)


def teardown_module():
    if _tmpdir and os.path.exists(_tmpdir):
        shutil.rmtree(_tmpdir)


# ---------- tests ----------


def test_consolidation_reduces_loss():
    """Training on correction episodes should reduce loss."""
    t = time.time()
    for i in range(10):
        component4.store_episode(
            f"What is {i} + {i}?",
            "wrong answer",
            f" The answer is {i + i}",
            t + i,
        )

    report = run_consolidation_cycle()
    assert isinstance(report, ConsolidationReport)
    assert report.episodes_processed > 0
    assert report.adapter_loss_after <= report.adapter_loss_before


def test_pruning_keeps_corrections():
    """Old, low-score episodes WITH corrections must never be pruned."""
    old_time = time.time() - (30 * 24 * 60 * 60)  # 30 days ago

    # Store an old episode with a correction and low score
    eid = component4.store_episode(
        "pruning test keep", "wrong", "important fix", old_time
    )
    ep = component4.get_episode_by_id(eid)
    ep.importance_score = 0.1  # very low score
    component4.update_episode(ep)

    _prune_episodes()

    # It should still be there (has correction)
    assert component4.get_episode_by_id(eid) is not None


def test_pruning_removes_old_low_score():
    """Old, low-score episodes WITHOUT corrections should be pruned."""
    old_time = time.time() - (30 * 24 * 60 * 60)  # 30 days ago

    eid = component4.store_episode(
        "pruning test remove", "response", None, old_time
    )
    ep = component4.get_episode_by_id(eid)
    ep.importance_score = 0.1  # below threshold
    component4.update_episode(ep)

    pruned = _prune_episodes()
    assert pruned >= 1
    assert component4.get_episode_by_id(eid) is None


def test_selection_picks_high_importance():
    """Episodes with importance_score > 0.5 should be selected."""
    fresh = tempfile.mkdtemp(prefix="sel_test_")
    try:
        component4.init(data_dir=fresh)
        t = time.time()

        # High importance (has correction → scored 0.6)
        eid_high = component4.store_episode("select high", "resp", "fix", t)
        # Low importance (no correction → scored 0.3)
        eid_low = component4.store_episode("select low", "resp", None, t + 1)

        selected = _select_episodes()
        selected_ids = {ep.id for ep in selected}

        assert eid_high in selected_ids, "High-importance episode should be selected"
    finally:
        shutil.rmtree(fresh)
        component4.init(data_dir=_tmpdir)


def test_trigger_episode_count():
    """Should trigger after TRIGGER_FIRST_COUNT (30) for first consolidation."""
    component6._episodes_since_last = 0
    component6._first_consolidation_done = False
    assert not should_consolidate()

    # Simulate 30 new episodes — should trigger first consolidation
    for _ in range(30):
        notify_new_episode()
    assert should_consolidate()

    # After first consolidation, threshold goes back to 100
    component6._episodes_since_last = 0
    component6._first_consolidation_done = True
    assert not should_consolidate()
    for _ in range(100):
        notify_new_episode()
    assert should_consolidate()


def test_report_has_all_fields():
    """ConsolidationReport should have all required fields."""
    fresh = tempfile.mkdtemp(prefix="report_test_")
    try:
        component4.init(data_dir=fresh)
        component4.store_episode("report test", "resp", "fix", time.time())

        report = run_consolidation_cycle()
        assert hasattr(report, "episodes_processed")
        assert hasattr(report, "episodes_pruned")
        assert hasattr(report, "adapter_loss_before")
        assert hasattr(report, "adapter_loss_after")
        assert hasattr(report, "duration_seconds")
        assert report.duration_seconds > 0
    finally:
        shutil.rmtree(fresh)
        component4.init(data_dir=_tmpdir)
