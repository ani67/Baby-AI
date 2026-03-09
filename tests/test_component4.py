"""Tests for Component 4: Episodic Store"""

import importlib
import json
import os
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import component4


# ---------- fixtures ----------

_tmpdir = None


def setup_module():
    """Each test run uses a fresh temp directory."""
    global _tmpdir
    _tmpdir = tempfile.mkdtemp(prefix="episodic_test_")
    component4.init(data_dir=_tmpdir)


def teardown_module():
    global _tmpdir
    if _tmpdir and os.path.exists(_tmpdir):
        shutil.rmtree(_tmpdir)


# ---------- tests ----------


def test_store_and_retrieve():
    """Storing an episode should make it retrievable."""
    eid = component4.store_episode("hello", "world", None, time.time())
    assert len(eid) == 16
    episodes = component4.get_recent_episodes(10)
    assert any(e.id == eid for e in episodes)


def test_episode_fields():
    """Episode should have correct defaults."""
    eid = component4.store_episode(
        "prompt", "response", "correction", time.time()
    )
    ep = component4.get_episode_by_id(eid)
    assert ep is not None
    assert ep.prompt == "prompt"
    assert ep.response == "response"
    assert ep.correction == "correction"
    # importance_score is 1.0 by default, or auto-scored by Component 5 if loaded
    assert ep.importance_score > 0
    assert ep.times_referenced == 0


def test_recent_episodes_ordering():
    """get_recent_episodes should return newest first."""
    t = time.time() + 10000  # future timestamps to guarantee these are most recent
    component4.store_episode("old_order_test", "old", None, t)
    component4.store_episode("new_order_test", "new", None, t + 100)
    episodes = component4.get_recent_episodes(2)
    assert episodes[0].prompt == "new_order_test"
    assert episodes[1].prompt == "old_order_test"


def test_episode_survives_restart():
    """Episodes must persist across a simulated restart (re-init)."""
    eid = component4.store_episode(
        "restart test", "test response", "test correction", time.time()
    )

    # Simulate restart: re-init from the same data directory
    component4.init(data_dir=_tmpdir)

    episodes = component4.get_recent_episodes(100)
    assert any(e.prompt == "restart test" for e in episodes)


def test_similarity_retrieval():
    """get_similar_episodes should find semantically related prompts."""
    # Fresh data dir so only our two episodes are present
    fresh_dir = tempfile.mkdtemp(prefix="episodic_sim_")
    try:
        component4.init(data_dir=fresh_dir)
        component4.store_episode("capital of France", "Paris", None, time.time())
        component4.store_episode("largest planet", "Jupiter", None, time.time())

        results = component4.get_similar_episodes(
            "what is the capital city of France?", n=1
        )
        assert len(results) >= 1
        assert "France" in results[0].prompt
    finally:
        shutil.rmtree(fresh_dir)
        # Restore original test dir
        component4.init(data_dir=_tmpdir)


def test_similarity_increments_times_referenced():
    """Retrieving by similarity should increment times_referenced."""
    fresh_dir = tempfile.mkdtemp(prefix="episodic_ref_")
    try:
        component4.init(data_dir=fresh_dir)
        eid = component4.store_episode(
            "unique referencing query", "answer", None, time.time()
        )
        ep_before = component4.get_episode_by_id(eid)
        assert ep_before.times_referenced == 0

        component4.get_similar_episodes("unique referencing query", n=1)

        ep_after = component4.get_episode_by_id(eid)
        assert ep_after.times_referenced >= 1
    finally:
        shutil.rmtree(fresh_dir)
        component4.init(data_dir=_tmpdir)


def test_json_file_exists():
    """episodes.json should be written to disk after store_episode."""
    fresh_dir = tempfile.mkdtemp(prefix="episodic_json_")
    try:
        component4.init(data_dir=fresh_dir)
        component4.store_episode("disk test", "response", None, time.time())
        episodes_file = os.path.join(fresh_dir, "episodes.json")
        assert os.path.exists(episodes_file)
        with open(episodes_file) as f:
            data = json.load(f)
        assert any(ep["prompt"] == "disk test" for ep in data)
    finally:
        shutil.rmtree(fresh_dir)
        component4.init(data_dir=_tmpdir)
