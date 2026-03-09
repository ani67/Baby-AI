"""Tests for Component 9: Knowledge Store"""

import importlib
import os
import shutil
import sys
import tempfile
import time
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import component4
import component9
from component9 import (
    Fact,
    store_fact,
    retrieve_facts,
    get_fact_count,
    _augment_prompt,
)

_tmpdir = None


def setup_module():
    global _tmpdir
    _tmpdir = tempfile.mkdtemp(prefix="knowledge_store_test_")
    component4.init(data_dir=_tmpdir)
    component9.init(data_dir=_tmpdir)


def teardown_module():
    if _tmpdir and os.path.exists(_tmpdir):
        shutil.rmtree(_tmpdir)


# ---------- tests ----------


def test_store_and_retrieve():
    """store_fact should store, retrieve_facts should find it."""
    fid = store_fact("The capital of Australia is Canberra", "test", 0.9)
    assert fid is not None

    facts = retrieve_facts("capital of Australia")
    assert len(facts) >= 1
    assert any("Canberra" in f.fact for f in facts)


def test_fact_fields():
    """Fact should have all required fields with correct types."""
    fid = store_fact("Water boils at 100 degrees Celsius", "test", 0.85)
    fact = component9._facts[fid]

    assert isinstance(fact, Fact)
    assert fact.id == fid
    assert fact.fact == "Water boils at 100 degrees Celsius"
    assert fact.source == "test"
    assert fact.confidence == 0.85
    assert fact.times_retrieved == 0
    assert fact.timestamp > 0


def test_get_fact_count():
    """get_fact_count should return the number of stored facts."""
    initial = get_fact_count()
    store_fact("Test count fact", "test", 0.5)
    assert get_fact_count() == initial + 1


def test_retrieve_increments_times_referenced():
    """retrieve_facts should increment times_retrieved on returned facts."""
    fid = store_fact("Unique fact for retrieval test XYZ123", "test", 0.9)
    assert component9._facts[fid].times_retrieved == 0

    retrieve_facts("Unique fact XYZ123")
    assert component9._facts[fid].times_retrieved >= 1


def test_fact_survives_restart():
    """Facts should survive a simulated restart (re-init from disk)."""
    fresh = tempfile.mkdtemp(prefix="restart_test_")
    try:
        component9.init(data_dir=fresh)
        store_fact("Persistent fact test", "test", 0.8)

        # Re-init from the same directory (simulates restart)
        component9.init(data_dir=fresh)
        facts = retrieve_facts("Persistent fact")
        assert any("Persistent fact test" in f.fact for f in facts)
    finally:
        shutil.rmtree(fresh)
        component9.init(data_dir=_tmpdir)


def test_json_file_exists():
    """facts.json should exist on disk after storing a fact."""
    fresh = tempfile.mkdtemp(prefix="json_test_")
    try:
        component9.init(data_dir=fresh)
        store_fact("JSON test fact", "test", 0.7)

        from pathlib import Path
        assert (Path(fresh) / "facts.json").exists()
    finally:
        shutil.rmtree(fresh)
        component9.init(data_dir=_tmpdir)


def test_augment_prompt_prepends_high_confidence():
    """Facts with confidence > 0.7 should be prepended to prompts."""
    fresh = tempfile.mkdtemp(prefix="augment_test_")
    try:
        component9.init(data_dir=fresh)
        store_fact("The capital of Australia is Canberra", "test", 0.9)

        augmented = _augment_prompt("What is the capital of Australia?")
        assert "Relevant facts:" in augmented
        assert "Canberra" in augmented
        assert "Now answer:" in augmented
    finally:
        shutil.rmtree(fresh)
        component9.init(data_dir=_tmpdir)


def test_augment_prompt_skips_low_confidence():
    """Facts with confidence <= 0.7 should NOT be prepended."""
    fresh = tempfile.mkdtemp(prefix="augment_low_test_")
    try:
        component9.init(data_dir=fresh)
        store_fact("Low confidence fact about cats", "test", 0.3)

        augmented = _augment_prompt("Tell me about cats")
        # Should return the original prompt unchanged
        assert augmented == "Tell me about cats"
    finally:
        shutil.rmtree(fresh)
        component9.init(data_dir=fresh)


def test_augment_prompt_no_facts():
    """With no facts stored, prompt should pass through unchanged."""
    fresh = tempfile.mkdtemp(prefix="augment_empty_test_")
    try:
        component9.init(data_dir=fresh)
        augmented = _augment_prompt("Hello world")
        assert augmented == "Hello world"
    finally:
        shutil.rmtree(fresh)
        component9.init(data_dir=_tmpdir)


def test_query_hook_installed():
    """component3.query should be wrapped by component9."""
    import component3
    assert component3.query is component9._augmented_query


def test_submit_correction_hook_installed():
    """component3.submit_correction should be wrapped by component9."""
    import component3
    assert component3.submit_correction is component9._augmented_submit_correction


def test_submit_correction_stores_fact():
    """submit_correction should auto-store the correction as a fact."""
    fresh = tempfile.mkdtemp(prefix="correction_fact_test_")
    try:
        component9.init(data_dir=fresh)
        initial_count = get_fact_count()

        # Call the augmented submit_correction (which queues training + stores fact)
        component9._augmented_submit_correction(
            "What is 2+2?", "The answer is 4"
        )

        assert get_fact_count() == initial_count + 1
        # The stored fact should be the correction text
        facts = retrieve_facts("answer is 4")
        assert any("The answer is 4" in f.fact for f in facts)
        # Should have user_correction source and 0.75 confidence
        correction_facts = [f for f in facts if f.source == "user_correction"]
        assert len(correction_facts) >= 1
        assert correction_facts[0].confidence == 0.75
    finally:
        shutil.rmtree(fresh)
        component9.init(data_dir=_tmpdir)


def test_separate_collection_from_episodes():
    """Knowledge store should use a different ChromaDB collection than episodes."""
    fresh = tempfile.mkdtemp(prefix="collection_test_")
    try:
        component4.init(data_dir=fresh)
        component9.init(data_dir=fresh)

        # Store an episode and a fact
        component4.store_episode("episode prompt", "response", None, time.time())
        store_fact("Fact text here", "test", 0.9)

        # Episode collection should have 1 entry
        assert component4._collection.count() == 1
        # Knowledge collection should have 1 entry
        assert component9._collection.count() == 1

        # They should be independent
        assert component4._collection.name == "episodes"
        assert component9._collection.name == "knowledge_store"
    finally:
        shutil.rmtree(fresh)
        component4.init(data_dir=_tmpdir)
        component9.init(data_dir=_tmpdir)
