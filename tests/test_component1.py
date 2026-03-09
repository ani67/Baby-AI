"""Tests for Component 1: Base Inference"""

import time
import sys
import os

# Ensure src/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from component1 import run_inference


def test_basic_inference():
    """Model returns a non-empty response with reasonable speed."""
    response, tps, count = run_inference("What is 2 + 2?")
    assert len(response) > 0, "Response should not be empty"
    assert tps > 5.0, f"Expected >5 tok/s on M1, got {tps:.1f}"
    assert count > 0, "Token count should be positive"


def test_model_loads_once():
    """Both calls should be fast — model was loaded once at import, not per call."""
    t1 = time.time()
    run_inference("hello", max_tokens=10)
    t1 = time.time() - t1

    t2 = time.time()
    run_inference("hello", max_tokens=10)
    t2 = time.time() - t2

    # Both calls should complete quickly since model is already loaded.
    # If it reloaded per call, each would take several seconds.
    assert t1 < 5.0, f"First call took {t1:.2f}s — model may be reloading per call"
    assert t2 < 5.0, f"Second call took {t2:.2f}s — model may be reloading per call"
    # Second call should be roughly the same speed (no reload penalty)
    assert abs(t1 - t2) < t1, "Calls should take similar time (no reload)"
