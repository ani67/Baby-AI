"""Tests for Component 3: Double-Buffer (Simultaneous Train + Inference)"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import component3


# ---------- helpers ----------


def _setup():
    """Start the double-buffer system (idempotent per test)."""
    component3.stop()          # clean up any previous run
    component3.start()


def _teardown():
    component3.stop()


# ---------- tests ----------


def test_inference_never_blocks():
    """query() must respond in <3 s even while corrections are queued."""
    _setup()
    try:
        # Submit 50 corrections rapidly (non-blocking)
        for i in range(50):
            component3.submit_correction(f"prompt {i}", f" completion {i}")

        # Inference should still be fast
        t = time.time()
        response = component3.query("What is 2 + 2?")
        elapsed = time.time() - t

        assert len(response) > 0, "Response should not be empty"
        assert elapsed < 3.0, (
            f"query() took {elapsed:.2f}s — must be <3 s even during heavy training"
        )
    finally:
        _teardown()


def test_submit_correction_is_nonblocking():
    """submit_correction() must return immediately (queue, not train)."""
    _setup()
    try:
        t = time.time()
        for i in range(20):
            component3.submit_correction(f"prompt {i}", f" completion {i}")
        elapsed = time.time() - t

        assert elapsed < 0.1, (
            f"Submitting 20 corrections took {elapsed:.3f}s — should be instant"
        )
    finally:
        _teardown()


def test_buffers_exist_on_disk():
    """Both buffer directories should be created on start()."""
    _setup()
    try:
        assert component3.BUFFER_A.exists(), "buffer_a/ should exist"
        assert component3.BUFFER_B.exists(), "buffer_b/ should exist"
        assert (component3.BUFFER_A / "adapters.safetensors").exists()
        assert (component3.BUFFER_B / "adapters.safetensors").exists()
    finally:
        _teardown()


def test_training_processes_queue():
    """Background thread should drain the correction queue over time."""
    _setup()
    try:
        for i in range(5):
            component3.submit_correction(f"prompt {i}", f" completion {i}")

        # Wait for the background thread to process (~0.23 s per step)
        time.sleep(5)

        remaining = component3.get_queue_size()
        assert remaining == 0, (
            f"Queue should be empty after processing, but has {remaining} items"
        )
        assert component3.get_training_step_count() >= 5, (
            f"Expected >=5 training steps, got {component3.get_training_step_count()}"
        )
    finally:
        _teardown()


def test_adapter_actually_updates():
    """After training steps, the adapter weights should change."""
    _setup()
    try:
        import mlx.core as mx
        from mlx.utils import tree_flatten

        # Snapshot weights before any training (under the lock)
        with component3._model_lock:
            weights_before = {
                name: mx.array(param)
                for name, param in tree_flatten(
                    component3._model.trainable_parameters()
                )
            }

        # Submit corrections and wait for processing
        for _ in range(5):
            component3.submit_correction(
                "The capital of Australia is", " Canberra"
            )
        time.sleep(5)

        # Stop training so we can safely inspect weights
        component3.stop()

        # Now safe to read weights (no training thread running)
        changed = False
        for name, param in tree_flatten(
            component3._model.trainable_parameters()
        ):
            if not mx.array_equal(param, weights_before[name]):
                changed = True
                break

        assert changed, "Adapter weights should have changed after training"
    finally:
        _teardown()
