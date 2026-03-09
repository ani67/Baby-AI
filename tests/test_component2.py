"""Tests for Component 2: LoRA Adapter Training"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from component2 import train_one_step, hash_base_model_weights


def test_adapter_trains():
    """A single training step should reduce loss and change adapter weights."""
    with tempfile.TemporaryDirectory() as tmpdir:
        adapter_path = os.path.join(tmpdir, "adapter")
        example = {
            "prompt": "The capital of Australia is",
            "completion": " Canberra",
        }
        loss_before, loss_after, changed = train_one_step(
            adapter_path, example
        )
        assert changed is True, "Adapter weights should have changed"
        assert loss_after < loss_before, (
            f"Loss should decrease: {loss_before:.4f} → {loss_after:.4f}"
        )


def test_base_model_unchanged():
    """Base model weights must not change during a training step."""
    hash_before = hash_base_model_weights()
    with tempfile.TemporaryDirectory() as tmpdir:
        adapter_path = os.path.join(tmpdir, "adapter")
        train_one_step(
            adapter_path,
            {"prompt": "test", "completion": " test"},
        )
    hash_after = hash_base_model_weights()
    assert hash_before == hash_after, "Base model weights must not change"


def test_adapter_saves_to_disk():
    """Adapter files should be written to the specified path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        adapter_path = os.path.join(tmpdir, "adapter")
        train_one_step(
            adapter_path,
            {"prompt": "hello", "completion": " world"},
        )
        assert os.path.exists(os.path.join(adapter_path, "adapters.safetensors"))
        assert os.path.exists(os.path.join(adapter_path, "adapter_config.json"))
