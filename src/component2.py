"""Component 2: LoRA Adapter Training

Applies a LoRA adapter (rank=8, alpha=16) to q_proj and v_proj layers
of the base model from Component 1. Provides train_one_step() for
single-example training with adapter persistence.

LoRA B matrix is zero-initialized by default in mlx_lm, so the adapter
starts as identity (no behaviour change until trained).
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from mlx_lm.tuner.utils import linear_to_lora_layers

# Import the base model and tokenizer from Component 1
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from component1 import _model, _tokenizer

# LoRA configuration
LORA_CONFIG = {
    "rank": 8,
    "dropout": 0.0,
    "scale": 16.0,       # alpha — spec says 16 (2× rank)
    "keys": ["self_attn.q_proj", "self_attn.v_proj"],
}
NUM_LORA_LAYERS = -1  # apply to all transformer layers

# Apply LoRA layers to the model (in-place, wraps base weights)
linear_to_lora_layers(_model, NUM_LORA_LAYERS, LORA_CONFIG)

# Freeze everything, then unfreeze only lora_a and lora_b
_model.freeze()
_model.unfreeze(keys="lora_a", strict=False)
_model.unfreeze(keys="lora_b", strict=False)


def _build_training_tokens(prompt: str, completion: str) -> mx.array:
    """Tokenize prompt+completion into a single sequence for training."""
    full_text = prompt + completion
    tokens = _tokenizer.encode(full_text)
    return mx.array(tokens)


def _compute_loss(model, tokens: mx.array) -> tuple:
    """Compute cross-entropy loss on the token sequence."""
    inputs = tokens[None, :-1]   # (1, seq_len-1)
    targets = tokens[None, 1:]   # (1, seq_len-1)

    logits = model(inputs)
    ce = nn.losses.cross_entropy(logits, targets)
    loss = ce.mean()
    return loss, targets.shape[1]


def hash_base_model_weights() -> str:
    """SHA256 hash of base (non-LoRA) model weights for integrity checks."""
    h = hashlib.sha256()
    for name, param in tree_flatten(_model.parameters()):
        if "lora_a" in name or "lora_b" in name:
            continue
        h.update(np.array(param, copy=False).tobytes())
    return h.hexdigest()


def _save_adapter(adapter_path: str):
    """Save trainable LoRA weights and config to disk."""
    path = Path(adapter_path)
    path.mkdir(parents=True, exist_ok=True)

    adapter_weights = dict(tree_flatten(_model.trainable_parameters()))
    mx.save_safetensors(str(path / "adapters.safetensors"), adapter_weights)

    config = {
        "lora_parameters": LORA_CONFIG,
        "num_layers": NUM_LORA_LAYERS,
        "fine_tune_type": "lora",
    }
    with open(path / "adapter_config.json", "w") as f:
        json.dump(config, f, indent=2)


def train_one_step(
    adapter_path: str,
    training_example: dict,
    learning_rate: float = 1e-4,
) -> tuple[float, float, bool]:
    """Train one step on a single example.

    Args:
        adapter_path: Directory to save adapter weights.
        training_example: {"prompt": str, "completion": str}
        learning_rate: Learning rate for this step.

    Returns:
        (loss_before, loss_after, adapter_changed)
    """
    prompt = training_example["prompt"]
    completion = training_example["completion"]
    tokens = _build_training_tokens(prompt, completion)

    # Snapshot adapter weights before training
    weights_before = {
        name: np.array(param, copy=True)
        for name, param in tree_flatten(_model.trainable_parameters())
    }

    # Compute loss before the update
    loss_before_val, _ = _compute_loss(_model, tokens)
    mx.eval(loss_before_val)
    loss_before = loss_before_val.item()

    # Forward + backward + optimizer step
    optimizer = optim.Adam(learning_rate=learning_rate)
    loss_and_grad = nn.value_and_grad(_model, _compute_loss)
    (loss_after_val, _), grads = loss_and_grad(_model, tokens)
    optimizer.update(_model, grads)
    mx.eval(_model.parameters(), optimizer.state)

    loss_after = loss_after_val.item()

    # Check if adapter weights actually changed
    adapter_changed = False
    for name, param in tree_flatten(_model.trainable_parameters()):
        if not np.array_equal(np.array(param, copy=False), weights_before[name]):
            adapter_changed = True
            break

    # Save adapter to disk after every update
    _save_adapter(adapter_path)

    return loss_before, loss_after, adapter_changed
