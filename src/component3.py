"""Component 3: Double-Buffer (Simultaneous Train + Inference)

Two adapter directories on disk: buffer_a (inference) and buffer_b (training).
Inference and training share one model in memory, protected by a lock so
weight updates never corrupt mid-generation. All MLX operations are serialized
through _model_lock because Metal is not thread-safe.

Training runs in a background thread fed by a queue. Every swap_interval
training steps, buffer_b is copied to buffer_a (atomic rename) and the
training step counter resets.
"""

import json
import queue
import shutil
import threading
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx_lm
from mlx.utils import tree_flatten
from mlx_lm.sample_utils import make_sampler

# Reuse model, tokenizer, LoRA setup, and helpers from Components 1 & 2
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from component1 import _model, _tokenizer
from component2 import (
    _build_training_tokens,
    _compute_loss,
    LORA_CONFIG,
    NUM_LORA_LAYERS,
)

# ---------- configuration ----------

SWAP_INTERVAL = 100
_BASE_DIR = Path(__file__).resolve().parent.parent / "adapters"
BUFFER_A = _BASE_DIR / "buffer_a"   # inference reads
BUFFER_B = _BASE_DIR / "buffer_b"   # training writes

# ---------- state ----------

_correction_queue: queue.Queue = queue.Queue()
_model_lock = threading.Lock()       # serialises ALL MLX model operations
_training_thread: threading.Thread | None = None
_stop_event = threading.Event()
_step_count = 0

# ---------- adapter I/O ----------


def _save_adapter_to(path: Path):
    """Save current trainable LoRA weights + config to a directory.

    Caller must hold _model_lock (touches model params via MLX).
    """
    path.mkdir(parents=True, exist_ok=True)
    weights = dict(tree_flatten(_model.trainable_parameters()))
    mx.save_safetensors(str(path / "adapters.safetensors"), weights)
    config = {
        "lora_parameters": LORA_CONFIG,
        "num_layers": NUM_LORA_LAYERS,
        "fine_tune_type": "lora",
    }
    with open(path / "adapter_config.json", "w") as f:
        json.dump(config, f, indent=2)


# ---------- buffer init ----------


def _init_buffers():
    """Seed both buffers with the current adapter weights."""
    with _model_lock:
        _save_adapter_to(BUFFER_A)
        _save_adapter_to(BUFFER_B)


# ---------- swap ----------


def _swap_buffers():
    """Atomic swap: buffer_b → buffer_a via rename, then recreate buffer_b.

    Caller must hold _model_lock so no inference is mid-generation.
    """
    tmp = _BASE_DIR / "_buffer_a_old"
    if tmp.exists():
        shutil.rmtree(tmp)

    # Atomic rename dance: A→tmp, B→A, copy A→B, remove tmp
    if BUFFER_A.exists():
        BUFFER_A.rename(tmp)
    BUFFER_B.rename(BUFFER_A)
    shutil.copytree(BUFFER_A, BUFFER_B)
    if tmp.exists():
        shutil.rmtree(tmp)


# ---------- training loop ----------


def _training_loop():
    """Background thread: drain the correction queue, train one step each."""
    global _step_count

    optimizer = optim.Adam(learning_rate=1e-4)
    loss_and_grad = nn.value_and_grad(_model, _compute_loss)

    while not _stop_event.is_set():
        try:
            prompt, completion = _correction_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        tokens = _build_training_tokens(prompt, completion)

        # All MLX operations under the lock (Metal is not thread-safe).
        # Each step is ~0.23 s — inference waits at most that long.
        with _model_lock:
            (loss, _), grads = loss_and_grad(_model, tokens)
            optimizer.update(_model, grads)
            mx.eval(_model.parameters(), optimizer.state)

            # Save to buffer_b (only this thread writes here)
            _save_adapter_to(BUFFER_B)

            _step_count += 1
            if _step_count >= SWAP_INTERVAL:
                _swap_buffers()
                _step_count = 0

        _correction_queue.task_done()


# ---------- public API ----------


def query(prompt: str, max_tokens: int = 256) -> str:
    """Run inference using the current adapter. Always responds in <3 s.

    Applies the instruct chat template for concise responses.
    """
    # Format with chat template (outside lock — pure Python)
    messages = [{"role": "user", "content": prompt}]
    formatted = _tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    sampler = make_sampler(temp=0.7)

    with _model_lock:
        response_text = ""
        for resp in mlx_lm.stream_generate(
            _model,
            _tokenizer,
            formatted,
            max_tokens=max_tokens,
            sampler=sampler,
        ):
            response_text += resp.text
    return response_text


def submit_correction(prompt: str, correct_completion: str) -> None:
    """Queue a correction for background training. Returns immediately."""
    _correction_queue.put((prompt, correct_completion))


# ---------- lifecycle ----------


def start():
    """Initialise buffers and start the background training thread."""
    global _training_thread, _step_count
    _stop_event.clear()
    _step_count = 0
    _init_buffers()
    _training_thread = threading.Thread(target=_training_loop, daemon=True)
    _training_thread.start()


def stop():
    """Signal the training thread to stop and wait for it to finish."""
    _stop_event.set()
    if _training_thread is not None:
        _training_thread.join(timeout=10)
    # Drain any remaining items so future starts are clean
    while not _correction_queue.empty():
        try:
            _correction_queue.get_nowait()
            _correction_queue.task_done()
        except queue.Empty:
            break


def get_training_step_count() -> int:
    """Expose step count for testing."""
    return _step_count


def get_queue_size() -> int:
    """Expose queue size for testing."""
    return _correction_queue.qsize()
