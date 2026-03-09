"""Component 6: Consolidation Loop (Sleep)

Periodically selects important episodes, trains the adapter on them,
prunes low-value episodes, and verifies the adapter didn't degrade.
Runs in a background thread; never blocks inference.
"""

import json
import logging
import random
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from component1 import _model, _tokenizer
from component2 import _build_training_tokens, _compute_loss
from component3 import _model_lock, _save_adapter_to, BUFFER_B
import component4
import component5
from component5 import learning_rate_for_episode

logger = logging.getLogger(__name__)

# ---------- configuration ----------

_SEVEN_DAYS = 7 * 24 * 60 * 60
_THIRTY_DAYS = 30 * 24 * 60 * 60
_TWENTY_FOUR_HOURS = 24 * 60 * 60

TRIGGER_EPISODE_COUNT = 100      # consolidate every N new episodes
TRIGGER_STORE_SIZE = 500         # consolidate when store exceeds this
LOSS_REVERT_THRESHOLD = 0.10     # revert if loss increases by >10%

_LOG_DIR = Path(__file__).resolve().parent.parent / "data"
_LOG_FILE = _LOG_DIR / "consolidation_log.jsonl"

# ---------- state ----------

_episodes_since_last = 0
_last_consolidation_time = time.time()
_consolidation_thread: threading.Thread | None = None
_stop_event = threading.Event()


# ---------- dataclass ----------


@dataclass
class ConsolidationReport:
    episodes_processed: int
    episodes_pruned: int
    adapter_loss_before: float
    adapter_loss_after: float
    duration_seconds: float


# ---------- selection ----------


def _select_episodes() -> list[component4.Episode]:
    """Select episodes for consolidation training."""
    component4._ensure_init()
    now = time.time()
    all_eps = list(component4._episodes.values())

    selected = []
    mid_tier = []

    for ep in all_eps:
        age = now - ep.timestamp
        # Exclude: very old + low score
        if age > _THIRTY_DAYS and ep.importance_score < 0.3:
            continue
        if ep.importance_score > 0.5:
            selected.append(ep)
        elif 0.2 <= ep.importance_score <= 0.5:
            mid_tier.append(ep)

    # Random 20% of mid-tier
    sample_size = max(1, len(mid_tier) // 5) if mid_tier else 0
    selected.extend(random.sample(mid_tier, min(sample_size, len(mid_tier))))

    # Sort by importance descending
    selected.sort(key=lambda e: e.importance_score, reverse=True)
    return selected


# ---------- loss measurement ----------


def _measure_average_loss(episodes: list[component4.Episode]) -> float:
    """Compute average loss over a set of episodes.

    Caller must hold _model_lock.
    """
    if not episodes:
        return 0.0

    total_loss = 0.0
    count = 0
    for ep in episodes:
        if ep.correction is not None:
            text = ep.prompt + ep.correction
        else:
            text = ep.prompt + ep.response
        tokens = mx.array(_tokenizer.encode(text))
        if tokens.shape[0] < 2:
            continue
        loss_val, _ = _compute_loss(_model, tokens)
        mx.eval(loss_val)
        total_loss += loss_val.item()
        count += 1

    return total_loss / count if count > 0 else 0.0


# ---------- adapter snapshot / revert ----------


def _snapshot_adapter_weights() -> dict[str, mx.array]:
    """Copy current trainable adapter weights. Caller must hold _model_lock."""
    return {
        name: mx.array(param)
        for name, param in tree_flatten(_model.trainable_parameters())
    }


def _restore_adapter_weights(snapshot: dict[str, mx.array]):
    """Restore adapter weights from snapshot. Caller must hold _model_lock."""
    _model.load_weights(list(snapshot.items()), strict=False)
    mx.eval(_model.parameters())


# ---------- pruning ----------


def _prune_episodes() -> int:
    """Remove low-value old episodes. Never prune corrections. Returns count pruned."""
    component4._ensure_init()
    now = time.time()
    to_prune = []

    for ep in list(component4._episodes.values()):
        age = now - ep.timestamp
        if (
            ep.importance_score < 0.2
            and age > _SEVEN_DAYS
            and ep.correction is None  # NEVER prune corrections
        ):
            to_prune.append(ep.id)

    for eid in to_prune:
        del component4._episodes[eid]
        # Also remove from ChromaDB
        try:
            component4._collection.delete(ids=[eid])
        except Exception:
            pass

    if to_prune:
        component4._save_episodes_to_disk()

    return len(to_prune)


# ---------- logging ----------


def _log_report(report: ConsolidationReport, reverted: bool = False):
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    entry = asdict(report)
    entry["reverted"] = reverted
    entry["timestamp"] = time.time()
    with open(_LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ---------- main consolidation cycle ----------


def run_consolidation_cycle() -> ConsolidationReport:
    """Run one full consolidation cycle. Can be called manually or by the background thread."""
    global _episodes_since_last, _last_consolidation_time

    t_start = time.time()

    # Select episodes
    selected = _select_episodes()
    if not selected:
        report = ConsolidationReport(0, 0, 0.0, 0.0, time.time() - t_start)
        _log_report(report)
        return report

    # Pick held-out examples for loss measurement (up to 5 from the selected set)
    held_out = selected[:min(5, len(selected))]

    with _model_lock:
        # Snapshot adapter weights
        snapshot = _snapshot_adapter_weights()

        # Measure loss before
        loss_before = _measure_average_loss(held_out)

        # Train on selected episodes
        loss_and_grad = nn.value_and_grad(_model, _compute_loss)

        for ep in selected:
            lr = learning_rate_for_episode(ep)
            optimizer = optim.Adam(learning_rate=lr)

            if ep.correction is not None:
                tokens = _build_training_tokens(ep.prompt, ep.correction)
            else:
                tokens = _build_training_tokens(ep.prompt, ep.response)

            (loss, _), grads = loss_and_grad(_model, tokens)
            optimizer.update(_model, grads)
            mx.eval(_model.parameters(), optimizer.state)

        # Measure loss after
        loss_after = _measure_average_loss(held_out)

        # Safety check: revert if loss increased by >10%
        reverted = False
        if loss_before > 0 and (loss_after - loss_before) / loss_before > LOSS_REVERT_THRESHOLD:
            _restore_adapter_weights(snapshot)
            loss_after = loss_before
            reverted = True
            logger.warning("Consolidation reverted: loss increased >10%%")
        else:
            # Save updated adapter to buffer_b
            _save_adapter_to(BUFFER_B)

    # Prune (no lock needed — pure Python on episode store)
    pruned = _prune_episodes()

    # Reset counters
    _episodes_since_last = 0
    _last_consolidation_time = time.time()

    report = ConsolidationReport(
        episodes_processed=len(selected),
        episodes_pruned=pruned,
        adapter_loss_before=loss_before,
        adapter_loss_after=loss_after,
        duration_seconds=time.time() - t_start,
    )
    _log_report(report, reverted=reverted)
    return report


# ---------- trigger check ----------


def should_consolidate() -> bool:
    """Check if any trigger condition is met."""
    component4._ensure_init()
    if _episodes_since_last >= TRIGGER_EPISODE_COUNT:
        return True
    if time.time() - _last_consolidation_time > _TWENTY_FOUR_HOURS:
        return True
    if component4.get_episode_count() > TRIGGER_STORE_SIZE:
        return True
    return False


def notify_new_episode():
    """Called when a new episode is stored, to track the trigger counter."""
    global _episodes_since_last
    _episodes_since_last += 1


# ---------- background thread ----------


def _consolidation_loop():
    """Background loop: check triggers periodically and run consolidation."""
    while not _stop_event.is_set():
        if should_consolidate():
            try:
                run_consolidation_cycle()
            except Exception as e:
                logger.error("Consolidation failed: %s", e)
        # Check every 60 seconds
        _stop_event.wait(60)


def start():
    """Start the background consolidation thread."""
    global _consolidation_thread
    _stop_event.clear()
    _consolidation_thread = threading.Thread(
        target=_consolidation_loop, daemon=True
    )
    _consolidation_thread.start()


def stop():
    """Stop the background consolidation thread."""
    _stop_event.set()
    if _consolidation_thread is not None:
        _consolidation_thread.join(timeout=10)
