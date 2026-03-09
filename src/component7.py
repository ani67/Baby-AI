"""Component 7: Internal State Monitor (Proto-Self)

Four continuously-computed metrics representing the system's sense of its
own condition. Logged to disk. Used to modulate learning (uncertainty →
importance boost) and trigger curiosity (uncertainty > 0.7 → teacher query).

Metrics:
  uncertainty  — mean token-level entropy of recent inferences
  performance  — 1 - (corrections / total) over rolling 50 interactions
  novelty      — how different recent prompts are from known patterns
  coherence    — consistency of responses on similar prompts
"""

import collections
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import component4
import component5

# ---------- dataclass ----------


@dataclass
class InternalState:
    uncertainty: float   # 0-1
    performance: float   # 0-1
    novelty: float       # 0-1
    coherence: float     # 0-1
    timestamp: float


# ---------- configuration ----------

_STATE_INTERVAL = 10            # recompute state every N inferences
_ENTROPY_WINDOW = 20            # last N inferences for uncertainty
_PERFORMANCE_WINDOW = 50        # rolling window for performance
_NOVELTY_WINDOW = 10            # last N inferences for novelty
_COHERENCE_WINDOW = 20          # last N prompts for coherence
_COHERENCE_SIM_THRESHOLD = 0.9  # cosine sim threshold for "similar" prompts
_RESPONSE_SIM_THRESHOLD = 0.5   # cosine sim threshold for "consistent" responses
_MAX_ENTROPY = math.log(128256) # max possible entropy for vocab size

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_STATE_LOG = _DATA_DIR / "state_log.jsonl"

# ---------- ring buffers ----------

_entropy_buffer: collections.deque = collections.deque(maxlen=_ENTROPY_WINDOW)
_interaction_buffer: collections.deque = collections.deque(maxlen=_PERFORMANCE_WINDOW)
# Each entry: {"had_correction": bool}

_novelty_buffer: collections.deque = collections.deque(maxlen=_NOVELTY_WINDOW)
# Each entry: float (cosine distance to centroid)

_recent_prompts: collections.deque = collections.deque(maxlen=_COHERENCE_WINDOW)
# Each entry: {"prompt": str, "response": str}

# ---------- state ----------

_inference_count = 0
_current_state = InternalState(0.5, 1.0, 0.5, 1.0, time.time())
uncertainty_flag = False  # read by Component 8

# Embedding function (same as ChromaDB uses internally)
_embed_fn = DefaultEmbeddingFunction()

# ---------- notification hooks ----------


def notify_inference(prompt: str, response: str, token_entropies: list[float]):
    """Called after every inference. Accumulates data, recomputes state every N calls.

    Args:
        prompt: the user prompt
        response: the model's response text
        token_entropies: list of per-token entropies from this inference
    """
    global _inference_count

    # Record entropy (mean over tokens in this inference)
    if token_entropies:
        mean_entropy = sum(token_entropies) / len(token_entropies)
        normalised = min(1.0, mean_entropy / _MAX_ENTROPY)
        _entropy_buffer.append(normalised)

    # Record interaction (no correction yet — marked later if corrected)
    _interaction_buffer.append({"had_correction": False})

    # Record for novelty
    _novelty_buffer.append(_compute_novelty_distance(prompt))

    # Record for coherence
    _recent_prompts.append({"prompt": prompt, "response": response})

    _inference_count += 1
    if _inference_count % _STATE_INTERVAL == 0:
        _recompute_state()


def notify_correction(prompt: str, correction: str):
    """Called when a correction is submitted. Marks the most recent matching interaction."""
    # Mark the most recent interaction as corrected
    if _interaction_buffer:
        _interaction_buffer[-1]["had_correction"] = True


# ---------- metric computations ----------


def _compute_uncertainty() -> float:
    """Mean normalised entropy over recent inferences."""
    if not _entropy_buffer:
        return 0.5
    return sum(_entropy_buffer) / len(_entropy_buffer)


def _compute_performance() -> float:
    """1 - (corrections / total) over the rolling window."""
    if not _interaction_buffer:
        return 1.0
    corrections = sum(1 for i in _interaction_buffer if i["had_correction"])
    return 1.0 - corrections / len(_interaction_buffer)


def _compute_novelty_distance(prompt: str) -> float:
    """Cosine distance from prompt embedding to centroid of known embeddings."""
    component4._ensure_init()

    # Get all stored episode prompts
    all_eps = list(component4._episodes.values())
    if not all_eps:
        return 0.5  # no history → moderate novelty

    # Embed the new prompt
    prompt_vec = np.array(_embed_fn([prompt])[0], dtype=np.float32)

    # Compute centroid of stored prompts (sample up to 100 for speed)
    sample = all_eps[:100] if len(all_eps) > 100 else all_eps
    stored_vecs = np.array(
        _embed_fn([ep.prompt for ep in sample]), dtype=np.float32
    )
    centroid = stored_vecs.mean(axis=0)

    # Cosine distance = 1 - cosine_similarity
    cos_sim = np.dot(prompt_vec, centroid) / (
        np.linalg.norm(prompt_vec) * np.linalg.norm(centroid) + 1e-10
    )
    distance = 1.0 - float(cos_sim)
    return max(0.0, min(1.0, distance))


def _compute_novelty() -> float:
    """Mean novelty over recent inferences."""
    if not _novelty_buffer:
        return 0.5
    return sum(_novelty_buffer) / len(_novelty_buffer)


def _compute_coherence() -> float:
    """Proportion of similar-prompt pairs with consistent responses."""
    if len(_recent_prompts) < 2:
        return 1.0

    prompts_list = list(_recent_prompts)
    prompt_texts = [p["prompt"] for p in prompts_list]
    response_texts = [p["response"] for p in prompts_list]

    # Embed all prompts and responses
    prompt_vecs = np.array(_embed_fn(prompt_texts), dtype=np.float32)
    response_vecs = np.array(_embed_fn(response_texts), dtype=np.float32)

    # Normalise for cosine similarity
    p_norms = np.linalg.norm(prompt_vecs, axis=1, keepdims=True) + 1e-10
    r_norms = np.linalg.norm(response_vecs, axis=1, keepdims=True) + 1e-10
    prompt_normed = prompt_vecs / p_norms
    response_normed = response_vecs / r_norms

    # Prompt similarity matrix
    prompt_sim = prompt_normed @ prompt_normed.T

    # Find similar prompt pairs (above threshold, excluding self-pairs)
    n = len(prompts_list)
    similar_pairs = 0
    consistent_pairs = 0

    for i in range(n):
        for j in range(i + 1, n):
            if prompt_sim[i, j] >= _COHERENCE_SIM_THRESHOLD:
                similar_pairs += 1
                # Check response consistency
                resp_sim = float(response_normed[i] @ response_normed[j])
                if resp_sim >= _RESPONSE_SIM_THRESHOLD:
                    consistent_pairs += 1

    if similar_pairs == 0:
        return 1.0  # no similar pairs → coherent by default
    return consistent_pairs / similar_pairs


# ---------- state recomputation ----------


def _recompute_state():
    """Recompute all four metrics and log the new state."""
    global _current_state, uncertainty_flag

    _current_state = InternalState(
        uncertainty=round(_compute_uncertainty(), 4),
        performance=round(_compute_performance(), 4),
        novelty=round(_compute_novelty(), 4),
        coherence=round(_compute_coherence(), 4),
        timestamp=time.time(),
    )

    # Set uncertainty flag for Component 8
    uncertainty_flag = _current_state.uncertainty > 0.7

    # Modulate Component 5 importance scoring when uncertain
    if _current_state.uncertainty > 0.7:
        _install_uncertainty_boost()
    else:
        _remove_uncertainty_boost()

    # Log to disk
    _log_state(_current_state)


def _log_state(state: InternalState):
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(_STATE_LOG, "a") as f:
        f.write(json.dumps(asdict(state)) + "\n")


# ---------- importance score modulation ----------

_original_score_fn = None
_boost_installed = False


def _boosted_score_fn(episode):
    """Wrapper that multiplies importance by 1.5 when uncertainty is high."""
    base = component5.score_episode(episode)
    return round(min(1.0, base * 1.5), 4)


def _install_uncertainty_boost():
    """Replace component4's scoring hook with the boosted version."""
    global _boost_installed
    if not _boost_installed:
        component4._score_fn = _boosted_score_fn
        _boost_installed = True


def _remove_uncertainty_boost():
    """Restore component5's original scoring function."""
    global _boost_installed
    if _boost_installed:
        component4._score_fn = component5.score_episode
        _boost_installed = False


# ---------- public API ----------


def get_current_state() -> InternalState:
    """Return the most recently computed internal state."""
    return _current_state


def reset():
    """Reset all buffers and state. Useful for testing."""
    global _inference_count, _current_state, uncertainty_flag, _boost_installed
    _entropy_buffer.clear()
    _interaction_buffer.clear()
    _novelty_buffer.clear()
    _recent_prompts.clear()
    _inference_count = 0
    _current_state = InternalState(0.5, 1.0, 0.5, 1.0, time.time())
    uncertainty_flag = False
    _remove_uncertainty_boost()
    _boost_installed = False
