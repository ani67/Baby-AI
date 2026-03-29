"""
MetricsTracker — lightweight rolling-window metrics for distillation,
generation, and reasoning accuracy.

All data is stored in deques with bounded maxlen, so memory is constant
regardless of how long the system runs.
"""

from collections import deque


class MetricsTracker:
    """Tracks distillation progress, generation quality, and reasoning accuracy."""

    def __init__(self, window: int = 100):
        self._window = window

        # Distillation
        self._text_distill_sims: deque[float] = deque(maxlen=window)
        self._vision_distill_sims: deque[float] = deque(maxlen=window)
        self._text_distill_total: int = 0
        self._vision_distill_total: int = 0

        # Generation
        self._generation_relevance: deque[float] = deque(maxlen=window)
        self._generation_words: deque[set[str]] = deque(maxlen=window)
        self._vocab_used: set[str] = set()

        # Reasoning (per task type)
        self._reasoning_accuracy: dict[str, deque[float]] = {}
        self._reasoning_total: int = 0

        # Spatial selectivity (vision patch entropy)
        self._spatial_selectivity: deque[float] = deque(maxlen=window)

    # ── Recording ──

    def record_text_distill(self, cosine_sim: float) -> None:
        """Record a native-vs-CLIP text similarity."""
        self._text_distill_sims.append(cosine_sim)
        self._text_distill_total += 1

    def record_vision_distill(self, cosine_sim: float) -> None:
        """Record a native-vs-CLIP vision similarity."""
        self._vision_distill_sims.append(cosine_sim)
        self._vision_distill_total += 1

    def record_generation(self, input_text: str, generated_text: str, relevance: float) -> None:
        """Record a generation event with input/output text and relevance score."""
        self._generation_relevance.append(relevance)

        words = set(generated_text.lower().split())
        self._generation_words.append(words)
        self._vocab_used.update(words)

    def record_reasoning(self, task_type: str, correct: bool, similarity: float) -> None:
        """Record a reasoning task result."""
        if task_type not in self._reasoning_accuracy:
            self._reasoning_accuracy[task_type] = deque(maxlen=self._window)
        self._reasoning_accuracy[task_type].append(1.0 if correct else 0.0)
        self._reasoning_total += 1

    def record_spatial_selectivity(self, patch_entropy: float) -> None:
        """Record spatial patch entropy from vision encoder."""
        self._spatial_selectivity.append(patch_entropy)

    # ── Snapshot ──

    def snapshot(self) -> dict:
        """Return current metrics as a JSON-serializable dict."""
        return {
            "distillation": self._distillation_snapshot(),
            "generation": self._generation_snapshot(),
            "reasoning": self._reasoning_snapshot(),
        }

    def _distillation_snapshot(self) -> dict:
        text_sims = list(self._text_distill_sims)
        vision_sims = list(self._vision_distill_sims)

        return {
            "text_cosine_sim": _safe_mean(text_sims),
            "text_cosine_sim_trend": _trend(text_sims),
            "text_samples": self._text_distill_total,
            "vision_cosine_sim": _safe_mean(vision_sims),
            "vision_cosine_sim_trend": _trend(vision_sims),
            "vision_samples": self._vision_distill_total,
        }

    def _generation_snapshot(self) -> dict:
        relevance = list(self._generation_relevance)

        # Unique words across the last N generations
        recent_words: set[str] = set()
        for word_set in self._generation_words:
            recent_words.update(word_set)

        return {
            "response_relevance": _safe_mean(relevance),
            "vocab_size": len(self._vocab_used),
            "unique_words_last_100": len(recent_words),
        }

    def _reasoning_snapshot(self) -> dict:
        task_types = [
            "comparison", "sequence", "analogy",
            "memory_retrieval", "odd_one_out",
        ]

        result: dict = {}
        all_scores: list[float] = []

        for task in task_types:
            scores = list(self._reasoning_accuracy.get(task, []))
            acc = _safe_mean(scores)
            result[f"{task}_accuracy"] = acc
            all_scores.extend(scores)

        # Include any extra task types not in the standard list
        for task, scores_deque in self._reasoning_accuracy.items():
            if task not in task_types:
                result[f"{task}_accuracy"] = _safe_mean(list(scores_deque))
                all_scores.extend(scores_deque)

        result["overall_accuracy"] = _safe_mean(all_scores)
        result["total_tasks"] = self._reasoning_total

        return result


# ── Helpers ──

def _safe_mean(values: list[float]) -> float | None:
    """Mean of a list, or None if empty."""
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def _trend(values: list[float]) -> float | None:
    """
    Compare last 10 values to the previous 10.
    Returns the difference (positive = improving).
    None if fewer than 20 samples.
    """
    if len(values) < 20:
        return None
    recent = values[-10:]
    previous = values[-20:-10]
    recent_mean = sum(recent) / len(recent)
    previous_mean = sum(previous) / len(previous)
    return round(recent_mean - previous_mean, 4)
