from collections import deque

import torch


class CuriosityScorer:
    """
    Computes a curiosity score for a curriculum item.
    curiosity = uncertainty x novelty x stage_relevance
    """

    def __init__(self, window: int = 50):
        self._recent_inputs: deque = deque(maxlen=window)
        self._prediction_errors: dict = {}

    def score(self, item, model) -> float:
        uncertainty = self._uncertainty(item, model)
        novelty = self._novelty(item)
        stage_relevance = item.stage_relevance

        score = uncertainty * novelty * stage_relevance
        if item.input_vector is not None:
            self._recent_inputs.append(item.input_vector)
        return float(score)

    def _uncertainty(self, item, model) -> float:
        if item.input_vector is None:
            return 0.8
        output, _ = model.forward(item.input_vector)
        if item.expected_vector is None:
            # Use mean node activation_variance across clusters as proxy
            active_clusters = [c for c in model.graph.clusters if not c.dormant]
            if not active_clusters:
                return 0.8
            total_var = 0.0
            count = 0
            for c in active_clusters:
                for n in c.nodes:
                    if n.alive:
                        total_var += n.activation_variance
                        count += 1
            variance = total_var / max(count, 1)
            # Early on, variance is near-zero — default to high uncertainty
            if variance < 0.001:
                return 0.8
            return min(1.0, variance * 2)
        similarity = torch.dot(output, item.expected_vector).item()
        return 1.0 - max(0.0, similarity)

    def _novelty(self, item) -> float:
        if not self._recent_inputs or item.input_vector is None:
            return 1.0
        similarities = [
            torch.dot(item.input_vector, prev).item()
            for prev in self._recent_inputs
        ]
        max_similarity = max(similarities)
        return 1.0 - max(0.0, max_similarity)
