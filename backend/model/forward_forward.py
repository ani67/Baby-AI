"""
Forward-Forward local learning rule.

The update logic lives on Node.ff_update() and Cluster.ff_update().
This module provides the PlasticitySchedule that controls
global learning rate decay over time.
"""

import math

from .cluster import Cluster


class PlasticitySchedule:
    """
    Controls the global learning rate over time.
    Pure exponential decay — no stage gating.
    The model self-regulates through weight normalization and Oja's rule.
    """

    def current_rate(self, step: int, stage: int = 0) -> float:
        # Warmup for first 500 steps, then exponential decay
        # Prevents wild early weight swings before structure forms
        base = 0.01
        floor = 0.001
        decay = 0.0003
        warmup_steps = 500
        rate = max(floor, base * math.exp(-decay * step))
        if step < warmup_steps:
            rate *= step / warmup_steps
        return rate

    def cluster_rate(self, cluster: Cluster, global_rate: float) -> float:
        # Homeostatic per-cluster LR: clusters with high error learn faster,
        # accurate clusters stabilize. Young clusters also learn faster.
        age_factor = min(1.0, cluster.age / 500)
        age_boost = 2.0 - age_factor  # 2x for new clusters, 1x for old

        # Error-driven boost: high error → up to 2x LR, low error → 0.5x LR
        if cluster._error_history and len(cluster._error_history) >= 10:
            mean_error = sum(cluster._error_history) / len(cluster._error_history)
            error_boost = 0.5 + 1.5 * mean_error  # 0.5 (accurate) to 2.0 (wrong)
        else:
            error_boost = 1.0  # neutral until enough history

        return global_rate * age_boost * error_boost
