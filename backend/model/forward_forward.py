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
        # Young clusters learn faster regardless of global rate
        age_factor = min(1.0, cluster.age / 500)
        return global_rate * (2.0 - age_factor)
