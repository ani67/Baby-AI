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
    Early stages: high plasticity.
    Later stages: lower plasticity.
    """

    def current_rate(self, step: int) -> float:
        # Exponential decay from 0.01 to 0.001 over 10,000 steps
        base = 0.01
        floor = 0.001
        decay = 0.0003
        return max(floor, base * math.exp(-decay * step))

    def cluster_rate(self, cluster: Cluster, global_rate: float) -> float:
        # Young clusters learn faster regardless of global rate
        age_factor = min(1.0, cluster.age / 500)
        return global_rate * (2.0 - age_factor)
