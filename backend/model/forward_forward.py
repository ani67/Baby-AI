"""
Forward-Forward local learning rule.

The update logic lives on Node.ff_update() and Cluster.ff_update().
This module provides the PlasticitySchedule that controls
global learning rate decay over time.
"""

import math

from .cluster import Cluster

# Stage-aware LR multipliers: later stages need lower rates
# because the graph is larger and updates have wider impact.
STAGE_LR_SCALE = {
    0: 1.0,
    1: 0.5,
    2: 0.1,
    3: 0.1,
    4: 0.1,
}


class PlasticitySchedule:
    """
    Controls the global learning rate over time and by stage.
    Early stages: high plasticity.
    Later stages: lower plasticity.
    """

    def current_rate(self, step: int, stage: int = 0) -> float:
        # Exponential decay from 0.01 to 0.001 over 10,000 steps
        base = 0.01
        floor = 0.001
        decay = 0.0003
        time_rate = max(floor, base * math.exp(-decay * step))
        # Apply stage scaling
        stage_scale = STAGE_LR_SCALE.get(stage, 0.1)
        return time_rate * stage_scale

    def cluster_rate(self, cluster: Cluster, global_rate: float) -> float:
        # Young clusters learn faster regardless of global rate
        age_factor = min(1.0, cluster.age / 500)
        return global_rate * (2.0 - age_factor)
