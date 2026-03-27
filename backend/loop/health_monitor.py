"""
HealthMonitor — automatic parameter tuning based on system health metrics.
Runs every 50 steps, adjusts model parameters to keep learning stable.
"""

from collections import deque


# Hard limits per parameter
LIMITS = {
    "resonance_threshold":    (0.005, 0.15),
    "resonance_min_pass":     (4, 30),
    "inhibition_radius":      (0.80, 0.98),
    "bud_cooldown":           (200, 1500),
}

# Healthy ranges for each metric
HEALTHY = {
    "positive_rate":       (0.35, 0.65),
    "growth_rate":         (0, 15),       # new clusters per 50 steps
    "active_per_step":     (4, 40),       # clusters active per step
    "edge_ratio":          (1.5, 6.0),    # edges / active_clusters
    "similarity_trend":    (-0.05, 0.5),  # mean similarity (not a delta)
}


class HealthMonitor:
    def __init__(self):
        self._interval = 50
        self._last_check_step = -50
        # Track consecutive out-of-range readings per metric
        self._consecutive: dict[str, int] = {k: 0 for k in HEALTHY}
        # Per-parameter cooldown: step when last adjusted
        self._param_cooldown: dict[str, int] = {k: -100 for k in LIMITS}
        # Metric history for trend detection
        self._active_per_step: deque = deque(maxlen=50)
        self._cluster_count_history: deque = deque(maxlen=100)
        print("[health] monitor initialized", flush=True)

    def record_step(self, active_count: int, total_clusters: int) -> None:
        """Called every step to accumulate per-step metrics."""
        self._active_per_step.append(active_count)
        self._cluster_count_history.append(total_clusters)

    def check(self, step: int, stage: int, model, positive_history, similarity_history) -> None:
        """Run health check. Call from orchestrator after growth_check."""
        if step - self._last_check_step < self._interval:
            return
        self._last_check_step = step

        # Stages collapsed — no emergency freeze. Growth self-regulates via
        # z-score resonance, Oja's rule, and BUD rate limiting (clusters//50).

        metrics = self._compute_metrics(model, positive_history, similarity_history)
        issues = []

        for metric, (lo, hi) in HEALTHY.items():
            val = metrics.get(metric)
            if val is None:
                self._consecutive[metric] = 0
                continue
            if val < lo or val > hi:
                self._consecutive[metric] += 1
            else:
                self._consecutive[metric] = 0

            if self._consecutive[metric] >= 2:
                issues.append((metric, val, lo, hi))

        if issues:
            for metric, val, lo, hi in issues:
                self._apply_fix(step, metric, val, lo, hi, model)
            names = [f"{m}={v:.2f}" for m, v, _, _ in issues]
            print(f"[health] step={step} out of range: {', '.join(names)}", flush=True)
        else:
            print(f"[health] step={step} OK", flush=True)

    def _compute_metrics(self, model, positive_history, similarity_history) -> dict:
        metrics = {}

        # Positive rate
        if len(positive_history) >= 10:
            metrics["positive_rate"] = sum(positive_history) / len(positive_history)

        # Growth rate: cluster count delta over last 50 steps
        if len(self._cluster_count_history) >= 50:
            old = list(self._cluster_count_history)[-50]
            new = list(self._cluster_count_history)[-1]
            metrics["growth_rate"] = new - old

        # Active clusters per step (mean over window)
        if len(self._active_per_step) >= 10:
            metrics["active_per_step"] = sum(self._active_per_step) / len(self._active_per_step)

        # Edge ratio
        active = sum(1 for c in model.graph.clusters if not c.dormant)
        if active > 0:
            metrics["edge_ratio"] = len(model.graph.edges) / active

        # Similarity trend (mean recent similarity)
        if len(similarity_history) >= 10:
            metrics["similarity_trend"] = sum(similarity_history) / len(similarity_history)

        return metrics

    def _apply_fix(self, step: int, metric: str, val: float, lo: float, hi: float, model) -> None:
        """Map an out-of-range metric to parameter adjustments."""
        too_low = val < lo
        monitor = model._growth_monitor

        if metric == "positive_rate":
            # Too few positives -> lower resonance threshold (more clusters -> better predictions)
            # Too many positives -> raise threshold (tighten)
            if too_low:
                self._adjust(step, "resonance_threshold", model, "resonance_threshold", -0.15)
                self._adjust(step, "resonance_min_pass", model, "resonance_min_pass", 0.15)
            else:
                self._adjust(step, "resonance_threshold", model, "resonance_threshold", 0.10)
                self._adjust(step, "resonance_min_pass", model, "resonance_min_pass", -0.10)

        elif metric == "growth_rate":
            # Too much growth -> increase bud cooldown
            if not too_low:
                self._adjust(step, "bud_cooldown", monitor, "bud_cooldown_steps", 0.20)

        elif metric == "active_per_step":
            if too_low:
                # Too few active -> lower resonance threshold, increase min_pass
                self._adjust(step, "resonance_threshold", model, "resonance_threshold", -0.20)
                self._adjust(step, "resonance_min_pass", model, "resonance_min_pass", 0.20)
            else:
                # Too many active -> raise threshold
                self._adjust(step, "resonance_threshold", model, "resonance_threshold", 0.15)
                self._adjust(step, "resonance_min_pass", model, "resonance_min_pass", -0.15)

        elif metric == "edge_ratio":
            if too_low:
                # Too few edges -> increase resonance_min_pass (more clusters = more coactivation pairs)
                self._adjust(step, "resonance_min_pass", model, "resonance_min_pass", 0.20)
            else:
                # Too many edges -> decrease min_pass, slow growth
                self._adjust(step, "resonance_min_pass", model, "resonance_min_pass", -0.15)
                self._adjust(step, "bud_cooldown", monitor, "bud_cooldown_steps", 0.15)

        elif metric == "similarity_trend":
            if too_low:
                # Similarity stuck near zero -> lower resonance threshold, more clusters participate
                self._adjust(step, "resonance_threshold", model, "resonance_threshold", -0.15)
                self._adjust(step, "resonance_min_pass", model, "resonance_min_pass", 0.15)

    def _adjust(self, step: int, param_key: str, obj, attr: str, direction: float) -> None:
        """
        Adjust a parameter by up to 20% in the given direction.
        direction > 0 means increase, < 0 means decrease.
        Respects 100-step cooldown and hard limits.
        """
        if step - self._param_cooldown[param_key] < 100:
            return

        lo, hi = LIMITS[param_key]
        old_val = getattr(obj, attr)

        # Compute change (max 20% of current value)
        max_change = abs(old_val) * 0.20
        if max_change < 0.001:
            max_change = 0.001

        change = max_change * (1.0 if direction > 0 else -1.0)
        # Scale by direction magnitude if < 1.0
        if abs(direction) < 1.0:
            change *= abs(direction) / 0.20

        new_val = old_val + change

        # For integer params, round
        if isinstance(old_val, int):
            new_val = int(round(new_val))

        # Clamp to hard limits
        new_val = max(lo, min(hi, new_val))

        # Skip if no effective change
        if new_val == old_val:
            return

        setattr(obj, attr, new_val)
        self._param_cooldown[param_key] = step

        if isinstance(new_val, float):
            print(f"[health] step={step} param={param_key}: {old_val:.4f} -> {new_val:.4f} (metric out of range)", flush=True)
        else:
            print(f"[health] step={step} param={param_key}: {old_val} -> {new_val} (metric out of range)", flush=True)
