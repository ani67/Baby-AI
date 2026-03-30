"""
Benchmark: BrainState (current) vs BrainV2 (distributed architecture).

Compares learning quality, memory, throughput, and stability on synthetic
category-learning task. See doc/distributed-brain-reflect.md for context.

Run with pytest:
    cd backend && python -m pytest tests/test_brain_v2_benchmark.py -v -s

Run standalone:
    cd backend && python tests/test_brain_v2_benchmark.py
"""

import sys
import time
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from model.brain import BrainState

try:
    from model.brain_v2 import BrainV2
except ImportError:
    BrainV2 = None


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DIM = 512
INITIAL_NEURONS = 500
NUM_CATEGORIES = 20
ITEMS_PER_CATEGORY = 50
HELD_OUT_PER_CATEGORY = 5
TOTAL_STEPS = 5000
METRIC_INTERVAL = 500
SEED = 42
NOISE_STD = 0.08  # within-cluster noise (keeps clusters separable)


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_data(seed: int = SEED):
    """Create clustered synthetic data.

    Returns:
        train_items:   list of (vector, category_index)  -- 1000 items
        held_out:      list of (vector, category_index)  --  100 items
        centers:       (NUM_CATEGORIES, DIM) tensor of category centroids
    """
    rng = torch.Generator().manual_seed(seed)

    centers = F.normalize(torch.randn(NUM_CATEGORIES, DIM, generator=rng), dim=1)

    train_items: list[tuple[torch.Tensor, int]] = []
    held_out: list[tuple[torch.Tensor, int]] = []

    for cat in range(NUM_CATEGORIES):
        center = centers[cat]
        n_total = ITEMS_PER_CATEGORY + HELD_OUT_PER_CATEGORY
        noise = torch.randn(n_total, DIM, generator=rng) * NOISE_STD
        vecs = F.normalize(center.unsqueeze(0) + noise, dim=1)

        for i in range(ITEMS_PER_CATEGORY):
            train_items.append((vecs[i], cat))
        for i in range(ITEMS_PER_CATEGORY, n_total):
            held_out.append((vecs[i], cat))

    return train_items, held_out, centers


# ---------------------------------------------------------------------------
# Brain factories
# ---------------------------------------------------------------------------

def make_brain_state(seed: int = SEED) -> BrainState:
    """Create a BrainState with INITIAL_NEURONS neurons, seeded for reproducibility."""
    torch.manual_seed(seed)
    brain = BrainState(dim=DIM, initial_size=INITIAL_NEURONS, device="cpu")
    return brain


def make_brain_v2(seed: int = SEED):
    """Create a BrainV2 with identical initial parameters."""
    if BrainV2 is None:
        return None
    torch.manual_seed(seed)
    brain = BrainV2(dim=DIM, initial_size=INITIAL_NEURONS, device="cpu")
    return brain


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class Snapshot:
    step: int = 0
    category_accuracy: float = 0.0
    avg_cosine_sim: float = 0.0
    memory_bytes: int = 0
    forward_time_ms: float = 0.0
    neuron_utilization: float = 0.0
    edge_count: int = 0
    active_neurons: int = 0


METRIC_NAMES = [
    "category_accuracy",
    "avg_cosine_sim",
    "memory_bytes",
    "forward_time_ms",
    "neuron_utilization",
    "edge_count",
    "active_neurons",
]


def _memory_bytes(brain: BrainState) -> int:
    """Estimate total memory: weights + edges."""
    n = brain.n
    weight_bytes = n * brain.dim * 4  # float32
    edge_bytes = len(brain._edge_strengths) * 12  # 2 ints + 1 float per edge
    return weight_bytes + edge_bytes


def _forward_time_ms(brain: BrainState, items: list[tuple[torch.Tensor, int]]) -> float:
    """Average forward time over held-out items (ms)."""
    start = time.perf_counter()
    for vec, _ in items:
        brain.forward(vec)
    elapsed = time.perf_counter() - start
    return (elapsed / len(items)) * 1000.0


def _category_accuracy(
    brain: BrainState,
    held_out: list[tuple[torch.Tensor, int]],
    centers: torch.Tensor,
) -> float:
    """Forward each held-out item, find nearest category center, check correctness."""
    correct = 0
    for vec, true_cat in held_out:
        pred, _ = brain.forward(vec)
        pred_norm = F.normalize(pred.unsqueeze(0), dim=1)
        sims = (pred_norm @ centers.T).squeeze(0)
        predicted_cat = sims.argmax().item()
        if predicted_cat == true_cat:
            correct += 1
    return correct / len(held_out)


def _avg_cosine_sim(
    brain: BrainState,
    held_out: list[tuple[torch.Tensor, int]],
    centers: torch.Tensor,
) -> float:
    """Average cosine similarity between prediction and true category center."""
    total = 0.0
    for vec, true_cat in held_out:
        pred, _ = brain.forward(vec)
        sim = F.cosine_similarity(pred.unsqueeze(0), centers[true_cat].unsqueeze(0)).item()
        total += sim
    return total / len(held_out)


def _neuron_utilization(brain: BrainState, items: list[tuple[torch.Tensor, int]]) -> float:
    """Fraction of non-dormant neurons that fire on at least 1 of the given inputs."""
    n = brain.n
    ever_fired = torch.zeros(n, dtype=torch.bool)

    for vec, _ in items:
        brain.forward(vec)
        if brain._last_fired is not None:
            fired = brain._last_fired[:n]
            ever_fired |= fired.cpu()

    active_mask = ~brain.dormant[:n].cpu()
    n_active = active_mask.sum().item()
    if n_active == 0:
        return 0.0
    fired_active = (ever_fired & active_mask).sum().item()
    return fired_active / n_active


def collect_metrics(
    brain: BrainState,
    step: int,
    held_out: list[tuple[torch.Tensor, int]],
    centers: torch.Tensor,
) -> Snapshot:
    """Collect all metrics for a brain at a given step."""
    s = Snapshot(step=step)
    s.category_accuracy = _category_accuracy(brain, held_out, centers)
    s.avg_cosine_sim = _avg_cosine_sim(brain, held_out, centers)
    s.memory_bytes = _memory_bytes(brain)
    s.forward_time_ms = _forward_time_ms(brain, held_out)

    # Neuron utilization on a random 100-item subset of held-out
    s.neuron_utilization = _neuron_utilization(brain, held_out)

    summary = brain.summary()
    s.edge_count = summary["edge_count"]
    s.active_neurons = summary["cluster_count"]
    return s


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_brain(
    brain: BrainState,
    train_items: list[tuple[torch.Tensor, int]],
    held_out: list[tuple[torch.Tensor, int]],
    centers: torch.Tensor,
    total_steps: int = TOTAL_STEPS,
    label: str = "BrainState",
    use_reflect: bool = False,
) -> list[Snapshot]:
    """Run training loop, collecting metrics at intervals.

    Args:
        use_reflect: if True, call brain.reflect() after each forward (BrainV2 only).
    """
    rng = torch.Generator().manual_seed(SEED + 7)  # separate shuffle seed
    n_items = len(train_items)
    snapshots: list[Snapshot] = []

    print(f"\n{'='*60}")
    print(f"Training {label}: {total_steps} steps, {n_items} items")
    print(f"{'='*60}")

    t0 = time.perf_counter()

    for step in range(1, total_steps + 1):
        # Pick item (deterministic cycle with shuffle every epoch)
        idx = (step - 1) % n_items
        if idx == 0 and step > 1:
            # Shuffle order each epoch (same generator for both brains)
            pass  # we use a fixed permutation per epoch below
        vec, cat = train_items[idx]
        target = centers[cat]

        # Forward
        pred, _ = brain.forward(vec)

        # Reflect (BrainV2 only)
        if use_reflect and hasattr(brain, "reflect"):
            error = target - pred
            brain.reflect(error)

        # Update: train toward category center
        brain.update(vec, target)

        # Growth check every 500 steps
        if step % 500 == 0:
            brain.growth_check(step)

        # Collect metrics
        if step % METRIC_INTERVAL == 0:
            snap = collect_metrics(brain, step, held_out, centers)
            snapshots.append(snap)
            elapsed = time.perf_counter() - t0
            steps_per_sec = step / elapsed
            print(
                f"  [{label}] step {step:5d} | "
                f"acc={snap.category_accuracy:.3f} "
                f"cos={snap.avg_cosine_sim:.3f} "
                f"mem={snap.memory_bytes / 1e6:.1f}MB "
                f"fwd={snap.forward_time_ms:.2f}ms "
                f"util={snap.neuron_utilization:.2f} "
                f"edges={snap.edge_count} "
                f"neurons={snap.active_neurons} "
                f"| {steps_per_sec:.0f} steps/s"
            )

    total_time = time.perf_counter() - t0
    print(f"  [{label}] done in {total_time:.1f}s ({total_steps / total_time:.0f} steps/s)")
    return snapshots


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def _winner(v1: float, v2: float, metric: str) -> str:
    """Determine winner. For memory and forward_time, lower is better."""
    lower_is_better = metric in ("memory_bytes", "forward_time_ms")
    if abs(v1 - v2) < 1e-9:
        return "TIE"
    if lower_is_better:
        return "V1" if v1 < v2 else "V2"
    return "V1" if v1 > v2 else "V2"


def _fmt(value: float, metric: str) -> str:
    if metric == "memory_bytes":
        return f"{value / 1e6:.1f}MB"
    if metric == "forward_time_ms":
        return f"{value:.2f}ms"
    if metric in ("edge_count", "active_neurons"):
        return f"{int(value)}"
    return f"{value:.4f}"


def print_comparison(v1_snaps: list[Snapshot], v2_snaps: list[Snapshot]):
    """Print the comparison table."""
    header = f"{'Step':>5} | {'Metric':<20} | {'BrainState':>12} | {'BrainV2':>12} | {'Winner':>6}"
    sep = "-" * len(header)

    print(f"\n{sep}")
    print(header)
    print(sep)

    for s1, s2 in zip(v1_snaps, v2_snaps):
        assert s1.step == s2.step
        for m in METRIC_NAMES:
            v1_val = getattr(s1, m)
            v2_val = getattr(s2, m)
            w = _winner(v1_val, v2_val, m)
            print(
                f"{s1.step:>5} | {m:<20} | {_fmt(v1_val, m):>12} | {_fmt(v2_val, m):>12} | {w:>6}"
            )
        print(sep)


# ---------------------------------------------------------------------------
# Pytest tests
# ---------------------------------------------------------------------------

def test_brainstate_solo_benchmark():
    """Run BrainState alone (always available). Validates the harness works."""
    train_items, held_out, centers = generate_data()
    brain = make_brain_state()
    snapshots = train_brain(brain, train_items, held_out, centers, label="BrainState")

    # Sanity: brain should learn something
    final = snapshots[-1]
    assert final.category_accuracy > 0.0, "BrainState learned nothing"
    assert final.avg_cosine_sim > -1.0, "Cosine similarity is degenerate"
    assert final.active_neurons > 0, "No active neurons"


def test_brain_v2_benchmark():
    """Full comparison benchmark: BrainState vs BrainV2."""
    if BrainV2 is None:
        import pytest
        pytest.skip("BrainV2 not yet implemented (model.brain_v2 not found)")

    train_items, held_out, centers = generate_data()

    # Train both on identical data in identical order
    brain_v1 = make_brain_state()
    v1_snaps = train_brain(brain_v1, train_items, held_out, centers, label="BrainState")

    brain_v2 = make_brain_v2()
    v2_snaps = train_brain(
        brain_v2, train_items, held_out, centers,
        label="BrainV2", use_reflect=True,
    )

    print_comparison(v1_snaps, v2_snaps)

    # -- Pass/fail criteria (from design doc) --
    final_v1 = v1_snaps[-1]
    final_v2 = v2_snaps[-1]

    # 1. Memory bounded: BrainV2 never exceeds 50MB
    for snap in v2_snaps:
        assert snap.memory_bytes <= 50 * 1024 * 1024, (
            f"BrainV2 memory exceeded 50MB at step {snap.step}: "
            f"{snap.memory_bytes / 1e6:.1f}MB"
        )

    # 2. Learning quality: BrainV2 category_accuracy >= BrainState at final step
    assert final_v2.category_accuracy >= final_v1.category_accuracy, (
        f"BrainV2 accuracy ({final_v2.category_accuracy:.4f}) < "
        f"BrainState ({final_v1.category_accuracy:.4f})"
    )

    # 3. Throughput: BrainV2 steps/sec >= 0.8x BrainState
    #    (measured via forward_time_ms as proxy)
    slowdown = final_v2.forward_time_ms / max(final_v1.forward_time_ms, 1e-6)
    assert slowdown <= 1.25, (
        f"BrainV2 too slow: {final_v2.forward_time_ms:.2f}ms vs "
        f"BrainState {final_v1.forward_time_ms:.2f}ms "
        f"(slowdown {slowdown:.2f}x, max allowed 1.25x)"
    )

    # 4. Stability: no crashes (if we got here, it didn't crash)
    print("\nAll pass/fail criteria met.")


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train_items, held_out, centers = generate_data()

    # Always run BrainState
    brain_v1 = make_brain_state()
    v1_snaps = train_brain(brain_v1, train_items, held_out, centers, label="BrainState")

    if BrainV2 is not None:
        brain_v2 = make_brain_v2()
        v2_snaps = train_brain(
            brain_v2, train_items, held_out, centers,
            label="BrainV2", use_reflect=True,
        )
        print_comparison(v1_snaps, v2_snaps)
    else:
        print("\nBrainV2 not available (model.brain_v2 not found).")
        print("Running BrainState only. Final metrics:")
        final = v1_snaps[-1]
        for m in METRIC_NAMES:
            print(f"  {m:<20}: {_fmt(getattr(final, m), m)}")

    sys.exit(0)
