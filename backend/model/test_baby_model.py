import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn.functional as F

from model.baby_model import BabyModel
from model.growth import bud
from model.serializer import ModelSerializer


class MockStateStore:
    """Mock store for testing growth_check."""
    def __init__(self):
        self.events = []

    def log_graph_event(self, step, event_type, cluster_a, cluster_b, metadata):
        self.events.append({
            "step": step,
            "event_type": event_type,
            "cluster_a": cluster_a,
            "cluster_b": cluster_b,
            "metadata": metadata,
        })
        return len(self.events)


# ── Tests ──


def test_initial_state():
    model = BabyModel(initial_clusters=4, nodes_per_cluster=4)
    summary = model.graph.summary()
    assert summary["cluster_count"] == 4
    assert summary["node_count"] == 16
    assert summary["edge_count"] == 0
    print("PASS: test_initial_state")


def test_forward_returns_512_dim():
    model = BabyModel()
    x = F.normalize(torch.randn(512), dim=0)
    output, _ = model.forward(x)
    assert output.shape == (512,), f"Expected (512,), got {output.shape}"
    print("PASS: test_forward_returns_512_dim")


def test_forward_output_is_normalized():
    model = BabyModel()
    x = F.normalize(torch.randn(512), dim=0)
    output, _ = model.forward(x)
    norm = torch.norm(output).item()
    assert abs(norm - 1.0) < 1e-4, f"Norm = {norm}, expected ~1.0"
    print("PASS: test_forward_output_is_normalized")


def test_update_changes_weights():
    model = BabyModel()
    x = F.normalize(torch.randn(512), dim=0)
    before = model._cluster_weight_snapshot(model.graph.clusters[0])
    model.forward(x)
    model.update(x, is_positive=True)
    after = model._cluster_weight_snapshot(model.graph.clusters[0])
    assert not torch.allclose(before, after), "Weights should change after update"
    print("PASS: test_update_changes_weights")


def test_connect_triggered_by_coactivation():
    model = BabyModel(initial_clusters=4, growth_check_interval=10)
    x = F.normalize(torch.randn(512), dim=0)
    # Force high correlation by running same input 200 times
    # Use high learning rate so node weights converge toward x,
    # producing strong activations that trigger co-activation detection
    for _ in range(200):
        model.forward(x)
        model.update(x, is_positive=True, learning_rate=0.1)
    # Ensure step is a multiple of growth_check_interval
    model.step = (model.step // 10) * 10
    store_mock = MockStateStore()
    events = model.growth_check(store_mock)
    connect_events = [e for e in events if e["event_type"] == "CONNECT"]
    assert len(connect_events) > 0, (
        f"Expected CONNECT events, got {[e['event_type'] for e in events]}"
    )
    print(f"PASS: test_connect_triggered_by_coactivation ({len(connect_events)} connects)")


def test_bud_splits_cluster():
    model = BabyModel(initial_clusters=1, nodes_per_cluster=16)
    cluster = model.graph.clusters[0]
    cluster.age = 300

    # Force bimodal by setting half weights to opposite directions
    half = len(cluster.nodes) // 2
    for i, node in enumerate(cluster.nodes):
        if i < half:
            node.weights = F.normalize(torch.ones(512), dim=0)
        else:
            node.weights = F.normalize(-torch.ones(512), dim=0)

    result = bud(cluster, model.graph)
    assert result is not None, "BUD should not return None"
    child_a, child_b = result
    assert child_a is not None
    assert child_b is not None
    # Parent removed, two children added
    assert len(model.graph.clusters) == 2
    assert model.graph.edge_exists(child_a.id, child_b.id)
    print("PASS: test_bud_splits_cluster")


def test_serialization_round_trip():
    model = BabyModel()
    x = F.normalize(torch.randn(512), dim=0)
    for _ in range(50):
        model.forward(x)
        model.update(x, is_positive=True)

    serializer = ModelSerializer()
    serializer.save(model, "/tmp/test_baby_checkpoint")

    model2 = BabyModel()
    serializer.load(model2, "/tmp/test_baby_checkpoint")

    out1, _ = model.forward(x)
    out2, _ = model2.forward(x)
    assert torch.allclose(out1, out2, atol=1e-5), (
        f"Outputs differ after load: dist={torch.dist(out1, out2).item():.6f}"
    )
    print("PASS: test_serialization_round_trip")


def test_dormant_cluster_excluded():
    model = BabyModel(initial_clusters=4)
    cluster = model.graph.clusters[-1]
    cluster.dormant = True

    x = F.normalize(torch.randn(512), dim=0)
    _, activations = model.forward(x, return_activations=True)
    assert cluster.id not in activations, (
        f"Dormant cluster {cluster.id} should not appear in activations"
    )
    print("PASS: test_dormant_cluster_excluded")


if __name__ == "__main__":
    test_initial_state()
    test_forward_returns_512_dim()
    test_forward_output_is_normalized()
    test_update_changes_weights()
    test_connect_triggered_by_coactivation()
    test_bud_splits_cluster()
    test_serialization_round_trip()
    test_dormant_cluster_excluded()
    print("\nAll 8 tests passed.")
