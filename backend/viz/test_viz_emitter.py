import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn.functional as F

from viz.emitter import VizEmitter
from viz.diff import compute_diff
from viz.projector import Projector
from model.node import Node
from model.cluster import Cluster
from model.graph import Graph


# ── Mock WebSocket ──


class MockWebSocket:
    def __init__(self):
        self.sent_messages = []
        self.accepted = False

    async def accept(self):
        self.accepted = True

    async def send_json(self, data: dict):
        self.sent_messages.append(data)


class FailingWebSocket:
    def __init__(self):
        self.accepted = False

    async def accept(self):
        self.accepted = True

    async def send_json(self, data: dict):
        raise ConnectionError("client disconnected")


# ── Helpers ──


def minimal_graph_json():
    return {
        "nodes": [
            {"id": "n_000", "cluster": "c_00", "activation_mean": 0.0,
             "activation_variance": 0.0, "age_steps": 0, "pos": None,
             "alive": True, "plasticity": 1.0}
        ],
        "clusters": [
            {"id": "c_00", "type": "arbitration", "density": 0.0,
             "node_count": 1, "layer_index": 0, "label": None,
             "dormant": False, "plasticity": 1.0, "age": 0}
        ],
        "edges": [],
    }


def minimal_graph():
    """Creates a real Graph object with a few nodes."""
    graph = Graph()
    nodes = []
    for i in range(4):
        n = Node(
            id=graph.next_node_id(),
            cluster_id="",
            weights=F.normalize(torch.randn(512), dim=0),
            bias=torch.zeros(1),
        )
        nodes.append(n)
    cluster = Cluster(id=graph.next_cluster_id(), nodes=nodes, layer_index=0)
    for n in cluster.nodes:
        n.cluster_id = cluster.id
    graph.add_cluster(cluster)
    return graph


def minimal_graph_with_nodes(n: int):
    """Creates a Graph with exactly n living nodes."""
    graph = Graph()
    nodes = []
    for i in range(n):
        node = Node(
            id=graph.next_node_id(),
            cluster_id="",
            weights=F.normalize(torch.randn(512), dim=0),
            bias=torch.zeros(1),
        )
        # Add pos attribute (projector writes to it)
        node.pos = None
        nodes.append(node)
    cluster = Cluster(id=graph.next_cluster_id(), nodes=nodes, layer_index=0)
    for node in cluster.nodes:
        node.cluster_id = cluster.id
    graph.add_cluster(cluster)
    return graph


# ── Tests ──


async def test_connect_sends_snapshot():
    emitter = VizEmitter()
    emitter._last_graph_json = minimal_graph_json()

    ws = MockWebSocket()
    await emitter.connect(ws)
    assert len(ws.sent_messages) == 1
    assert ws.sent_messages[0]["type"] == "snapshot"
    print("PASS: test_connect_sends_snapshot")


async def test_no_snapshot_on_connect_if_empty():
    emitter = VizEmitter()
    ws = MockWebSocket()
    await emitter.connect(ws)
    assert len(ws.sent_messages) == 0
    print("PASS: test_no_snapshot_on_connect_if_empty")


async def test_emit_step_broadcasts_to_all_clients():
    emitter = VizEmitter()
    ws1, ws2 = MockWebSocket(), MockWebSocket()
    await emitter.connect(ws1)
    await emitter.connect(ws2)
    ws1.sent_messages.clear()
    ws2.sent_messages.clear()

    graph = minimal_graph()
    await emitter.emit_step(
        step=1, stage=0, graph=graph,
        activations={"c_00": 0.5},
        last_question="test?", last_answer="test.",
        curiosity_score=0.7, is_positive=True,
        growth_events=[],
    )
    assert len(ws1.sent_messages) >= 1
    assert len(ws2.sent_messages) >= 1
    print("PASS: test_emit_step_broadcasts_to_all_clients")


async def test_dead_client_removed_on_broadcast():
    emitter = VizEmitter()
    good = MockWebSocket()
    dead = FailingWebSocket()
    await emitter.connect(good)
    await emitter.connect(dead)

    graph = minimal_graph()
    await emitter.emit_step(
        step=1, stage=0, graph=graph,
        activations={}, last_question="q", last_answer="a",
        curiosity_score=0.5, is_positive=True, growth_events=[],
    )
    assert dead not in emitter._clients
    assert good in emitter._clients
    print("PASS: test_dead_client_removed_on_broadcast")


async def test_snapshot_sent_on_interval():
    emitter = VizEmitter(snapshot_interval=5)
    graph = minimal_graph()
    ws = MockWebSocket()
    await emitter.connect(ws)
    ws.sent_messages.clear()

    for i in range(1, 11):
        await emitter.emit_step(
            step=i, stage=0, graph=graph,
            activations={}, last_question="q", last_answer="a",
            curiosity_score=0.5, is_positive=True, growth_events=[],
        )

    snapshot_messages = [
        m for m in ws.sent_messages if m["type"] == "snapshot"
    ]
    assert len(snapshot_messages) == 2, (
        f"Expected 2 snapshots (steps 5, 10), got {len(snapshot_messages)}"
    )
    print("PASS: test_snapshot_sent_on_interval")


def test_diff_detects_new_edge():
    old = {"nodes": [], "clusters": [], "edges": []}
    new = {
        "nodes": [],
        "clusters": [],
        "edges": [{"from": "c_00", "to": "c_01", "strength": 0.1}],
    }
    diff = compute_diff(old, new)
    assert len(diff["edges_added"]) == 1
    assert len(diff["edges_removed"]) == 0
    print("PASS: test_diff_detects_new_edge")


def test_diff_detects_removed_cluster():
    old = {"nodes": [], "edges": [],
           "clusters": [{"id": "c_00", "type": "integration"}]}
    new = {"nodes": [], "edges": [], "clusters": []}
    diff = compute_diff(old, new)
    assert len(diff["clusters_removed"]) == 1
    print("PASS: test_diff_detects_removed_cluster")


def test_diff_ignores_small_strength_change():
    edge = {"from": "c_00", "to": "c_01", "strength": 0.50}
    old = {"nodes": [], "clusters": [], "edges": [edge]}
    new_edge = {"from": "c_00", "to": "c_01", "strength": 0.51}
    new = {"nodes": [], "clusters": [], "edges": [new_edge]}
    diff = compute_diff(old, new)
    assert len(diff["edges_updated"]) == 0
    print("PASS: test_diff_ignores_small_strength_change")


async def test_projector_runs_without_error():
    projector = Projector()
    graph = minimal_graph_with_nodes(n=10)
    await projector.reproject(graph)
    for cluster in graph.clusters:
        for node in cluster.nodes:
            assert node.pos is not None, f"Node {node.id} has no pos"
            assert len(node.pos) == 3, f"Node {node.id} pos has {len(node.pos)} dims"
    print("PASS: test_projector_runs_without_error")


async def test_projector_handles_fewer_than_4_nodes():
    projector = Projector()
    graph = minimal_graph_with_nodes(n=2)
    await projector.reproject(graph)
    for cluster in graph.clusters:
        for node in cluster.nodes:
            assert node.pos == [0.0, 0.0, 0.0]
    print("PASS: test_projector_handles_fewer_than_4_nodes")


if __name__ == "__main__":
    async def run_all():
        # Sync tests
        test_diff_detects_new_edge()
        test_diff_detects_removed_cluster()
        test_diff_ignores_small_strength_change()

        # Async tests
        await test_connect_sends_snapshot()
        await test_no_snapshot_on_connect_if_empty()
        await test_emit_step_broadcasts_to_all_clients()
        await test_dead_client_removed_on_broadcast()
        await test_snapshot_sent_on_interval()
        await test_projector_runs_without_error()
        await test_projector_handles_fewer_than_4_nodes()

        print("\nAll 10 tests passed.")

    asyncio.run(run_all())
