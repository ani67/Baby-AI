"""
Save/load Baby Model state to .pt + .json files.
"""

import json

import torch
import torch.nn.functional as F

from .node import Node
from .cluster import Cluster
from .graph import Graph, Edge


class ModelSerializer:
    """
    Two files per checkpoint:
      step_{N}.pt    — PyTorch state dict (all node weights and biases)
      step_{N}.json  — Graph structure (cluster membership, edges, metadata)
    """

    def save(self, model, path_prefix: str) -> None:
        # 1. Collect all node weights into a state dict
        state_dict = {}
        for cluster in model.graph.clusters:
            for node in cluster.nodes:
                state_dict[f"{node.id}.weights"] = node.weights
                state_dict[f"{node.id}.bias"] = node.bias
        torch.save(state_dict, f"{path_prefix}.pt")

        # 2. Save graph structure as JSON
        graph_json = self._graph_to_json(model)
        with open(f"{path_prefix}.json", "w") as f:
            json.dump(graph_json, f)

    def load(self, model, path_prefix: str) -> None:
        # 1. Load graph structure first
        with open(f"{path_prefix}.json") as f:
            graph_json = json.load(f)
        self._json_to_graph(graph_json, model)

        # 2. Load weights into the nodes
        state_dict = torch.load(f"{path_prefix}.pt", map_location="cpu",
                                weights_only=True)
        for cluster in model.graph.clusters:
            for node in cluster.nodes:
                if f"{node.id}.weights" in state_dict:
                    node.weights = state_dict[f"{node.id}.weights"]
                    node.bias = state_dict[f"{node.id}.bias"]

    def _graph_to_json(self, model) -> dict:
        graph = model.graph
        clusters = []
        for c in graph.clusters:
            nodes = []
            for n in c.nodes:
                nodes.append({
                    "id": n.id,
                    "plasticity": n.plasticity,
                    "age": n.age,
                    "alive": n.alive,
                })
            clusters.append({
                "id": c.id,
                "layer_index": c.layer_index,
                "plasticity": c.plasticity,
                "age": c.age,
                "dormant": c.dormant,
                "nodes": nodes,
                "internal_edges": c.internal_edges,
                "interface_nodes": c.interface_nodes,
            })

        edges = []
        for e in graph.edges:
            edges.append({
                "from_id": e.from_id,
                "to_id": e.to_id,
                "strength": e.strength,
                "age": e.age,
                "direction": e.direction,
                "steps_since_activation": e.steps_since_activation,
            })

        return {
            "step": model.step,
            "stage": model.stage,
            "node_counter": graph._node_counter,
            "cluster_counter": graph._cluster_counter,
            "clusters": clusters,
            "edges": edges,
        }

    def _json_to_graph(self, data: dict, model) -> None:
        model.step = data["step"]
        model.stage = data["stage"]
        model.graph = Graph()
        model.graph._node_counter = data["node_counter"]
        model.graph._cluster_counter = data["cluster_counter"]

        for cd in data["clusters"]:
            nodes = []
            for nd in cd["nodes"]:
                node = Node(
                    id=nd["id"],
                    cluster_id=cd["id"],
                    weights=torch.zeros(512),  # loaded from .pt next
                    bias=torch.zeros(1),
                    plasticity=nd["plasticity"],
                    age=nd["age"],
                    alive=nd["alive"],
                )
                nodes.append(node)
            cluster = Cluster(
                id=cd["id"],
                nodes=nodes,
                layer_index=cd["layer_index"],
                plasticity=cd["plasticity"],
                age=cd["age"],
                dormant=cd["dormant"],
                internal_edges=cd.get("internal_edges", {}),
                interface_nodes=cd.get("interface_nodes", []),
            )
            model.graph.add_cluster(cluster)

        for ed in data["edges"]:
            edge = Edge(
                from_id=ed["from_id"],
                to_id=ed["to_id"],
                strength=ed["strength"],
                age=ed["age"],
                direction=ed["direction"],
                steps_since_activation=ed["steps_since_activation"],
            )
            model.graph.edges.append(edge)
