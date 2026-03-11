from dataclasses import dataclass, field

from .cluster import Cluster
from .node import Node


@dataclass
class Edge:
    from_id: str
    to_id: str
    strength: float = 0.1
    age: int = 0
    direction: str = "bidirectional"
    steps_since_activation: int = 0

    def hebbian_update(
        self,
        from_activation: float,
        to_activation: float,
        decay: float = 0.001,
    ) -> None:
        delta = 0.01 * from_activation * to_activation - decay
        self.strength = max(0.0, min(1.0, self.strength + delta))
        if from_activation > 0.1 and to_activation > 0.1:
            self.steps_since_activation = 0
        else:
            self.steps_since_activation += 1
        self.age += 1


class Graph:
    """Registry for all clusters and edges."""

    def __init__(self):
        self.clusters: list[Cluster] = []
        self.edges: list[Edge] = []
        self._cluster_index: dict[str, Cluster] = {}
        self._node_counter: int = 0
        self._cluster_counter: int = 0

    def add_cluster(self, cluster: Cluster, source: str = "unknown") -> None:
        self.clusters.append(cluster)
        self._cluster_index[cluster.id] = cluster
        total = len(self.clusters)
        print(f"[cluster_create] id={cluster.id} layer={cluster.layer_index} source={source} total={total}", flush=True)

    def remove_cluster(self, cluster_id: str) -> None:
        cluster = self._cluster_index.pop(cluster_id, None)
        if cluster:
            self.clusters = [c for c in self.clusters if c.id != cluster_id]
            self.edges = [
                e for e in self.edges
                if e.from_id != cluster_id and e.to_id != cluster_id
            ]

    def get_cluster(self, cluster_id: str) -> Cluster | None:
        return self._cluster_index.get(cluster_id)

    def add_edge(self, from_id: str, to_id: str, strength: float = 0.1) -> None:
        edge = Edge(from_id=from_id, to_id=to_id, strength=strength)
        self.edges.append(edge)

    def remove_edge(self, edge: Edge) -> None:
        self.edges = [e for e in self.edges if e is not edge]

    def edge_exists(self, from_id: str, to_id: str) -> bool:
        for e in self.edges:
            if (e.from_id == from_id and e.to_id == to_id) or \
               (e.from_id == to_id and e.to_id == from_id):
                return True
        return False

    def incoming_edges(self, cluster_id: str) -> list[Edge]:
        return [e for e in self.edges
                if e.to_id == cluster_id or
                (e.direction == "bidirectional" and e.from_id == cluster_id)]

    def outgoing_edges(self, cluster_id: str) -> list[Edge]:
        result = []
        for e in self.edges:
            if e.from_id == cluster_id and e.to_id != cluster_id:
                result.append(e)
            elif e.direction == "bidirectional" and e.to_id == cluster_id and e.from_id != cluster_id:
                # For bidirectional, also treat reverse as outgoing
                result.append(Edge(
                    from_id=cluster_id, to_id=e.from_id,
                    strength=e.strength, age=e.age,
                    direction=e.direction,
                    steps_since_activation=e.steps_since_activation
                ))
        return result

    def entry_clusters(self) -> list[Cluster]:
        return [c for c in self.clusters if c.layer_index == 0 and not c.dormant]

    def top_layer_clusters(self) -> list[Cluster]:
        if not self.clusters:
            return []
        max_layer = max(c.layer_index for c in self.clusters if not c.dormant)
        return [c for c in self.clusters
                if c.layer_index == max_layer and not c.dormant]

    def adjacent_pairs(self) -> list[tuple]:
        """Pairs of clusters with edges between them."""
        pairs = []
        seen = set()
        for e in self.edges:
            key = tuple(sorted([e.from_id, e.to_id]))
            if key not in seen:
                seen.add(key)
                a = self.get_cluster(e.from_id)
                b = self.get_cluster(e.to_id)
                if a and b:
                    pairs.append((a, b))
        return pairs

    def replace_cluster(self, old: Cluster, new_clusters: list[Cluster], source: str = "bud") -> None:
        """Replace old cluster with new clusters, transferring external edges."""
        for nc in new_clusters:
            for node in nc.nodes:
                node.cluster_id = nc.id
            self.add_cluster(nc, source=source)

        # Transfer external edges to all new clusters
        new_edges = []
        for e in self.edges:
            if e.from_id == old.id:
                for nc in new_clusters:
                    new_edges.append(Edge(
                        from_id=nc.id, to_id=e.to_id,
                        strength=e.strength, direction=e.direction
                    ))
            elif e.to_id == old.id:
                for nc in new_clusters:
                    new_edges.append(Edge(
                        from_id=e.from_id, to_id=nc.id,
                        strength=e.strength, direction=e.direction
                    ))
            else:
                new_edges.append(e)
        self.edges = new_edges

        # Remove old cluster
        self._cluster_index.pop(old.id, None)
        self.clusters = [c for c in self.clusters if c.id != old.id]

    def insert_cluster_between(
        self, before: Cluster, new: Cluster, after: Cluster
    ) -> None:
        for node in new.nodes:
            node.cluster_id = new.id
        self.add_cluster(new, source="insert")
        # Remove direct edge between before and after
        self.edges = [
            e for e in self.edges
            if not ((e.from_id == before.id and e.to_id == after.id) or
                    (e.from_id == after.id and e.to_id == before.id))
        ]
        # Add edges: before → new → after
        self.add_edge(before.id, new.id, strength=0.5)
        self.add_edge(new.id, after.id, strength=0.5)

    def next_node_id(self) -> str:
        nid = self._node_counter
        self._node_counter += 1
        return f"n_{nid:03d}"

    def next_cluster_id(self) -> str:
        cid = self._cluster_counter
        self._cluster_counter += 1
        return f"c_{cid:02d}"

    def to_json(self) -> dict:
        """Full serializable representation of the current graph."""
        nodes_json = []
        for c in self.clusters:
            for n in c.nodes:
                nodes_json.append({
                    "id": n.id,
                    "cluster": c.id,
                    "activation_mean": n.mean_activation,
                    "activation_variance": n.activation_variance,
                    "age_steps": n.age,
                    "pos": getattr(n, 'pos', None),
                    "alive": n.alive,
                    "plasticity": n.plasticity,
                })

        clusters_json = []
        for c in self.clusters:
            clusters_json.append({
                "id": c.id,
                "cluster_type": c.cluster_type,
                "density": c._compute_internal_density(),
                "node_count": len(c.nodes),
                "layer_index": c.layer_index,
                "label": None,
                "dormant": c.dormant,
                "plasticity": c.plasticity,
                "age": c.age,
            })

        edges_json = []
        for e in self.edges:
            edges_json.append({
                "from": e.from_id,
                "to": e.to_id,
                "strength": e.strength,
                "age_steps": e.age,
                "direction": e.direction,
                "steps_since_activation": e.steps_since_activation,
            })

        return {
            "nodes": nodes_json,
            "clusters": clusters_json,
            "edges": edges_json,
        }

    def summary(self) -> dict:
        return {
            "cluster_count": len(self.clusters),
            "node_count": sum(len(c.nodes) for c in self.clusters),
            "edge_count": len(self.edges),
            "dormant_count": sum(1 for c in self.clusters if c.dormant),
            "layer_count": len(set(
                c.layer_index for c in self.clusters if not c.dormant
            )),
        }
