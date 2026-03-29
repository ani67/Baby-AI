import asyncio
import math

import numpy as np
import torch


class Projector:
    """
    Projects nodes to 3D positions using force-directed layout.

    Clusters are treated as particles:
    - Connected clusters attract (spring force proportional to edge strength)
    - All clusters repel each other (electrostatic repulsion)
    - Layer index provides a gentle vertical bias (not rigid)
    - Nodes within each cluster are offset by PCA of their weight vectors

    This produces organic, t-SNE-like layouts where structure emerges from data.
    """

    def __init__(self):
        self._last_positions: dict[str, list[float]] = {}
        self._cluster_positions: dict[str, np.ndarray] = {}
        self._pending_node_positions: dict[str, list[float]] = {}

    async def reproject(self, graph) -> None:
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._compute_positions, graph)
            self.apply_pending_positions(graph)
        except Exception as e:
            print(f"[projector] ERROR: {e}", flush=True)
            import traceback
            traceback.print_exc()

    def apply_pending_positions(self, graph) -> None:
        """Apply buffered positions to nodes. Must be called from main thread."""
        if not self._pending_node_positions:
            return
        for cluster in graph.clusters:
            for node in cluster.nodes:
                if node.id in self._pending_node_positions:
                    node.pos = self._pending_node_positions[node.id]
        self._pending_node_positions = {}

    def _compute_positions(self, graph) -> None:
        active_clusters = [c for c in graph.clusters if not c.dormant]
        if not active_clusters:
            return

        n = len(active_clusters)
        cid_to_idx = {c.id: i for i, c in enumerate(active_clusters)}

        # Initialize positions — use previous positions if available, else seed
        pos = np.zeros((n, 3))
        for i, c in enumerate(active_clusters):
            if c.id in self._cluster_positions:
                pos[i] = self._cluster_positions[c.id]
            else:
                # Seed: layer gives y-bias, random x/z
                rng = np.random.RandomState(hash(c.id) % 2**31)
                pos[i] = [
                    rng.randn() * 1.5,
                    c.layer_index * 0.8,
                    rng.randn() * 1.5,
                ]

        # Build edge list with strengths
        edges = []
        for e in graph.edges:
            if e.from_id in cid_to_idx and e.to_id in cid_to_idx:
                edges.append((cid_to_idx[e.from_id], cid_to_idx[e.to_id], e.strength))

        # Force-directed simulation
        pos = self._force_directed(pos, edges, active_clusters, iterations=60)

        # Center the layout
        pos -= pos.mean(axis=0)

        # Scale to fit nicely in view
        max_extent = np.max(np.abs(pos)) + 1e-6
        if max_extent > 4.0:
            pos *= 3.5 / max_extent

        # Save cluster positions for next iteration (warm start)
        for i, c in enumerate(active_clusters):
            self._cluster_positions[c.id] = pos[i].copy()

        # Assign node positions: cluster center + PCA offset
        for i, cluster in enumerate(active_clusters):
            cx, cy, cz = pos[i]
            living = [nd for nd in cluster.nodes if nd.alive]
            if not living:
                continue

            if len(living) == 1:
                self._pending_node_positions[living[0].id] = [float(cx), float(cy), float(cz)]
                self._last_positions[living[0].id] = self._pending_node_positions[living[0].id]
                continue

            weights = torch.stack([nd.weights for nd in living]).detach().numpy()
            offsets = self._node_offsets(weights, spread=0.25)

            for node, offset in zip(living, offsets):
                ox = offset[0] if len(offset) > 0 else 0.0
                oy = offset[1] if len(offset) > 1 else 0.0
                oz = offset[2] if len(offset) > 2 else 0.0
                self._pending_node_positions[node.id] = [
                    float(cx + ox),
                    float(cy + oy),
                    float(cz + oz),
                ]
                self._last_positions[node.id] = self._pending_node_positions[node.id]

        # Clean up dead clusters from cache
        active_ids = {c.id for c in active_clusters}
        for dead_id in list(self._cluster_positions.keys()):
            if dead_id not in active_ids:
                del self._cluster_positions[dead_id]

    def _force_directed(
        self,
        pos: np.ndarray,
        edges: list[tuple],
        clusters: list,
        iterations: int = 60,
    ) -> np.ndarray:
        """Run force-directed layout simulation."""
        n = len(pos)
        if n <= 1:
            return pos

        # Params
        repulsion_strength = 2.0
        attraction_strength = 0.8
        layer_gravity = 0.15  # gentle pull toward layer y-position
        damping = 0.9
        dt = 0.05
        min_dist = 0.1

        # Compute layer y-targets (normalized)
        layer_indices = np.array([c.layer_index for c in clusters])
        unique_layers = np.unique(layer_indices)
        if len(unique_layers) > 1:
            layer_min, layer_max = unique_layers.min(), unique_layers.max()
            layer_y_target = (layer_indices - layer_min) / (layer_max - layer_min) * 3.0 - 1.5
        else:
            layer_y_target = np.zeros(n)

        velocity = np.zeros_like(pos)

        for iteration in range(iterations):
            forces = np.zeros_like(pos)

            # Temperature: decreases over iterations for convergence
            temp = 1.0 - iteration / iterations

            # ── Repulsion (all pairs) ──
            for i in range(n):
                for j in range(i + 1, n):
                    diff = pos[i] - pos[j]
                    dist = np.linalg.norm(diff) + 1e-6
                    if dist < min_dist:
                        dist = min_dist
                    # Inverse square repulsion
                    force_mag = repulsion_strength / (dist * dist)
                    force = (diff / dist) * force_mag
                    forces[i] += force
                    forces[j] -= force

            # ── Attraction (edges) ──
            for i, j, strength in edges:
                diff = pos[j] - pos[i]
                dist = np.linalg.norm(diff) + 1e-6
                # Spring force: pull together, proportional to distance and edge strength
                force_mag = attraction_strength * strength * dist * 0.3
                force = (diff / dist) * force_mag
                forces[i] += force
                forces[j] -= force

            # ── Layer gravity (gentle y-bias) ──
            for i in range(n):
                y_diff = layer_y_target[i] - pos[i, 1]
                forces[i, 1] += layer_gravity * y_diff

            # ── Integration ──
            velocity = velocity * damping + forces * dt * temp
            # Clamp velocity to prevent explosions
            speed = np.linalg.norm(velocity, axis=1, keepdims=True)
            max_speed = 0.5 * temp + 0.05
            velocity = np.where(speed > max_speed, velocity * max_speed / speed, velocity)
            pos += velocity

        return pos

    def _node_offsets(self, weight_matrix: np.ndarray, spread: float = 0.25) -> np.ndarray:
        """Compute 3D offsets for nodes within a cluster using PCA."""
        n = weight_matrix.shape[0]
        centered = weight_matrix - weight_matrix.mean(axis=0)
        try:
            U, S, Vt = np.linalg.svd(centered, full_matrices=False)
            min_dim = min(U.shape[1], 3)
            offsets = np.zeros((n, 3))
            offsets[:, :min_dim] = U[:, :min_dim] * spread
            if np.max(np.abs(offsets)) < 0.01:
                offsets = np.random.RandomState(42).randn(n, 3) * spread * 0.5
        except Exception:
            offsets = np.random.RandomState(42).randn(n, 3) * spread * 0.5
        return offsets
