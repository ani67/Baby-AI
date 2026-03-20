"""
VizEmitter — streams the living graph to the frontend over WebSocket.
All emit operations are non-blocking — the training loop never waits
on WebSocket sends or projection.
"""

import asyncio

from .diff import compute_diff
from .projector import Projector


class VizEmitter:
    def __init__(
        self,
        snapshot_interval: int = 50,
        projection_interval: int = 10,
    ):
        self._clients: set = set()
        self._last_graph_json: dict = {}
        self._prev_activations: dict[str, float] = {}
        self._prev_cluster_ids: set[str] = set()
        self._prev_edge_keys: set[tuple[str, str]] = set()
        self._projector = Projector()
        self._snapshot_interval = snapshot_interval
        self._projection_interval = projection_interval
        self._step = 0

    # ── Client management ──

    async def connect(self, ws) -> None:
        """
        Called when a new frontend client connects.
        Sends the current full snapshot immediately.
        """
        try:
            await ws.accept()
            self._clients.add(ws)
        except Exception:
            return

        if self._last_graph_json:
            snapshot = self._build_snapshot_from_json(self._last_graph_json)
            try:
                await ws.send_json(snapshot)
            except Exception:
                self._clients.discard(ws)

    async def disconnect(self, ws) -> None:
        self._clients.discard(ws)

    # ── Emit ──

    async def emit_step(
        self,
        step: int,
        stage: int,
        graph,
        activations: dict,
        last_question: str,
        last_answer: str,
        model_answer: str | None = None,
        curiosity_score: float = 0.0,
        is_positive: bool = True,
        growth_events: list | None = None,
        image_url: str | None = None,
    ) -> None:
        self._step = step
        growth_events = growth_events or []

        # Seed tracking state from last graph on first emission (avoids reporting all existing state as "new")
        if not self._prev_cluster_ids and self._last_graph_json:
            self._prev_cluster_ids = {c["id"] for c in self._last_graph_json.get("clusters", [])}
            self._prev_edge_keys = {(e["from"], e["to"]) for e in self._last_graph_json.get("edges", [])}
            self._prev_activations = dict(activations)

        # Reproject positions BEFORE capturing JSON so positions are included
        positions_stale = (step % self._projection_interval != 0)
        if not positions_stale:
            await self._projector.reproject(graph)

        # Capture current graph state
        current_json = graph.to_json()
        graph_summary = graph.summary() if hasattr(graph, 'summary') else {}

        # ── Compute delta ──
        # Activated / deactivated
        current_active = {cid for cid, v in activations.items() if v > 0.01}
        prev_active = {cid for cid, v in self._prev_activations.items() if v > 0.01}
        activated = list(current_active - prev_active)
        deactivated = list(prev_active - current_active)

        # Clusters added / dormanted
        current_cluster_ids = {c["id"] for c in current_json.get("clusters", []) if not c.get("dormant", False)}
        current_dormant_ids = {c["id"] for c in current_json.get("clusters", []) if c.get("dormant", False)}
        clusters_added_ids = current_cluster_ids - self._prev_cluster_ids
        clusters_dormanted = list(current_dormant_ids & self._prev_cluster_ids)  # was active, now dormant

        clusters_added = [
            c for c in current_json.get("clusters", [])
            if c["id"] in clusters_added_ids
        ]

        # Edges formed / pruned
        current_edge_keys = {(e["from"], e["to"]) for e in current_json.get("edges", [])}
        edge_map = {(e["from"], e["to"]): e for e in current_json.get("edges", [])}
        edges_formed = [
            (e["from"], e["to"], e.get("strength", 0.1))
            for k, e in edge_map.items() if k not in self._prev_edge_keys
        ]
        edges_pruned = [
            list(k) for k in self._prev_edge_keys if k not in current_edge_keys
        ]

        # Positions — only changed clusters (compare node positions)
        positions = {}
        if not positions_stale:
            for n in current_json.get("nodes", []):
                if n.get("pos") is not None:
                    positions[n["id"]] = n["pos"]

        # Build delta message
        delta = {
            "type": "delta",
            "step": step,
            "stage": stage,
            "activated": activated,
            "deactivated": deactivated,
            "activation_values": activations,
            "edges_formed": edges_formed,
            "edges_pruned": edges_pruned,
            "clusters_added": clusters_added,
            "clusters_dormanted": clusters_dormanted,
            "positions": positions,
            "dialogue": {
                "question": last_question,
                "answer": last_answer,
                "model_answer": model_answer,
                "curiosity_score": curiosity_score,
                "is_positive": is_positive,
                "image_url": image_url,
            },
            "growth_events": growth_events,
            "graph_summary": graph_summary,
        }

        # Also include nodes for newly added clusters
        if clusters_added_ids:
            delta["nodes_added"] = [
                n for n in current_json.get("nodes", [])
                if n.get("cluster") in clusters_added_ids or n.get("cluster_id") in clusters_added_ids
            ]

        print(
            f"[delta] step={step} activated={len(activated)} deactivated={len(deactivated)}"
            f" edges_formed={len(edges_formed)} edges_pruned={len(edges_pruned)}",
            flush=True,
        )

        # Update tracking state
        self._prev_activations = dict(activations)
        self._prev_cluster_ids = current_cluster_ids | current_dormant_ids
        self._prev_edge_keys = current_edge_keys
        self._last_graph_json = current_json

        await self._broadcast(delta)

        # Full snapshot on interval (unchanged)
        if step % self._snapshot_interval == 0:
            snapshot = self._build_snapshot(step, stage, graph)
            await self._broadcast(snapshot)

    async def emit_snapshot_to(self, ws) -> None:
        """Called by API handler on GET /snapshot."""
        if self._last_graph_json:
            snapshot = self._build_snapshot_from_json(self._last_graph_json)
            await ws.send_json(snapshot)

    def get_current_snapshot(self) -> dict:
        """Called by API handler for initial page load (REST)."""
        if self._last_graph_json:
            return self._build_snapshot_from_json(self._last_graph_json)
        return {
            "type": "snapshot",
            "step": 0,
            "stage": 0,
            "nodes": [],
            "clusters": [],
            "edges": [],
            "model_stats": {
                "total_nodes": 0,
                "total_clusters": 0,
                "total_edges": 0,
                "dormant_clusters": 0,
                "layer_count": 0,
                "step": 0,
                "stage": 0,
            },
        }

    # ── Internal ──

    async def _broadcast(self, message: dict) -> None:
        dead = set()
        for ws in list(self._clients):
            try:
                await asyncio.wait_for(ws.send_json(message), timeout=2.0)
            except Exception:
                dead.add(ws)
        self._clients -= dead

    def _build_snapshot(self, step: int, stage: int, graph) -> dict:
        graph_json = graph.to_json()
        return self._build_snapshot_from_json(graph_json, step, stage)

    def _build_snapshot_from_json(
        self, graph_json: dict, step: int = 0, stage: int = 0
    ) -> dict:
        nodes = graph_json.get("nodes", [])
        clusters = graph_json.get("clusters", [])
        edges = graph_json.get("edges", [])

        return {
            "type": "snapshot",
            "step": step or self._step,
            "stage": stage,
            "nodes": nodes,
            "clusters": clusters,
            "edges": edges,
            "model_stats": {
                "total_nodes": len(nodes),
                "total_clusters": len(clusters),
                "total_edges": len(edges),
                "dormant_clusters": sum(
                    1 for c in clusters if c.get("dormant", False)
                ),
                "layer_count": len(set(
                    c.get("layer_index", 0) for c in clusters
                    if not c.get("dormant", False)
                )),
                "step": step or self._step,
                "stage": stage,
            },
        }
