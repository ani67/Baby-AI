"""
VizEmitter — streams the living graph to the frontend over WebSocket.
"""

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

        # Reproject positions BEFORE capturing JSON so positions are included
        positions_stale = (step % self._projection_interval != 0)
        if not positions_stale:
            await self._projector.reproject(graph)

        # Compute diff against last known graph state
        current_json = graph.to_json()
        diff = compute_diff(self._last_graph_json, current_json)
        self._last_graph_json = current_json

        # Build step message
        graph_summary = graph.summary() if hasattr(graph, 'summary') else {}
        message = {
            "type": "step",
            "step": step,
            "stage": stage,
            "graph_diff": diff,
            "activations": activations,
            "growth_events": growth_events,
            "dialogue": {
                "question": last_question,
                "answer": last_answer,
                "model_answer": model_answer,
                "curiosity_score": curiosity_score,
                "is_positive": is_positive,
                "image_url": image_url,
            },
            "positions_stale": positions_stale,
            "graph_summary": graph_summary,
        }

        # On projection steps, include all node positions
        if not positions_stale:
            message["node_positions"] = {
                n["id"]: n["pos"] for n in current_json.get("nodes", [])
                if n.get("pos") is not None
            }

        await self._broadcast(message)

        # Full snapshot on interval
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
                await ws.send_json(message)
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
