"""
Quadtree-based neural graph with MPS-accelerated forward pass.

Each tile holds a 64×64 float32 identity texture (cluster weights as pixels).
Tiles subdivide when internal variance exceeds threshold.
Tiles collapse when children stay similar for 500+ steps.
Nearest-neighbor lookup is O(log n) via quadtree traversal.

API surface is identical to the flat-list Graph it replaces.
"""

from __future__ import annotations

import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F

from .cluster import Cluster
from .node import Node


# ---------------------------------------------------------------------------
# MPS helpers
# ---------------------------------------------------------------------------

_PHI = 0.618033988749895  # golden ratio conjugate for hashing
_MAX_QUADTREE_DEPTH = 32  # prevent infinite recursion on hash collisions


def _mps_available() -> bool:
    """Check if MPS is available without selecting it."""
    try:
        if torch.backends.mps.is_available():
            print("[mps] device available: True", flush=True)
            return True
    except Exception:
        pass
    print("[mps] device available: False", flush=True)
    return False


def _check_tensor(t: torch.Tensor, label: str) -> torch.Tensor:
    """Raise on NaN/inf — never silently fall back."""
    if torch.isnan(t).any() or torch.isinf(t).any():
        raise RuntimeError(f"[mps] ERROR: {label} contains NaN/inf")
    return t


# ---------------------------------------------------------------------------
# ID parsing helper
# ---------------------------------------------------------------------------

_ID_NUM_RE = re.compile(r"\d+")


def _id_to_num(cluster_id: str) -> int:
    """Extract the first integer from a cluster ID.
    Handles 'c_00', 'c_00a', 'c_12b', etc."""
    m = _ID_NUM_RE.search(cluster_id)
    return int(m.group()) if m else 0


# ---------------------------------------------------------------------------
# Edge (unchanged)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Quadtree tile
# ---------------------------------------------------------------------------

TILE_SIZE = 64  # 64×64 identity texture


@dataclass
class QuadTile:
    """
    One tile in the quadtree.  Holds a 64×64 identity texture that encodes
    the cluster's weight fingerprint as pixel values.

    Bounds define the tile's position in a normalised [0,1)×[0,1) layer space.
    """
    x0: float  # left
    y0: float  # top
    x1: float  # right
    y1: float  # bottom
    depth: int = 0

    # Identity texture — 64×64 float32, lazily allocated on device
    _texture: Optional[torch.Tensor] = field(default=None, repr=False)

    # The cluster stored at this leaf (None for interior nodes)
    cluster: Optional[Cluster] = field(default=None, repr=False)

    # Children (NW, NE, SW, SE) — None for leaf nodes
    children: Optional[list["QuadTile"]] = field(default=None, repr=False)

    # Variance tracking for split/collapse decisions
    _variance_history: deque = field(
        default_factory=lambda: deque(maxlen=600), repr=False
    )
    _similarity_streak: int = 0  # consecutive steps all children are similar

    @property
    def is_leaf(self) -> bool:
        return self.children is None

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2

    def contains(self, px: float, py: float) -> bool:
        return self.x0 <= px < self.x1 and self.y0 <= py < self.y1

    # -- texture management --------------------------------------------------

    def build_texture(self, device: torch.device) -> torch.Tensor:
        """
        Encode the cluster identity vector into a 64×64 texture.
        The 512-dim identity is tiled across 4096 pixels (512 values × 8 repeats).
        """
        if self.cluster is None:
            self._texture = torch.zeros(TILE_SIZE, TILE_SIZE, dtype=torch.float32, device=device)
        else:
            identity = self.cluster.identity.to(device)
            # Tile 512-dim vector into 4096 pixels (64×64)
            repeated = identity.repeat(TILE_SIZE * TILE_SIZE // identity.shape[0] + 1)[
                : TILE_SIZE * TILE_SIZE
            ]
            self._texture = repeated.reshape(TILE_SIZE, TILE_SIZE)
        return self._texture

    def get_texture(self, device: torch.device) -> torch.Tensor:
        if self._texture is None:
            return self.build_texture(device)
        return self._texture

    def invalidate_texture(self) -> None:
        self._texture = None

    # -- variance / split / collapse -----------------------------------------

    def compute_variance(self) -> float:
        """Internal variance of the identity texture."""
        if self._texture is None:
            return 0.0
        return self._texture.var().item()

    def record_variance(self) -> float:
        v = self.compute_variance()
        self._variance_history.append(v)
        return v

    def should_split(self, threshold: float) -> bool:
        """True if leaf, has a cluster, and variance exceeds threshold."""
        if not self.is_leaf or self.cluster is None:
            return False
        if self.depth >= _MAX_QUADTREE_DEPTH:
            return False
        return self.compute_variance() > threshold

    def should_collapse(self) -> bool:
        """True if interior node and all children have been similar for 500+ steps."""
        if self.is_leaf or self.children is None:
            return False
        return self._similarity_streak >= 500

    def update_similarity_streak(self, threshold: float = 0.05) -> None:
        """
        For interior nodes: check if all children textures are similar.
        Uses pairwise max-abs-diff < threshold.
        """
        if self.is_leaf or self.children is None:
            return
        textures = []
        for child in self.children:
            if child._texture is not None:
                textures.append(child._texture)
        if len(textures) < 2:
            self._similarity_streak = 0
            return
        all_similar = True
        for i in range(len(textures)):
            for j in range(i + 1, len(textures)):
                if (textures[i] - textures[j]).abs().max().item() > threshold:
                    all_similar = False
                    break
            if not all_similar:
                break
        if all_similar:
            self._similarity_streak += 1
        else:
            self._similarity_streak = 0


# ---------------------------------------------------------------------------
# Graph — quadtree + same API
# ---------------------------------------------------------------------------

class Graph:
    """Registry for all clusters and edges, backed by a quadtree of tiles."""

    def __init__(self, split_variance_threshold: float = 0.1):
        # Public state — same contract as the old flat Graph
        self.clusters: list[Cluster] = []
        self.edges: list[Edge] = []

        # Internal bookkeeping
        self._cluster_index: dict[str, Cluster] = {}
        self._node_counter: int = 0
        self._cluster_counter: int = 0

        # Quadtree
        self._root: QuadTile = QuadTile(x0=0.0, y0=0.0, x1=1.0, y1=1.0, depth=0)
        self._tile_index: dict[str, QuadTile] = {}  # cluster_id → tile
        self._split_threshold: float = split_variance_threshold

        # Device (MPS if available, else CPU — benchmarked on first forward)
        self._mps_available: bool = _mps_available()
        self._device: torch.device = torch.device("cpu")  # start on CPU, benchmark picks winner
        self._device_benchmarked: bool = False
        self._forward_step: int = 0
        self._latency_history: deque = deque(maxlen=10)
        self._slow_consecutive: int = 0

    # ------------------------------------------------------------------ #
    #  Cluster CRUD — same signatures as before                           #
    # ------------------------------------------------------------------ #

    def add_cluster(self, cluster: Cluster, source: str = "unknown") -> None:
        self.clusters.append(cluster)
        self._cluster_index[cluster.id] = cluster

        # Insert into quadtree
        pos = self._cluster_to_xy(cluster)
        tile = self._insert_into_tree(self._root, cluster, pos[0], pos[1])
        self._tile_index[cluster.id] = tile
        tile.build_texture(self._device)

        total = len(self.clusters)
        print(
            f"[cluster_create] id={cluster.id} layer={cluster.layer_index} "
            f"source={source} total={total}",
            flush=True,
        )

    def remove_cluster(self, cluster_id: str) -> None:
        cluster = self._cluster_index.pop(cluster_id, None)
        if cluster:
            self.clusters = [c for c in self.clusters if c.id != cluster_id]
            self.edges = [
                e for e in self.edges
                if e.from_id != cluster_id and e.to_id != cluster_id
            ]
            # Remove from quadtree
            tile = self._tile_index.pop(cluster_id, None)
            if tile:
                tile.cluster = None
                tile.invalidate_texture()

    def get_cluster(self, cluster_id: str) -> Cluster | None:
        return self._cluster_index.get(cluster_id)

    # ------------------------------------------------------------------ #
    #  Edge operations — unchanged                                        #
    # ------------------------------------------------------------------ #

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
        return [
            e for e in self.edges
            if e.to_id == cluster_id
            or (e.direction == "bidirectional" and e.from_id == cluster_id)
        ]

    def outgoing_edges(self, cluster_id: str) -> list[Edge]:
        result = []
        for e in self.edges:
            if e.from_id == cluster_id and e.to_id != cluster_id:
                result.append(e)
            elif (
                e.direction == "bidirectional"
                and e.to_id == cluster_id
                and e.from_id != cluster_id
            ):
                result.append(
                    Edge(
                        from_id=cluster_id,
                        to_id=e.from_id,
                        strength=e.strength,
                        age=e.age,
                        direction=e.direction,
                        steps_since_activation=e.steps_since_activation,
                    )
                )
        return result

    # ------------------------------------------------------------------ #
    #  Cluster queries                                                    #
    # ------------------------------------------------------------------ #

    def entry_clusters(self) -> list[Cluster]:
        return [c for c in self.clusters if c.layer_index == 0 and not c.dormant]

    def top_layer_clusters(self) -> list[Cluster]:
        if not self.clusters:
            return []
        max_layer = max(c.layer_index for c in self.clusters if not c.dormant)
        return [
            c for c in self.clusters
            if c.layer_index == max_layer and not c.dormant
        ]

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

    # ------------------------------------------------------------------ #
    #  Structural mutations                                               #
    # ------------------------------------------------------------------ #

    def replace_cluster(
        self, old: Cluster, new_clusters: list[Cluster], source: str = "bud"
    ) -> None:
        """Replace old cluster with new clusters, transferring external edges."""
        for nc in new_clusters:
            for node in nc.nodes:
                node.cluster_id = nc.id
            self.add_cluster(nc, source=source)

        new_edges = []
        for e in self.edges:
            if e.from_id == old.id:
                for nc in new_clusters:
                    new_edges.append(
                        Edge(
                            from_id=nc.id,
                            to_id=e.to_id,
                            strength=e.strength,
                            direction=e.direction,
                        )
                    )
            elif e.to_id == old.id:
                for nc in new_clusters:
                    new_edges.append(
                        Edge(
                            from_id=e.from_id,
                            to_id=nc.id,
                            strength=e.strength,
                            direction=e.direction,
                        )
                    )
            else:
                new_edges.append(e)
        self.edges = new_edges

        # Remove old cluster
        self._cluster_index.pop(old.id, None)
        self.clusters = [c for c in self.clusters if c.id != old.id]
        tile = self._tile_index.pop(old.id, None)
        if tile:
            tile.cluster = None
            tile.invalidate_texture()

    def insert_cluster_between(
        self, before: Cluster, new: Cluster, after: Cluster
    ) -> None:
        for node in new.nodes:
            node.cluster_id = new.id
        self.add_cluster(new, source="insert")
        self.edges = [
            e
            for e in self.edges
            if not (
                (e.from_id == before.id and e.to_id == after.id)
                or (e.from_id == after.id and e.to_id == before.id)
            )
        ]
        self.add_edge(before.id, new.id, strength=0.5)
        self.add_edge(new.id, after.id, strength=0.5)

    # ------------------------------------------------------------------ #
    #  ID generators                                                      #
    # ------------------------------------------------------------------ #

    def next_node_id(self) -> str:
        nid = self._node_counter
        self._node_counter += 1
        return f"n_{nid:03d}"

    def next_cluster_id(self) -> str:
        cid = self._cluster_counter
        self._cluster_counter += 1
        return f"c_{cid:02d}"

    # ------------------------------------------------------------------ #
    #  Device benchmark — run on first forward, pick faster device        #
    # ------------------------------------------------------------------ #

    def _benchmark_device(self, tile_count: int) -> None:
        """Run mps_forward-equivalent on both CPU and MPS with synthetic
        data matching current tile count. Pick whichever is faster."""
        if not self._mps_available:
            self._device = torch.device("cpu")
            self._device_benchmarked = True
            print(f"[mps] using device=cpu (MPS not available)", flush=True)
            return

        WARMUP = 2
        TRIALS = 5
        n = max(tile_count, 4)
        input_vec = torch.randn(TILE_SIZE * TILE_SIZE)

        results: dict[str, float] = {}
        for dev_name in ("cpu", "mps"):
            dev = torch.device(dev_name)
            inp = input_vec.to(dev).reshape(TILE_SIZE, TILE_SIZE)
            tiles = [torch.randn(TILE_SIZE, TILE_SIZE, device=dev) for _ in range(n)]

            # Warmup
            for _ in range(WARMUP):
                for start in range(0, n, 32):
                    chunk = torch.stack(tiles[start:start + 32])
                    _ = (chunk * inp.unsqueeze(0)).sum(dim=(1, 2))
                if dev_name == "mps":
                    torch.mps.synchronize()

            # Timed trials
            times = []
            for _ in range(TRIALS):
                t0 = time.perf_counter()
                for start in range(0, n, 32):
                    chunk = torch.stack(tiles[start:start + 32])
                    dots = (chunk * inp.unsqueeze(0)).sum(dim=(1, 2))
                    norms = chunk.reshape(chunk.shape[0], -1).norm(dim=1)
                    _ = dots / (norms * inp.reshape(-1).norm() + 1e-8)
                if dev_name == "mps":
                    torch.mps.synchronize()
                times.append((time.perf_counter() - t0) * 1000)

            median = sorted(times)[len(times) // 2]
            results[dev_name] = median
            print(f"[mps] benchmark {dev_name}: {median:.2f}ms (tiles={n})", flush=True)

        if results["mps"] < results["cpu"]:
            self._device = torch.device("mps")
            print(f"[mps] using device=mps (faster for {n} tiles)", flush=True)
        else:
            self._device = torch.device("cpu")
            print(f"[mps] using device=cpu (faster for {n} tiles)", flush=True)

        self._device_benchmarked = True

    # ------------------------------------------------------------------ #
    #  MPS forward pass — dot product on identity textures                #
    # ------------------------------------------------------------------ #

    def mps_forward(self, input_vec: torch.Tensor) -> dict[str, float]:
        """
        Compute dot-product similarity between input and every active
        cluster's identity texture.  Runs on MPS (or CPU fallback).

        Returns dict of cluster_id → similarity score.
        """
        self._forward_step += 1

        # Benchmark on first call to pick fastest device for current tile count
        if not self._device_benchmarked:
            active_count = sum(1 for c in self.clusters if not c.dormant)
            self._benchmark_device(active_count)
            # Rebuild textures on chosen device
            for tile in self._tile_index.values():
                tile.invalidate_texture()

        t0 = time.perf_counter()

        try:
            # Build input texture: tile 512-dim vector into 64×64
            inp = input_vec.to(self._device)
            _check_tensor(inp, "input_vec")

            if self._forward_step == 1 or self._forward_step % 100 == 0:
                print(
                    f"[mps] called — device={self._device} step={self._forward_step}",
                    flush=True,
                )

            inp_flat = inp.repeat(TILE_SIZE * TILE_SIZE // inp.shape[0] + 1)[
                : TILE_SIZE * TILE_SIZE
            ]
            input_texture = inp_flat.reshape(TILE_SIZE, TILE_SIZE)

            # Gather all active tile textures into a batch
            active_ids: list[str] = []
            textures: list[torch.Tensor] = []
            for cluster in self.clusters:
                if cluster.dormant:
                    continue
                tile = self._tile_index.get(cluster.id)
                if tile is None:
                    continue
                tex = tile.get_texture(self._device)
                _check_tensor(tex, f"texture({cluster.id})")
                textures.append(tex)
                active_ids.append(cluster.id)

            if not textures:
                return {}

            # Chunked dot product — process tiles in batches of 32
            # to keep individual MPS dispatches small and avoid memory pressure
            CHUNK = 32
            norm_input = input_texture.reshape(-1).norm()
            input_flat = input_texture.unsqueeze(0)  # (1, 64, 64)

            if self._forward_step == 1 or self._forward_step % 100 == 0:
                print(
                    f"[mps] called — device={input_texture.device} "
                    f"tiles={len(textures)} chunks={(len(textures) + CHUNK - 1) // CHUNK}",
                    flush=True,
                )

            all_sims: list[float] = []
            for start in range(0, len(textures), CHUNK):
                chunk = torch.stack(textures[start : start + CHUNK])  # (<=32, 64, 64)
                dots = (chunk * input_flat).sum(dim=(1, 2))
                _check_tensor(dots, "dot_products")
                norms_chunk = chunk.reshape(chunk.shape[0], -1).norm(dim=1)
                denom = norms_chunk * norm_input + 1e-8
                sims = (dots / denom).cpu()
                _check_tensor(sims, "similarities")
                all_sims.extend(sims.tolist())

            result = {
                cid: all_sims[i] for i, cid in enumerate(active_ids)
            }

        except RuntimeError as exc:
            if "NaN" in str(exc) or "inf" in str(exc) or "mps" in str(exc).lower():
                print(f"[mps] ERROR: {exc}", flush=True)
            raise

        # Latency tracking
        latency_ms = (time.perf_counter() - t0) * 1000
        self._latency_history.append(latency_ms)

        if latency_ms > 50:
            self._slow_consecutive += 1
        else:
            self._slow_consecutive = 0

        if self._slow_consecutive >= 10:
            print(
                f"[mps] WARNING: latency degradation — "
                f"{self._slow_consecutive} consecutive steps > 50ms",
                flush=True,
            )

        if self._forward_step % 100 == 0:
            try:
                alloc = torch.mps.current_allocated_memory() / (1024 * 1024)
            except Exception:
                alloc = -1
            print(
                f"[mps] step={self._forward_step} latency={latency_ms:.1f}ms "
                f"allocated={alloc:.1f}MB",
                flush=True,
            )

        return result

    # ------------------------------------------------------------------ #
    #  Quadtree: nearest-neighbor lookup — O(log n)                       #
    # ------------------------------------------------------------------ #

    def nearest_cluster(
        self, layer_index: float, position_hint: float = 0.5
    ) -> Cluster | None:
        """
        Find the nearest cluster to a given layer position via quadtree
        traversal.  O(log n) instead of O(n) scan.
        """
        px, py = self._layer_pos_to_xy(layer_index, position_hint)
        best = self._nearest_in_subtree(self._root, px, py, best_so_far=None)
        return best[0] if best else None

    def clusters_near(
        self, layer_index: float, position_hint: float = 0.5, radius: float = 0.1
    ) -> list[Cluster]:
        """Return all clusters within radius of the query point."""
        px, py = self._layer_pos_to_xy(layer_index, position_hint)
        results: list[tuple[Cluster, float]] = []
        self._range_query(self._root, px, py, radius, results)
        results.sort(key=lambda t: t[1])
        return [c for c, _ in results]

    # ------------------------------------------------------------------ #
    #  Quadtree maintenance — split / collapse / refresh textures         #
    # ------------------------------------------------------------------ #

    def maintain_quadtree(self) -> None:
        """
        Walk the tree once per call:
          - rebuild textures for all occupied leaves (keeps them fresh)
          - split leaves whose variance exceeds threshold
          - collapse interior nodes whose children have been similar 500+ steps
        """
        self._maintain_subtree(self._root)

        # Every 500 forward steps, log the 5 clusters closest to splitting
        if self._forward_step > 0 and self._forward_step % 500 == 0:
            variances: list[tuple[str, float]] = []
            for cid, tile in self._tile_index.items():
                if tile.cluster is not None and tile._texture is not None:
                    variances.append((cid, tile.compute_variance()))
            variances.sort(key=lambda t: t[1], reverse=True)
            top5 = variances[:5]
            print(
                f"[variance] step={self._forward_step} threshold={self._split_threshold} "
                f"top5={[(cid, round(v, 4)) for cid, v in top5]}",
                flush=True,
            )

    def _maintain_subtree(self, tile: QuadTile) -> None:
        if tile.is_leaf:
            if tile.cluster is not None:
                tile.build_texture(self._device)
                tile.record_variance()
                if tile.should_split(self._split_threshold):
                    self._split_tile(tile)
        else:
            for child in tile.children:
                self._maintain_subtree(child)
            tile.update_similarity_streak()
            if tile.should_collapse():
                self._collapse_tile(tile)

    def _split_tile(self, tile: QuadTile) -> None:
        """Subdivide a leaf into 4 children.  The cluster stays in the
        child whose quadrant contains its position."""
        if tile.depth >= _MAX_QUADTREE_DEPTH:
            return
        mx, my = tile.cx, tile.cy
        tile.children = [
            QuadTile(x0=tile.x0, y0=tile.y0, x1=mx, y1=my, depth=tile.depth + 1),  # NW
            QuadTile(x0=mx, y0=tile.y0, x1=tile.x1, y1=my, depth=tile.depth + 1),   # NE
            QuadTile(x0=tile.x0, y0=my, x1=mx, y1=tile.y1, depth=tile.depth + 1),   # SW
            QuadTile(x0=mx, y0=my, x1=tile.x1, y1=tile.y1, depth=tile.depth + 1),   # SE
        ]
        if tile.cluster is not None:
            cluster = tile.cluster
            pos = self._cluster_to_xy(cluster)
            placed = False
            for child in tile.children:
                if child.contains(pos[0], pos[1]):
                    child.cluster = cluster
                    child.build_texture(self._device)
                    self._tile_index[cluster.id] = child
                    placed = True
                    break
            if not placed:
                tile.children[0].cluster = cluster
                tile.children[0].build_texture(self._device)
                self._tile_index[cluster.id] = tile.children[0]
            tile.cluster = None
            tile.invalidate_texture()
        print(
            f"[quadtree] split tile depth={tile.depth} "
            f"bounds=({tile.x0:.2f},{tile.y0:.2f})-({tile.x1:.2f},{tile.y1:.2f})",
            flush=True,
        )

    def _collapse_tile(self, tile: QuadTile) -> None:
        """Merge children back into parent leaf.  Keep the first cluster found."""
        clusters_in_children: list[Cluster] = []
        self._collect_clusters(tile, clusters_in_children)
        tile.children = None
        tile._similarity_streak = 0
        if clusters_in_children:
            tile.cluster = clusters_in_children[0]
            tile.build_texture(self._device)
            self._tile_index[clusters_in_children[0].id] = tile
            for c in clusters_in_children[1:]:
                pos = self._cluster_to_xy(c)
                new_tile = self._insert_into_tree(self._root, c, pos[0], pos[1])
                self._tile_index[c.id] = new_tile
                new_tile.build_texture(self._device)
        print(
            f"[quadtree] collapse tile depth={tile.depth} "
            f"clusters={len(clusters_in_children)}",
            flush=True,
        )

    def _collect_clusters(
        self, tile: QuadTile, out: list[Cluster]
    ) -> None:
        if tile.cluster is not None:
            out.append(tile.cluster)
        if tile.children:
            for child in tile.children:
                self._collect_clusters(child, out)

    # ------------------------------------------------------------------ #
    #  Quadtree coordinate mapping                                        #
    # ------------------------------------------------------------------ #

    def _cluster_to_xy(self, cluster: Cluster) -> tuple[float, float]:
        """Map a cluster to a stable (x, y) in [0,1)×[0,1) quadtree space.

        Uses a fixed sigmoid mapping for y (layer_index) so that adding new
        layers does NOT shift the coordinates of existing clusters.
        x uses a deterministic hash of the full cluster ID string, so IDs
        like 'c_00a' work correctly without int-parsing.
        """
        # y: sigmoid-like mapping — layer 0 → ~0.12, layer 5 → ~0.73
        y = 1.0 / (1.0 + 2.0 ** (-cluster.layer_index + 2))

        # x: FNV-1a-inspired hash for good distribution on short strings
        h = 0x811c9dc5
        for ch in cluster.id:
            h = ((h ^ ord(ch)) * 0x01000193) & 0xFFFFFFFF
        x = (h * _PHI) % 1.0

        x = min(max(x, 0.001), 0.999)
        y = min(max(y, 0.001), 0.999)
        return (x, y)

    def _layer_pos_to_xy(
        self, layer_index: float, position_hint: float
    ) -> tuple[float, float]:
        """Convert a (layer, hint) query into quadtree coordinates."""
        y = 1.0 / (1.0 + 2.0 ** (-layer_index + 2))
        x = min(max(position_hint, 0.001), 0.999)
        y = min(max(y, 0.001), 0.999)
        return (x, y)

    def _insert_into_tree(
        self, tile: QuadTile, cluster: Cluster, px: float, py: float
    ) -> QuadTile:
        """Insert cluster into the correct leaf of the quadtree."""
        if tile.is_leaf:
            if tile.cluster is None:
                tile.cluster = cluster
                return tile
            # Leaf is occupied — split first, then insert
            if tile.depth >= _MAX_QUADTREE_DEPTH:
                # At max depth, just co-locate (store in a new empty child)
                # by forcing a split that places both clusters
                self._split_tile(tile)
                if tile.is_leaf:
                    # split was blocked — co-locate in this tile
                    # (shouldn't happen, but defensive)
                    tile.cluster = cluster
                    return tile
                return self._insert_into_tree(tile, cluster, px, py)
            self._split_tile(tile)
            return self._insert_into_tree(tile, cluster, px, py)

        # Interior node — recurse into correct child
        for child in tile.children:
            if child.contains(px, py):
                return self._insert_into_tree(child, cluster, px, py)

        # Fallback: first child
        return self._insert_into_tree(tile.children[0], cluster, px, py)

    def _nearest_in_subtree(
        self,
        tile: QuadTile,
        px: float,
        py: float,
        best_so_far: Optional[tuple[Cluster, float]],
    ) -> Optional[tuple[Cluster, float]]:
        """Recursive nearest-neighbor search in the quadtree."""
        if tile.is_leaf:
            if tile.cluster is not None and not tile.cluster.dormant:
                dist = (tile.cx - px) ** 2 + (tile.cy - py) ** 2
                if best_so_far is None or dist < best_so_far[1]:
                    return (tile.cluster, dist)
            return best_so_far

        # Interior: visit children, nearest-first
        children_by_dist = sorted(
            tile.children,
            key=lambda c: (c.cx - px) ** 2 + (c.cy - py) ** 2,
        )
        for child in children_by_dist:
            if best_so_far is not None:
                closest_x = max(child.x0, min(px, child.x1))
                closest_y = max(child.y0, min(py, child.y1))
                box_dist = (closest_x - px) ** 2 + (closest_y - py) ** 2
                if box_dist > best_so_far[1]:
                    continue
            best_so_far = self._nearest_in_subtree(child, px, py, best_so_far)
        return best_so_far

    def _range_query(
        self,
        tile: QuadTile,
        px: float,
        py: float,
        radius: float,
        results: list[tuple[Cluster, float]],
    ) -> None:
        """Collect all clusters within radius of (px, py)."""
        closest_x = max(tile.x0, min(px, tile.x1))
        closest_y = max(tile.y0, min(py, tile.y1))
        if (closest_x - px) ** 2 + (closest_y - py) ** 2 > radius ** 2:
            return

        if tile.is_leaf:
            if tile.cluster is not None and not tile.cluster.dormant:
                dist = ((tile.cx - px) ** 2 + (tile.cy - py) ** 2) ** 0.5
                if dist <= radius:
                    results.append((tile.cluster, dist))
        else:
            for child in tile.children:
                self._range_query(child, px, py, radius, results)

    # ------------------------------------------------------------------ #
    #  Serialization — same output format as before                       #
    # ------------------------------------------------------------------ #

    def to_json(self) -> dict:
        """Full serializable representation of the current graph."""
        nodes_json = []
        for c in self.clusters:
            for n in c.nodes:
                nodes_json.append(
                    {
                        "id": n.id,
                        "cluster": c.id,
                        "activation_mean": n.mean_activation,
                        "activation_variance": n.activation_variance,
                        "age_steps": n.age,
                        "pos": getattr(n, "pos", None),
                        "alive": n.alive,
                        "plasticity": n.plasticity,
                    }
                )

        clusters_json = []
        for c in self.clusters:
            clusters_json.append(
                {
                    "id": c.id,
                    "cluster_type": c.cluster_type,
                    "density": c._compute_internal_density(),
                    "node_count": len(c.nodes),
                    "layer_index": c.layer_index,
                    "label": None,
                    "dormant": c.dormant,
                    "plasticity": c.plasticity,
                    "age": c.age,
                }
            )

        edges_json = []
        for e in self.edges:
            edges_json.append(
                {
                    "from": e.from_id,
                    "to": e.to_id,
                    "strength": e.strength,
                    "age_steps": e.age,
                    "direction": e.direction,
                    "steps_since_activation": e.steps_since_activation,
                }
            )

        return {
            "nodes": nodes_json,
            "clusters": clusters_json,
            "edges": edges_json,
        }

    def summary(self) -> dict:
        active_tiles = sum(
            1 for t in self._tile_index.values() if t.cluster and not t.cluster.dormant
        )
        return {
            "cluster_count": len(self.clusters),
            "node_count": sum(len(c.nodes) for c in self.clusters),
            "edge_count": len(self.edges),
            "dormant_count": sum(1 for c in self.clusters if c.dormant),
            "layer_count": len(
                set(c.layer_index for c in self.clusters if not c.dormant)
            ),
            "active_tiles": active_tiles,
            "quadtree_depth": self._max_depth(self._root),
        }

    def _max_depth(self, tile: QuadTile) -> int:
        if tile.is_leaf:
            return tile.depth
        return max(self._max_depth(c) for c in tile.children)

    def to_tree_json(self) -> dict:
        """Serialize the quadtree as a flat list of nodes with parent pointers.

        Only includes tiles that contain a cluster or are ancestors of one.
        This keeps the response small even for deep trees.

        Returns::

            {
                "nodes": [
                    {
                        "id": "tile_0",
                        "parent": null,
                        "depth": 0,
                        "cluster_id": "c_00"|null,
                        "dormant": false,
                        "cluster_type": "integration"|null,
                        "child_index": 0,
                    },
                    ...
                ],
                "max_depth": 7,
            }
        """
        nodes: list[dict] = []
        self._walk_tree_json(self._root, parent_id=None, child_index=0, nodes=nodes, counter=[0])
        return {
            "nodes": nodes,
            "max_depth": self._max_depth(self._root),
        }

    def _subtree_has_cluster(self, tile: QuadTile) -> bool:
        """Return True if this tile or any descendant contains a cluster."""
        if tile.cluster is not None:
            return True
        if tile.children:
            return any(self._subtree_has_cluster(c) for c in tile.children)
        return False

    def _walk_tree_json(
        self,
        tile: QuadTile,
        parent_id: str | None,
        child_index: int,
        nodes: list[dict],
        counter: list[int],
    ) -> None:
        # Skip entire subtrees that contain no clusters
        if not self._subtree_has_cluster(tile):
            return
        tid = f"tile_{counter[0]}"
        counter[0] += 1
        cluster_id = tile.cluster.id if tile.cluster else None
        dormant = tile.cluster.dormant if tile.cluster else False
        cluster_type = tile.cluster.cluster_type if tile.cluster else None
        nodes.append({
            "id": tid,
            "parent": parent_id,
            "depth": tile.depth,
            "cluster_id": cluster_id,
            "dormant": dormant,
            "cluster_type": cluster_type,
            "child_index": child_index,
        })
        if tile.children:
            for i, child in enumerate(tile.children):
                self._walk_tree_json(child, tid, i, nodes, counter)
