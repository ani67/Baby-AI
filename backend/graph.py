"""Concept Graph (Component C).

What exists:
  - ConceptNode with a non-null AffectTrace (load-bearing).
  - One-shot write on surprise, with find-or-match dedup against existing nodes.
  - Strengthen path for matched writes (activation_count, last_activated, EWMA on
    running affect trace).
  - Brute-force cosine search over an L2-normalized embedding matrix
    (kept as fallback). Phase 7 perf fix routes nearest() through FAISS.
  - The full 10-type edge taxonomy (is_a, has_property, causes, precedes,
    similar_to, opposite_of, context_of, refers_to, expresses, part_of).
    Edges carry source_id, target_id, type, weight, created_at, last_activated,
    activation_count. add_edge / get_edges / strengthen_edge are exposed.

What does NOT exist (deferred to later phases):
  - Spreading activation (no caller until F lands in Phase 3).
  - Abstraction formation, forget loop, persistence.
  - peak_state / peak_magnitude on AffectTrace.
  - encoder_id on ConceptNode (multi-encoder swap is later).
  - Auto edge laying inside write_on_surprise — Phase 3 hooks F to provide
    `context_active`; until then edges are added only by explicit caller.
"""
from __future__ import annotations

# Phase 8 (v0.7): faiss-cpu is gone; the index is now PyTorch matmul
# on MPS. The OMP env vars stay as belt-and-braces against any future
# multi-libomp dep regression.
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from dataclasses import dataclass
from enum import Enum

import numpy as np
import torch

from backend.config import (
    AUTO_LINK_K,
    AUTO_LINK_MIN_COSINE,
    CONCEPT_CEILING,
    D_REP,
    PRUNE_TO,
    R_MATCH,
)


# ============================================================
# MPSConceptIndex — replaces FAISS IndexFlatIP
# ============================================================
#
# Three problems FAISS gave us, all solved by torch:
#   1. faiss-cpu's libomp co-loaded with torch's libomp killed the api
#      at cd.generate() (silent kernel kill, no Python traceback). With
#      torch as the only matmul lib, libomp is single-vendor.
#   2. faiss IndexFlatIP at small-to-medium N (5-50K) was actually
#      slower per-query than numpy BLAS on M1 (BLAS wins on small
#      matrices; the FAISS speedup for incremental write+search was
#      from avoiding full rebuild on every dirty bit, not from the
#      search itself).
#   3. faiss-cpu doesn't use the GPU. torch can pin the matrix to MPS
#      and do batched matmul on the M1 unified-memory GPU at no
#      transfer cost.
#
# Same incremental-add semantics as the old FAISS-backed version:
# write_on_surprise calls add(); strengthen path is a no-op (embeddings
# don't change). _remove_node calls remove() during prune; rebuild()
# fires once after MindPersistence.load to populate from saved nodes.
# ============================================================
class MPSConceptIndex:
    """Cosine-similarity index over a torch tensor on MPS (with CPU
    fallback when MPS isn't available). Inputs are L2-normalized at add
    time so search is plain inner-product matmul.
    """

    def __init__(self, d_rep: int = D_REP) -> None:
        self._device = (
            torch.device("mps")
            if torch.backends.mps.is_available()
            else torch.device("cpu")
        )
        self._d_rep = d_rep
        self._matrix: torch.Tensor | None = None   # (N, D) on device
        self._id_map: list[int] = []               # row index → concept_id

    @property
    def ntotal(self) -> int:
        return len(self._id_map)

    @property
    def device(self) -> str:
        return str(self._device)

    def add(self, embedding: np.ndarray, concept_id: int) -> None:
        vec = torch.tensor(
            embedding, dtype=torch.float32, device=self._device,
        ).unsqueeze(0)
        if self._matrix is None:
            self._matrix = vec
        else:
            self._matrix = torch.cat([self._matrix, vec], dim=0)
        self._id_map.append(int(concept_id))

    def remove(self, concept_id: int) -> None:
        try:
            idx = self._id_map.index(int(concept_id))
        except ValueError:
            return
        self._id_map.pop(idx)
        if self._matrix is None:
            return
        keep = list(range(self._matrix.shape[0]))
        keep.pop(idx)
        if keep:
            self._matrix = self._matrix[keep]
        else:
            self._matrix = None

    def remove_many(self, concept_ids: set[int]) -> None:
        """Bulk remove. Single mask + index, much cheaper than N calls
        to remove() during a prune that drops thousands of nodes."""
        if not concept_ids:
            return
        if self._matrix is None:
            self._id_map = [c for c in self._id_map if c not in concept_ids]
            return
        keep_indices = [
            i for i, cid in enumerate(self._id_map)
            if cid not in concept_ids
        ]
        if not keep_indices:
            self._matrix = None
            self._id_map = []
            return
        self._matrix = self._matrix[keep_indices]
        self._id_map = [self._id_map[i] for i in keep_indices]

    def search(self, query: np.ndarray, k: int = 1) -> tuple[list[float], list[int]]:
        if self._matrix is None or not self._id_map:
            return [], []
        q = torch.tensor(
            query, dtype=torch.float32, device=self._device,
        ).unsqueeze(0)
        sims = (self._matrix @ q.T).squeeze(1)
        kk = min(int(k), len(self._id_map))
        top_sims, top_idx = sims.topk(kk)
        return (
            top_sims.cpu().tolist(),
            [self._id_map[int(i)] for i in top_idx.cpu().tolist()],
        )

    def search_batch(
        self, queries: np.ndarray, k: int = 1,
    ) -> list[tuple[float, int]]:
        if self._matrix is None or not self._id_map:
            return [(0.0, -1)] * int(np.asarray(queries).shape[0])
        Q = torch.tensor(queries, dtype=torch.float32, device=self._device)
        sims = Q @ self._matrix.T
        best_sims, best_idx = sims.max(dim=1)
        bs = best_sims.cpu().tolist()
        bi = best_idx.cpu().tolist()
        return [(float(bs[i]), self._id_map[int(bi[i])]) for i in range(len(bs))]

    def rebuild(self, nodes: dict) -> None:
        if not nodes:
            self._matrix = None
            self._id_map = []
            return
        ids = sorted(nodes.keys())
        rows = np.empty((len(ids), self._d_rep), dtype=np.float32)
        for i, cid in enumerate(ids):
            emb = nodes[cid].embedding
            n = float(np.linalg.norm(emb))
            rows[i] = emb / n if n >= 1e-9 else 0.0
        self._matrix = torch.tensor(rows, dtype=torch.float32, device=self._device)
        self._id_map = ids

    def to_numpy_matrix(self) -> np.ndarray | None:
        if self._matrix is None:
            return None
        return self._matrix.cpu().numpy().astype(np.float32, copy=False)

    @classmethod
    def from_numpy_matrix(
        cls, matrix: np.ndarray, id_map: list[int], d_rep: int = D_REP,
    ) -> "MPSConceptIndex":
        idx = cls(d_rep=d_rep)
        if matrix is not None and len(id_map) > 0:
            idx._matrix = torch.tensor(matrix, dtype=torch.float32, device=idx._device)
            idx._id_map = list(id_map)
        return idx


class EdgeType(Enum):
    """The closed 10-type edge taxonomy from SYNTHESIS.

    New types require explicit migration — runtime-invented types break query
    stability for B, D, E, F, G, H.
    """
    IS_A         = "is_a"          # taxonomy / abstraction
    HAS_PROPERTY = "has_property"  # attribution
    CAUSES       = "causes"        # directional causal claim
    PRECEDES     = "precedes"      # temporal order
    SIMILAR_TO   = "similar_to"    # symmetric associative tie
    OPPOSITE_OF  = "opposite_of"   # symmetric contrastive
    CONTEXT_OF   = "context_of"    # situational co-occurrence
    REFERS_TO    = "refers_to"     # indexicality (E uses for self-reference)
    EXPRESSES    = "expresses"     # concept → surface form (G's hot path)
    PART_OF      = "part_of"       # mereology, distinct from is_a


# Default weight for a freshly-added edge. Leaves room for strengthen passes
# to grow it toward 1.0 as the relationship is observed repeatedly.
DEFAULT_EDGE_WEIGHT = 0.5

# Strengthen rate per traversal — bounded growth to 1.0.
EDGE_STRENGTHEN_RATE = 0.05


class SpreadMode(Enum):
    """The angle through which spreading activation is queried.

    Affects which edge types are weighted up. Phase 3 minimal: all modes
    accepted, all edge types weighted equally; per-mode tables land later.
    """
    PERCEIVE = "perceive"   # input recognition
    PREDICT  = "predict"    # mid-thought anticipation
    SIMULATE = "simulate"   # rollout under D


# Spread tunables — Phase 3 minimal defaults. Synthesis flags these as
# empirically calibrated; first-run values are placeholders.
SPREAD_K_BASE        = 4     # min top_k at saturating arousal
SPREAD_K_AROUSAL     = 28    # additional top_k as arousal → 0
SPREAD_PROPAGATE_MIN = 0.05  # edge must clear this (weight) to propagate
SPREAD_ACTIVATE_MIN  = 0.05  # post-propagation activation cutoff
SPREAD_DEPTH_DAMP    = 0.6   # per-step path-length decay
SPREAD_TRAVERSED_CAP = 64    # top traversed edges retained in result


@dataclass
class SpreadResult:
    """Output of a single spreading-activation invocation.

    active_set       — concept_id → max activation seen at that node.
                       Includes seeds (clamped to their seed values) plus
                       any concept whose propagated activation crossed
                       SPREAD_ACTIVATE_MIN.
    traversed_edges  — top SPREAD_TRAVERSED_CAP propagation events by
                       activation, each ((src, dst, type), propagated).
                       F uses this for HabitPath tracking.
    """
    active_set: dict[int, float]
    traversed_edges: list[tuple[tuple[int, int, "EdgeType"], float]]


@dataclass
class AffectTrace:
    """The affective imprint of a concept.

    Phase 1 carries only the two vectors the success condition exercises:
      birth_state    — composite affect at write time. Immutable.
      running_state  — EWMA over composite affect on each strengthen event.
    """
    birth_state: np.ndarray        # float32[N_AFF]
    running_state: np.ndarray      # float32[N_AFF]
    last_affect_update: float      # wall-clock epoch seconds


@dataclass
class ConceptNode:
    concept_id: int
    name: str
    embedding: np.ndarray          # float32[D_REP], stored at full precision
    affect_trace: AffectTrace
    activation_count: int
    last_activated: float
    created_at: float
    surprise_at_birth: float


@dataclass
class Edge:
    """A typed, directed, weighted connection between two ConceptNodes.

    Phase 2 carries exactly the seven fields in the brief. Synthesis-locked
    fields not yet present (edge_id, confidence, affect_at_birth,
    last_traversed) will be added when the caller (D for replay, F for
    spreading) actually requires them — we don't fake fields no one reads.
    """
    source_id: int
    target_id: int
    type: EdgeType
    weight: float
    created_at: float
    last_activated: float          # epoch seconds; equals created_at until first strengthen
    activation_count: int          # 0 at creation; bumped on each strengthen


class ConceptGraph:
    """In-memory concept store with brute-force nearest-neighbor + typed edges.

    Single-writer assumed. Node mutations go through `write_on_surprise`;
    edge mutations through `add_edge` / `strengthen_edge`.
    """

    def __init__(self) -> None:
        self.nodes: dict[int, ConceptNode] = {}
        self._next_concept_id: int = 1

        # Edge storage: keyed by (src, dst, type). Src/dst pair may have many
        # edges of different types — the type is part of the identity.
        self._edges: dict[tuple[int, int, EdgeType], Edge] = {}
        # Out- and in-adjacency indexes for O(1) neighborhood queries.
        self._out_edges: dict[int, list[Edge]] = {}
        self._in_edges:  dict[int, list[Edge]] = {}

        # Pins — concept_ids immune to forgetting. Maps cid → reason for
        # diagnostic; the forget loop (Phase 3+) checks `is_pinned`. E
        # maintains its own richer PinRecord registry (with timestamps and
        # reason enums); C just enforces the immune-set contract.
        self._pins: dict[int, str] = {}

        # Lazily rebuilt L2-normalized matrix used for cosine search.
        # Kept around as a fallback for callers that need the full
        # similarity vector (rare; mostly the LM trainer's top-K lookup).
        # _matrix_ids[i] is the concept_id whose unit-vector lives at row i.
        self._matrix: np.ndarray | None = None
        self._matrix_ids: list[int] = []
        self._matrix_dirty: bool = True

        # MPSConceptIndex (Phase 8 / v0.7) replaces FAISS. Single torch-
        # backed index works equally in api, curriculum, and trainer
        # processes — no libomp conflict. Always initialised; api too.
        self._index = MPSConceptIndex(d_rep=D_REP)

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def edge_count(self) -> int:
        return len(self._edges)

    @property
    def pin_count(self) -> int:
        return len(self._pins)

    # ---- queries ----

    def nearest(self, representation: np.ndarray) -> tuple[int, float] | None:
        """Return (concept_id, cosine_similarity) for the nearest node,
        or None if the graph is empty. Goes through MPSConceptIndex —
        torch matmul on the M1 unified-memory GPU, flat curve up to
        N=50K+."""
        if not self.nodes or self._index.ntotal == 0:
            return None
        rep = representation.astype(np.float32, copy=False)
        n = float(np.linalg.norm(rep))
        if n < 1e-9:
            return None
        sims, ids = self._index.search(rep / n, k=1)
        if not sims:
            return None
        return int(ids[0]), float(sims[0])

    def nearest_k(
        self, representation: np.ndarray, k: int,
    ) -> list[tuple[int, float]]:
        """Top-k (concept_id, cosine) descending."""
        if not self.nodes or k <= 0 or self._index.ntotal == 0:
            return []
        rep = representation.astype(np.float32, copy=False)
        n = float(np.linalg.norm(rep))
        if n < 1e-9:
            return []
        sims, ids = self._index.search(rep / n, k=int(k))
        return [(int(ids[i]), float(sims[i])) for i in range(len(ids))]

    def find_or_match(self, representation: np.ndarray, threshold: float = R_MATCH) -> int | None:
        """Return concept_id of the nearest node with cosine ≥ threshold, else None."""
        result = self.nearest(representation)
        if result is None:
            return None
        cid, sim = result
        return cid if sim >= threshold else None

    # ---- writes ----

    def write_on_surprise(
        self,
        representation: np.ndarray,
        surprise: float,
        current_affect: np.ndarray,
        name_hint: str,
        now: float,
        force_write: bool = False,
    ) -> tuple[int, bool]:
        """One-shot write or strengthen-existing-match.

        Returns (concept_id, is_new). is_new=False indicates an existing node
        within R_MATCH was strengthened instead of a new node being created.

        `force_write=True` bypasses the find_or_match dedup. Phase 4 sleep
        uses this so an abstraction-parent's centroid embedding doesn't
        silently merge back into one of its own members (centroids of
        tightly clustered unit vectors typically sit within R_MATCH of the
        cluster).

        Phase 1/2 callers leave force_write at the default False.
        """
        if not force_write:
            match = self.find_or_match(representation, threshold=R_MATCH)
            if match is not None:
                self._strengthen(match, current_affect, now)
                return match, False

        cid = self._next_concept_id
        self._next_concept_id += 1

        embedding = representation.astype(np.float32, copy=True)
        affect = current_affect.astype(np.float32, copy=True)

        self.nodes[cid] = ConceptNode(
            concept_id=cid,
            name=name_hint,
            embedding=embedding,
            affect_trace=AffectTrace(
                birth_state=affect.copy(),
                running_state=affect.copy(),
                last_affect_update=now,
            ),
            activation_count=1,
            last_activated=now,
            created_at=now,
            surprise_at_birth=float(surprise),
        )
        self._matrix_dirty = True
        # Register in MPSConceptIndex — incremental add so nearest() never
        # has to rebuild on the hot path. Strengthen path doesn't change
        # the embedding, so it never invalidates.
        n = float(np.linalg.norm(embedding))
        if n >= 1e-9:
            self._index.add(
                (embedding / n).astype(np.float32, copy=False), cid,
            )
        return cid, True

    # ---- edges ----

    def add_edge(
        self,
        source_id: int,
        target_id: int,
        type: EdgeType,
        weight: float = DEFAULT_EDGE_WEIGHT,
        now: float = 0.0,
    ) -> Edge:
        """Create a typed directed edge from source to target.

        Idempotent: if an edge with the same (source_id, target_id, type)
        triple already exists, it is returned unchanged. Strengthening an
        existing edge is `strengthen_edge`'s job — keeping `add_edge` pure
        means callers can re-assert structural relationships safely.

        Raises KeyError if either concept does not exist.
        """
        if source_id not in self.nodes:
            raise KeyError(f"source concept {source_id} not in graph")
        if target_id not in self.nodes:
            raise KeyError(f"target concept {target_id} not in graph")
        if not isinstance(type, EdgeType):
            raise TypeError(f"type must be EdgeType, got {type!r}")

        key = (source_id, target_id, type)
        existing = self._edges.get(key)
        if existing is not None:
            return existing

        edge = Edge(
            source_id=source_id,
            target_id=target_id,
            type=type,
            weight=float(weight),
            created_at=float(now),
            last_activated=float(now),
            activation_count=0,
        )
        self._edges[key] = edge
        self._out_edges.setdefault(source_id, []).append(edge)
        self._in_edges.setdefault(target_id, []).append(edge)
        return edge

    def get_edges(
        self,
        concept_id: int,
        type: EdgeType | None = None,
        direction: str = "out",
    ) -> list[Edge]:
        """Return edges incident on `concept_id`.

        direction: 'out' (default) = edges where concept_id is source;
                   'in'            = edges where concept_id is target;
                   'both'          = union of the above (duplicates removed).
        type: if provided, filter to that EdgeType. None returns all types.
        """
        if direction == "out":
            edges = list(self._out_edges.get(concept_id, ()))
        elif direction == "in":
            edges = list(self._in_edges.get(concept_id, ()))
        elif direction == "both":
            seen: set[tuple[int, int, EdgeType]] = set()
            edges = []
            for src in (self._out_edges.get(concept_id, ()), self._in_edges.get(concept_id, ())):
                for e in src:
                    k = (e.source_id, e.target_id, e.type)
                    if k not in seen:
                        seen.add(k)
                        edges.append(e)
        else:
            raise ValueError(f"direction must be 'out' / 'in' / 'both', got {direction!r}")

        if type is not None:
            edges = [e for e in edges if e.type == type]
        return edges

    def strengthen_edge(
        self,
        source_id: int,
        target_id: int,
        type: EdgeType,
        now: float,
    ) -> Edge:
        """Bump weight (capped at 1.0), refresh last_activated, increment
        activation_count. Raises KeyError if the edge does not exist —
        strengthening a missing edge is always a caller bug.
        """
        key = (source_id, target_id, type)
        edge = self._edges.get(key)
        if edge is None:
            raise KeyError(f"no edge {key!r} in graph")
        edge.weight = min(1.0, edge.weight + EDGE_STRENGTHEN_RATE)
        edge.last_activated = float(now)
        edge.activation_count += 1
        return edge

    # ---- pinning (E owns the policy; C enforces the contract) ----

    def pin(self, concept_id: int, reason: str = "") -> None:
        """Mark a concept immune to forgetting. Idempotent. Raises KeyError
        if the concept does not exist (you cannot pin a phantom).
        """
        if concept_id not in self.nodes:
            raise KeyError(f"cannot pin missing concept {concept_id}")
        self._pins[concept_id] = reason

    def unpin(self, concept_id: int) -> None:
        """Remove the pin if present. Silent no-op otherwise."""
        self._pins.pop(concept_id, None)

    def is_pinned(self, concept_id: int) -> bool:
        return concept_id in self._pins

    # ---- forgetting / pruning (synthesis "forgetting is curation") ----

    def prune_to_ceiling(self, now: float) -> int:
        """Remove weakest concepts until node_count <= PRUNE_TO. Called
        from MainLoop.sleep after _form_abstractions (post-merge so
        abstraction parents are protected by their inbound IS_A edges).
        Returns the number of nodes removed.

        Pruning criteria (weakest first), multiplicative score:

            score = activation_count
                  * affect_trace_magnitude
                  * max(edge_count, 1)
                  * surprise_at_birth

        A factor of zero anywhere collapses the score: a never-activated
        concept (activation_count == 0 — but the write path sets it to
        1, so this is mostly a guard), a flat-affect concept, an
        edgeless concept, or a low-surprise birth all sink. The product
        ranks "everything important" above "any one missing factor."

        Immune from pruning:
          - pinned concepts
          - concepts incident on any IS_A edge (parents AND members of
            abstractions; once clustered, the cluster is load-bearing)
        """
        if self.node_count <= CONCEPT_CEILING:
            return 0

        # ---- immunity sets ----
        protected: set[int] = set(self._pins.keys())
        for edge in self._edges.values():
            if edge.type is EdgeType.IS_A:
                protected.add(edge.source_id)   # member of an abstraction
                protected.add(edge.target_id)   # abstraction parent

        # ---- score remaining concepts ----
        # Out + in degree per node, computed once.
        out_deg: dict[int, int] = {}
        in_deg: dict[int, int] = {}
        for e in self._edges.values():
            out_deg[e.source_id] = out_deg.get(e.source_id, 0) + 1
            in_deg[e.target_id] = in_deg.get(e.target_id, 0) + 1

        candidates: list[tuple[float, int]] = []
        for cid, node in self.nodes.items():
            if cid in protected:
                continue
            edge_count = out_deg.get(cid, 0) + in_deg.get(cid, 0)
            affect_magnitude = float(np.linalg.norm(
                node.affect_trace.running_state
            ))
            score = (
                float(node.activation_count)
                * affect_magnitude
                * max(edge_count, 1)
                * float(node.surprise_at_birth)
            )
            candidates.append((score, cid))

        if not candidates:
            return 0

        # Ascending score — weakest first.
        candidates.sort(key=lambda x: x[0])

        # Drop until we hit PRUNE_TO. Never overflow into the immune set.
        n_to_drop = min(self.node_count - PRUNE_TO, len(candidates))
        drop_set: set[int] = {cid for _, cid in candidates[:n_to_drop]}

        # Bulk-remove from the MPS index (single mask op) — much cheaper
        # than calling self._index.remove(cid) per-cid.
        self._index.remove_many(drop_set)

        # Drop nodes + their incident edges. Use a single edge-key sweep
        # since we know all the dropped cids up front.
        for cid in drop_set:
            self.nodes.pop(cid, None)
            self._pins.pop(cid, None)
        survived: dict[tuple[int, int, EdgeType], Edge] = {}
        for k, e in self._edges.items():
            if e.source_id in drop_set or e.target_id in drop_set:
                continue
            survived[k] = e
        self._edges = survived
        # Rebuild adjacency indexes from survivors (single pass).
        self._out_edges = {}
        self._in_edges = {}
        for e in self._edges.values():
            self._out_edges.setdefault(e.source_id, []).append(e)
            self._in_edges.setdefault(e.target_id, []).append(e)

        # Dense matrix invalidated.
        self._matrix = None
        self._matrix_ids = []
        self._matrix_dirty = True

        return len(drop_set)

    def _remove_node(self, cid: int) -> None:
        """Remove one concept and its incident edges. Used in tests +
        single-cid removal paths. The bulk-prune path inside
        prune_to_ceiling does NOT route through here — it handles the
        sweep itself for performance."""
        if cid not in self.nodes:
            return
        del self.nodes[cid]
        self._pins.pop(cid, None)

        keys_to_remove = [
            k for k in self._edges
            if k[0] == cid or k[1] == cid
        ]
        for k in keys_to_remove:
            del self._edges[k]
        self._out_edges = {}
        self._in_edges = {}
        for e in self._edges.values():
            self._out_edges.setdefault(e.source_id, []).append(e)
            self._in_edges.setdefault(e.target_id, []).append(e)

        # Drop from the MPS index.
        self._index.remove(cid)
        # Dense matrix invalidated.
        self._matrix_dirty = True

    # ---- auto-linking (Phase 4 — densifies the graph as concepts arrive) ----

    def link_to_nearest_neighbors(
        self,
        concept_id: int,
        *,
        top_k: int = AUTO_LINK_K,
        min_cosine: float = AUTO_LINK_MIN_COSINE,
        edge_type: EdgeType = EdgeType.SIMILAR_TO,
        now: float = 0.0,
    ) -> list[Edge]:
        """Lay symmetric similar_to edges from `concept_id` to its top-K
        cosine-nearest existing neighbors above `min_cosine`. Skips self,
        existing edges of the same type are returned unchanged (idempotent
        via add_edge). Edge weight = the cosine itself (clamped to [0, 1]).

        Pure structural augmentation — no affect side effects, no surprise.
        Used after any concept write so the graph builds horizontal density
        as it grows. Without this every new concept is an island.

        Phase 7 perf fix: uses FAISS `nearest_k(top_k+1)` instead of
        scanning the full similarity vector. The +1 reserves one slot
        for self (write_on_surprise registers the new embedding before
        this method runs, so the FAISS top-1 result is the new node
        itself). top_k defaults to AUTO_LINK_K=1.
        """
        if concept_id not in self.nodes:
            return []
        # Skip if there's nothing else to link to.
        if self._index.ntotal <= 1:
            return []

        rep = self.nodes[concept_id].embedding
        # Pull top_k + 1 from FAISS — the freshly-added concept itself
        # almost always sits at rank 0 with sim≈1.0; we filter it out.
        candidates = self.nearest_k(rep, top_k + 1)
        out: list[Edge] = []
        for peer_id, sim in candidates:
            if peer_id == concept_id:
                continue
            if sim < min_cosine:
                break
            weight = max(0.0, min(1.0, sim))
            out.append(self.add_edge(concept_id, peer_id, edge_type, weight=weight, now=now))
            out.append(self.add_edge(peer_id, concept_id, edge_type, weight=weight, now=now))
            if len(out) >= top_k * 2:
                break
        return out

    # ---- spreading activation ----

    def spread(
        self,
        seeds: dict[int, float],
        affect,
        composite_affect: np.ndarray,
        arousal: float,
        max_steps: int = 3,
        budget: int = 256,
        mode: SpreadMode = SpreadMode.PERCEIVE,
    ) -> SpreadResult:
        """Affect-gated spreading activation from a set of seeds.

        Sparsity envelope (top_k expansions per step) is set by arousal:
        high arousal → small top_k (narrow, intense); low arousal → large
        top_k (broad, diffuse). Per-step path-length damping prevents
        runaway. budget caps total node touches across all steps.

        For each candidate traversal, A.gate_attention is called to combine
        affect alignment, semantic relevance, and predictive prior into one
        scalar gate. The propagated activation is `seed_activation × gate ×
        edge_weight × type_weight × depth_decay`. If it clears
        SPREAD_ACTIVATE_MIN, the destination joins the next-step active set.

        Returns a SpreadResult with the final active set and the top
        SPREAD_TRAVERSED_CAP traversal events for habit-tracking by F.

        Phase 3 minimal:
          - All edge types weighted equally; mode is accepted but doesn't
            differentiate yet (per-mode tables land later).
          - Predictive score is fixed at 1.0; F's predict_prior dict will
            raise it once B advertises priors.
          - No spread_restricted variant yet.
        """
        if not seeds or not self.nodes:
            return SpreadResult({}, [])

        # Sparsity envelope from arousal — clamped so 0 ≤ arousal ≤ 1
        # shapes top_k between SPREAD_K_BASE and SPREAD_K_BASE+SPREAD_K_AROUSAL.
        a = min(1.0, max(0.0, arousal))
        top_k = int(round(SPREAD_K_BASE + SPREAD_K_AROUSAL * (1.0 - a)))

        # active_set holds the max activation seen at each concept across all steps.
        active: dict[int, float] = {cid: float(v) for cid, v in seeds.items() if cid in self.nodes}
        # frontier is the working set for the current step.
        frontier: dict[int, float] = dict(active)
        visited_for_expansion: set[int] = set()
        traversed: list[tuple[tuple[int, int, EdgeType], float]] = []
        budget_remaining = int(budget)

        for step in range(max_steps):
            if budget_remaining <= 0 or not frontier:
                break

            # Take the top_k most-active frontier nodes for expansion this step.
            candidates = sorted(frontier.items(), key=lambda x: -x[1])[:top_k]
            next_frontier: dict[int, float] = {}

            for src_id, src_activation in candidates:
                if src_id in visited_for_expansion:
                    continue
                visited_for_expansion.add(src_id)

                out = self._out_edges.get(src_id)
                if not out:
                    continue

                # Walk outbound edges, strongest first, until budget runs out.
                for edge in sorted(out, key=lambda e: -e.weight):
                    if budget_remaining <= 0:
                        break
                    if edge.weight < SPREAD_PROPAGATE_MIN:
                        continue

                    type_w = self._type_weight(edge.type, mode)
                    if type_w <= 0.0:
                        continue

                    dst_id = edge.target_id
                    dst_node = self.nodes.get(dst_id)
                    if dst_node is None:
                        continue

                    # Phase 3 minimal: semantic score is the edge's own weight.
                    # Predictive score is the synthesis-locked baseline 1.0.
                    semantic_score = edge.weight
                    predictive_score = 1.0
                    gate = affect.gate_attention(
                        dst_node.affect_trace,
                        composite_affect,
                        arousal,
                        semantic_score,
                        predictive_score,
                    )

                    decay = SPREAD_DEPTH_DAMP ** step
                    propagated = src_activation * gate * type_w * decay

                    budget_remaining -= 1

                    if propagated >= SPREAD_ACTIVATE_MIN:
                        prev = next_frontier.get(dst_id, 0.0)
                        if propagated > prev:
                            next_frontier[dst_id] = float(propagated)
                        traversed.append(((src_id, dst_id, edge.type), float(propagated)))

            # Merge next_frontier into active (max activation across steps),
            # then make it the new frontier.
            for dst_id, val in next_frontier.items():
                if val > active.get(dst_id, 0.0):
                    active[dst_id] = val
            frontier = next_frontier

        # Cap traversed_edges to top by activation magnitude.
        traversed.sort(key=lambda t: -t[1])
        traversed = traversed[:SPREAD_TRAVERSED_CAP]

        # Final active set: drop anything that didn't clear ACTIVATE_MIN, but
        # always retain seed values regardless of magnitude (they are inputs).
        final: dict[int, float] = {}
        for cid, val in active.items():
            if cid in seeds or val >= SPREAD_ACTIVATE_MIN:
                final[cid] = val

        return SpreadResult(active_set=final, traversed_edges=traversed)

    @staticmethod
    def _type_weight(edge_type: EdgeType, mode: SpreadMode) -> float:
        """Per-mode edge-type weighting. Phase 3 minimal: all types equal.

        Synthesis spec calls for PERCEIVE to favor similar_to / context_of /
        has_property, PREDICT to favor causes / precedes / is_a, and
        SIMULATE to favor causes / precedes — all of which only matter
        once enough heterogeneous edges exist to differentiate.
        """
        return 1.0

    # ---- internals ----

    def _strengthen(self, concept_id: int, current_affect: np.ndarray, now: float) -> None:
        node = self.nodes[concept_id]
        node.activation_count += 1
        node.last_activated = now

        # EWMA on running affect, alpha matched to A's mood timescale (slow).
        # Synthesis lists mood half-life = 7200 s; one observation per ~minute
        # means alpha ≈ 0.05 maps cleanly to mood-band integration.
        alpha = 0.05
        running = node.affect_trace.running_state
        delta = current_affect.astype(np.float32) - running
        running += alpha * delta
        node.affect_trace.last_affect_update = now

    def _cosine_to_all(self, representation: np.ndarray) -> np.ndarray:
        if self._matrix_dirty or self._matrix is None:
            self._rebuild_matrix()
        if self._matrix is None:
            return np.zeros(0, dtype=np.float32)

        rep = representation.astype(np.float32, copy=False)
        rep_norm = float(np.linalg.norm(rep))
        if rep_norm < 1e-9:
            return np.zeros(len(self._matrix_ids), dtype=np.float32)
        return self._matrix @ (rep / rep_norm)

    def _rebuild_matrix(self) -> None:
        if not self.nodes:
            self._matrix = None
            self._matrix_ids = []
            self._matrix_dirty = False
            return

        ids = sorted(self.nodes.keys())
        rows = np.empty((len(ids), self.nodes[ids[0]].embedding.shape[0]), dtype=np.float32)
        for i, cid in enumerate(ids):
            emb = self.nodes[cid].embedding
            n = float(np.linalg.norm(emb))
            rows[i] = emb / n if n >= 1e-9 else 0.0
        self._matrix = rows
        self._matrix_ids = ids
        self._matrix_dirty = False

    def _rebuild_index(self) -> None:
        """Rebuild the MPS concept index over all current nodes. Called
        from MindPersistence._restore_graph after the bulk node load.
        Single GPU upload; flat curve to N=50K+ on M1 Pro.
        """
        self._index.rebuild(self.nodes)

    # Backwards-compat alias — older code paths call _rebuild_faiss_index
    # (run_curriculum.load_or_construct, ingest_book, scripts/prune_graph,
    # diagnostic scripts). Keep one name reachable for now.
    _rebuild_faiss_index = _rebuild_index
