"""Concept Graph (Component C).

What exists:
  - ConceptNode with a non-null AffectTrace (load-bearing).
  - One-shot write on surprise, with find-or-match dedup against existing nodes.
  - Strengthen path for matched writes (activation_count, last_activated, EWMA on
    running affect trace).
  - Brute-force cosine search over an L2-normalized embedding matrix.
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

import os
# faiss-cpu ships its own libomp on macOS, which collides with the libomp
# that PyTorch / NumPy may have already loaded. The mode change is
# documented as "unsafe" by the OMP runtime but is practically harmless
# for our single-process M1 workload (no nested-parallel hot loops between
# the two libraries). Set BEFORE importing faiss so the runtime sees it.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from dataclasses import dataclass
from enum import Enum

import faiss
import numpy as np

from backend.config import AUTO_LINK_K, AUTO_LINK_MIN_COSINE, D_REP, R_MATCH


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

        # FAISS exact-IP index (Phase 7 perf fix). On L2-normalized vectors
        # inner product equals cosine similarity, so IndexFlatIP gives
        # identical results to the brute-force matmul but in vectorized C.
        # Add-only — strengthen does not change embeddings, so the index
        # stays in sync without invalidation. Rebuilt only at __init__ and
        # after MindPersistence.load.
        self._faiss_index = faiss.IndexFlatIP(D_REP)
        self._faiss_id_map: list[int] = []

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
        """Return (concept_id, cosine_similarity) for the nearest node, or
        None if the graph is empty. Caller decides whether the similarity
        is good enough for their purpose.

        Phase 7 perf fix: FAISS IndexFlatIP. On L2-normalized vectors this
        is bit-exact equivalent to the prior brute-force `_cosine_to_all`
        + argmax — same result, vectorized C kernel, sublinear in practice
        once the SIMD path lights up.
        """
        if self._faiss_index.ntotal == 0:
            return None
        rep = representation.astype(np.float32, copy=False)
        n = float(np.linalg.norm(rep))
        if n < 1e-9:
            return None
        rep_unit = (rep / n).reshape(1, -1).astype(np.float32, copy=False)
        D, I = self._faiss_index.search(rep_unit, 1)
        idx = int(I[0][0])
        if idx < 0:
            return None
        return self._faiss_id_map[idx], float(D[0][0])

    def nearest_k(
        self, representation: np.ndarray, k: int,
    ) -> list[tuple[int, float]]:
        """Return up to k (concept_id, cosine) pairs sorted descending.
        Used by the auto-link hot path so it doesn't need a full
        similarity vector — FAISS top-K is sublinear in N.
        """
        if self._faiss_index.ntotal == 0 or k <= 0:
            return []
        rep = representation.astype(np.float32, copy=False)
        n = float(np.linalg.norm(rep))
        if n < 1e-9:
            return []
        rep_unit = (rep / n).reshape(1, -1).astype(np.float32, copy=False)
        kk = min(int(k), self._faiss_index.ntotal)
        D, I = self._faiss_index.search(rep_unit, kk)
        out: list[tuple[int, float]] = []
        for d, i in zip(D[0].tolist(), I[0].tolist()):
            if i < 0:
                continue
            out.append((self._faiss_id_map[int(i)], float(d)))
        return out

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
        # Register in FAISS — kept in sync incrementally so nearest() never
        # needs to rebuild on the hot path. Strengthen path doesn't change
        # the embedding, so it never invalidates.
        n = float(np.linalg.norm(embedding))
        if n >= 1e-9:
            unit = (embedding / n).reshape(1, -1).astype(np.float32, copy=False)
            self._faiss_index.add(unit)
            self._faiss_id_map.append(cid)
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
        if self._faiss_index.ntotal <= 1:
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

    def _rebuild_faiss_index(self) -> None:
        """Rebuild the FAISS index from scratch over all current nodes.

        Called at __init__ (no-op on empty graph) and after
        MindPersistence.load restores nodes in bulk. The hot path
        (write_on_surprise) keeps the index in sync incrementally so
        runtime never needs to call this.
        """
        index = faiss.IndexFlatIP(D_REP)
        id_map: list[int] = []
        if self.nodes:
            ids = sorted(self.nodes.keys())
            rows = np.empty((len(ids), D_REP), dtype=np.float32)
            for i, cid in enumerate(ids):
                emb = self.nodes[cid].embedding
                n = float(np.linalg.norm(emb))
                rows[i] = emb / n if n >= 1e-9 else 0.0
                id_map.append(cid)
            index.add(rows)
        self._faiss_index = index
        self._faiss_id_map = id_map
