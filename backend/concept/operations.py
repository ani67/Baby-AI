"""Operations on the ConceptGraph — spreading activation, inference,
composition, random walks, analogy discovery, selection, and context."""

from __future__ import annotations

import math
import random
import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torch.nn.functional import cosine_similarity

if TYPE_CHECKING:
    from concept.graph import ConceptGraph

from concept.node import ConceptNode, TypedEdge


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class ActivationResult:
    """Output of spread_activation."""
    activated: dict[str, float]           # node_id -> activation level
    paths: dict[str, list[str]]           # node_id -> path from seed


@dataclass
class QueryIntent:
    """Parsed intent from a natural-language question."""
    target_concept: str | None = None
    target_relation: str | None = None
    query_type: str = "general"           # what_is, property, quantity, yes_no, general


@dataclass
class WalkResult:
    """Output of wander()."""
    path: list[str]                       # node ids visited
    edge_types: list[str]                 # relation of each step
    interestingness: float                # 0-1 score


# ---------------------------------------------------------------------------
# Conversation context
# ---------------------------------------------------------------------------

class ConversationContext:
    """Tracks multi-turn conversation state for pronoun resolution
    and activation decay."""

    def __init__(self) -> None:
        self.topic_stack: list[str] = []          # concept ids, most recent last
        self.recent_mentions: deque[str] = deque(maxlen=20)
        self.pronoun_map: dict[str, str] = {}     # "it" -> concept name
        self.turn_count: int = 0

    # -- public API --------------------------------------------------------

    def resolve_query(self, raw_query: str) -> str:
        """Replace pronouns with their mapped concept names."""
        resolved = raw_query
        for pronoun, name in self.pronoun_map.items():
            pattern = re.compile(rf"\b{re.escape(pronoun)}\b", re.IGNORECASE)
            resolved = pattern.sub(name, resolved)
        return resolved

    def update(
        self,
        query_concepts: list[str],
        response_concepts: list[str],
    ) -> None:
        """Update topic stack and pronoun map after a turn.

        *query_concepts* and *response_concepts* are concept names (not ids).
        """
        self.turn_count += 1

        all_mentioned = query_concepts + response_concepts
        for name in all_mentioned:
            self.recent_mentions.append(name)

        # Push the most prominent response concept onto the topic stack.
        if response_concepts:
            self.topic_stack.append(response_concepts[0])

        # Pronoun heuristics: last singular noun -> "it" / "its"
        if all_mentioned:
            last = all_mentioned[-1]
            self.pronoun_map.update({
                "it": last,
                "its": last,
                "that": last,
                "this": last,
            })
        # "they" / "them" -> first plural set if available
        if len(all_mentioned) >= 2:
            joined = " and ".join(all_mentioned[:2])
            self.pronoun_map.update({"they": joined, "them": joined})

    def apply_decay(
        self,
        graph: ConceptGraph,
        decay_factor: float = 0.3,
    ) -> None:
        """Decay all node activations by *decay_factor*.  Call once per turn."""
        for node in graph.nodes.values():
            node.activation *= (1.0 - decay_factor)
            if node.activation < 0.01:
                node.activation = 0.0


# ---------------------------------------------------------------------------
# Spreading activation
# ---------------------------------------------------------------------------

def spread_activation(
    graph: ConceptGraph,
    seeds: list[tuple[str, float]],
    max_hops: int = 3,
    damping: float = 0.5,
    threshold: float = 0.05,
    top_k: int = 20,
    edge_filter: str | None = None,
) -> ActivationResult:
    """BFS spreading activation from *seeds*.

    Each hop: ``activation * edge_strength * damping``.
    Kills activations below *threshold*.
    Returns top-K activated nodes.
    """
    activated: dict[str, float] = {}
    paths: dict[str, list[str]] = {}

    # Initialise seeds.
    frontier: deque[tuple[str, float, list[str]]] = deque()
    for node_id, strength in seeds:
        if node_id not in graph.nodes:
            continue
        activated[node_id] = strength
        paths[node_id] = [node_id]
        graph.nodes[node_id].activation = strength
        frontier.append((node_id, strength, [node_id]))

    for _ in range(max_hops):
        next_frontier: deque[tuple[str, float, list[str]]] = deque()
        while frontier:
            src_id, src_act, path = frontier.popleft()
            edges = graph.get_edges_from(src_id)
            for edge in edges:
                if edge_filter is not None and edge.relation != edge_filter:
                    continue
                new_act = src_act * edge.strength * damping
                if new_act < threshold:
                    continue
                tid = edge.target_id
                if tid not in graph.nodes:
                    continue
                if tid not in activated or new_act > activated[tid]:
                    activated[tid] = new_act
                    paths[tid] = path + [tid]
                    graph.nodes[tid].activation = max(
                        graph.nodes[tid].activation, new_act
                    )
                    next_frontier.append((tid, new_act, paths[tid]))
        frontier = next_frontier

    # Top-K selection.
    sorted_items = sorted(activated.items(), key=lambda x: x[1], reverse=True)
    top_items = sorted_items[:top_k]
    top_ids = {nid for nid, _ in top_items}

    return ActivationResult(
        activated={nid: act for nid, act in top_items},
        paths={nid: paths[nid] for nid in top_ids if nid in paths},
    )


# ---------------------------------------------------------------------------
# Typed inference
# ---------------------------------------------------------------------------

def typed_inference(
    graph: ConceptGraph,
    constraints: list[tuple[str, str, str]],
) -> list[tuple[str, float]]:
    """Constraint-based inference using the graph's reverse index.

    Each constraint is ``(variable_position, relation, fixed_value)`` where
    *variable_position* is ``"source"`` or ``"target"``.

    ``("source", "color", "red_id")`` means: find all sources X such that
    X --color--> red_id.

    Returns matching node ids with confidence scores.
    """
    candidate_sets: list[set[str]] = []

    for var_pos, relation, fixed_value in constraints:
        matches: set[str] = set()
        if var_pos == "source":
            # Use reverse_index: (relation, target) -> set(source_ids)
            key = (relation, fixed_value)
            if hasattr(graph, "reverse_index") and key in graph.reverse_index:
                matches = set(graph.reverse_index[key])
            else:
                # Fallback: scan edges
                for edge in graph.get_all_edges():
                    if edge.relation == relation and edge.target_id == fixed_value:
                        matches.add(edge.source_id)
        else:
            # var_pos == "target": find all targets where fixed_value --rel--> ?
            for edge in graph.get_edges_from(fixed_value):
                if edge.relation == relation:
                    matches.add(edge.target_id)
        candidate_sets.append(matches)

    if not candidate_sets:
        return []

    # Intersect all constraint sets.
    result_ids = candidate_sets[0]
    for s in candidate_sets[1:]:
        result_ids = result_ids & s

    # Score by node confidence.
    scored: list[tuple[str, float]] = []
    for nid in result_ids:
        node = graph.nodes.get(nid)
        score = node.confidence if node else 0.0
        scored.append((nid, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


# ---------------------------------------------------------------------------
# Composition (Option C: edge-type-guided dimension blending)
# ---------------------------------------------------------------------------

def compose(
    graph: ConceptGraph,
    base_concept_id: str,
    modifier_concept_id: str,
    modifier_relation: str | None = None,
    persist: bool = False,
) -> ConceptNode:
    """Create a composed concept by blending base + modifier vectors.

    Strategy (Option C from design):
    1. If base has an existing edge of *modifier_relation*, replace that
       direction component in the vector.
    2. Otherwise add the modifier as an offset.
    3. Return a temporary node (or persist it in the graph).
    """
    base = graph.nodes[base_concept_id]
    modifier = graph.nodes[modifier_concept_id]

    base_vec = base.vector.clone().float()
    mod_vec = modifier.vector.clone().float()

    if modifier_relation is not None:
        # Find the current target of this relation on base.
        existing_target_id: str | None = None
        for edge in graph.get_edges_from(base_concept_id):
            if edge.relation == modifier_relation:
                existing_target_id = edge.target_id
                break

        if existing_target_id is not None and existing_target_id in graph.nodes:
            # Replace: subtract old direction, add new.
            old_vec = graph.nodes[existing_target_id].vector.float()
            direction = old_vec - base_vec
            direction_norm = direction / (direction.norm() + 1e-8)
            # Remove old direction component from base.
            projection = (base_vec * direction_norm).sum() * direction_norm
            base_vec = base_vec - projection
            # Add modifier direction.
            mod_direction = mod_vec - base_vec
            mod_direction_norm = mod_direction / (mod_direction.norm() + 1e-8)
            mod_projection = (mod_vec * mod_direction_norm).sum() * mod_direction_norm
            base_vec = base_vec + 0.5 * mod_projection
        else:
            # No existing edge of this type: add offset.
            diff = mod_vec - base_vec
            base_vec = base_vec + 0.3 * diff
    else:
        # No relation specified: simple interpolation.
        base_vec = 0.7 * base_vec + 0.3 * mod_vec

    # Normalise to unit sphere.
    composed_vec = base_vec / (base_vec.norm() + 1e-8)

    base_name = base.name or base.id
    mod_name = modifier.name or modifier.id
    composed_name = f"{mod_name}_{base_name}"

    node = ConceptNode(
        vector=composed_vec,
        name=composed_name,
        modalities=base.modalities | modifier.modalities,
        confidence=min(base.confidence, modifier.confidence),
    )

    if persist:
        graph.nodes[node.id] = node
        # Link back to sources.
        rel = modifier_relation or "composed_from"
        graph.add_edge(TypedEdge(
            source_id=node.id,
            target_id=base_concept_id,
            relation="composed_from",
            strength=0.8,
        ))
        graph.add_edge(TypedEdge(
            source_id=node.id,
            target_id=modifier_concept_id,
            relation=rel,
            strength=0.8,
        ))

    return node


# ---------------------------------------------------------------------------
# Random walk (wander)
# ---------------------------------------------------------------------------

def wander(
    graph: ConceptGraph,
    start: str | None = None,
    steps: int = 20,
) -> WalkResult:
    """Weighted random walk following edges.

    Interestingness is scored by cluster crossings + vector similarity
    of distant nodes (low similarity between far-apart nodes = interesting).
    """
    if not graph.nodes:
        return WalkResult(path=[], edge_types=[], interestingness=0.0)

    if start is None:
        start = random.choice(list(graph.nodes.keys()))

    path: list[str] = [start]
    edge_types: list[str] = []
    cluster_crossings = 0

    current = start
    for _ in range(steps):
        edges = graph.get_edges_from(current)
        if not edges:
            break

        # Weighted selection by edge strength.
        weights = [e.strength for e in edges]
        total = sum(weights)
        if total == 0:
            break
        chosen = random.choices(edges, weights=weights, k=1)[0]

        next_id = chosen.target_id
        if next_id not in graph.nodes:
            break

        # Track cluster crossing.
        cur_cluster = graph.nodes[current].cluster_id
        nxt_cluster = graph.nodes[next_id].cluster_id
        if cur_cluster is not None and nxt_cluster is not None and cur_cluster != nxt_cluster:
            cluster_crossings += 1

        path.append(next_id)
        edge_types.append(chosen.relation)
        current = next_id

    # Score interestingness.
    interestingness = 0.0
    actual_steps = len(path) - 1
    if actual_steps > 0:
        # Component 1: cluster crossings normalised by steps.
        crossing_score = cluster_crossings / actual_steps

        # Component 2: vector dissimilarity between start and end.
        start_vec = graph.nodes[path[0]].vector.float().unsqueeze(0)
        end_vec = graph.nodes[path[-1]].vector.float().unsqueeze(0)
        sim = cosine_similarity(start_vec, end_vec).item()
        dissim_score = max(0.0, 1.0 - sim)  # low sim = high interest

        interestingness = 0.5 * crossing_score + 0.5 * dissim_score

    return WalkResult(
        path=path,
        edge_types=edge_types,
        interestingness=interestingness,
    )


# ---------------------------------------------------------------------------
# Analogy discovery
# ---------------------------------------------------------------------------

def discover_analogies(
    graph: ConceptGraph,
    num_walks: int = 50,
    walk_length: int = 20,
    min_pattern_length: int = 2,
    min_occurrences: int = 2,
) -> list[tuple[tuple[str, ...], list[str]]]:
    """Run random walks, extract edge-type subsequences, find repeated patterns.

    Returns list of ``(pattern_tuple, [starting_node_ids])`` for patterns
    that appear from 2+ distinct starting points.
    """
    # pattern -> set of starting node ids where it was observed
    pattern_origins: dict[tuple[str, ...], set[str]] = defaultdict(set)

    for _ in range(num_walks):
        result = wander(graph, start=None, steps=walk_length)
        if len(result.edge_types) < min_pattern_length:
            continue

        start_id = result.path[0]
        types = result.edge_types

        # Extract all contiguous subsequences of valid length.
        for length in range(min_pattern_length, len(types) + 1):
            for i in range(len(types) - length + 1):
                pattern = tuple(types[i : i + length])
                pattern_origins[pattern].add(start_id)

    # Filter to patterns with enough distinct origins.
    analogies: list[tuple[tuple[str, ...], list[str]]] = []
    for pattern, origins in pattern_origins.items():
        if len(origins) >= min_occurrences:
            analogies.append((pattern, sorted(origins)))

    # Sort by number of origins (most common patterns first).
    analogies.sort(key=lambda x: len(x[1]), reverse=True)
    return analogies


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def select(
    graph: ConceptGraph,
    activated: dict[str, float],
    query_vector: Tensor,
    query_intent: QueryIntent | None = None,
    top_k: int = 10,
) -> list[tuple[str, float]]:
    """Layered selection: activation x query_sim x intent_boost x recency.

    Combines multiple signals to rank activated nodes for response
    generation.
    """
    now = int(time.time())
    scored: list[tuple[str, float]] = []
    query_vec = query_vector.float().unsqueeze(0)

    for nid, act_level in activated.items():
        node = graph.nodes.get(nid)
        if node is None:
            continue

        # Query similarity: 0.5 + 0.5 * cosine_sim (range 0.0 to 1.0).
        node_vec = node.vector.float().unsqueeze(0)
        sim = cosine_similarity(query_vec, node_vec).item()
        query_factor = 0.5 + 0.5 * sim

        # Intent boost: 1.5x if node has an edge matching the target relation.
        intent_boost = 1.0
        if query_intent is not None and query_intent.target_relation is not None:
            for edge in graph.get_edges_from(nid):
                if edge.relation == query_intent.target_relation:
                    intent_boost = 1.5
                    break

        # Recency: exponential decay over time (half-life ~1 hour).
        age = max(0, now - node.last_accessed)
        recency = math.exp(-age / 3600.0)

        score = act_level * query_factor * intent_boost * recency
        scored.append((nid, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


# ---------------------------------------------------------------------------
# Query intent parsing
# ---------------------------------------------------------------------------

# Relation keywords for pattern-based intent extraction.
_RELATION_PATTERNS: list[tuple[str, str]] = [
    (r"\bwhat\s+colou?r\b", "color"),
    (r"\bwhat\s+size\b", "size"),
    (r"\bwhat\s+shape\b", "shape"),
    (r"\bwhere\b", "location"),
    (r"\bwhen\b", "temporal"),
    (r"\bwho\b", "agent"),
    (r"\bwhy\b", "cause"),
    (r"\bhow\s+many\b", "quantity"),
    (r"\bhow\s+much\b", "quantity"),
    (r"\bhow\s+big\b", "size"),
    (r"\bpart\s+of\b", "part_of"),
    (r"\bmade\s+of\b", "material"),
    (r"\bkind\s+of\b", "is_a"),
    (r"\btype\s+of\b", "is_a"),
]


def parse_query_intent(question: str) -> QueryIntent:
    """Simple pattern-based intent extraction from a question string."""
    q = question.lower().strip()

    # Check relation-specific patterns first.
    for pattern, relation in _RELATION_PATTERNS:
        if re.search(pattern, q):
            return QueryIntent(
                target_relation=relation,
                query_type="property",
            )

    # General patterns.
    if re.search(r"\bwhat\s+is\b", q):
        return QueryIntent(query_type="what_is")

    if re.search(r"\bis\s+(it|there|this|that)\b", q):
        return QueryIntent(query_type="yes_no")

    if re.search(r"\b(can|does|do|did|will|would|should|could)\b", q):
        return QueryIntent(query_type="yes_no")

    if re.search(r"\blist\b|\bname\b|\bwhich\b", q):
        return QueryIntent(query_type="enumerate")

    return QueryIntent(query_type="general")
