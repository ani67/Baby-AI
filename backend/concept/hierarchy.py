"""Hierarchy module — emergent clustering and multi-level management for ConceptGraph."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import torch
from torch import Tensor

from concept.node import ConceptNode, TypedEdge

if __import__("typing").TYPE_CHECKING:
    from concept.graph import ConceptGraph


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------

@dataclass
class HierarchyLevel:
    level: int
    nodes: list[str]
    parent_map: dict[str, str]  # child_id -> parent_id


@dataclass
class HierarchyTree:
    levels: list[HierarchyLevel]
    root: str | None


# ------------------------------------------------------------------
# K-means (pure torch, no external deps)
# ------------------------------------------------------------------

def _kmeans_torch(
    vectors: Tensor,
    k: int,
    max_iter: int = 20,
) -> Tensor:
    """Run k-means on (N, D) matrix. Returns (N,) int tensor of assignments."""
    n, d = vectors.shape
    k = min(k, n)

    # Initialize centroids via k-means++ seeding.
    indices = [torch.randint(n, (1,)).item()]
    for _ in range(1, k):
        dists = torch.cdist(vectors, vectors[indices])  # (N, len(indices))
        min_dists = dists.min(dim=1).values              # (N,)
        probs = min_dists / (min_dists.sum() + 1e-12)
        idx = torch.multinomial(probs, 1).item()
        indices.append(idx)

    centroids = vectors[indices].clone()  # (k, D)

    assignments = torch.zeros(n, dtype=torch.long)
    for _ in range(max_iter):
        # Assign each point to nearest centroid.
        dists = torch.cdist(vectors, centroids)  # (N, k)
        assignments = dists.argmin(dim=1)         # (N,)

        # Update centroids.
        new_centroids = torch.zeros_like(centroids)
        counts = torch.zeros(k)
        for c in range(k):
            mask = assignments == c
            if mask.any():
                new_centroids[c] = vectors[mask].mean(dim=0)
                counts[c] = mask.sum()
            else:
                # Empty cluster: re-seed to a random point.
                new_centroids[c] = vectors[torch.randint(n, (1,)).item()]
                counts[c] = 0

        # Early stop if centroids converged.
        shift = (new_centroids - centroids).norm(dim=1).max().item()
        centroids = new_centroids
        if shift < 1e-6:
            break

    return assignments


# ------------------------------------------------------------------
# cluster_concepts
# ------------------------------------------------------------------

def cluster_concepts(
    graph: ConceptGraph,
    gamma: float = 1.0,
    min_cluster_size: int = 3,
) -> dict[str, list[str]]:
    """Cluster all concept nodes via torch k-means.

    Returns {cluster_label: [node_ids]}.
    gamma scales the number of clusters (higher = more clusters).
    """
    node_ids = list(graph._nodes.keys())
    n = len(node_ids)
    if n == 0:
        return {}

    # Build normalized vector matrix.
    vecs = torch.stack([graph._nodes[nid].vector for nid in node_ids])
    vecs = vecs / (vecs.norm(dim=1, keepdim=True) + 1e-8)

    # Determine k.
    k = max(5, int(math.sqrt(n / 2) * gamma))
    k = min(k, 100, n)

    if n <= min_cluster_size:
        return {"cluster_0": node_ids}

    assignments = _kmeans_torch(vecs, k, max_iter=20)

    # Group by cluster.
    raw_clusters: dict[int, list[str]] = defaultdict(list)
    for i, cid in enumerate(assignments.tolist()):
        raw_clusters[cid].append(node_ids[i])

    # Post-process: merge small clusters into nearest neighbor cluster.
    # Compute cluster centroids first.
    centroid_map: dict[int, Tensor] = {}
    for cid, members in raw_clusters.items():
        member_vecs = torch.stack([graph._nodes[m].vector for m in members])
        centroid_map[cid] = member_vecs.mean(dim=0)
        centroid_map[cid] = centroid_map[cid] / (centroid_map[cid].norm() + 1e-8)

    large_clusters = {c for c, m in raw_clusters.items() if len(m) >= min_cluster_size}
    small_clusters = {c for c, m in raw_clusters.items() if len(m) < min_cluster_size}

    # If ALL clusters are small, just keep them as-is (no valid merge target).
    if large_clusters:
        large_ids = sorted(large_clusters)
        large_centroids = torch.stack([centroid_map[c] for c in large_ids])

        for sc in small_clusters:
            sc_centroid = centroid_map[sc]
            sims = large_centroids @ sc_centroid
            best_idx = sims.argmax().item()
            target = large_ids[best_idx]
            raw_clusters[target].extend(raw_clusters[sc])
            del raw_clusters[sc]

    # Build output with string keys.
    result: dict[str, list[str]] = {}
    for i, (_, members) in enumerate(sorted(raw_clusters.items())):
        result[f"cluster_{i}"] = members

    return result


# ------------------------------------------------------------------
# create_manager_nodes
# ------------------------------------------------------------------

def create_manager_nodes(
    graph: ConceptGraph,
    clusters: dict[str, list[str]],
) -> list[tuple[str, Tensor, list[str]]]:
    """Create a manager ConceptNode for each cluster.

    Returns list of (manager_id, manager_vector, member_ids).
    """
    managers: list[tuple[str, Tensor, list[str]]] = []

    for cluster_label, member_ids in clusters.items():
        if not member_ids:
            continue

        # Compute centroid.
        member_vecs = torch.stack([graph._nodes[m].vector for m in member_ids])
        centroid = member_vecs.mean(dim=0)
        centroid = centroid / (centroid.norm() + 1e-8)

        # Most central concept (nearest to centroid).
        sims = member_vecs @ centroid
        central_idx = sims.argmax().item()
        central_node = graph._nodes[member_ids[central_idx]]
        cluster_name = central_node.name or member_ids[central_idx]

        manager_id = f"mgr_{cluster_name}"

        # Create manager node in graph.
        manager_node = ConceptNode(
            id=manager_id,
            vector=centroid,
            name=f"[{cluster_name}]",
            modalities={"abstract"},
            confidence=0.8,
            observation_count=len(member_ids),
            cluster_id=None,
        )
        graph._nodes[manager_id] = manager_node
        if manager_node.name:
            graph._name_index[manager_node.name].append(manager_id)
        graph._append_vector(manager_id, centroid)

        # Assign cluster_id on members.
        for mid in member_ids:
            node = graph._nodes.get(mid)
            if node:
                node.cluster_id = manager_id

        # Add part_of edges from members to manager.
        for mid in member_ids:
            graph.add_edge(mid, manager_id, "part_of", strength=0.7)

        managers.append((manager_id, centroid, member_ids))

    # Build manager-to-manager edges based on external connectivity.
    manager_member_set: dict[str, set[str]] = {}
    member_to_manager: dict[str, str] = {}
    for mgr_id, _, mids in managers:
        manager_member_set[mgr_id] = set(mids)
        for mid in mids:
            member_to_manager[mid] = mgr_id

    for mgr_id, _, mids in managers:
        # Count external edges by target manager.
        external_relations: dict[str, Counter] = defaultdict(Counter)
        for mid in mids:
            for edge in graph.get_edges(mid, direction="outgoing"):
                target_mgr = member_to_manager.get(edge.target_id)
                if target_mgr and target_mgr != mgr_id:
                    external_relations[target_mgr][edge.relation] += 1

        # Top-3 relations per target manager.
        for target_mgr, rel_counts in external_relations.items():
            for rel, count in rel_counts.most_common(3):
                strength = min(1.0, count * 0.1)
                graph.add_edge(mgr_id, target_mgr, rel, strength=strength)

    return managers


# ------------------------------------------------------------------
# build_hierarchy
# ------------------------------------------------------------------

def build_hierarchy(
    graph: ConceptGraph,
    max_depth: int = 4,
    min_clusters_to_recurse: int = 5,
) -> HierarchyTree:
    """Build a fractal hierarchy: concepts -> clusters -> regions -> systems."""
    all_node_ids = [
        nid for nid in graph._nodes
        if not nid.startswith("mgr_")
    ]

    levels: list[HierarchyLevel] = []
    levels.append(HierarchyLevel(level=0, nodes=list(all_node_ids), parent_map={}))

    current_ids = list(all_node_ids)

    for depth in range(1, max_depth + 1):
        if len(current_ids) < min_clusters_to_recurse:
            break

        clusters = cluster_concepts(graph, min_cluster_size=3)

        # Filter clusters to only include current_ids.
        current_set = set(current_ids)
        filtered: dict[str, list[str]] = {}
        for label, members in clusters.items():
            filtered_members = [m for m in members if m in current_set]
            if filtered_members:
                filtered[label] = filtered_members

        if len(filtered) < 2:
            break

        mgr_info = create_manager_nodes(graph, filtered)

        parent_map: dict[str, str] = {}
        next_ids: list[str] = []
        for mgr_id, _, member_ids in mgr_info:
            next_ids.append(mgr_id)
            for mid in member_ids:
                parent_map[mid] = mgr_id

        levels.append(HierarchyLevel(
            level=depth,
            nodes=next_ids,
            parent_map=parent_map,
        ))

        current_ids = next_ids

        if len(current_ids) < min_clusters_to_recurse:
            break

    root = current_ids[0] if len(current_ids) == 1 else None
    return HierarchyTree(levels=levels, root=root)


# ------------------------------------------------------------------
# incremental_assign
# ------------------------------------------------------------------

def incremental_assign(
    graph: ConceptGraph,
    new_concept_ids: list[str],
) -> dict[str, str]:
    """Assign new concepts to existing manager clusters by vector similarity.

    Returns {concept_id: manager_id}.
    """
    # Collect all manager nodes.
    mgr_ids = [nid for nid in graph._nodes if nid.startswith("mgr_")]
    if not mgr_ids:
        return {}

    mgr_vecs = torch.stack([graph._nodes[m].vector for m in mgr_ids])
    mgr_vecs = mgr_vecs / (mgr_vecs.norm(dim=1, keepdim=True) + 1e-8)

    assignments: dict[str, str] = {}
    for cid in new_concept_ids:
        node = graph._nodes.get(cid)
        if node is None:
            continue

        query = node.vector / (node.vector.norm() + 1e-8)
        sims = mgr_vecs @ query
        best_idx = sims.argmax().item()
        best_mgr = mgr_ids[best_idx]

        node.cluster_id = best_mgr
        graph.add_edge(cid, best_mgr, "part_of", strength=0.5)
        assignments[cid] = best_mgr

    return assignments


# ------------------------------------------------------------------
# merge_graphs
# ------------------------------------------------------------------

def merge_graphs(
    target_graph: ConceptGraph,
    source_graph: ConceptGraph,
    similarity_threshold: float = 0.85,
) -> dict[str, int]:
    """Merge source_graph into target_graph.

    Matching nodes (cosine >= threshold) are merged via EMA vectors + edge union.
    Non-matching nodes are added as new.
    Returns {"merged": N, "added": N, "edges_added": N}.
    """
    merged_count = 0
    added_count = 0
    edges_added = 0

    # Map source node IDs to their target counterparts (or new IDs).
    id_remap: dict[str, str] = {}

    for src_id, src_node in source_graph._nodes.items():
        results = target_graph.find_similar(src_node.vector, k=1)
        if results and results[0][1] >= similarity_threshold:
            tgt_node = results[0][0]
            # EMA merge vectors.
            total = tgt_node.observation_count + src_node.observation_count
            w_tgt = tgt_node.observation_count / max(total, 1)
            w_src = src_node.observation_count / max(total, 1)
            tgt_node.vector = w_tgt * tgt_node.vector + w_src * src_node.vector
            tgt_node.vector = tgt_node.vector / (tgt_node.vector.norm() + 1e-8)
            tgt_node.observation_count = total
            tgt_node.confidence = max(tgt_node.confidence, src_node.confidence)
            tgt_node.modalities |= src_node.modalities
            target_graph._update_vector_row(tgt_node.id, tgt_node.vector)
            id_remap[src_id] = tgt_node.id
            merged_count += 1
        else:
            # Add as new node.
            new_node = ConceptNode(
                vector=src_node.vector.clone(),
                name=src_node.name,
                modalities=set(src_node.modalities),
                modality_vectors={k: v.clone() for k, v in src_node.modality_vectors.items()},
                modality_weights=dict(src_node.modality_weights),
                confidence=src_node.confidence,
                observation_count=src_node.observation_count,
            )
            target_graph._nodes[new_node.id] = new_node
            if new_node.name:
                target_graph._name_index[new_node.name].append(new_node.id)
            target_graph._append_vector(new_node.id, new_node.vector)
            id_remap[src_id] = new_node.id
            added_count += 1

    # Remap and add edges.
    for edge in source_graph._edges.values():
        new_src = id_remap.get(edge.source_id)
        new_tgt = id_remap.get(edge.target_id)
        if new_src and new_tgt:
            target_graph.add_edge(new_src, new_tgt, edge.relation, edge.strength)
            edges_added += 1

    return {"merged": merged_count, "added": added_count, "edges_added": edges_added}


# ------------------------------------------------------------------
# remove_region
# ------------------------------------------------------------------

def remove_region(
    graph: ConceptGraph,
    cluster_id: str,
    dangling_strategy: str = "retarget_strong",
) -> dict[str, int]:
    """Remove all members of a cluster and its manager.

    dangling_strategy:
      - "retarget_strong": edges with strength >= 0.3 get retargeted to nearest
        surviving concept; weaker ones are dropped.
      - "drop": all dangling edges are removed.

    Returns {"removed_nodes": N, "retargeted_edges": N, "dropped_edges": N}.
    """
    # Find all members of this cluster.
    member_ids = {
        nid for nid, node in graph._nodes.items()
        if node.cluster_id == cluster_id
    }
    member_ids.add(cluster_id)  # Include the manager itself.

    # Filter to only IDs that actually exist.
    member_ids = {m for m in member_ids if m in graph._nodes}
    if not member_ids:
        return {"removed_nodes": 0, "retargeted_edges": 0, "dropped_edges": 0}

    surviving_ids = [nid for nid in graph._nodes if nid not in member_ids]

    retargeted = 0
    dropped = 0

    # Handle dangling edges: edges from surviving nodes to doomed nodes,
    # and edges from doomed nodes to surviving nodes.
    dangling_keys = []
    for mid in member_ids:
        for key in list(graph._adjacency_in.get(mid, [])):
            if key in graph._edges and key[0] not in member_ids:
                dangling_keys.append(key)
        for key in list(graph._adjacency_out.get(mid, [])):
            if key in graph._edges and key[1] not in member_ids:
                dangling_keys.append(key)

    for key in dangling_keys:
        edge = graph._edges.get(key)
        if edge is None:
            continue

        if dangling_strategy == "retarget_strong" and edge.strength >= 0.3 and surviving_ids:
            # Retarget to nearest surviving concept.
            if edge.source_id in member_ids:
                # Outgoing from doomed node: find nearest surviving to source.
                src_vec = graph._nodes[edge.source_id].vector
            else:
                # Incoming to doomed node: find nearest surviving to target.
                src_vec = graph._nodes[edge.target_id].vector

            surv_vecs = torch.stack([graph._nodes[s].vector for s in surviving_ids])
            surv_vecs = surv_vecs / (surv_vecs.norm(dim=1, keepdim=True) + 1e-8)
            query = src_vec / (src_vec.norm() + 1e-8)
            sims = surv_vecs @ query
            best_idx = sims.argmax().item()
            best_surv = surviving_ids[best_idx]

            # Remove old edge, add retargeted one.
            graph._remove_edge(key)
            if edge.source_id in member_ids:
                graph.add_edge(best_surv, edge.target_id, edge.relation, edge.strength)
            else:
                graph.add_edge(edge.source_id, best_surv, edge.relation, edge.strength)
            retargeted += 1
        else:
            graph._remove_edge(key)
            dropped += 1

    # Remove all cluster members.
    removed_count = 0
    for mid in list(member_ids):
        # Remove remaining edges (internal to the cluster).
        for key in list(graph._adjacency_out.get(mid, [])):
            graph._remove_edge(key)
        for key in list(graph._adjacency_in.get(mid, [])):
            graph._remove_edge(key)
        graph._remove_node(mid)
        removed_count += 1

    return {
        "removed_nodes": removed_count,
        "retargeted_edges": retargeted,
        "dropped_edges": dropped,
    }


# ------------------------------------------------------------------
# import_region
# ------------------------------------------------------------------

def import_region(
    graph: ConceptGraph,
    region_graph: ConceptGraph,
    auto_connect: bool = True,
    sim_threshold: float = 0.75,
) -> dict[str, int]:
    """Import a region (sub-graph) into the main graph.

    Overlapping concepts are merged; unique ones are added.
    If auto_connect, new concepts get a weak similar_to edge to their nearest
    existing neighbor.

    Returns {"merged": N, "added": N, "edges_added": N, "auto_edges": N}.
    """
    merged_count = 0
    added_count = 0
    edges_added = 0
    auto_edges = 0

    id_remap: dict[str, str] = {}
    new_ids: list[str] = []

    for src_id, src_node in region_graph._nodes.items():
        results = graph.find_similar(src_node.vector, k=1)
        if results and results[0][1] >= sim_threshold:
            tgt_node = results[0][0]
            graph.merge_concepts(tgt_node.id, src_id) if src_id in graph._nodes else None
            # EMA merge manually since source isn't in target graph.
            total = tgt_node.observation_count + src_node.observation_count
            w_tgt = tgt_node.observation_count / max(total, 1)
            w_src = src_node.observation_count / max(total, 1)
            tgt_node.vector = w_tgt * tgt_node.vector + w_src * src_node.vector
            tgt_node.vector = tgt_node.vector / (tgt_node.vector.norm() + 1e-8)
            tgt_node.observation_count = total
            tgt_node.modalities |= src_node.modalities
            graph._update_vector_row(tgt_node.id, tgt_node.vector)
            id_remap[src_id] = tgt_node.id
            merged_count += 1
        else:
            new_node = ConceptNode(
                vector=src_node.vector.clone(),
                name=src_node.name,
                modalities=set(src_node.modalities),
                modality_vectors={k: v.clone() for k, v in src_node.modality_vectors.items()},
                modality_weights=dict(src_node.modality_weights),
                confidence=src_node.confidence,
                observation_count=src_node.observation_count,
            )
            graph._nodes[new_node.id] = new_node
            if new_node.name:
                graph._name_index[new_node.name].append(new_node.id)
            graph._append_vector(new_node.id, new_node.vector)
            id_remap[src_id] = new_node.id
            new_ids.append(new_node.id)
            added_count += 1

    # Add remapped edges.
    for edge in region_graph._edges.values():
        new_src = id_remap.get(edge.source_id)
        new_tgt = id_remap.get(edge.target_id)
        if new_src and new_tgt:
            graph.add_edge(new_src, new_tgt, edge.relation, edge.strength)
            edges_added += 1

    # Auto-connect: link new concepts to nearest existing concept.
    if auto_connect and new_ids:
        existing_ids = [nid for nid in graph._nodes if nid not in set(new_ids)]
        if existing_ids:
            exist_vecs = torch.stack([graph._nodes[e].vector for e in existing_ids])
            exist_vecs = exist_vecs / (exist_vecs.norm(dim=1, keepdim=True) + 1e-8)

            for nid in new_ids:
                query = graph._nodes[nid].vector
                query = query / (query.norm() + 1e-8)
                sims = exist_vecs @ query
                best_idx = sims.argmax().item()
                best_sim = sims[best_idx].item()
                if best_sim > 0.0:
                    graph.add_edge(
                        nid, existing_ids[best_idx], "similar_to",
                        strength=max(0.1, best_sim * 0.3),
                    )
                    auto_edges += 1

    return {
        "merged": merged_count,
        "added": added_count,
        "edges_added": edges_added,
        "auto_edges": auto_edges,
    }
