"""ConceptGraph — the unified substrate for Concept Brain v3."""

from __future__ import annotations

import sqlite3
from collections import defaultdict

import torch
from torch import Tensor

from concept.node import ConceptNode, TypedEdge, _now_step


class ConceptGraph:
    """Single data structure that IS the store, hierarchy, and inference engine."""

    def __init__(self) -> None:
        self._nodes: dict[str, ConceptNode] = {}
        self._name_index: dict[str, list[str]] = defaultdict(list)

        # Dense vector matrix for brute-force similarity search.
        self._vector_matrix: Tensor | None = None  # (N, 512)
        self._vector_id_map: list[str] = []         # row index -> node id

        # Edges keyed by (source_id, target_id, relation).
        self._edges: dict[tuple[str, str, str], TypedEdge] = {}
        self._adjacency_out: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
        self._adjacency_in: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
        self._reverse_index: dict[tuple[str, str], set[str]] = defaultdict(set)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        return len(self._edges)

    @property
    def relation_types(self) -> set[str]:
        return {key[2] for key in self._edges}

    # ------------------------------------------------------------------
    # Core: find_or_create
    # ------------------------------------------------------------------

    def find_or_create(
        self,
        name: str | None,
        vector: Tensor,
        modality: str = "text",
    ) -> tuple[ConceptNode, bool]:
        """Find an existing concept or create a new one.

        Match priority:
        1. Same name AND vector cosine similarity >= 0.85.
        2. Any node with vector cosine similarity >= 0.85 (name-agnostic).
        3. No match -> create new.

        Returns (node, is_new).
        """
        vector = vector.detach().float()
        if vector.dim() == 0:
            vector = torch.zeros(512)

        best_node: ConceptNode | None = None
        best_sim: float = -1.0

        # Fast path: exact name match with single candidate — skip vector search entirely.
        if name and name in self._name_index:
            candidates = self._name_index[name]
            if len(candidates) == 1:
                # Unambiguous name match — no matmul needed
                best_node = self._nodes[candidates[0]]
                best_sim = 1.0
            else:
                # Multiple concepts with same name (homonyms) — use vector to pick
                for nid in candidates:
                    node = self._nodes[nid]
                    sim = self._cosine(node.vector, vector)
                    if sim >= 0.85 and sim > best_sim:
                        best_sim = sim
                        best_node = node

        # If no name match, try global vector search.
        if best_node is None and self._vector_matrix is not None and self._vector_matrix.shape[0] > 0:
            sims = self._cosine_batch(vector)
            if sims.numel() > 0:
                max_sim, max_idx = sims.max(dim=0)
                if max_sim.item() >= 0.85:
                    best_sim = max_sim.item()
                    best_node = self._nodes[self._vector_id_map[max_idx.item()]]

        if best_node is not None:
            return self._update_existing(best_node, vector, modality), False

        return self._create_new(name, vector, modality), True

    # ------------------------------------------------------------------
    # Core: add_edge
    # ------------------------------------------------------------------

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        relation: str,
        strength: float = 0.5,
    ) -> TypedEdge:
        """Add or reinforce a typed edge between two concepts."""
        key = (source_id, target_id, relation)
        if key in self._edges:
            edge = self._edges[key]
            edge.evidence += 1
            edge.strength = min(1.0, edge.strength + 0.1 * (1.0 - edge.strength))
            edge.last_used = _now_step()
            return edge

        edge = TypedEdge(
            source_id=source_id,
            target_id=target_id,
            relation=relation,
            strength=strength,
        )
        self._edges[key] = edge
        self._adjacency_out[source_id].append(key)
        self._adjacency_in[target_id].append(key)
        self._reverse_index[(relation, target_id)].add(source_id)
        return edge

    # ------------------------------------------------------------------
    # Core: write (batch convenience)
    # ------------------------------------------------------------------

    def write(
        self,
        triples: list[tuple[str, str, str]],
        vectors: dict[str, Tensor],
        modality: str = "text",
    ) -> list[ConceptNode]:
        """Ingest a list of (subject, relation, object) triples.

        `vectors` maps concept names to their encoded vectors.
        Returns the list of all nodes touched.
        """
        touched: dict[str, ConceptNode] = {}

        for subj, rel, obj in triples:
            for concept_name in (subj, obj):
                if concept_name not in touched:
                    vec = vectors.get(concept_name, torch.randn(512))
                    node, _ = self.find_or_create(concept_name, vec, modality)
                    touched[concept_name] = node

            self.add_edge(touched[subj].id, touched[obj].id, rel)

        return list(touched.values())

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get_node(self, node_id: str) -> ConceptNode | None:
        return self._nodes.get(node_id)

    def get_by_name(self, name: str) -> list[ConceptNode]:
        return [self._nodes[nid] for nid in self._name_index.get(name, [])]

    def get_edges(
        self,
        node_id: str,
        relation: str | None = None,
        direction: str = "outgoing",
    ) -> list[TypedEdge]:
        """Get edges for a node, optionally filtered by relation and direction."""
        if direction == "outgoing":
            keys = self._adjacency_out.get(node_id, [])
        elif direction == "incoming":
            keys = self._adjacency_in.get(node_id, [])
        else:
            keys = self._adjacency_out.get(node_id, []) + self._adjacency_in.get(node_id, [])

        edges = [self._edges[k] for k in keys if k in self._edges]
        if relation is not None:
            edges = [e for e in edges if e.relation == relation]
        return edges

    # ------------------------------------------------------------------
    # Similarity search
    # ------------------------------------------------------------------

    def find_similar(self, query_vector: Tensor, k: int = 10) -> list[tuple[ConceptNode, float]]:
        """Brute-force cosine similarity search against all concepts."""
        if self._vector_matrix is None or self._vector_matrix.shape[0] == 0:
            return []

        sims = self._cosine_batch(query_vector.detach().float())
        k = min(k, sims.shape[0])
        topk_sims, topk_idxs = sims.topk(k)

        results = []
        for sim, idx in zip(topk_sims.tolist(), topk_idxs.tolist()):
            node = self._nodes[self._vector_id_map[idx]]
            results.append((node, sim))
        return results

    # ------------------------------------------------------------------
    # Maintenance: merge, dedup, prune
    # ------------------------------------------------------------------

    def merge_concepts(self, keep_id: str, absorb_id: str) -> None:
        """Merge absorb_id into keep_id. Redirect edges, delete absorbed node."""
        keep = self._nodes.get(keep_id)
        absorb = self._nodes.get(absorb_id)
        if keep is None or absorb is None:
            return

        # Weighted vector merge.
        total = keep.observation_count + absorb.observation_count
        if total > 0:
            w_keep = keep.observation_count / total
            w_absorb = absorb.observation_count / total
            keep.vector = w_keep * keep.vector + w_absorb * absorb.vector
            keep.vector = keep.vector / (keep.vector.norm() + 1e-8)

        keep.observation_count = total
        keep.confidence = max(keep.confidence, absorb.confidence)
        keep.modalities |= absorb.modalities

        # Merge modality vectors.
        for mod, vec in absorb.modality_vectors.items():
            if mod not in keep.modality_vectors:
                keep.modality_vectors[mod] = vec
                keep.modality_weights[mod] = absorb.modality_weights.get(mod, 1.0)

        # Redirect edges from absorbed node.
        for key in list(self._adjacency_out.get(absorb_id, [])):
            edge = self._edges.pop(key, None)
            if edge is None:
                continue
            new_key = (keep_id, edge.target_id, edge.relation)
            if new_key in self._edges:
                existing = self._edges[new_key]
                existing.evidence += edge.evidence
                existing.strength = max(existing.strength, edge.strength)
            else:
                edge.source_id = keep_id
                self._edges[new_key] = edge
                self._adjacency_out[keep_id].append(new_key)
                self._adjacency_in[edge.target_id].append(new_key)
                self._reverse_index[(edge.relation, edge.target_id)].add(keep_id)

        for key in list(self._adjacency_in.get(absorb_id, [])):
            edge = self._edges.pop(key, None)
            if edge is None:
                continue
            new_key = (edge.source_id, keep_id, edge.relation)
            if new_key in self._edges:
                existing = self._edges[new_key]
                existing.evidence += edge.evidence
                existing.strength = max(existing.strength, edge.strength)
            else:
                edge.target_id = keep_id
                self._edges[new_key] = edge
                self._adjacency_out[edge.source_id].append(new_key)
                self._adjacency_in[keep_id].append(new_key)
                self._reverse_index[(edge.relation, keep_id)].add(edge.source_id)

        # Clean up absorbed node from reverse index.
        for (rel, tgt), sources in self._reverse_index.items():
            sources.discard(absorb_id)

        # Remove absorbed node.
        self._remove_node(absorb_id)

    def dedup_pass(self, threshold: float = 0.95) -> int:
        """Find and merge near-duplicate concepts. Returns merge count."""
        if self._vector_matrix is None or self._vector_matrix.shape[0] < 2:
            return 0

        merged = 0
        visited: set[str] = set()

        # Compute full similarity matrix.
        norms = self._vector_matrix / (self._vector_matrix.norm(dim=1, keepdim=True) + 1e-8)
        sim_matrix = norms @ norms.T

        n = sim_matrix.shape[0]
        for i in range(n):
            if self._vector_id_map[i] in visited:
                continue
            for j in range(i + 1, n):
                if self._vector_id_map[j] in visited:
                    continue
                if sim_matrix[i, j].item() >= threshold:
                    keep_id = self._vector_id_map[i]
                    absorb_id = self._vector_id_map[j]
                    visited.add(absorb_id)
                    self.merge_concepts(keep_id, absorb_id)
                    merged += 1
                    break  # Rebuild indices changed; restart inner loop next i.

        return merged

    def prune_weak_edges(self, min_strength: float = 0.01, min_age: int = 5000) -> int:
        """Remove edges weaker than min_strength and older than min_age steps."""
        now = _now_step()
        to_remove = []
        for key, edge in self._edges.items():
            age = now - edge.created_at
            if edge.strength < min_strength and age > min_age:
                to_remove.append(key)

        for key in to_remove:
            self._remove_edge(key)

        return len(to_remove)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, db_path: str, tensor_path: str) -> None:
        """Save graph: SQLite for metadata/edges, torch .pt for vectors."""
        # Save vectors.
        tensor_data = {
            "vectors": {nid: node.vector for nid, node in self._nodes.items()},
            "modality_vectors": {
                nid: node.modality_vectors for nid, node in self._nodes.items()
            },
        }
        torch.save(tensor_data, tensor_path)

        # Save metadata + edges to SQLite.
        db = sqlite3.connect(db_path)
        db.execute("DROP TABLE IF EXISTS nodes")
        db.execute("DROP TABLE IF EXISTS edges")
        db.execute(
            """CREATE TABLE nodes (
                id TEXT PRIMARY KEY,
                name TEXT,
                modalities TEXT,
                modality_weights TEXT,
                confidence REAL,
                observation_count INTEGER,
                activation REAL,
                cluster_id TEXT,
                created_at INTEGER,
                last_accessed INTEGER
            )"""
        )
        db.execute(
            """CREATE TABLE edges (
                source_id TEXT,
                target_id TEXT,
                relation TEXT,
                strength REAL,
                evidence INTEGER,
                created_at INTEGER,
                last_used INTEGER,
                PRIMARY KEY (source_id, target_id, relation)
            )"""
        )

        for node in self._nodes.values():
            db.execute(
                "INSERT INTO nodes VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    node.id,
                    node.name,
                    ",".join(sorted(node.modalities)),
                    ",".join(f"{k}:{v}" for k, v in node.modality_weights.items()),
                    node.confidence,
                    node.observation_count,
                    node.activation,
                    node.cluster_id,
                    node.created_at,
                    node.last_accessed,
                ),
            )

        for edge in self._edges.values():
            db.execute(
                "INSERT INTO edges VALUES (?,?,?,?,?,?,?)",
                (
                    edge.source_id,
                    edge.target_id,
                    edge.relation,
                    edge.strength,
                    edge.evidence,
                    edge.created_at,
                    edge.last_used,
                ),
            )

        db.commit()
        db.close()

    @classmethod
    def load(cls, db_path: str, tensor_path: str) -> ConceptGraph:
        """Restore a graph from SQLite + tensor file."""
        graph = cls()
        tensor_data = torch.load(tensor_path, weights_only=False)
        vectors = tensor_data["vectors"]
        modality_vectors = tensor_data.get("modality_vectors", {})

        db = sqlite3.connect(db_path)

        for row in db.execute("SELECT * FROM nodes"):
            nid, name, mods_str, mw_str, conf, obs, act, cid, cat, la = row
            modalities = set(mods_str.split(",")) if mods_str else set()

            modality_weights = {}
            if mw_str:
                for pair in mw_str.split(","):
                    if ":" in pair:
                        k, v = pair.split(":", 1)
                        modality_weights[k] = float(v)

            node = ConceptNode(
                id=nid,
                vector=vectors.get(nid, torch.zeros(512)),
                name=name,
                modalities=modalities,
                modality_vectors=modality_vectors.get(nid, {}),
                modality_weights=modality_weights,
                confidence=conf,
                observation_count=obs,
                activation=act,
                cluster_id=cid,
                created_at=cat,
                last_accessed=la,
            )
            graph._nodes[nid] = node
            if name:
                graph._name_index[name].append(nid)
            graph._vector_id_map.append(nid)

        # Rebuild vector matrix.
        if graph._vector_id_map:
            graph._vector_matrix = torch.stack(
                [graph._nodes[nid].vector for nid in graph._vector_id_map]
            )

        for row in db.execute("SELECT * FROM edges"):
            sid, tid, rel, strength, evidence, cat, lu = row
            edge = TypedEdge(
                source_id=sid,
                target_id=tid,
                relation=rel,
                strength=strength,
                evidence=evidence,
                created_at=cat,
                last_used=lu,
            )
            key = (sid, tid, rel)
            graph._edges[key] = edge
            graph._adjacency_out[sid].append(key)
            graph._adjacency_in[tid].append(key)
            graph._reverse_index[(rel, tid)].add(sid)

        db.close()
        return graph

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        return {
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "relation_types": sorted(self.relation_types),
            "modalities": sorted(
                {m for n in self._nodes.values() for m in n.modalities}
            ),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _create_new(self, name: str | None, vector: Tensor, modality: str) -> ConceptNode:
        vec = vector / (vector.norm() + 1e-8)
        node = ConceptNode(
            vector=vec,
            name=name,
            modalities={modality},
            modality_vectors={modality: vec.clone()},
            modality_weights={modality: 1.0},
            confidence=0.1,
            observation_count=1,
        )
        self._nodes[node.id] = node
        if name:
            self._name_index[name].append(node.id)
        self._append_vector(node.id, vec)
        return node

    def _update_existing(self, node: ConceptNode, vector: Tensor, modality: str) -> ConceptNode:
        node.observation_count += 1
        alpha = 1.0 / (1.0 + node.observation_count)

        # EMA update main vector.
        node.vector = (1 - alpha) * node.vector + alpha * vector
        node.vector = node.vector / (node.vector.norm() + 1e-8)

        # Update modality vector.
        if modality in node.modality_vectors:
            mv = node.modality_vectors[modality]
            node.modality_vectors[modality] = (1 - alpha) * mv + alpha * vector
            node.modality_vectors[modality] /= node.modality_vectors[modality].norm() + 1e-8
        else:
            node.modality_vectors[modality] = vector / (vector.norm() + 1e-8)
            node.modality_weights[modality] = 1.0

        node.modalities.add(modality)
        node.confidence = min(1.0, node.confidence + 0.05)
        node.last_accessed = _now_step()

        # Update row in vector matrix.
        self._update_vector_row(node.id, node.vector)
        return node

    def _append_vector(self, node_id: str, vector: Tensor) -> None:
        row = vector.unsqueeze(0)
        if self._vector_matrix is None:
            self._vector_matrix = row
        else:
            self._vector_matrix = torch.cat([self._vector_matrix, row], dim=0)
        self._vector_id_map.append(node_id)

    def _update_vector_row(self, node_id: str, vector: Tensor) -> None:
        try:
            idx = self._vector_id_map.index(node_id)
            self._vector_matrix[idx] = vector
        except ValueError:
            self._append_vector(node_id, vector)

    def _remove_node(self, node_id: str) -> None:
        node = self._nodes.pop(node_id, None)
        if node is None:
            return

        if node.name and node.name in self._name_index:
            ids = self._name_index[node.name]
            if node_id in ids:
                ids.remove(node_id)
            if not ids:
                del self._name_index[node.name]

        # Remove from vector matrix.
        if node_id in self._vector_id_map:
            idx = self._vector_id_map.index(node_id)
            self._vector_id_map.pop(idx)
            if self._vector_matrix is not None and self._vector_matrix.shape[0] > 0:
                self._vector_matrix = torch.cat(
                    [self._vector_matrix[:idx], self._vector_matrix[idx + 1:]], dim=0
                )
                if self._vector_matrix.shape[0] == 0:
                    self._vector_matrix = None

        # Clean adjacency lists.
        self._adjacency_out.pop(node_id, None)
        self._adjacency_in.pop(node_id, None)

    def _remove_edge(self, key: tuple[str, str, str]) -> None:
        edge = self._edges.pop(key, None)
        if edge is None:
            return

        src, tgt, rel = key
        if src in self._adjacency_out and key in self._adjacency_out[src]:
            self._adjacency_out[src].remove(key)
        if tgt in self._adjacency_in and key in self._adjacency_in[tgt]:
            self._adjacency_in[tgt].remove(key)
        if (rel, tgt) in self._reverse_index:
            self._reverse_index[(rel, tgt)].discard(src)

    @staticmethod
    def _cosine(a: Tensor, b: Tensor) -> float:
        a_n = a / (a.norm() + 1e-8)
        b_n = b / (b.norm() + 1e-8)
        return (a_n @ b_n).item()

    def _cosine_batch(self, query: Tensor) -> Tensor:
        """Cosine similarity of query against all rows in _vector_matrix."""
        q = query / (query.norm() + 1e-8)
        norms = self._vector_matrix / (self._vector_matrix.norm(dim=1, keepdim=True) + 1e-8)
        return norms @ q
