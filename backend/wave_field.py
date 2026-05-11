"""
WaveField: the concept graph as a continuous wave medium.

Not a sequential pipeline. Not fixed ticks.
A physical wave propagating through the concept graph simultaneously
across all N nodes, updated every dt seconds on GPU.

Wave equation on graph:
  acceleration_i = wave_speed * sum_j A_ij * activation_j
                 - wave_speed * activation_i
                 + top_down_i
                 - damping * velocity_i
  velocity_i    += acceleration_i * dt
  activation_i  += velocity_i * dt

Activation has momentum (velocity). It doesn't jump instantly. It builds,
peaks, and decays like a physical wave. Multiple waves interfere.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


@dataclass
class WaveConfig:
    wave_speed: float = 0.15
    damping: float = 0.85
    top_down_strength: float = 0.3
    affect_gate_strength: float = 0.4
    dt: float = 0.05
    convergence_threshold: float = 0.0005
    max_settle_steps: int = 300

    similar_weight: float = 1.0
    causal_weight: float = 1.4
    hierarchy_weight: float = 0.7
    contrast_weight: float = 0.3
    agent_weight: float = 0.5


class WaveField:
    """Concept graph as living wave medium. All N concepts active simultaneously."""

    def __init__(self, graph, config: Optional[WaveConfig] = None):
        self.graph = graph
        self.config = config or WaveConfig()
        self.device = (
            torch.device('mps') if torch.backends.mps.is_available()
            else torch.device('cpu')
        )

        nodes = list(graph.nodes.values())
        N = len(nodes)
        self.N = N
        self.activation = torch.zeros(N, device=self.device)
        self.velocity = torch.zeros(N, device=self.device)
        self.affect_gate = torch.full((N,), 0.5, device=self.device)
        self.top_down = torch.zeros(N, device=self.device)

        self._node_to_idx = {n.concept_id: i for i, n in enumerate(nodes)}
        self._idx_to_node = {i: n.concept_id for i, n in enumerate(nodes)}

        emb_dim = nodes[0].embedding.shape[0] if nodes else 512
        self.node_matrix = torch.tensor(
            np.stack([n.embedding for n in nodes]) if nodes
            else np.zeros((0, emb_dim), dtype=np.float32),
            dtype=torch.float32, device=self.device,
        )

        self._build_adjacency_matrices()
        self.total_steps = 0
        self.last_inject_t = 0.0
        self._last_settle_steps = 0
        self._adj_dirty = False

    def _build_adjacency_matrices(self):
        from backend.graph import EdgeType

        N = self.N
        cfg = self.config

        buckets = {
            'forward':   ([], [], []),
            'backward':  ([], [], []),
            'causal':    ([], [], []),
            'hierarchy': ([], [], []),
        }
        type_map = {
            EdgeType.SIMILAR_TO:  ('forward',   'backward', cfg.similar_weight),
            EdgeType.CAUSES:      ('causal',    'forward',  cfg.causal_weight),
            EdgeType.IS_A:        ('hierarchy', 'backward', cfg.hierarchy_weight),
            EdgeType.OPPOSITE_OF: ('forward',   None,       cfg.contrast_weight),
            EdgeType.REFERS_TO:   ('forward',   None,       cfg.agent_weight),
        }
        for (src, dst, etype), edge in self.graph._edges.items():
            if src not in self._node_to_idx or dst not in self._node_to_idx:
                continue
            si = self._node_to_idx[src]
            di = self._node_to_idx[dst]
            w = float(edge.weight)
            if etype not in type_map:
                continue
            fwd_bucket, bwd_bucket, type_w = type_map[etype]
            effective_w = w * type_w
            rows, cols, weights = buckets[fwd_bucket]
            rows.append(si); cols.append(di); weights.append(effective_w)
            if bwd_bucket:
                rows, cols, weights = buckets[bwd_bucket]
                rows.append(di); cols.append(si); weights.append(effective_w * 0.6)

        def make_sparse(rows, cols, weights):
            if not rows:
                return torch.sparse_coo_tensor(
                    torch.zeros((2, 0), dtype=torch.long),
                    torch.zeros(0, dtype=torch.float32),
                    (N, N), device=self.device,
                ).coalesce()
            indices = torch.tensor([rows, cols], dtype=torch.long)
            values = torch.tensor(weights, dtype=torch.float32)
            return torch.sparse_coo_tensor(
                indices, values, (N, N), device=self.device,
            ).coalesce()

        self.A_forward   = make_sparse(*buckets['forward'])
        self.A_backward  = make_sparse(*buckets['backward'])
        self.A_causal    = make_sparse(*buckets['causal'])
        self.A_hierarchy = make_sparse(*buckets['hierarchy'])

    def inject(self, concept_ids: list[int], strength: float = 1.0,
               mode: str = 'velocity'):
        for cid in concept_ids:
            idx = self._node_to_idx.get(cid)
            if idx is None:
                continue
            if mode == 'velocity':
                self.velocity[idx] += strength
            else:
                self.activation[idx] += strength
        self.last_inject_t = time.time()

    def inject_representation(self, representation: np.ndarray,
                              strength: float = 1.0):
        if self.N == 0:
            return
        rep = torch.tensor(representation, dtype=torch.float32, device=self.device)
        n = rep.norm()
        if n > 0:
            rep = rep / n
        sims = (self.node_matrix @ rep).clamp(0.0, 1.0)
        top_k = min(5, self.N)
        top_sims, top_idx = sims.topk(top_k)
        for sim, idx in zip(top_sims, top_idx):
            self.velocity[idx] += strength * float(sim)

    def update_affect_gate(self, affect_vector: np.ndarray):
        """Project N_AFF affect vector via affect.W into D_REP space, then
        compute alignment with each concept embedding for gating."""
        # affect_vector is N_AFF (12); node_matrix is (N, D_REP).
        # We need to project affect to D_REP first using the W pseudoinverse
        # from the caller. Since we don't have W here, accept a D_REP-space
        # affect vector directly. Caller projects.
        if affect_vector.shape[0] != self.node_matrix.shape[1]:
            # caller passed N_AFF — leave gate at default
            return
        affect = torch.tensor(affect_vector, dtype=torch.float32, device=self.device)
        alignment = self.node_matrix @ affect
        self.affect_gate = torch.sigmoid(alignment * self.config.affect_gate_strength)

    def set_top_down(self, top_down_activation: torch.Tensor):
        self.top_down = top_down_activation.to(self.device)

    def step(self):
        cfg = self.config
        if self.N == 0:
            return
        act = self.activation.unsqueeze(1)
        fwd  = torch.sparse.mm(self.A_forward,   act).squeeze(1)
        bwd  = torch.sparse.mm(self.A_backward,  act).squeeze(1)
        caus = torch.sparse.mm(self.A_causal,    act).squeeze(1)
        hier = torch.sparse.mm(self.A_hierarchy, act).squeeze(1)

        acceleration = (
            cfg.wave_speed * (fwd - self.activation)
            + cfg.top_down_strength * (bwd + self.top_down - self.activation)
            + 0.5 * caus
            + 0.3 * hier
            - (1 - cfg.damping) * self.velocity
        )
        acceleration = acceleration * self.affect_gate
        self.velocity = self.velocity + acceleration * cfg.dt
        self.activation = self.activation + self.velocity * cfg.dt
        self.activation = torch.clamp(self.activation, 0.0, 1.0)
        self.velocity = torch.clamp(self.velocity, -2.0, 2.0)
        self.activation = self.activation * 0.995
        self.total_steps += 1

    def step_n(self, n: int):
        for _ in range(n):
            self.step()

    def run_until_settled(self) -> int:
        cfg = self.config
        steps = 0
        prev = self.activation.clone()
        for steps in range(cfg.max_settle_steps):
            self.step()
            delta = torch.mean(torch.abs(self.activation - prev)).item()
            if delta < cfg.convergence_threshold:
                break
            prev = self.activation.clone()
        self._last_settle_steps = steps + 1
        return steps + 1

    def get_top_concepts(self, k: int = 50) -> list[tuple[int, float]]:
        if self.N == 0:
            return []
        k = min(int(k), self.N)
        top_act, top_idx = self.activation.topk(k)
        out = []
        for act, idx in zip(top_act.cpu(), top_idx.cpu()):
            cid = self._idx_to_node.get(int(idx))
            if cid is not None:
                out.append((cid, float(act)))
        return out

    def get_field_centroid(self) -> np.ndarray:
        if self.N == 0:
            return np.zeros(self.node_matrix.shape[1], dtype=np.float32)
        weights = self.activation.unsqueeze(1)
        centroid = (self.node_matrix * weights).sum(0)
        n = centroid.norm()
        if n > 0:
            centroid = centroid / n
        return centroid.cpu().numpy()

    def get_bridge_concepts(self, domain_a_ids: list[int],
                            domain_b_ids: list[int],
                            settle_steps: int = 100) -> list[tuple[int, float]]:
        saved_act = self.activation.clone()
        saved_vel = self.velocity.clone()
        self.activation = torch.zeros(self.N, device=self.device)
        self.velocity = torch.zeros(self.N, device=self.device)
        self.inject(domain_a_ids, strength=1.0)
        self.inject(domain_b_ids, strength=1.0)
        self.step_n(settle_steps)
        seed_set = set(domain_a_ids) | set(domain_b_ids)
        top = self.get_top_concepts(k=100)
        bridges = [(cid, act) for cid, act in top if cid not in seed_set]
        self.activation = saved_act
        self.velocity = saved_vel
        return sorted(bridges, key=lambda x: -x[1])[:20]

    def add_node(self, concept_id: int, embedding: np.ndarray):
        idx = self.N
        self._node_to_idx[concept_id] = idx
        self._idx_to_node[idx] = concept_id
        self.N += 1
        zero = torch.zeros(1, device=self.device)
        self.activation = torch.cat([self.activation, zero])
        self.velocity = torch.cat([self.velocity, zero])
        self.affect_gate = torch.cat(
            [self.affect_gate, torch.tensor([0.5], device=self.device)])
        self.top_down = torch.cat([self.top_down, zero])
        emb = torch.tensor(embedding, dtype=torch.float32,
                           device=self.device).unsqueeze(0)
        self.node_matrix = torch.cat([self.node_matrix, emb], dim=0)
        self._adj_dirty = True

    def rebuild_if_dirty(self):
        if getattr(self, '_adj_dirty', False):
            self._build_adjacency_matrices()
            self._adj_dirty = False

    @property
    def energy(self) -> float:
        if self.N == 0:
            return 0.0
        return float(self.velocity.abs().mean())

    @property
    def peak_activation(self) -> float:
        if self.N == 0:
            return 0.0
        return float(self.activation.max())
