"""
BrainState — the v2 parallel engine.

The entire brain as GPU-resident matrices. No Python loops in the hot path.
Every neuron evaluates simultaneously. Self-activation thresholds replace
the centralized resonance manager. Message passing replaces serial BFS.

Analogy: a stadium wave, not an announcer calling seat numbers.
"""

import torch
import torch.nn.functional as F


class BrainState:
    def __init__(
        self,
        dim: int = 512,
        initial_size: int = 4,
        max_size: int = 50000,
        device: str = "cpu",
    ):
        self.dim = dim
        self.max_size = max_size
        self.device = torch.device(device)
        self.n = 0  # current number of active neurons

        # Pre-allocated GPU tensors (grow by doubling when needed)
        cap = max(initial_size * 2, 64)
        self._cap = cap
        self.weights = torch.zeros(cap, dim, device=self.device)
        self.thresholds = torch.full((cap,), 0.15, device=self.device)
        self.fire_rates = torch.zeros(cap, device=self.device)
        self.layer_indices = torch.zeros(cap, device=self.device)
        self.dormant = torch.zeros(cap, dtype=torch.bool, device=self.device)
        self.ages = torch.zeros(cap, dtype=torch.long, device=self.device)

        # Sparse edge matrix — (N, N) but stored as dict for growth flexibility
        self._edge_strengths: dict[tuple[int, int], float] = {}
        self._edge_matrix_cache: torch.Tensor | None = None
        self._edge_matrix_n: int = 0

        # Metadata
        self.cluster_ids: list[str] = []
        self._id_to_idx: dict[str, int] = {}

        # Config
        self.target_fire_rate = 0.05
        self.threshold_adapt_rate = 0.001
        self.lr = 0.01
        self.max_rounds = 3

        # Cached state from last forward (for update)
        self._last_fired: torch.Tensor | None = None
        self._last_scores: torch.Tensor | None = None
        self._last_prediction: torch.Tensor | None = None

        # Activation buffer (carried from v1)
        self.activation_buffer = torch.zeros(dim, device=self.device)
        self.buffer_decay = 0.9
        self.buffer_weight = 0.15

        # Initialize neurons
        for i in range(initial_size):
            self.add_neuron(
                cluster_id=f"c_{i:02d}",
                weights=F.normalize(torch.randn(dim), dim=0),
                layer_index=0 if i < initial_size // 2 else 1,
            )

    # ── Core State Management ──

    def _ensure_capacity(self, needed: int):
        """Double capacity if needed."""
        if needed <= self._cap:
            return
        new_cap = max(self._cap * 2, needed)
        for attr in ['weights', 'thresholds', 'fire_rates', 'layer_indices', 'ages']:
            old = getattr(self, attr)
            new = torch.zeros(new_cap, *old.shape[1:], dtype=old.dtype, device=self.device)
            new[:self._cap] = old
            setattr(self, attr, new)
        old_d = self.dormant
        new_d = torch.zeros(new_cap, dtype=torch.bool, device=self.device)
        new_d[:self._cap] = old_d
        self.dormant = new_d
        # Thresholds for new slots default to 0.15
        self.thresholds[self._cap:new_cap] = 0.15
        self._cap = new_cap

    def add_neuron(
        self,
        cluster_id: str,
        weights: torch.Tensor,
        layer_index: float = 0,
        threshold: float = 0.15,
    ) -> int:
        """Add a neuron. Returns its index."""
        idx = self.n
        self._ensure_capacity(idx + 1)
        self.weights[idx] = F.normalize(weights.to(self.device), dim=0)
        self.thresholds[idx] = threshold
        self.layer_indices[idx] = layer_index
        self.fire_rates[idx] = self.target_fire_rate  # start at target
        self.ages[idx] = 0
        self.dormant[idx] = False
        self.cluster_ids.append(cluster_id)
        self._id_to_idx[cluster_id] = idx
        self.n += 1
        return idx

    def remove_neuron(self, cluster_id: str):
        """Mark a neuron as dormant (don't compact — indices are stable)."""
        idx = self._id_to_idx.get(cluster_id)
        if idx is not None:
            self.dormant[idx] = True

    # ── Edges ──

    def add_edge(self, from_id: str, to_id: str, strength: float = 0.1):
        from_idx = self._id_to_idx.get(from_id)
        to_idx = self._id_to_idx.get(to_id)
        if from_idx is not None and to_idx is not None:
            self._edge_strengths[(from_idx, to_idx)] = strength
            self._invalidate_edge_cache()

    def edge_exists(self, from_id: str, to_id: str) -> bool:
        from_idx = self._id_to_idx.get(from_id)
        to_idx = self._id_to_idx.get(to_id)
        if from_idx is None or to_idx is None:
            return False
        return (from_idx, to_idx) in self._edge_strengths

    def _build_edge_matrix(self) -> torch.Tensor:
        """Build dense edge matrix for message passing. Cached until edges change."""
        if self._edge_matrix_cache is not None and self._edge_matrix_n == self.n:
            return self._edge_matrix_cache
        n = self.n
        mat = torch.zeros(n, n, device=self.device)
        for (i, j), s in self._edge_strengths.items():
            if i < n and j < n:
                mat[i, j] = s
        self._edge_matrix_cache = mat
        self._edge_matrix_n = n
        return mat

    def _invalidate_edge_cache(self):
        self._edge_matrix_cache = None

    # ── Forward Pass: The Stadium Wave ──

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """
        Parallel forward: all neurons evaluate simultaneously.
        Self-activation thresholds. Message passing for multi-round thinking.
        Returns (prediction (512,), activations {cluster_id: score}).
        """
        x = x.to(self.device)
        n = self.n

        # Apply buffer (normalized — v1 lesson)
        buf_norm = self.activation_buffer.norm()
        if buf_norm > 1e-6:
            buf_dir = self.activation_buffer / buf_norm
            effective_x = F.normalize(x + self.buffer_weight * buf_dir, dim=0)
        else:
            effective_x = F.normalize(x, dim=0)

        # 1. SENSE — all neurons evaluate (1 matmul)
        identities = F.normalize(self.weights[:n], dim=1)
        scores = identities @ effective_x  # (N,)

        # 2. FIRE — self-activation
        active = ~self.dormant[:n]
        fired = (scores > self.thresholds[:n]) & active

        # Guarantee minimum firing (at least 4 neurons)
        if fired.sum() < 4:
            top_scores, top_idx = scores.topk(min(4, n))
            for idx in top_idx:
                if active[idx]:
                    fired[idx] = True

        # 3. THINK — message passing rounds (skip most steps for speed)
        self._step_count = getattr(self, '_step_count', 0) + 1
        do_message_pass = (self._step_count % 5 == 0) and self._edge_strengths and fired.sum() > 0
        if do_message_pass:
            edge_mat = self._build_edge_matrix()
            for _ in range(self.max_rounds):
                fired_idx = fired.nonzero().squeeze(1)
                if len(fired_idx) == 0:
                    break

                # Messages from fired neurons propagate along edges
                # edge_mat[i, j] = strength of j→i connection
                # messages[i] = sum of edge_mat[i, fired] * scores[fired]
                message_strength = edge_mat[:n, :][:, fired_idx] @ scores[fired_idx]  # (N,)
                new_scores = scores + 0.05 * message_strength

                # Check for new firings
                newly_fired = (new_scores > self.thresholds[:n]) & active & ~fired
                if newly_fired.sum() == 0:
                    break
                fired = fired | newly_fired
                scores = new_scores

        # 4. OUTPUT — weighted aggregate of fired neurons
        fired_idx = fired.nonzero().squeeze(1)
        if len(fired_idx) == 0:
            prediction = torch.zeros(self.dim, device=self.device)
        else:
            active_weights = self.weights[fired_idx]  # (K, 512)
            active_scores = scores[fired_idx]  # (K,)
            # Softmax-weighted combination (gentle — let multiple neurons contribute)
            attn = F.softmax(active_scores * 2.0, dim=0)
            prediction = attn @ active_weights  # (512,)
            prediction = F.normalize(prediction, dim=0)

        # Cache for update
        self._last_fired = fired
        self._last_scores = scores
        self._last_prediction = prediction

        # Update buffer
        self._update_buffer(fired, scores)

        # Age neurons
        self.ages[:n] += 1

        # Build activations dict (API compat)
        activations = {}
        for idx in fired_idx:
            activations[self.cluster_ids[idx.item()]] = scores[idx].item()

        return prediction.cpu(), activations

    # ── Learning: Parallel Correction-Based ──

    def update(self, x: torch.Tensor, teacher_vec: torch.Tensor):
        """
        All fired neurons learn in parallel.
        Error = teacher - prediction. Each neuron's update scaled by its activation.
        """
        if self._last_fired is None or self._last_scores is None:
            return

        x = x.to(self.device)
        teacher_vec = F.normalize(teacher_vec.to(self.device), dim=0)

        fired = self._last_fired
        scores = self._last_scores
        fired_idx = fired.nonzero().squeeze(1)

        if len(fired_idx) == 0:
            return

        # Error = what teacher said - what baby said
        prediction = self._last_prediction.to(self.device)
        error = teacher_vec - prediction  # (512,)

        # All fired neurons update in parallel
        # Each neuron's update = its score × the error direction
        active_scores = scores[fired_idx].unsqueeze(1)  # (K, 1)
        updates = active_scores * error.unsqueeze(0)  # (K, 512)
        self.weights[fired_idx] = self.weights[fired_idx] + self.lr * updates

        # Re-normalize (keep unit vectors)
        self.weights[fired_idx] = F.normalize(self.weights[fired_idx], dim=1)

        # Hebbian edge update: strengthen connections between co-fired neurons
        self._hebbian_update(fired_idx, scores)

    def _hebbian_update(self, fired_idx: torch.Tensor, scores: torch.Tensor):
        """Strengthen edges between neurons that fire together. Batched for speed."""
        if len(fired_idx) < 2:
            return
        # Only update existing edges, sample top-5 pairs for speed
        fired_list = fired_idx[:5].tolist()  # cap at 5 to avoid O(N^2)
        for i in range(len(fired_list)):
            for j in range(i + 1, len(fired_list)):
                idx_i, idx_j = fired_list[i], fired_list[j]
                co_act = scores[idx_i].item() * scores[idx_j].item()
                for pair in [(idx_i, idx_j), (idx_j, idx_i)]:
                    if pair in self._edge_strengths:
                        self._edge_strengths[pair] = min(
                            self._edge_strengths[pair] + 0.001 * co_act, 1.0
                        )

    # ── Homeostatic Threshold Adaptation ──

    def adapt_thresholds(self):
        """Adjust thresholds so each neuron fires at target rate."""
        n = self.n
        fired = self._last_fired
        if fired is None:
            return

        # Update firing rates (EMA)
        self.fire_rates[:n] = 0.999 * self.fire_rates[:n] + 0.001 * fired.float()

        # Adjust: too active → raise threshold, too quiet → lower
        deviation = self.fire_rates[:n] - self.target_fire_rate
        self.thresholds[:n] = self.thresholds[:n] + self.threshold_adapt_rate * deviation
        self.thresholds[:n].clamp_(0.01, 0.95)

    # ── Buffer ──

    def _update_buffer(self, fired: torch.Tensor, scores: torch.Tensor):
        """Decay buffer, add top-K fired neuron identities."""
        self.activation_buffer *= self.buffer_decay
        fired_idx = fired.nonzero().squeeze(1)
        if len(fired_idx) == 0:
            return
        # Top-5 by score
        k = min(5, len(fired_idx))
        fired_scores = scores[fired_idx]
        top_k_local = fired_scores.topk(k).indices
        top_k_idx = fired_idx[top_k_local]
        for idx in top_k_idx:
            act = scores[idx].item()
            self.activation_buffer += act * F.normalize(self.weights[idx], dim=0)

    # ── Growth ──

    def bud(self, cluster_id: str) -> tuple[str, str] | None:
        """Split a neuron into two. Returns (child_a_id, child_b_id) or None."""
        idx = self._id_to_idx.get(cluster_id)
        if idx is None:
            return None

        parent_w = self.weights[idx].clone()
        layer = self.layer_indices[idx].item()
        threshold = self.thresholds[idx].item()

        # Two children: parent weight + small random perturbations
        noise = 0.1 * torch.randn(self.dim, device=self.device)
        child_a_w = F.normalize(parent_w + noise, dim=0)
        child_b_w = F.normalize(parent_w - noise, dim=0)

        child_a_id = f"{cluster_id}a"
        child_b_id = f"{cluster_id}b"

        # Children at different depths — creates hierarchy over time
        idx_a = self.add_neuron(child_a_id, child_a_w, layer, threshold)
        idx_b = self.add_neuron(child_b_id, child_b_w, layer + 0.5, threshold)

        # Transfer edges from parent to children
        to_remove = []
        to_add = []
        for (i, j), s in self._edge_strengths.items():
            if i == idx:
                to_add.append(((idx_a, j), s))
                to_add.append(((idx_b, j), s))
                to_remove.append((i, j))
            elif j == idx:
                to_add.append(((i, idx_a), s))
                to_add.append(((i, idx_b), s))
                to_remove.append((i, j))
        for k in to_remove:
            del self._edge_strengths[k]
        for k, s in to_add:
            self._edge_strengths[k] = s
        # Add edge between siblings
        self._edge_strengths[(idx_a, idx_b)] = 0.5
        self._edge_strengths[(idx_b, idx_a)] = 0.5

        # Dormant parent
        self.dormant[idx] = True

        return child_a_id, child_b_id

    def growth_check(self, step: int) -> list[dict]:
        """Check for growth triggers. Returns list of events."""
        events = []
        n = self.n

        # BUD: neurons with high fire rate and high age → split
        for i in range(n):
            if self.dormant[i]:
                continue
            if self.ages[i] < 200:
                continue
            if self.fire_rates[i] > self.target_fire_rate * 2:
                # This neuron is overworked → split
                cid = self.cluster_ids[i]
                result = self.bud(cid)
                if result:
                    events.append({
                        "event_type": "BUD",
                        "cluster_a": result[0],
                        "cluster_b": result[1],
                        "metadata": {"parent": cid, "step": step},
                    })
                    if len(events) >= 4:
                        break  # rate limit

        # DORMANCY: neurons that never fire → sleep
        for i in range(n):
            if self.dormant[i]:
                continue
            if self.ages[i] > 500 and self.fire_rates[i] < self.target_fire_rate * 0.1:
                self.dormant[i] = True
                events.append({
                    "event_type": "DORMANT",
                    "cluster_a": self.cluster_ids[i],
                    "cluster_b": None,
                    "metadata": {"step": step, "fire_rate": self.fire_rates[i].item()},
                })

        # CONNECT: frequently co-firing neurons that aren't connected
        # (simplified — just check top co-fired pairs from last forward)
        if self._last_fired is not None:
            fired_idx = self._last_fired.nonzero().squeeze(1)
            if len(fired_idx) >= 2:
                for i in range(min(len(fired_idx), 5)):
                    for j in range(i + 1, min(len(fired_idx), 5)):
                        ii, jj = fired_idx[i].item(), fired_idx[j].item()
                        if (ii, jj) not in self._edge_strengths:
                            self._edge_strengths[(ii, jj)] = 0.1
                            self._edge_strengths[(jj, ii)] = 0.1
                            events.append({
                                "event_type": "CONNECT",
                                "cluster_a": self.cluster_ids[ii],
                                "cluster_b": self.cluster_ids[jj],
                                "metadata": {"step": step},
                            })

        # PRUNE: weak edges
        to_prune = [(k, s) for k, s in self._edge_strengths.items() if s < 0.01]
        for k, _ in to_prune[:10]:
            del self._edge_strengths[k]

        return events

    # ── Serialization ──

    def state_dict(self) -> dict:
        return {
            "weights": self.weights[:self.n].clone(),
            "thresholds": self.thresholds[:self.n].clone(),
            "fire_rates": self.fire_rates[:self.n].clone(),
            "layer_indices": self.layer_indices[:self.n].clone(),
            "dormant": self.dormant[:self.n].clone(),
            "ages": self.ages[:self.n].clone(),
            "cluster_ids": list(self.cluster_ids),
            "edge_strengths": dict(self._edge_strengths),
            "activation_buffer": self.activation_buffer.clone(),
            "n": self.n,
        }

    def load_state_dict(self, d: dict):
        n = d["n"]
        self._ensure_capacity(n)
        self.n = n
        self.weights[:n] = d["weights"]
        self.thresholds[:n] = d["thresholds"]
        self.fire_rates[:n] = d["fire_rates"]
        self.layer_indices[:n] = d["layer_indices"]
        self.dormant[:n] = d["dormant"]
        self.ages[:n] = d["ages"]
        self.cluster_ids = d["cluster_ids"]
        self._id_to_idx = {cid: i for i, cid in enumerate(self.cluster_ids)}
        self._edge_strengths = d["edge_strengths"]
        self.activation_buffer = d["activation_buffer"]
        print(f"[brain] restored {n} neurons, {len(self._edge_strengths)} edges", flush=True)

    # ── Stats ──

    def summary(self) -> dict:
        n = self.n
        active = (~self.dormant[:n]).sum().item()
        layers = set()
        for i in range(n):
            if not self.dormant[i]:
                layers.add(int(self.layer_indices[i].item()))
        return {
            "cluster_count": active,
            "node_count": active,  # 1:1 in v2
            "edge_count": len(self._edge_strengths),
            "dormant_count": self.dormant[:n].sum().item(),
            "layer_count": len(layers),
            "active_tiles": active,
            "quadtree_depth": 0,
        }
