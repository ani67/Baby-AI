"""
BrainV2 — distributed representation engine with REFLECT.

Key differences from BrainState (brain.py):

  1. REFLECT: backward error propagation through edges (not backprop — no
     chain rule). Error flows top-to-bottom through the same edge network,
     giving every neuron a directional correction signal.

  2. Hybrid learning: weight_delta = alpha * FF_signal + beta * REFLECT_signal.
     FF stays dominant (local goodness), REFLECT adds cross-layer coordination.

  3. Developmental phases:
     - Phase 1 (step < 50K): rapid neuron budding up to MAX_NEURONS
     - Phase 2 (step >= 50K): zero budding, edge growth only, neuron recycling

  4. Dense edge network: target 200 edges/active neuron. Edges are the primary
     learning substrate. Edge formation guided by REFLECT error correlations.

  5. Edge lifecycle tracking: last_used timestamps for intelligent pruning.
     Edges unused for 5000 steps with strength < 0.01 get pruned.

  6. Capped THINK rounds (2 instead of 3) — dense edges carry more signal
     per round, so fewer rounds needed.

Same interface as BrainState: forward(x), update(x, teacher_vec), summary(),
state_dict(), load_state_dict(). Drop-in replacement.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .working_memory import WorkingMemory


_MPS_AVAILABLE = torch.backends.mps.is_available()
_MPS_NEURON_THRESHOLD = 2000


class BrainV2:
    def __init__(
        self,
        dim: int = 512,
        initial_size: int = 4,
        max_neurons: int = 10000,
        device: str = "cpu",
        alpha: float = 0.7,
        beta: float = 0.3,
        phase_boundary: int = 50000,
        target_edges_per_neuron: int = 200,
    ):
        self.dim = dim
        self.max_neurons = max_neurons
        self.max_size = max_neurons  # backward compat alias
        self.device = torch.device(device)
        self.n = 0  # current number of neurons (including dormant)

        # Hybrid learning coefficients
        self.alpha = alpha
        self.beta = beta

        # Developmental phase config
        self.phase_boundary = phase_boundary
        self.target_edges_per_neuron = target_edges_per_neuron

        # Pre-allocated GPU tensors (grow by doubling)
        cap = max(initial_size * 2, 64)
        self._cap = cap
        self.weights = torch.zeros(cap, dim, device=self.device)
        self.thresholds = torch.full((cap,), 0.15, device=self.device)
        self.fire_rates = torch.zeros(cap, device=self.device)
        self.layer_indices = torch.zeros(cap, device=self.device)
        self.dormant = torch.zeros(cap, dtype=torch.bool, device=self.device)
        self.ages = torch.zeros(cap, dtype=torch.long, device=self.device)

        # Edges: dict for growth flexibility, with lifecycle tracking
        self._edge_strengths: dict[tuple[int, int], float] = {}
        self._edge_last_used: dict[tuple[int, int], int] = {}
        self._edge_matrix_cache: torch.Tensor | None = None
        self._edge_matrix_n: int = 0

        # Metadata
        self.cluster_ids: list[str] = []
        self._id_to_idx: dict[str, int] = {}

        # Config
        self.target_fire_rate = 0.05
        self.threshold_adapt_rate = 0.001
        self.lr = 0.01
        self.max_rounds = 2  # fewer rounds than v1 — dense edges compensate

        # Cached state from last forward
        self._last_fired: torch.Tensor | None = None
        self._last_scores: torch.Tensor | None = None
        self._last_prediction: torch.Tensor | None = None

        # Learned projection (residual linear transform)
        self.projection = torch.zeros(dim, dim, device=self.device)
        self.projection_alpha = 0.0
        self._update_count = 0

        # Prediction weights (predictive coding)
        self.prediction_weights = torch.zeros(cap, dim, device=self.device)

        # Activation buffer
        self.activation_buffer = torch.zeros(dim, device=self.device)
        self.buffer_decay = 0.9
        self.buffer_weight = 0.15

        # REFLECT: per-neuron error buffer (persists across reflect calls for smoothing)
        self._neuron_error_buffer = torch.zeros(cap, device=self.device)

        # Step counter (global)
        self._step_count = 0

        # Initialize neurons
        for i in range(initial_size):
            self.add_neuron(
                cluster_id=f"c_{i:02d}",
                weights=F.normalize(torch.randn(dim), dim=0),
                layer_index=0 if i < initial_size // 2 else 1,
            )

    # ── Properties ──

    @property
    def phase(self) -> int:
        """Current developmental phase. 1 = foundation (budding), 2 = refinement (edges only)."""
        return 1 if self._step_count < self.phase_boundary else 2

    # ── Core State Management ──

    def _ensure_capacity(self, needed: int):
        """Double capacity if needed."""
        if needed <= self._cap:
            return
        new_cap = max(self._cap * 2, needed)
        for attr in [
            "weights", "thresholds", "fire_rates", "layer_indices",
            "ages", "prediction_weights",
        ]:
            old = getattr(self, attr)
            new = torch.zeros(new_cap, *old.shape[1:], dtype=old.dtype, device=self.device)
            new[: self._cap] = old
            setattr(self, attr, new)
        # Bool tensor (dormant)
        old_d = self.dormant
        new_d = torch.zeros(new_cap, dtype=torch.bool, device=self.device)
        new_d[: self._cap] = old_d
        self.dormant = new_d
        # Error buffer
        old_err = self._neuron_error_buffer
        new_err = torch.zeros(new_cap, device=self.device)
        new_err[: self._cap] = old_err
        self._neuron_error_buffer = new_err
        # Defaults for new slots
        self.thresholds[self._cap : new_cap] = 0.15
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
        self.fire_rates[idx] = self.target_fire_rate
        self.ages[idx] = 0
        self.dormant[idx] = False
        self._neuron_error_buffer[idx] = 0.0
        self.cluster_ids.append(cluster_id)
        self._id_to_idx[cluster_id] = idx
        self.n += 1
        return idx

    def remove_neuron(self, cluster_id: str):
        """Mark a neuron as dormant."""
        idx = self._id_to_idx.get(cluster_id)
        if idx is not None:
            self.dormant[idx] = True

    def _maybe_migrate_to_mps(self):
        """Switch to MPS when neuron count crosses threshold."""
        if not _MPS_AVAILABLE or self.device.type == "mps":
            return
        active = int((~self.dormant[: self.n]).sum().item())
        if active < _MPS_NEURON_THRESHOLD:
            return
        print(f"[brain-v2] migrating to MPS (active={active} > {_MPS_NEURON_THRESHOLD})", flush=True)
        new_device = torch.device("mps")
        for attr in [
            "weights", "thresholds", "fire_rates", "layer_indices", "ages",
            "prediction_weights", "_neuron_error_buffer",
        ]:
            setattr(self, attr, getattr(self, attr).to(new_device))
        self.dormant = self.dormant.to(new_device)
        self.activation_buffer = self.activation_buffer.to(new_device)
        self.projection = self.projection.to(new_device)
        self._invalidate_edge_cache()
        self.device = new_device

    # ── Edges ──

    def add_edge(self, from_id: str, to_id: str, strength: float = 0.1):
        from_idx = self._id_to_idx.get(from_id)
        to_idx = self._id_to_idx.get(to_id)
        if from_idx is not None and to_idx is not None:
            key = (from_idx, to_idx)
            self._edge_strengths[key] = strength
            self._edge_last_used[key] = self._step_count
            self._invalidate_edge_cache()

    def _add_edge_by_idx(self, from_idx: int, to_idx: int, strength: float = 0.1):
        """Add edge by neuron index (internal use — skips id lookup)."""
        key = (from_idx, to_idx)
        self._edge_strengths[key] = strength
        self._edge_last_used[key] = self._step_count
        self._invalidate_edge_cache()

    def edge_exists(self, from_id: str, to_id: str) -> bool:
        from_idx = self._id_to_idx.get(from_id)
        to_idx = self._id_to_idx.get(to_id)
        if from_idx is None or to_idx is None:
            return False
        return (from_idx, to_idx) in self._edge_strengths

    def _build_edge_matrix(self) -> torch.Tensor:
        """Build sparse COO edge matrix. Cached until edges change."""
        if self._edge_matrix_cache is not None and self._edge_matrix_n == self.n:
            return self._edge_matrix_cache
        n = self.n
        if not self._edge_strengths:
            mat = torch.sparse_coo_tensor(
                torch.empty(2, 0, dtype=torch.long, device=self.device),
                torch.empty(0, device=self.device),
                size=(n, n),
            ).coalesce()
        else:
            indices = []
            values = []
            for (i, j), s in self._edge_strengths.items():
                if i < n and j < n:
                    indices.append((i, j))
                    values.append(s)
            if indices:
                idx_t = torch.tensor(indices, dtype=torch.long, device=self.device).t()
                val_t = torch.tensor(values, dtype=torch.float32, device=self.device)
                mat = torch.sparse_coo_tensor(idx_t, val_t, size=(n, n)).coalesce()
            else:
                mat = torch.sparse_coo_tensor(
                    torch.empty(2, 0, dtype=torch.long, device=self.device),
                    torch.empty(0, device=self.device),
                    size=(n, n),
                ).coalesce()
        self._edge_matrix_cache = mat
        self._edge_matrix_n = n
        return mat

    def _invalidate_edge_cache(self):
        self._edge_matrix_cache = None

    # ── REFLECT: Backward Error Propagation ──

    @torch.no_grad()
    def reflect(self, error: torch.Tensor) -> torch.Tensor:
        """
        Propagate error backward through edges in ONE sparse matmul.

        The edge matrix transpose naturally reverses signal flow (j->i instead
        of i->j), so a single matmul propagates error from output neurons back
        through the entire edge network.

        Returns per-neuron error magnitudes as a tensor of shape (n,).
        """
        n = self.n
        if n == 0 or not self._edge_strengths:
            return torch.zeros(max(n, 1), device=self.device)

        error = error.to(self.device)

        # Initialize error at output neurons: fired neurons get error proportional
        # to their contribution (score-weighted)
        neuron_errors = torch.zeros(n, device=self.device)

        if self._last_fired is not None and self._last_scores is not None:
            fired_idx = self._last_fired.nonzero().squeeze(1)
            if len(fired_idx) > 0:
                fired_weights = F.normalize(self.weights[fired_idx], dim=1)
                per_neuron_err = (fired_weights * error.unsqueeze(0)).sum(dim=1).abs()
                neuron_errors[fired_idx] = per_neuron_err

        # Single sparse matmul: transpose reverses edge direction for backward flow
        edge_mat = self._build_edge_matrix()
        backflow = torch.sparse.mm(
            edge_mat.t(), neuron_errors.unsqueeze(1)
        ).squeeze(1)
        neuron_errors = neuron_errors + backflow

        # Normalize to prevent explosion (L2 norm cap)
        err_norm = neuron_errors.norm()
        if err_norm > 1.0:
            neuron_errors = neuron_errors / err_norm

        # Update persistent error buffer (EMA smoothing)
        self._neuron_error_buffer[:n] = (
            0.9 * self._neuron_error_buffer[:n] + 0.1 * neuron_errors
        )

        return neuron_errors

    # ── Forward Pass ──

    @torch.no_grad()
    def forward(self, x: torch.Tensor, _pre_sensed=None) -> tuple[torch.Tensor, dict]:
        """
        Parallel forward: SENSE → FIRE → THINK → OUTPUT.
        Same interface as BrainState.forward().
        THINK capped at 2 rounds (dense edges = more signal per round).
        """
        x = x.to(self.device)
        n = self.n
        self._step_count += 1

        # Apply buffer
        buf_norm = self.activation_buffer.norm()
        if buf_norm > 1e-6:
            buf_dir = self.activation_buffer / buf_norm
            effective_x = F.normalize(x + self.buffer_weight * buf_dir, dim=0)
        else:
            effective_x = F.normalize(x, dim=0)

        # PROJECT — learned residual transform
        if self.projection_alpha > 0:
            projected = F.normalize(
                effective_x + self.projection_alpha * (self.projection @ effective_x),
                dim=0,
            )
            effective_x = projected

        # 1. SENSE
        active_mask = ~self.dormant[:n]
        if _pre_sensed is not None:
            active_weights, active_idx = _pre_sensed
            n_active = len(active_idx)
        else:
            active_idx = active_mask.nonzero().squeeze(1)
            n_active = len(active_idx)
            active_weights = F.normalize(self.weights[active_idx], dim=1)

        active_scores = active_weights @ effective_x

        scores = torch.zeros(n, device=self.device)
        scores[active_idx] = active_scores

        # 2. FIRE
        fired = torch.zeros(n, dtype=torch.bool, device=self.device)
        active_fired = active_scores > self.thresholds[active_idx]
        fired[active_idx] = active_fired

        if fired.sum() < 4:
            _, top_local = active_scores.topk(min(4, n_active))
            for li in top_local:
                fired[active_idx[li]] = True

        # 3. THINK — message passing (capped at 2 rounds for dense edges)
        thresholds_n = self.thresholds[:n]
        confidence = torch.zeros(n, device=self.device)
        confidence[fired] = (scores[fired] - thresholds_n[fired]) / thresholds_n[fired].clamp(min=0.01)

        do_message_pass = (self._step_count % 20 == 0) and self._edge_strengths and fired.sum() > 0
        if do_message_pass:
            edge_mat = self._build_edge_matrix()
            for _ in range(self.max_rounds):
                fired_idx = fired.nonzero().squeeze(1)
                if len(fired_idx) == 0:
                    break

                fired_scores = scores[fired_idx]
                mean_s = fired_scores.mean()
                std_s = fired_scores.std()
                if std_s > 1e-6:
                    surprise = (scores - mean_s) / std_s
                else:
                    surprise = torch.zeros(n, device=self.device)

                send_weight = scores * surprise.clamp(min=0.0)
                score_vec = torch.zeros(n, device=self.device)
                score_vec[fired_idx] = send_weight[fired_idx]
                message_strength = torch.sparse.mm(
                    edge_mat, score_vec.unsqueeze(1)
                ).squeeze(1)
                new_scores = scores + 0.05 * message_strength

                confidence = torch.zeros(n, device=self.device)
                confidence[fired] = (
                    (new_scores[fired] - thresholds_n[fired])
                    / thresholds_n[fired].clamp(min=0.01)
                )

                newly_fired = (new_scores > thresholds_n) & active_mask & ~fired
                if newly_fired.sum() == 0:
                    break
                fired = fired | newly_fired
                confidence[newly_fired] = (
                    (new_scores[newly_fired] - thresholds_n[newly_fired])
                    / thresholds_n[newly_fired].clamp(min=0.01)
                )
                scores = new_scores

            # Mark traversed edges as used (via sparse tensor indices)
            fired_set_t = fired.nonzero().squeeze(1)
            if len(fired_set_t) > 0 and edge_mat._nnz() > 0:
                edge_indices = edge_mat.indices()  # (2, E)
                # Check if either endpoint is fired
                fired_lookup = torch.zeros(n, dtype=torch.bool, device=self.device)
                fired_lookup[fired_set_t] = True
                src_fired = fired_lookup[edge_indices[0]]
                dst_fired = fired_lookup[edge_indices[1]]
                touched = src_fired | dst_fired
                touched_idx = touched.nonzero().squeeze(1)
                if len(touched_idx) > 0:
                    step = self._step_count
                    src_list = edge_indices[0][touched_idx].cpu().tolist()
                    dst_list = edge_indices[1][touched_idx].cpu().tolist()
                    for si, di in zip(src_list, dst_list):
                        self._edge_last_used[(si, di)] = step

        # 4. OUTPUT — confidence-weighted aggregate
        fired_idx = fired.nonzero().squeeze(1)
        if len(fired_idx) == 0:
            prediction = torch.zeros(self.dim, device=self.device)
        else:
            fw = self.weights[fired_idx]
            active_conf = confidence[fired_idx].clamp(min=0.0)
            attn = F.softmax(active_conf * 2.0, dim=0)
            prediction = F.normalize(attn @ fw, dim=0)

        # Cache for update
        self._last_fired = fired
        self._last_scores = scores
        self._last_prediction = prediction

        # Update buffer
        self.activation_buffer *= self.buffer_decay
        if len(fired_idx) > 0:
            k = min(5, len(fired_idx))
            _, top_local = scores[fired_idx].topk(k)
            top_idx = fired_idx[top_local]
            top_scores = scores[top_idx]
            top_weights = F.normalize(self.weights[top_idx], dim=1)
            self.activation_buffer += (top_scores.unsqueeze(1) * top_weights).sum(dim=0)

        # Age neurons
        self.ages[:n] += 1

        # Build activations dict
        if len(fired_idx) > 0:
            fired_cids = [self.cluster_ids[i] for i in fired_idx.cpu().tolist()]
            fired_vals = scores[fired_idx].cpu().tolist()
            activations = dict(zip(fired_cids, fired_vals))
        else:
            activations = {}

        return prediction.cpu(), activations

    def pre_sense(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Cache normalized active weights for batched SENSE."""
        n = self.n
        active_mask = ~self.dormant[:n]
        active_idx = active_mask.nonzero().squeeze(1)
        active_weights = F.normalize(self.weights[active_idx], dim=1)
        return active_weights, active_idx

    @torch.no_grad()
    def score_candidates(self, candidates: torch.Tensor) -> torch.Tensor:
        """Batched read-only scoring: how well does the brain already know each candidate?

        Takes (N, dim) tensor, returns (N,) similarity scores.
        No state mutation — no buffer updates, no age increments, no step_count changes.
        """
        candidates = F.normalize(candidates.to(self.device), dim=1)

        if self.projection_alpha > 0:
            candidates = F.normalize(
                candidates + self.projection_alpha * (candidates @ self.projection.T),
                dim=1,
            )

        active_mask = ~self.dormant[:self.n]
        active_idx = active_mask.nonzero().squeeze(1)
        n_active = len(active_idx)
        if n_active == 0:
            return torch.zeros(candidates.shape[0], device=self.device)

        active_weights = F.normalize(self.weights[active_idx], dim=1)
        active_thresholds = self.thresholds[active_idx]

        scores = active_weights @ candidates.T  # (A, N)

        fired = scores > active_thresholds.unsqueeze(1)
        fire_counts = fired.sum(dim=0)
        low_fire = (fire_counts < 4).nonzero().squeeze(1)
        if len(low_fire) > 0:
            _, top_k = scores[:, low_fire].topk(min(4, n_active), dim=0)
            for col_i, cand_i in enumerate(low_fire):
                fired[top_k[:, col_i], cand_i] = True

        confidence = (scores - active_thresholds.unsqueeze(1)) / active_thresholds.clamp(min=0.01).unsqueeze(1)
        confidence[~fired] = -1e9
        attn = F.softmax(confidence * 2.0, dim=0)
        attn[~fired] = 0.0

        predictions = (attn.T @ active_weights)  # (N, dim)
        predictions = F.normalize(predictions, dim=1)

        return (predictions * candidates).sum(dim=1)

    # ── Learning: FF + REFLECT Hybrid ──

    @torch.no_grad()
    def update(self, x: torch.Tensor, teacher_vec: torch.Tensor):
        """
        Hybrid update: alpha * FF_signal + beta * REFLECT_signal.

        FF signal: per-neuron differentiated learning (same as BrainState).
        REFLECT signal: backward error gives directional correction from output.
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

        prediction = self._last_prediction.to(self.device)
        error = teacher_vec - prediction  # (dim,)

        # ── FF SIGNAL (same as BrainState) ──
        fired_weights = self.weights[fired_idx]
        neuron_teacher_sim = (fired_weights * teacher_vec.unsqueeze(0)).sum(dim=1)
        signs = torch.where(neuron_teacher_sim > 0, 1.0, -0.5)

        active_scores = scores[fired_idx]
        total_score = active_scores.sum().clamp(min=1e-6)
        shares = (active_scores / total_score).unsqueeze(1)
        local_targets = F.normalize(fired_weights + shares * error.unsqueeze(0), dim=1)
        ff_deltas = signs.unsqueeze(1) * (local_targets - fired_weights)

        # ── REFLECT SIGNAL ──
        neuron_errors = self.reflect(error)  # tensor of shape (n,)

        # Vectorized: direction toward teacher scaled by per-neuron error magnitude
        err_magnitudes = neuron_errors[fired_idx]  # (num_fired,)
        directions = teacher_vec.unsqueeze(0) - self.weights[fired_idx]  # (num_fired, dim)
        reflect_deltas = err_magnitudes.unsqueeze(1) * directions

        # ── HYBRID COMBINE ──
        combined = self.alpha * ff_deltas + self.beta * reflect_deltas
        self.weights[fired_idx] = F.normalize(
            self.weights[fired_idx] + self.lr * combined, dim=1
        )

        # ── REFLECT-GUIDED EDGE FORMATION ──
        # If neuron A has high backward error AND neuron B has high activation
        # AND no edge exists → create edge
        self._reflect_edge_formation(neuron_errors, fired_idx, scores)

        # Train projection
        self._update_count += 1
        self.projection += 0.0001 * torch.outer(error, x)
        self.projection_alpha = min(1.0, self._update_count / 10000)

        # Hebbian update + edge maintenance (every 10 steps)
        if self._update_count % 10 == 0:
            self._hebbian_update(fired_idx, scores)

    def _reflect_edge_formation(
        self,
        neuron_errors: torch.Tensor,
        fired_idx: torch.Tensor,
        scores: torch.Tensor,
    ):
        """Form new edges based on REFLECT error correlations.

        Neurons with high backward error that lack connections to high-activation
        neurons get wired together. Uses tensor operations to find candidate pairs.
        """
        n = self.n
        if n == 0 or neuron_errors.sum() < 1e-6:
            return

        active_count = int((~self.dormant[:n]).sum().item())
        current_edge_density = len(self._edge_strengths) / max(active_count, 1)
        if current_edge_density >= self.target_edges_per_neuron:
            return

        # High error neurons: top 10% by error magnitude
        error_threshold = neuron_errors.quantile(0.9)
        high_err_idx = (neuron_errors > error_threshold.clamp(min=1e-6)).nonzero().squeeze(1)
        if len(high_err_idx) == 0:
            return

        # High activation neurons: scores > 0.3 among fired
        high_act_mask = scores > 0.3
        high_act_idx = high_act_mask.nonzero().squeeze(1)
        if len(high_act_idx) == 0:
            return

        # Limit candidates to keep this bounded
        if len(high_err_idx) > 10:
            high_err_idx = high_err_idx[:10]
        if len(high_act_idx) > 10:
            high_act_idx = high_act_idx[:10]

        # Score candidate pairs by error * activation product
        err_vals = neuron_errors[high_err_idx]  # (E,)
        act_vals = scores[high_act_idx]  # (A,)
        pair_scores = err_vals.unsqueeze(1) * act_vals.unsqueeze(0)  # (E, A)

        # Flatten and get top-K
        max_new = 50
        flat_scores = pair_scores.flatten()
        k = min(max_new, len(flat_scores))
        _, top_flat = flat_scores.topk(k)
        err_indices = top_flat // len(high_act_idx)
        act_indices = top_flat % len(high_act_idx)

        # Convert back to global neuron indices and add edges
        err_global = high_err_idx[err_indices].cpu().tolist()
        act_global = high_act_idx[act_indices].cpu().tolist()

        new_edges = 0
        step = self._step_count
        for a, b in zip(err_global, act_global):
            if a == b:
                continue
            if (a, b) not in self._edge_strengths:
                self._edge_strengths[(a, b)] = 0.1
                self._edge_last_used[(a, b)] = step
                new_edges += 1

        if new_edges > 0:
            self._invalidate_edge_cache()

    def _hebbian_update(self, fired_idx: torch.Tensor, scores: torch.Tensor):
        """Synaptic plasticity + edge lifecycle management."""
        # 1. Global decay + age-based pruning (batch via sparse tensor)
        self._decay_counter = getattr(self, "_decay_counter", 0) + 1
        if self._decay_counter % 100 == 0 and self._edge_strengths:
            edge_mat = self._build_edge_matrix()
            vals = edge_mat.values()
            decayed = vals * 0.99
            indices = edge_mat.indices()  # (2, E)

            # Build last_used tensor for age check
            step = self._step_count
            edge_keys = list(zip(indices[0].cpu().tolist(), indices[1].cpu().tolist()))
            last_used_t = torch.tensor(
                [self._edge_last_used.get(k, 0) for k in edge_keys],
                device=self.device, dtype=torch.float32,
            )
            ages = step - last_used_t

            # Prune: (strength < 0.01 AND age > 5000) OR strength < 0.005
            prune_mask = ((decayed < 0.01) & (ages > 5000)) | (decayed < 0.005)
            keep_mask = ~prune_mask

            if prune_mask.any():
                # Remove pruned edges from dicts
                prune_indices = prune_mask.nonzero().squeeze(1).cpu().tolist()
                for pi in prune_indices:
                    k = edge_keys[pi]
                    self._edge_strengths.pop(k, None)
                    self._edge_last_used.pop(k, None)

            # Update surviving edge strengths in dict
            if keep_mask.any():
                keep_indices = keep_mask.nonzero().squeeze(1).cpu().tolist()
                decayed_cpu = decayed.cpu()
                for ki in keep_indices:
                    k = edge_keys[ki]
                    self._edge_strengths[k] = decayed_cpu[ki].item()

            self._invalidate_edge_cache()

        # 2. Hebbian reinforcement — co-fired edges strengthen (small loop, max 5 neurons)
        if len(fired_idx) < 2:
            return
        fired_list = fired_idx[:5].tolist()
        step = self._step_count
        for i in range(len(fired_list)):
            for j in range(i + 1, len(fired_list)):
                idx_i, idx_j = fired_list[i], fired_list[j]
                co_act = scores[idx_i].item() * scores[idx_j].item()
                for pair in [(idx_i, idx_j), (idx_j, idx_i)]:
                    if pair in self._edge_strengths:
                        self._edge_strengths[pair] = min(
                            self._edge_strengths[pair] + 0.001 * co_act, 1.0
                        )
                        self._edge_last_used[pair] = step

        # 3. Predictive coding (uses edge matrix for neighbor lookup)
        if self._decay_counter % 50 == 0 and len(fired_idx) >= 2:
            fired_set = set(fired_idx[:10].tolist())
            for idx_i in list(fired_set)[:5]:
                pred_w = self.prediction_weights[idx_i]
                if pred_w.norm() < 1e-6:
                    self.prediction_weights[idx_i] = self.weights[idx_i].detach().clone()
                    continue
                # Find neighbors of idx_i via edge dict (bounded by fired set size)
                for idx_j in list(fired_set):
                    if idx_j == idx_i:
                        continue
                    if (idx_i, idx_j) not in self._edge_strengths:
                        continue
                    neighbor_w = self.weights[idx_j]
                    predicted = torch.dot(pred_w, neighbor_w).item()
                    actual = 1.0
                    if abs(predicted - actual) > 0.3:
                        direction = neighbor_w if actual > predicted else -neighbor_w
                        self.prediction_weights[idx_i] = F.normalize(
                            pred_w + 0.001 * direction, dim=0
                        )

    # ── Homeostatic Threshold Adaptation ──

    def adapt_thresholds(self):
        """Adjust thresholds so each neuron fires at target rate."""
        n = self.n
        fired = self._last_fired
        if fired is None:
            return
        self.fire_rates[:n] = 0.999 * self.fire_rates[:n] + 0.001 * fired.float()
        deviation = self.fire_rates[:n] - self.target_fire_rate
        self.thresholds[:n] = self.thresholds[:n] + self.threshold_adapt_rate * deviation
        self.thresholds[:n].clamp_(0.01, 0.95)

    # ── Growth ──

    def _recycle_neuron(self, cluster_id: str, weights: torch.Tensor, layer_index: float) -> int | None:
        """Reclaim the most dormant neuron and overwrite it. Phase 2 only.

        Selection criteria (in order):
          1. Must be dormant
          2. Longest dormancy (highest age since going dormant)
          3. Lowest fire_rate
          4. Fewest edges

        Returns recycled neuron index or None if no candidates.
        """
        n = self.n
        dormant_idx = self.dormant[:n].nonzero().squeeze(1)
        if len(dormant_idx) == 0:
            return None

        # Score each dormant neuron: higher = better candidate for recycling
        best_idx = None
        best_score = -1.0
        dormant_list = dormant_idx.cpu().tolist()

        for idx in dormant_list:
            age = self.ages[idx].item()
            fr = self.fire_rates[idx].item()
            # Count edges involving this neuron
            edge_count = sum(1 for (i, j) in self._edge_strengths if i == idx or j == idx)
            # Score: prefer old dormant neurons with low fire rate and few edges
            score = age * 1.0 + (1.0 - fr) * 1000.0 + (1.0 / (edge_count + 1)) * 100.0
            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx is None:
            return None

        # Remove old edges
        dead_edges = [k for k in self._edge_strengths if k[0] == best_idx or k[1] == best_idx]
        for k in dead_edges:
            del self._edge_strengths[k]
            self._edge_last_used.pop(k, None)
        if dead_edges:
            self._invalidate_edge_cache()

        # Overwrite the neuron
        old_cid = self.cluster_ids[best_idx]
        self.weights[best_idx] = F.normalize(weights.to(self.device), dim=0)
        self.thresholds[best_idx] = 0.15
        self.layer_indices[best_idx] = layer_index
        self.fire_rates[best_idx] = self.target_fire_rate
        self.ages[best_idx] = 0
        self.dormant[best_idx] = False
        self._neuron_error_buffer[best_idx] = 0.0
        self.prediction_weights[best_idx] = torch.zeros(self.dim, device=self.device)
        self.cluster_ids[best_idx] = cluster_id
        del self._id_to_idx[old_cid]
        self._id_to_idx[cluster_id] = best_idx

        return best_idx

    def bud(self, cluster_id: str) -> tuple[str, str] | None:
        """Split a neuron into two. Phase 1 only (phase 2 uses recycling)."""
        idx = self._id_to_idx.get(cluster_id)
        if idx is None:
            return None

        parent_w = self.weights[idx].clone()
        layer = self.layer_indices[idx].item()
        threshold = self.thresholds[idx].item()

        noise = 0.1 * torch.randn(self.dim, device=self.device)
        child_a_w = F.normalize(parent_w + noise, dim=0)
        child_b_w = F.normalize(parent_w - noise, dim=0)

        child_a_id = f"{cluster_id}a"
        child_b_id = f"{cluster_id}b"

        if self.phase == 1:
            # Normal budding: allocate new slots
            idx_a = self.add_neuron(child_a_id, child_a_w, layer, threshold)
            idx_b = self.add_neuron(child_b_id, child_b_w, layer + 0.5, threshold)
        else:
            # Phase 2: recycle dormant neurons
            idx_a = self._recycle_neuron(child_a_id, child_a_w, layer)
            idx_b = self._recycle_neuron(child_b_id, child_b_w, layer + 0.5)
            if idx_a is None or idx_b is None:
                return None  # no dormant neurons available to recycle

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
            self._edge_last_used.pop(k, None)
        for k, s in to_add:
            self._edge_strengths[k] = s
            self._edge_last_used[k] = self._step_count
        # Sibling edge
        self._edge_strengths[(idx_a, idx_b)] = 0.5
        self._edge_strengths[(idx_b, idx_a)] = 0.5
        self._edge_last_used[(idx_a, idx_b)] = self._step_count
        self._edge_last_used[(idx_b, idx_a)] = self._step_count

        # Dormant parent
        self.dormant[idx] = True

        self._invalidate_edge_cache()
        return child_a_id, child_b_id

    def growth_check(self, step: int) -> list[dict]:
        """Check for growth triggers. Returns list of events.

        Phase 1: neuron budding + edge formation.
        Phase 2: edge growth only + neuron recycling.
        """
        check_interval = min(200 + self.n // 5, 500)
        last_check = getattr(self, "_last_growth_step", -check_interval)
        if step - last_check < check_interval:
            return []

        self._maybe_migrate_to_mps()
        self._last_growth_step = step

        events = []
        n = self.n
        active_count = int((~self.dormant[:n]).sum().item())
        active_fr = self.fire_rates[:n][~self.dormant[:n]]
        edge_density = len(self._edge_strengths) / max(active_count, 1)

        print(
            f"[brain-v2-growth] step={step} phase={self.phase} active={active_count} "
            f"total={n} edges={len(self._edge_strengths)} "
            f"density={edge_density:.1f} "
            f"fire_rate min={active_fr.min():.4f} max={active_fr.max():.4f} "
            f"mean={active_fr.mean():.4f}",
            flush=True,
        )

        # BUD: phase 1 only, and only up to max_neurons
        if self.phase == 1 and active_count < self.max_neurons:
            max_buds = max(4, active_count // 25)
            bud_count = 0
            bud_threshold = self.target_fire_rate * 1.5
            for i in range(n):
                if self.dormant[i]:
                    continue
                if self.ages[i] < 200:
                    continue
                if self.fire_rates[i] > bud_threshold:
                    cid = self.cluster_ids[i]
                    result = self.bud(cid)
                    if result:
                        events.append({
                            "event_type": "BUD",
                            "cluster_a": result[0],
                            "cluster_b": result[1],
                            "metadata": {"parent": cid, "step": step},
                        })
                        bud_count += 1
                        if bud_count >= max_buds:
                            break
                        # Re-check active count against max
                        active_count = int((~self.dormant[:self.n]).sum().item())
                        if active_count >= self.max_neurons:
                            break
            if bud_count > 0:
                print(f"[brain-v2-growth] BUD {bud_count} neurons", flush=True)

        # DORMANCY: neurons that never fire → sleep
        dormancy_threshold = self.target_fire_rate * 0.02 * (100.0 / max(active_count, 100))
        for i in range(n):
            if self.dormant[i]:
                continue
            if self.ages[i] > 2000 and self.fire_rates[i] < dormancy_threshold:
                self.dormant[i] = True
                events.append({
                    "event_type": "DORMANT",
                    "cluster_a": self.cluster_ids[i],
                    "cluster_b": None,
                    "metadata": {"step": step, "fire_rate": self.fire_rates[i].item()},
                })
        dormant_count = sum(1 for e in events if e["event_type"] == "DORMANT")
        if dormant_count > 0:
            print(
                f"[brain-v2-growth] DORMANT {dormant_count} neurons "
                f"(threshold={dormancy_threshold:.6f})",
                flush=True,
            )

        # CONNECT: co-firing neurons that aren't connected
        if self._last_fired is not None:
            fired_idx = self._last_fired.nonzero().squeeze(1)
            if len(fired_idx) >= 2:
                for i in range(min(len(fired_idx), 5)):
                    for j in range(i + 1, min(len(fired_idx), 5)):
                        ii, jj = fired_idx[i].item(), fired_idx[j].item()
                        if (ii, jj) not in self._edge_strengths:
                            self._edge_strengths[(ii, jj)] = 0.1
                            self._edge_strengths[(jj, ii)] = 0.1
                            self._edge_last_used[(ii, jj)] = step
                            self._edge_last_used[(jj, ii)] = step
                            events.append({
                                "event_type": "CONNECT",
                                "cluster_a": self.cluster_ids[ii],
                                "cluster_b": self.cluster_ids[jj],
                                "metadata": {"step": step},
                            })
                self._invalidate_edge_cache()

        return events

    # ── Serialization ──

    def state_dict(self) -> dict:
        """Checkpoint. Always saves to CPU for portability."""
        return {
            "weights": self.weights[: self.n].cpu().clone(),
            "thresholds": self.thresholds[: self.n].cpu().clone(),
            "fire_rates": self.fire_rates[: self.n].cpu().clone(),
            "layer_indices": self.layer_indices[: self.n].cpu().clone(),
            "dormant": self.dormant[: self.n].cpu().clone(),
            "ages": self.ages[: self.n].cpu().clone(),
            "cluster_ids": list(self.cluster_ids),
            "edge_strengths": dict(self._edge_strengths),
            "edge_last_used": dict(self._edge_last_used),
            "activation_buffer": self.activation_buffer.cpu().clone(),
            "prediction_weights": self.prediction_weights[: self.n].cpu().clone(),
            "projection": self.projection.cpu().clone(),
            "projection_alpha": self.projection_alpha,
            "update_count": self._update_count,
            "step_count": self._step_count,
            "neuron_error_buffer": self._neuron_error_buffer[: self.n].cpu().clone(),
            "n": self.n,
            # Config for reproducibility
            "alpha": self.alpha,
            "beta": self.beta,
            "phase_boundary": self.phase_boundary,
            "max_neurons": self.max_neurons,
        }

    def load_state_dict(self, d: dict):
        """Restore from checkpoint."""
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
        self._edge_last_used = d.get("edge_last_used", {})
        self.activation_buffer = d["activation_buffer"].to(self.device)
        if "prediction_weights" in d:
            self.prediction_weights[:n] = d["prediction_weights"]
        if "projection" in d:
            self.projection = d["projection"].to(self.device)
            self.projection_alpha = d["projection_alpha"]
            self._update_count = d["update_count"]
        if "step_count" in d:
            self._step_count = d["step_count"]
        if "neuron_error_buffer" in d:
            self._neuron_error_buffer[:n] = d["neuron_error_buffer"]
        # Restore config
        if "alpha" in d:
            self.alpha = d["alpha"]
        if "beta" in d:
            self.beta = d["beta"]
        if "phase_boundary" in d:
            self.phase_boundary = d["phase_boundary"]
        if "max_neurons" in d:
            self.max_neurons = d["max_neurons"]
            self.max_size = d["max_neurons"]

        print(
            f"[brain-v2] restored {n} neurons, {len(self._edge_strengths)} edges, "
            f"phase={self.phase}, step={self._step_count}, device={self.device}",
            flush=True,
        )
        self._maybe_migrate_to_mps()

    # ── Stats ──

    def summary(self) -> dict:
        n = self.n
        dormant = self.dormant[:n]
        active = int((~dormant).sum().item())
        active_layers = self.layer_indices[:n][~dormant].cpu()
        layer_count = int(active_layers.unique().numel())
        edge_density = len(self._edge_strengths) / max(active, 1)
        return {
            "cluster_count": active,
            "node_count": active,
            "edge_count": len(self._edge_strengths),
            "dormant_count": n - active,
            "layer_count": layer_count,
            "active_tiles": active,
            "quadtree_depth": 0,
            # V2-specific
            "phase": self.phase,
            "step_count": self._step_count,
            "edge_density": round(edge_density, 1),
            "max_neurons": self.max_neurons,
            "alpha": self.alpha,
            "beta": self.beta,
        }
