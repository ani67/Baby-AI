"""
BrainState — the v2 parallel engine.

The entire brain as GPU-resident matrices. No Python loops in the hot path.
Every neuron evaluates simultaneously. Self-activation thresholds replace
the centralized resonance manager. Message passing replaces serial BFS.

Analogy: a stadium wave, not an announcer calling seat numbers.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .working_memory import WorkingMemory


_MPS_AVAILABLE = torch.backends.mps.is_available()
# MPS crossover: GPU wins at 2K+ neurons. Below that, CPU is faster due to
# Metal dispatch overhead (~0.8ms/call). Benchmarked on M1 Pro.
_MPS_NEURON_THRESHOLD = 2000


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

        # Learned projection layer (ported from v1 Phase D)
        # Residual linear transform: P(x) = normalize(x + α·Δ·x)
        # α ramps 0→1 over 10K steps so early learning isn't distorted.
        self.projection = torch.zeros(dim, dim, device=self.device)
        self.projection_alpha = 0.0
        self._update_count = 0

        # Neighbor prediction (predictive coding)
        # Each neuron predicts whether connected neighbors will fire.
        # prediction_weights[i] = 512-dim vector: dot(pred_w[i], neighbor_w[j]) → expected co-firing
        # Trained when predictions are wrong → pressure toward consistent, predictable behavior.
        self.prediction_weights = torch.zeros(cap, dim, device=self.device)

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
        for attr in ['weights', 'thresholds', 'fire_rates', 'layer_indices', 'ages', 'prediction_weights']:
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

    def _maybe_migrate_to_mps(self):
        """Switch to MPS when neuron count crosses threshold. One-time migration."""
        if not _MPS_AVAILABLE or self.device.type == "mps":
            return
        active = int((~self.dormant[:self.n]).sum().item())
        if active < _MPS_NEURON_THRESHOLD:
            return
        print(f"[brain] migrating to MPS (active={active} > {_MPS_NEURON_THRESHOLD})", flush=True)
        new_device = torch.device("mps")
        for attr in ['weights', 'thresholds', 'fire_rates', 'layer_indices', 'ages']:
            setattr(self, attr, getattr(self, attr).to(new_device))
        self.dormant = self.dormant.to(new_device)
        self.activation_buffer = self.activation_buffer.to(new_device)
        self.projection = self.projection.to(new_device)
        self.prediction_weights = self.prediction_weights.to(new_device)
        if hasattr(self, '_working_memory') and self._working_memory is not None:
            self._working_memory.to(new_device)
        self._invalidate_edge_cache()
        self.device = new_device

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
        """Build sparse COO edge matrix for message passing. Cached until edges change.
        At 10K neurons: sparse ~2MB vs dense ~400MB. Message passing stays O(E) not O(N²).
        """
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

        # 0. PROJECT — learned residual transform (Phase D from v1)
        if self.projection_alpha > 0:
            projected = F.normalize(
                effective_x + self.projection_alpha * (self.projection @ effective_x),
                dim=0,
            )
            effective_x = projected

        # 1. SENSE — only active neurons evaluate (skip dormant = major speedup)
        active_mask = ~self.dormant[:n]
        active_idx = active_mask.nonzero().squeeze(1)
        n_active = len(active_idx)

        # Compact: gather only active weights for the matmul
        active_weights = F.normalize(self.weights[active_idx], dim=1)  # (n_active, 512)
        active_scores = active_weights @ effective_x  # (n_active,)

        # Scatter back to full-size arrays (dormant neurons get score 0)
        scores = torch.zeros(n, device=self.device)
        scores[active_idx] = active_scores

        # 2. FIRE — self-activation (only among active neurons)
        fired = torch.zeros(n, dtype=torch.bool, device=self.device)
        active_fired = active_scores > self.thresholds[active_idx]
        fired[active_idx] = active_fired

        # Guarantee minimum firing (at least 4 neurons)
        if fired.sum() < 4:
            _, top_local = active_scores.topk(min(4, n_active))
            for li in top_local:
                fired[active_idx[li]] = True

        # 3. THINK — confidence-weighted message passing (ported from v1 deliberation)
        #
        # V1 insight: raw scores let generalists dominate. Instead, compute
        # "surprise" — a z-scored confidence that measures how well each neuron
        # matches THIS input relative to the cohort. Only surprised (above-average)
        # neurons send messages, weighted by their surprise. This creates diverse
        # communities instead of one mega-cluster.
        #
        # confidence = margin above threshold (how strongly it fired)
        # surprise   = z-score of scores across fired neurons (input-specific signal)
        thresholds_n = self.thresholds[:n]
        confidence = torch.zeros(n, device=self.device)
        confidence[fired] = (scores[fired] - thresholds_n[fired]) / thresholds_n[fired].clamp(min=0.01)

        self._step_count = getattr(self, '_step_count', 0) + 1
        do_message_pass = (self._step_count % 5 == 0) and self._edge_strengths and fired.sum() > 0
        if do_message_pass:
            edge_mat = self._build_edge_matrix()  # sparse COO
            for _ in range(self.max_rounds):
                fired_idx = fired.nonzero().squeeze(1)
                if len(fired_idx) == 0:
                    break

                # Surprise: z-score of fired scores (input-specific, not absolute)
                fired_scores = scores[fired_idx]
                mean_s = fired_scores.mean()
                std_s = fired_scores.std()
                if std_s > 1e-6:
                    surprise = (scores - mean_s) / std_s
                else:
                    surprise = torch.zeros(n, device=self.device)

                # Only positive-surprise neurons send messages (above-average match).
                # Message strength = score * clamp(surprise, min=0) — confident
                # experts influence uncertain neighbors, not the reverse.
                send_weight = scores * surprise.clamp(min=0.0)
                score_vec = torch.zeros(n, device=self.device)
                score_vec[fired_idx] = send_weight[fired_idx]
                message_strength = torch.sparse.mm(edge_mat, score_vec.unsqueeze(1)).squeeze(1)
                new_scores = scores + 0.05 * message_strength

                # Update confidence with neighbor reinforcement
                confidence = torch.zeros(n, device=self.device)
                confidence[fired] = (new_scores[fired] - thresholds_n[fired]) / thresholds_n[fired].clamp(min=0.01)

                # Check for new firings
                newly_fired = (new_scores > thresholds_n) & active_mask & ~fired
                if newly_fired.sum() == 0:
                    break
                fired = fired | newly_fired
                confidence[newly_fired] = (new_scores[newly_fired] - thresholds_n[newly_fired]) / thresholds_n[newly_fired].clamp(min=0.01)
                scores = new_scores

        # 4. OUTPUT — confidence-weighted aggregate (v1: "confident experts contribute more")
        #
        # Instead of uniform softmax over raw scores, weight by post-deliberation
        # confidence: neurons that fired strongly AND were reinforced by neighbors
        # contribute more. Confidence = margin above threshold — input-specific,
        # not dominated by generalists.
        fired_idx = fired.nonzero().squeeze(1)
        if len(fired_idx) == 0:
            prediction = torch.zeros(self.dim, device=self.device)
        else:
            active_weights = self.weights[fired_idx]  # (K, 512)
            active_conf = confidence[fired_idx].clamp(min=0.0)  # (K,)
            # Gentle softmax on confidence preserves multi-contributor property
            # while letting confident experts lead.
            attn = F.softmax(active_conf * 2.0, dim=0)
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

    # ── Multi-Step Reasoning ──

    def reason(
        self,
        x: torch.Tensor,
        steps: int = 5,
        memory: WorkingMemory | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Multi-step inference: iterate forward passes, feeding prediction back
        as input. The activation buffer accumulates across iterations, building
        up temporal context with each step.

        If memory is provided:
          - Before each forward: read from memory and blend with input
          - After each forward: write prediction to memory

        This does NOT interfere with normal forward() — it calls forward()
        internally. Each forward() updates the activation buffer, so later
        iterations benefit from richer context.

        Returns: (final_prediction, final_activations)
        """
        x = F.normalize(x.to(self.device), dim=0)

        prediction = x
        activations: dict = {}

        for _ in range(steps):
            # Memory read: blend stored context into input
            if memory is not None and memory.occupancy > 0:
                mem_context = memory.read(x)
                x = F.normalize(0.8 * x + 0.2 * mem_context, dim=0)

            # Forward pass (updates activation buffer internally)
            prediction, activations = self.forward(x)

            # Memory write: store prediction for future steps
            if memory is not None:
                memory.write(prediction)

            # Blend prediction back as next input (keep some of original query)
            x = F.normalize(0.7 * prediction + 0.3 * x, dim=0)

        return prediction, activations

    # ── Learning: Parallel Correction-Based ──

    def update(self, x: torch.Tensor, teacher_vec: torch.Tensor):
        """
        Per-neuron differentiated learning.

        Each fired neuron independently asks: "am I aligned with the teacher?"
          - Yes (positive dot product) → move toward teacher (attractive)
          - No  (negative dot product) → move away from teacher (repulsive)

        This is the critical force that creates specialization. Without it,
        all neurons chase the same target and converge into one blob.
        Like electrons: attraction alone = collapse. Repulsion creates structure.
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

        # Per-neuron signal: each neuron decides its own relationship to teacher.
        # Neurons aligned with teacher → attract (move toward).
        # Neurons misaligned → repel (move away). This IS the repulsive force.
        fired_weights = self.weights[fired_idx]  # (K, 512)
        neuron_teacher_sim = (fired_weights * teacher_vec.unsqueeze(0)).sum(dim=1)  # (K,)
        signs = torch.where(neuron_teacher_sim > 0, 1.0, -0.5)  # asymmetric: repel gently

        # Distributed error: each neuron fixes its SHARE of the error, proportional
        # to its activation. A neuron contributing 10% of total activation corrects
        # 10% of the gap. This prevents convergence collapse — neurons near the target
        # get small corrections, neurons far away get large ones.
        prediction = self._last_prediction.to(self.device)
        error = teacher_vec - prediction  # (512,)
        active_scores = scores[fired_idx]  # (K,)
        total_score = active_scores.sum().clamp(min=1e-6)
        shares = (active_scores / total_score).unsqueeze(1)  # (K, 1) — each neuron's share
        local_targets = F.normalize(fired_weights + shares * error.unsqueeze(0), dim=1)  # (K, 512)
        deltas = local_targets - fired_weights  # direction toward local target
        updates = signs.unsqueeze(1) * deltas  # attract/repel preserved
        self.weights[fired_idx] = self.weights[fired_idx] + self.lr * updates

        # Re-normalize (keep unit vectors)
        self.weights[fired_idx] = F.normalize(self.weights[fired_idx], dim=1)

        # ── ERROR BACKFLOW ──
        # Propagate error backward along edges. If neuron A sent a message
        # to neuron B during THINK, and B had high error, A should learn too.
        # This is NOT backprop (no chain rule). It's local: each neuron sees
        # only its direct neighbors' error, weighted by edge strength.
        #
        # Forward (THINK): messages = edges @ scores      (columns → rows)
        # Backward:        upstream_err = edges.T @ error  (rows → columns)
        #
        # Only runs when message passing ran (every 5 steps).
        if getattr(self, '_step_count', 0) % 5 == 0 and self._edge_strengths:
            n = self.n
            # Per-neuron error magnitude: how far was each fired neuron from teacher?
            neuron_errors = torch.zeros(n, device=self.device)
            neuron_errors[fired_idx] = 1.0 - neuron_teacher_sim.clamp(-1, 1)  # 0=perfect, 2=opposite

            # Backflow: upstream neurons get weighted error from downstream
            edge_mat = self._build_edge_matrix()  # sparse COO, cached
            backflow = torch.sparse.mm(
                edge_mat.t(),  # transpose: reverse direction
                neuron_errors.unsqueeze(1),
            ).squeeze(1)  # (N,)

            # Neurons with high backflow error and that fired → small correction
            # toward teacher. Scale is tiny (0.1× lr) — this is a hint, not a command.
            backflow_mask = (backflow > 0.1) & fired & (backflow > neuron_errors)
            if backflow_mask.sum() > 0:
                bf_idx = backflow_mask.nonzero().squeeze(1)
                bf_weights = self.weights[bf_idx]
                bf_teacher_sim = (bf_weights * teacher_vec.unsqueeze(0)).sum(dim=1)
                # Only correct neurons that are somewhat aligned (avoid pushing random neurons)
                aligned = bf_teacher_sim > -0.3
                if aligned.sum() > 0:
                    aligned_idx = bf_idx[aligned]
                    bf_scale = backflow[aligned_idx].clamp(max=1.0).unsqueeze(1) * 0.1
                    bf_direction = teacher_vec.unsqueeze(0) - self.weights[aligned_idx]
                    self.weights[aligned_idx] += self.lr * bf_scale * bf_direction
                    self.weights[aligned_idx] = F.normalize(self.weights[aligned_idx], dim=1)

        # Train projection: outer product of error ⊗ input, scaled small
        self._update_count += 1
        self.projection += 0.0001 * torch.outer(error, x)
        self.projection_alpha = min(1.0, self._update_count / 10000)

        # Hebbian edge update + predictive coding: co-firing patterns emerge
        # over hundreds of steps, so running every 10th update is sufficient.
        # This skips ~1.1ms of Python dict/list iteration on 9 out of 10 steps.
        if self._update_count % 10 == 0:
            self._hebbian_update(fired_idx, scores)

    def _hebbian_update(self, fired_idx: torch.Tensor, scores: torch.Tensor):
        """Synaptic plasticity: co-firing strengthens, silence weakens.
        Every edge decays slightly each step. Only co-fired edges get reinforced.
        Net effect: unused synapses die naturally — no explicit prune needed."""
        # 1. Global decay — all edges weaken a tiny bit (use it or lose it)
        # 0.9999 per step = half-life of ~7000 steps. Fast enough to cull,
        # slow enough that meaningful edges survive between co-firings.
        self._decay_counter = getattr(self, '_decay_counter', 0) + 1
        if self._decay_counter % 100 == 0:  # batch decay every 100 steps for speed
            dead = []
            for k in self._edge_strengths:
                self._edge_strengths[k] *= 0.99  # 0.99^1 per 100 steps ≈ 0.9999/step
                if self._edge_strengths[k] < 0.005:
                    dead.append(k)
            for k in dead:
                del self._edge_strengths[k]
            if dead:
                self._invalidate_edge_cache()

        # 2. Hebbian reinforcement — co-fired edges strengthen
        if len(fired_idx) < 2:
            return
        fired_list = fired_idx[:5].tolist()
        for i in range(len(fired_list)):
            for j in range(i + 1, len(fired_list)):
                idx_i, idx_j = fired_list[i], fired_list[j]
                co_act = scores[idx_i].item() * scores[idx_j].item()
                for pair in [(idx_i, idx_j), (idx_j, idx_i)]:
                    if pair in self._edge_strengths:
                        self._edge_strengths[pair] = min(
                            self._edge_strengths[pair] + 0.001 * co_act, 1.0
                        )

        # 3. Predictive coding — each neuron learns to predict its neighbors
        # If neuron i predicted neuron j would fire and j didn't (or vice versa),
        # nudge i's prediction_weights toward j's actual weight vector.
        # This creates pressure for CONSISTENT co-firing patterns = specialization.
        if self._decay_counter % 50 == 0 and len(fired_idx) >= 2:
            fired_set = set(fired_idx[:10].tolist())
            for idx_i in list(fired_set)[:5]:
                pred_w = self.prediction_weights[idx_i]
                if pred_w.norm() < 1e-6:
                    # Initialize prediction weights from neuron's own weights
                    self.prediction_weights[idx_i] = self.weights[idx_i].detach().clone()
                    continue
                # Check connected neighbors
                for (a, b), s in list(self._edge_strengths.items())[:20]:
                    if a != idx_i:
                        continue
                    neighbor_w = self.weights[b]
                    # Predicted co-firing: dot(pred_w, neighbor_w)
                    predicted = torch.dot(pred_w, neighbor_w).item()
                    actual = 1.0 if b in fired_set else 0.0
                    # Prediction error → nudge prediction weights
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
        # Check interval: capped at 500 steps so growth doesn't stall at scale.
        check_interval = min(200 + self.n // 5, 500)
        last_check = getattr(self, '_last_growth_step', -check_interval)
        if step - last_check < check_interval:
            return []

        # Auto-migrate to MPS when large enough to benefit
        self._maybe_migrate_to_mps()
        self._last_growth_step = step

        events = []
        n = self.n
        active_count = int((~self.dormant[:n]).sum().item())
        active_fr = self.fire_rates[:n][~self.dormant[:n]]
        print(
            f"[brain-growth] step={step} active={active_count} total={n} "
            f"edges={len(self._edge_strengths)} "
            f"fire_rate min={active_fr.min():.4f} max={active_fr.max():.4f} mean={active_fr.mean():.4f}",
            flush=True,
        )

        # BUD: neurons with high fire rate and high age → split
        # Threshold at 1.5x target (not 2x) — homeostatic adaptation keeps fire rates
        # tightly around target, so 2x is unreachable. 1.5x still means overworked.
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
        if bud_count > 0:
            print(f"[brain-growth] BUD {bud_count} neurons (max_buds={max_buds})", flush=True)

        # DORMANCY: neurons that never fire → sleep
        # Gentle threshold that scales down as brain grows — never choke growth.
        # At 100 active: 0.05 * 0.02 * 1.0   = 0.001  (50x below target)
        # At 200 active: 0.05 * 0.02 * 0.5   = 0.0005 (100x below target)
        # At 1K active:  0.05 * 0.02 * 0.1   = 0.0001 (500x below target)
        # Min age 2000 — give new neurons plenty of time to find their niche.
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
            print(f"[brain-growth] DORMANT {dormant_count} neurons (threshold={dormancy_threshold:.6f})", flush=True)

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

        # PRUNE: no longer needed — edge decay in _hebbian_update handles this naturally.
        # Unused synapses weaken and die on their own (use-it-or-lose-it).

        return events

    # ── Serialization ──

    def state_dict(self) -> dict:
        # Always save to CPU for checkpoint portability
        return {
            "weights": self.weights[:self.n].cpu().clone(),
            "thresholds": self.thresholds[:self.n].cpu().clone(),
            "fire_rates": self.fire_rates[:self.n].cpu().clone(),
            "layer_indices": self.layer_indices[:self.n].cpu().clone(),
            "dormant": self.dormant[:self.n].cpu().clone(),
            "ages": self.ages[:self.n].cpu().clone(),
            "cluster_ids": list(self.cluster_ids),
            "edge_strengths": dict(self._edge_strengths),
            "activation_buffer": self.activation_buffer.cpu().clone(),
            "prediction_weights": self.prediction_weights[:self.n].cpu().clone(),
            "projection": self.projection.cpu().clone(),
            "projection_alpha": self.projection_alpha,
            "update_count": self._update_count,
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
        self.activation_buffer = d["activation_buffer"].to(self.device)
        if "prediction_weights" in d:
            self.prediction_weights[:n] = d["prediction_weights"]
        # Projection layer (backward-compatible with older checkpoints)
        if "projection" in d:
            self.projection = d["projection"].to(self.device)
            self.projection_alpha = d["projection_alpha"]
            self._update_count = d["update_count"]
        print(f"[brain] restored {n} neurons, {len(self._edge_strengths)} edges, device={self.device}", flush=True)
        # Check if we should be on MPS given restored size
        self._maybe_migrate_to_mps()

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
