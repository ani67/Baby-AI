"""
BabyModelV2Reflect -- adapter that slots BrainV2 into the BabyModelV2 shell.

Inherits BabyModelV2 and replaces:
  - self.brain: BrainState -> BrainV2
  - update_batch(): adds brain.reflect(error) after each forward+update

Everything else (forward, growth_check, checkpoint, restore, graph adapter)
works unchanged because BrainV2 exposes the same interface as BrainState.

Usage:
    from config_brain_v2 import USE_BRAIN_V2
    if USE_BRAIN_V2:
        from model.baby_model_v2_reflect import BabyModelV2Reflect as BabyModel
    else:
        from model.baby_model_v2 import BabyModelV2 as BabyModel

    model = BabyModel(...)  # same constructor signature

train_worker.py internals that access brain attributes directly
-------------------------------------------------------------
All of these are present on BrainV2 with identical types/shapes:

    brain.n                  int             -- neuron count
    brain.weights            (cap, dim)      -- weight matrix
    brain.dormant            (cap,) bool     -- dormancy mask
    brain.cluster_ids        list[str]       -- per-neuron cluster IDs
    brain.fire_rates         (cap,)          -- running fire-rate EMA
    brain.ages               (cap,) long     -- per-neuron age counter
    brain.layer_indices      (cap,)          -- layer assignments
    brain.thresholds         (cap,)          -- firing thresholds
    brain.activation_buffer  (dim,)          -- temporal context buffer
    brain._edge_strengths    dict[(i,j),f]   -- edge weight dict
    brain._id_to_idx         dict[str,int]   -- cluster_id -> index
    brain.device             torch.device    -- current device

    brain.forward(x)                -> (prediction, activations)
    brain.update(x, teacher_vec)    -> None
    brain.adapt_thresholds()        -> None
    brain.pre_sense()               -> (active_weights, active_idx)
    brain.growth_check(step)        -> list[dict]
    brain.state_dict()              -> dict
    brain.load_state_dict(d)        -> None
    brain.summary()                 -> dict

Methods on BrainState that BrainV2 does NOT have (added by this adapter):
    brain.forward_sequence(vectors, memory)  -> (prediction, activations)
    brain.reason(x, steps, memory)           -> (prediction, activations)
"""

import torch
import torch.nn.functional as F

from .baby_model_v2 import BabyModelV2, GraphAdapter
from .brain_v2 import BrainV2
from .working_memory import WorkingMemory


class BabyModelV2Reflect(BabyModelV2):
    """
    Drop-in replacement for BabyModelV2 that uses BrainV2 (FF + REFLECT).
    Same API surface -- train_worker.py needs zero changes.
    """

    def __init__(
        self,
        input_dim: int = 512,
        initial_clusters: int = 4,
        nodes_per_cluster: int = 8,
        initial_plasticity: float = 1.0,
        growth_check_interval: int = 50,
        snapshot_interval: int = 50,
    ):
        # Skip BabyModelV2.__init__ -- we need to wire BrainV2 instead of BrainState.
        # Reproduce the same setup with BrainV2.
        self.input_dim = input_dim
        self.step = 0
        self.stage = 0
        self.snapshot_interval = snapshot_interval
        self.growth_check_interval = growth_check_interval

        # V2 REFLECT engine (replaces BrainState)
        self.brain = BrainV2(dim=input_dim, initial_size=initial_clusters)

        # Patch missing methods onto BrainV2 instance so callers
        # (orchestrator, train_worker) that call brain.forward_sequence
        # or brain.reason don't break.
        self.brain.forward_sequence = self._forward_sequence
        self.brain.reason = self._reason

        # Working memory for multi-step reasoning
        self._working_memory = WorkingMemory(
            slots=8, dim=input_dim, device=self.brain.device.type,
        )

        # Graph adapter for API compatibility (works with BrainV2 -- same attrs)
        self.graph = GraphAdapter(self.brain)

        # Buffer alias (orchestrator/dashboard access these directly)
        self._activation_buffer = self.brain.activation_buffer
        self.buffer_decay = self.brain.buffer_decay
        self.buffer_weight = self.brain.buffer_weight
        self.buffer_top_k = 5

        # Growth monitor stub
        from .baby_model_v2 import _GrowthMonitorStub
        self._growth_monitor = _GrowthMonitorStub()
        self._last_bud_step = -20
        self._restore_step = -200

        # V1 compat attributes (health monitor reads/writes these via setattr)
        self.resonance_threshold = 0.10
        self.resonance_min_pass = 12
        self.inhibition_radius = 0.92
        self.suppression_factor = 0.5
        self.per_cluster_signal = True
        self.per_cluster_global_steps = 5000
        self.per_cluster_blend_steps = 10000
        self.gate_activation_step = float('inf')
        self.exp_per_cluster_sign = False
        self.exp_error_direction = False
        self.exp_contrastive_pairs = False
        self.exp_multi_target = False
        self.growth_warning_threshold = 256

    # ── Batch compute with REFLECT ──

    def update_batch(
        self,
        samples: list[tuple],
    ) -> tuple[dict, list[dict]]:
        """
        Process a batch of samples, same as BabyModelV2.update_batch but
        adds brain.reflect(error) after each update to propagate error
        backward through the edge network.

        REFLECT adds cross-layer coordination without backprop. The hybrid
        alpha/beta weighting inside BrainV2.update() blends FF + REFLECT
        signals automatically -- this method just ensures reflect() is
        called with the right error vector.
        """
        all_activations = []
        _pre = self.brain.pre_sense()

        for sample in samples:
            x = sample[0]
            teacher_vec = sample[2] if len(sample) > 2 else None

            if isinstance(x, list):
                prediction, activations = self._forward_sequence(
                    x, memory=self._working_memory,
                )
                all_activations.append(activations)
                if teacher_vec is not None:
                    self.brain.update(x[-1], teacher_vec)
                    # REFLECT: propagate error backward through edges
                    error = F.normalize(teacher_vec, dim=0) - F.normalize(prediction, dim=0)
                    self.brain.reflect(error)
            else:
                prediction, activations = self.brain.forward(x, _pre_sensed=_pre)
                all_activations.append(activations)
                if teacher_vec is not None:
                    self.brain.update(x, teacher_vec)
                    # REFLECT: propagate error backward through edges
                    error = F.normalize(teacher_vec, dim=0) - F.normalize(prediction, dim=0)
                    self.brain.reflect(error)

            self.brain.adapt_thresholds()

        self.step += len(samples)
        self._activation_buffer = self.brain.activation_buffer

        return {}, all_activations

    # ── Missing BrainState methods (forward_sequence, reason) ──

    @torch.no_grad()
    def _forward_sequence(
        self,
        vectors: list[torch.Tensor],
        memory: WorkingMemory | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Process a sequence of vectors, building temporal context via the
        activation buffer. Mirrors BrainState.forward_sequence().
        """
        if not vectors:
            return torch.zeros(self.brain.dim, device=self.brain.device), {}

        if len(vectors) == 1:
            return self.brain.forward(vectors[0])

        # Temporarily adjust buffer for sequential input
        orig_decay = self.brain.buffer_decay
        orig_weight = self.brain.buffer_weight
        self.brain.buffer_decay = 0.95
        self.brain.buffer_weight = 0.30

        try:
            prediction = torch.zeros(self.brain.dim, device=self.brain.device)
            activations: dict = {}

            for i, vec in enumerate(vectors):
                if memory is not None and memory.occupancy > 0:
                    mem_context = memory.read(vec)
                    vec = F.normalize(0.8 * vec + 0.2 * mem_context, dim=0)

                prediction, activations = self.brain.forward(vec)

                if memory is not None:
                    memory.write(prediction)
        finally:
            self.brain.buffer_decay = orig_decay
            self.brain.buffer_weight = orig_weight

        return prediction, activations

    @torch.no_grad()
    def _reason(
        self,
        x: torch.Tensor,
        steps: int = 5,
        memory: WorkingMemory | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Multi-step reasoning: iterate forward passes, feeding prediction
        back as input. Mirrors BrainState.reason().
        """
        x = F.normalize(x.to(self.brain.device), dim=0)
        prediction = x
        activations: dict = {}

        for _ in range(steps):
            if memory is not None and memory.occupancy > 0:
                mem_context = memory.read(x)
                x = F.normalize(0.8 * x + 0.2 * mem_context, dim=0)

            prediction, activations = self.brain.forward(x)

            if memory is not None:
                memory.write(prediction)

            x = F.normalize(0.7 * prediction + 0.3 * x, dim=0)

        return prediction, activations
