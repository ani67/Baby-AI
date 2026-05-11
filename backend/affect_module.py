"""AffectModule: differentiable affect integrated into the transformer."""
import math, time
import torch, torch.nn as nn, torch.nn.functional as F

N_AFF = 12
N_HEADS = 8
D_REP = 512

HALF_LIVES_S = {
    'reaction':    2.0,
    'working':     180.0,
    'mood':        7200.0,
    'disposition': 1_209_600.0,
    'character':   6_307_200.0,
}
COMPOSITE_WEIGHTS = [0.30, 0.30, 0.20, 0.15, 0.05]


class AffectTimescales(nn.Module):
    """5 EMA buffers, not differentiable."""
    def __init__(self, n_aff=N_AFF):
        super().__init__()
        for name in HALF_LIVES_S:
            self.register_buffer(f'{name}_state',
                                 torch.zeros(n_aff, dtype=torch.float32))
        # MPS does not support float64; use float32 for last_t scalar timestamp.
        # Sub-second precision is preserved over deltas; the absolute epoch value
        # is only ever read via .item() into Python float, never used in device ops.
        self.register_buffer('last_t', torch.tensor(0.0, dtype=torch.float32))

    def _apply(self, fn, recurse=True):
        # Timescale buffers stay on CPU regardless of .to(device) on the parent.
        # The spec's update path explicitly does affect_vec.float().cpu(), so the
        # EMA state lives on CPU off the hot path. Snapshot dtypes/devices by
        # name and restore (PyTorch replaces buffer tensors in self._buffers).
        snapshot = {n: (b.device, b.dtype) for n, b in self._buffers.items()}
        super()._apply(fn, recurse)
        for name, (dev, dt) in snapshot.items():
            b = self._buffers[name]
            if b is not None and (b.device != dev or b.dtype != dt):
                self._buffers[name] = b.to(device=dev, dtype=dt)
        return self

    @torch.no_grad()
    def update(self, affect_vec: torch.Tensor, now: float):
        dt = max(now - float(self.last_t.item()), 0.0)
        vec = affect_vec.detach().float().cpu()
        for name, hl in HALF_LIVES_S.items():
            alpha = 1.0 - math.exp(-dt * math.log(2) / hl)
            alpha = max(0.0, min(1.0, alpha))
            state = getattr(self, f'{name}_state')
            new = (1 - alpha) * state + alpha * vec
            setattr(self, f'{name}_state', new)
        self.last_t = torch.tensor(now, dtype=torch.float32, device=self.last_t.device)

    def composite(self) -> torch.Tensor:
        states = [getattr(self, f'{n}_state') for n in HALF_LIVES_S]
        return sum(w * s for w, s in zip(COMPOSITE_WEIGHTS, states))


class AffectModule(nn.Module):
    def __init__(self, d_model=D_REP, n_aff=N_AFF, n_heads=N_HEADS):
        super().__init__()
        self.n_heads = n_heads
        self.affect_net = nn.Sequential(
            nn.Linear(d_model, 256), nn.GELU(),
            nn.Linear(256, 64), nn.GELU(),
            nn.Linear(64, n_aff), nn.Tanh(),
        )
        self.bias_proj = nn.Linear(n_aff, n_heads)
        self.valence_head = nn.Sequential(
            nn.Linear(n_aff, 16), nn.GELU(),
            nn.Linear(16, 1), nn.Tanh(),
        )
        self.timescales = AffectTimescales(n_aff)

    def _apply(self, fn, recurse=True):
        # Design intent (spec § 4.2 + MPS rules): the affect MLP runs in fp32
        # even when the surrounding module is cast to bfloat16. Snapshot the
        # fp32 sub-modules' dtypes and restore after any .to() / .cuda() / etc.
        fp32_mods = (self.affect_net, self.bias_proj, self.valence_head)
        snapshot = [
            [(p, p.dtype) for p in mod.parameters()] for mod in fp32_mods
        ]
        super()._apply(fn, recurse)
        for entries in snapshot:
            for p, dt in entries:
                if p.is_floating_point() and p.dtype != dt:
                    p.data = p.data.to(dt)
        return self

    def forward(
        self,
        memory_state: torch.Tensor,   # (M, D) — already mixed/combined slots
        memory_weights: torch.Tensor = None,
    ) -> dict:
        if memory_weights is not None:
            w = F.softmax(memory_weights.float(), dim=0)
            pooled = (w.unsqueeze(-1).to(memory_state.dtype)
                      * memory_state).sum(0)
        else:
            pooled = memory_state.mean(0)
        pooled_f = pooled.float()
        aff = self.affect_net(pooled_f)
        bias = self.bias_proj(aff).to(memory_state.dtype)
        valence = self.valence_head(aff).squeeze()
        arousal = torch.clamp(aff.norm() / math.sqrt(N_AFF), 0, 1)
        return {'affect_vector': aff, 'attention_bias': bias,
                'valence': valence, 'arousal': arousal}

    def update_timescales(self, aff: torch.Tensor, now: float):
        self.timescales.update(aff, now)

    def composite(self) -> torch.Tensor:
        return self.timescales.composite()

    def character(self) -> torch.Tensor:
        return self.timescales.character_state.clone()
