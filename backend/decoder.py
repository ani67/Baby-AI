"""UnifiedDecoder: expression via cross-attention to memory.

Replaces native_head.py and expression_graph.py (legacy kept). Standard
autoregressive decoder. Cross-attn over top-K=256 active memory slots
(same pool as encoder). Weight tying with encoder.text_embedding via the
LM head.

All MPS rules from spec § 3 apply:
  - bfloat16 params/activations, float32 only for loss math (lm_head on x.float())
  - no in-place ops on autograd-tracked tensors
  - no dynamic shapes inside layers (mask cached per-T)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from backend.memory_bank import PersistentMemoryBank
from backend.unified_config import D_REP, N_HEADS, N_DEC_LAYERS, TOP_K_ACTIVE  # noqa: F401

MAX_GEN_LEN = 64
D_FF = D_REP * 4


class CausalSelfAttn(nn.Module):
    def __init__(self, d_model=D_REP, n_heads=N_HEADS):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self._mask_cache: dict[int, torch.Tensor] = {}

    def _mask(self, T, device):
        if T not in self._mask_cache:
            self._mask_cache[T] = torch.tril(
                torch.ones(T, T, device=device, dtype=torch.bool))
        return self._mask_cache[T]

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        sc = (q @ k.transpose(-2, -1)) * self.scale
        sc = sc.masked_fill(~self._mask(T, x.device), float('-inf'))
        a = F.softmax(sc, dim=-1)
        return self.out((a @ v).transpose(1, 2).reshape(B, T, D))


class MemoryCrossAttn(nn.Module):
    def __init__(self, d_model=D_REP, n_heads=N_HEADS):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, mem_active: torch.Tensor):
        # x: (B, T, D); mem_active: (K, D)
        B, T, D = x.shape
        K = mem_active.shape[0]
        Q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        Kp = self.k_proj(mem_active).view(1, K, self.n_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(mem_active).view(1, K, self.n_heads, self.d_head).transpose(1, 2)
        sc = Q @ Kp.transpose(-2, -1) * self.scale
        a = F.softmax(sc, dim=-1)
        return self.out_proj((a @ V).transpose(1, 2).reshape(B, T, D))


class DecoderLayer(nn.Module):
    def __init__(self, d_model=D_REP):
        super().__init__()
        self.self_attn = CausalSelfAttn(d_model)
        self.cross_attn = MemoryCrossAttn(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, D_FF), nn.GELU(),
            nn.Linear(D_FF, d_model))
        self.n1 = nn.LayerNorm(d_model)
        self.n2 = nn.LayerNorm(d_model)
        self.n3 = nn.LayerNorm(d_model)

    def forward(self, x, mem_active):
        x = x + self.self_attn(self.n1(x))
        x = x + self.cross_attn(self.n2(x), mem_active)
        x = x + self.ff(self.n3(x))
        return x


class UnifiedDecoder(nn.Module):
    def __init__(
        self,
        memory_bank: PersistentMemoryBank,
        vocab_size: int,
        shared_embedding: Optional[nn.Embedding] = None,
        n_layers: int = N_DEC_LAYERS,
        d_model: int = D_REP,
    ):
        super().__init__()
        self.bank = memory_bank
        self.vocab_size = vocab_size
        self.token_embedding = shared_embedding or nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(MAX_GEN_LEN + 2, d_model)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(d_model)
        # weight-tied LM head
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, input_ids: torch.Tensor, mem_active: torch.Tensor):
        B, T = input_ids.shape
        device = input_ids.device
        pos = torch.arange(T, device=device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.pos_embedding(pos)
        x = x.to(torch.bfloat16)
        for layer in self.layers:
            x = layer(x, mem_active)
        x = self.final_norm(x)
        logits = self.lm_head(x.float())
        return logits, x

    @torch.no_grad()
    def generate(
        self,
        mem_active: torch.Tensor,
        tokenizer,
        max_new_tokens: int = 40,
        temperature: float = 0.8,
        top_p: float = 0.9,
        bos_id: int = 1,
        eos_id: int = 2,
    ) -> tuple[str, torch.Tensor]:
        device = mem_active.device
        ids = torch.tensor([[bos_id]], device=device, dtype=torch.long)
        gen: list[int] = []
        for _ in range(max_new_tokens):
            logits, _ = self.forward(ids, mem_active)
            nl = logits[0, -1, :] / max(temperature, 1e-3)
            sv, si = torch.sort(nl, descending=True)
            cp = torch.cumsum(F.softmax(sv, dim=-1), dim=-1)
            rm = cp > top_p
            rm[1:] = rm[:-1].clone()
            rm[0] = False
            nl[si[rm]] = float('-inf')
            probs = F.softmax(nl, dim=-1)
            nxt = int(torch.multinomial(probs, 1))
            if nxt == eos_id:
                break
            gen.append(nxt)
            ids = torch.cat([ids, torch.tensor([[nxt]], device=device)], dim=1)
        return tokenizer.decode(gen), torch.tensor(gen, device=device)


def compute_expression_gap(
    gen_ids: torch.Tensor,
    bank_centroid: torch.Tensor,
    encoder,
    tokenizer,
) -> float:
    surface = tokenizer.decode(gen_ids.tolist())
    with torch.no_grad():
        r = encoder(text=surface)
        gen_rep = F.normalize(r['input_rep'].float(), dim=-1)
        c = F.normalize(bank_centroid.float(), dim=-1)
        return 1.0 - float(torch.dot(gen_rep, c))
