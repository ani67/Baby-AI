"""SparseMemoryTransformer: sparse self-attn over the memory bank.

Replaces WaveField. Each transformer layer = one wave step.
Sparse attention: each slot attends to top-K=64 neighbors via gather-scatter.
Affect bias added to attention scores per head.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from backend.memory_bank import PersistentMemoryBank
from backend.unified_config import (
    D_REP,
    N_HEADS,
    N_MEM_LAYERS,
    TOP_K_NBR,
)

# spec § 4.4 declares DROPOUT here (the spec's import-line walrus is invalid
# Python; see note immediately below the snippet in the spec).
DROPOUT = 0.1


class SparseSelfAttention(nn.Module):
    def __init__(self, d_model=D_REP, n_heads=N_HEADS, top_k=TOP_K_NBR,
                 dropout=DROPOUT):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.top_k = top_k
        self.scale = self.d_head ** -0.5
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,                       # (M, D) full memory state
        memory_bank: PersistentMemoryBank,     # for neighbor search
        affect_bias: torch.Tensor = None,      # (n_heads,)
        chunk_size: int = 4096,
    ) -> torch.Tensor:
        M, D = x.shape
        Q = self.q_proj(x).view(M, self.n_heads, self.d_head)
        K = self.k_proj(x).view(M, self.n_heads, self.d_head)
        V = self.v_proj(x).view(M, self.n_heads, self.d_head)

        out = torch.zeros_like(x)
        for s in range(0, M, chunk_size):
            e = min(s + chunk_size, M)
            q_c = Q[s:e]                       # (chunk, H, d_head)
            chunk_len = e - s

            # neighbor search: query with the raw x[s:e], not projected.
            # we want geometric neighbors in representation space.
            with torch.no_grad():
                _, top_idx = memory_bank.search(
                    x[s:e].detach(), k=self.top_k,
                    q_chunk=min(chunk_len, 1024))
            # top_idx: (chunk, K)

            flat = top_idx.reshape(-1)              # (chunk*K,)
            k_g = K[flat].view(chunk_len, self.top_k, self.n_heads,
                               self.d_head)
            v_g = V[flat].view(chunk_len, self.top_k, self.n_heads,
                               self.d_head)

            # scores: (chunk, H, K)
            # q_c: (chunk, H, d_head); k_g: (chunk, K, H, d_head)
            scores = torch.einsum('chd,ckhd->chk', q_c, k_g) * self.scale
            if affect_bias is not None:
                scores = scores + affect_bias.view(1, -1, 1)
            attn = F.softmax(scores, dim=-1)
            attn = self.dropout(attn)
            # weighted sum: (chunk, H, K) × (chunk, K, H, d_head) → (chunk, H, d_head)
            o = torch.einsum('chk,ckhd->chd', attn, v_g)
            out[s:e] = o.reshape(chunk_len, D)

        return self.out_proj(out)


class MemTransformerLayer(nn.Module):
    def __init__(self, d_model=D_REP, n_heads=N_HEADS, d_ff=D_REP * 4):
        super().__init__()
        self.attn = SparseSelfAttention(d_model, n_heads)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(DROPOUT),
            nn.Linear(d_ff, d_model), nn.Dropout(DROPOUT))
        self.n1 = nn.LayerNorm(d_model)
        self.n2 = nn.LayerNorm(d_model)

    def forward(self, x, bank, affect_bias=None):
        x = x + self.attn(self.n1(x), bank, affect_bias)
        x = x + self.ff(self.n2(x))
        return x


class SparseMemoryTransformer(nn.Module):
    def __init__(self, memory_bank: PersistentMemoryBank,
                 n_layers=N_MEM_LAYERS, use_checkpoint=True):
        super().__init__()
        self.bank = memory_bank
        self.use_checkpoint = use_checkpoint
        self.layers = nn.ModuleList(
            [MemTransformerLayer() for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(D_REP)

    def forward(self, memory_state=None, affect_bias=None) -> torch.Tensor:
        if memory_state is None:
            memory_state = self.bank.slots
        x = memory_state
        for layer in self.layers:
            if self.use_checkpoint and self.training:
                x = checkpoint(layer, x, self.bank, affect_bias,
                               use_reentrant=False)
            else:
                x = layer(x, self.bank, affect_bias)
        return self.final_norm(x)
