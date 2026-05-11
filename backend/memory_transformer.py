"""SparseMemoryTransformer: self-attn over the memory bank.

Replaces WaveField. Each transformer layer = one wave step. Affect bias
added to attention scores per head.

Two attention modes, selected by M_SLOTS vs. SPARSE_ATTENTION_THRESHOLD:
  - Dense: single (H, M, M) matmul. Fast on M1 for M ≤ 16384 because
    one MPS kernel launch beats 32 gather-scatter launches.
  - Sparse: top-K=64 gather-scatter via memory_bank.search(). Required
    when M is too large for the (H, M, M) tensor to fit.
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
    SPARSE_ATTENTION_THRESHOLD,
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
        memory_bank: PersistentMemoryBank,     # for neighbor search (sparse only)
        affect_bias: torch.Tensor = None,      # (n_heads,)
        chunk_size: int = 4096,
    ) -> torch.Tensor:
        M, D = x.shape

        # Dense path: chunked over Q rows. Per chunk peak is
        # (H, q_chunk, M) bf16 ≈ 128MB at q_chunk=1024, H=8, M=8192.
        # Manual chunked is needed because PyTorch's MPS SDPA backend
        # falls back to the naïve (H, M, M) materialization which OOMs.
        # Kernel-launch count: M / q_chunk per matmul phase, ~8 for
        # M=8192 — far less than the ~32 launches the sparse gather-
        # scatter path issues.
        if M <= SPARSE_ATTENTION_THRESHOLD:
            Q = (self.q_proj(x).view(M, self.n_heads, self.d_head)
                 .transpose(0, 1))                # (H, M, d_head)
            K = (self.k_proj(x).view(M, self.n_heads, self.d_head)
                 .transpose(0, 1))                # (H, M, d_head)
            V = (self.v_proj(x).view(M, self.n_heads, self.d_head)
                 .transpose(0, 1))                # (H, M, d_head)
            K_T = K.transpose(-2, -1)             # (H, d_head, M)

            out_h = torch.empty(self.n_heads, M, self.d_head,
                                device=x.device, dtype=x.dtype)
            q_chunk = 1024
            for s in range(0, M, q_chunk):
                e = min(s + q_chunk, M)
                # (H, chunk, d_head) @ (H, d_head, M) → (H, chunk, M)
                scores = (Q[:, s:e] @ K_T) * self.scale
                if affect_bias is not None:
                    scores = scores + affect_bias.view(self.n_heads, 1, 1)
                attn = F.softmax(scores, dim=-1)
                attn = self.dropout(attn)
                # (H, chunk, M) @ (H, M, d_head) → (H, chunk, d_head)
                out_h[:, s:e] = attn @ V

            out = out_h.transpose(0, 1).reshape(M, D).contiguous()
            return self.out_proj(out)

        # Sparse path: top-K=64 gather-scatter. Used when the (H, M, M)
        # tensor would exceed budget. Chunked over Q to bound peak memory.
        Q = self.q_proj(x).view(M, self.n_heads, self.d_head)
        K = self.k_proj(x).view(M, self.n_heads, self.d_head)
        V = self.v_proj(x).view(M, self.n_heads, self.d_head)

        out = torch.zeros_like(x)
        for s in range(0, M, chunk_size):
            e = min(s + chunk_size, M)
            q_c = Q[s:e]                       # (chunk, H, d_head)
            chunk_len = e - s

            with torch.no_grad():
                _, top_idx = memory_bank.search(
                    x[s:e].detach(), k=self.top_k,
                    q_chunk=min(chunk_len, 1024),
                    m_chunk=memory_bank.m_slots)

            flat = top_idx.reshape(-1)
            k_g = K[flat].view(chunk_len, self.top_k, self.n_heads,
                               self.d_head)
            v_g = V[flat].view(chunk_len, self.top_k, self.n_heads,
                               self.d_head)

            scores = torch.einsum('chd,ckhd->chk', q_c, k_g) * self.scale
            if affect_bias is not None:
                scores = scores + affect_bias.view(1, -1, 1)
            attn = F.softmax(scores, dim=-1)
            attn = self.dropout(attn)
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
