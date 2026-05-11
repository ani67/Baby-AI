"""PersistentMemoryBank: differentiable memory store for v2.0 unified mind.

Replaces the storage role of the v0.9/v1.1 concept graph (graph.py kept on
disk for legacy API back-compat — not touched here).

Memory is split into two tensors:
  - trained_slots    : nn.Parameter, gradient-updated by the optimizer.
  - experience_slots : buffer, soft-write-updated by surprise drift.

The "active" memory exposed via the .slots property is a learned convex
combination of the two (alpha · trained + (1-α) · experience), L2-normalized.
This resolves the in-place-write-on-Parameter autograd conflict that bit the
earlier draft.

All MPS rules from spec § 3 apply:
  - bfloat16 params/activations, float32 only for loss math
  - no in-place ops on autograd-tracked tensors
  - search() chunks BOTH the query dim and the memory dim (max 4096)
  - empty_cache guarded by device.type check
  - no torch.sparse
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.unified_config import (
    D_REP,
    DRIFT_RATE_BASE,
    INITIAL_TRAINED_ALPHA,
    M_SLOTS,
    MAX_DRIFT_RATE,
    N_AFF,
    TOP_K_NBR,
)


class PersistentMemoryBank(nn.Module):
    def __init__(self, m_slots=M_SLOTS, d_rep=D_REP, device=None):
        super().__init__()
        self.m_slots = m_slots
        self.d_rep = d_rep
        self.device = device or torch.device(
            'mps' if torch.backends.mps.is_available() else 'cpu'
        )

        # trained component — gradient-updated.
        # Random init (not zeros): the .slots property F.normalizes the mixed
        # tensor, and zero vectors would survive with zero norm, breaking the
        # invariant that active slots are unit-norm.
        self.trained_slots = nn.Parameter(
            torch.randn(m_slots, d_rep, dtype=torch.bfloat16) * (1.0 / math.sqrt(d_rep))
        )

        # experience component — soft-write-updated, NOT a Parameter.
        # Mirror trained init so uninitialized slots still yield unit-norm
        # output through the mixing/normalize path.
        self.register_buffer(
            'experience_slots',
            torch.randn(m_slots, d_rep, dtype=torch.bfloat16) * (1.0 / math.sqrt(d_rep))
        )

        # learned mixing weight (scalar, sigmoid-bounded)
        self.alpha_logit = nn.Parameter(
            torch.tensor(math.log(INITIAL_TRAINED_ALPHA /
                                  (1 - INITIAL_TRAINED_ALPHA)))
        )

        # metadata buffers (not differentiable)
        self.register_buffer('activation_count',
                             torch.zeros(m_slots, dtype=torch.long))
        self.register_buffer('last_written',
                             torch.full((m_slots,), -1, dtype=torch.long))
        self.register_buffer('surprise_at_write',
                             torch.zeros(m_slots, dtype=torch.float32))
        self.register_buffer('affect_traces',
                             torch.zeros(m_slots, N_AFF, dtype=torch.float32))
        self.register_buffer('n_written',
                             torch.tensor(0, dtype=torch.long))

        self.to(self.device)

    @property
    def slots(self) -> torch.Tensor:
        """The 'active' memory: convex combination of trained + experience."""
        alpha = torch.sigmoid(self.alpha_logit)
        # bfloat16 mixing, F.normalize'd
        mixed = (alpha * self.trained_slots
                 + (1 - alpha) * self.experience_slots)
        return F.normalize(mixed, dim=-1)

    def initialize_from_concept_graph(
        self,
        concept_embeddings: np.ndarray,         # (N, 512)
        concept_affect_traces: np.ndarray = None,  # (N, 12)
        concept_activation_counts: np.ndarray = None,  # (N,)
    ) -> int:
        """Populate first N slots from existing concepts.

        If the corpus has more concepts than M_SLOTS, we keep the most-
        activated ones (the slots that defined the mind's character). When
        no activation counts are given, fall back to insertion order.
        Both trained and experience start from these embeddings.
        """
        if len(concept_embeddings) > self.m_slots:
            if concept_activation_counts is not None:
                # top-M_SLOTS by activation count
                top_idx = np.argsort(concept_activation_counts)[-self.m_slots:]
            else:
                top_idx = np.arange(self.m_slots)
            concept_embeddings = concept_embeddings[top_idx]
            if concept_affect_traces is not None:
                concept_affect_traces = concept_affect_traces[top_idx]
            if concept_activation_counts is not None:
                concept_activation_counts = concept_activation_counts[top_idx]

        N = min(len(concept_embeddings), self.m_slots)
        emb = torch.tensor(concept_embeddings[:N], dtype=torch.bfloat16,
                           device=self.device)
        emb = F.normalize(emb, dim=-1)
        with torch.no_grad():
            self.trained_slots.data[:N] = emb
            self.experience_slots[:N] = emb
            self.n_written.fill_(N)
            if concept_affect_traces is not None:
                self.affect_traces[:N] = torch.tensor(
                    concept_affect_traces[:N], dtype=torch.float32,
                    device=self.device)
            if concept_activation_counts is not None:
                self.activation_count[:N] = torch.tensor(
                    concept_activation_counts[:N], dtype=torch.long,
                    device=self.device)
        return N

    @torch.no_grad()
    def soft_write(
        self,
        representation: torch.Tensor,    # (D,) or (B, D)
        surprise_magnitude: torch.Tensor,
        affect_vector: torch.Tensor = None,  # (12,)
        step: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Drift experience_slots toward `representation` at most-similar slot."""
        if representation.dim() == 1:
            representation = representation.unsqueeze(0)
        B = representation.shape[0]
        rep_norm = F.normalize(
            representation.to(torch.bfloat16), dim=-1)

        sims, top_idx = self.search(rep_norm, k=1)
        best_slot = top_idx[:, 0]  # (B,)

        if isinstance(surprise_magnitude, (int, float)):
            surprise_magnitude = torch.tensor(
                [surprise_magnitude] * B, device=self.device)
        elif surprise_magnitude.dim() == 0:
            surprise_magnitude = surprise_magnitude.unsqueeze(0).expand(B)

        drift = torch.clamp(
            surprise_magnitude.float() * DRIFT_RATE_BASE,
            max=MAX_DRIFT_RATE)

        for b in range(B):
            idx = int(best_slot[b])
            dr = float(drift[b])
            old = self.experience_slots[idx]
            new = old + dr * (rep_norm[b] - old)
            self.experience_slots[idx] = F.normalize(new, dim=-1)
            self.activation_count[idx] += 1
            self.last_written[idx] = step
            self.surprise_at_write[idx] = float(surprise_magnitude[b])
            if affect_vector is not None:
                self.affect_traces[idx] = (
                    0.9 * self.affect_traces[idx]
                    + 0.1 * affect_vector.detach()
                                          .to(self.affect_traces.device)
                                          .float())

        # experience_slots was mutated; bump the cache version
        self.invalidate_slot_cache()
        return best_slot, drift

    def invalidate_slot_cache(self) -> None:
        """Drop the cached normalized slots. Call after soft_write or any
        explicit write to bank tensors."""
        self._cached_slots_norm = None
        self._cache_bump = getattr(self, '_cache_bump', 0) + 1

    @torch.no_grad()
    def _get_slots_norm(self, dtype: torch.dtype) -> torch.Tensor:
        """Cached normalized slots.

        Invalidated by `invalidate_slot_cache()` AND by any in-place change
        to `trained_slots` / `experience_slots` via PyTorch's tensor
        `_version` attribute. Optimizer.step() bumps the trained_slots
        version, so we pick it up automatically; soft_write bumps
        `_cache_bump` explicitly.

        Plain integer comparison — no `.item()` on a CUDA/MPS tensor.
        """
        version = (
            self.trained_slots._version,
            self.experience_slots._version,
            self.alpha_logit._version,
            getattr(self, '_cache_bump', 0),
            dtype,
        )
        cached_norm = getattr(self, '_cached_slots_norm', None)
        cached_ver = getattr(self, '_cached_slots_norm_cache_ver', None)
        if cached_norm is not None and cached_ver == version:
            return cached_norm
        norm = F.normalize(self.slots.to(dtype), dim=-1)
        self._cached_slots_norm = norm
        self._cached_slots_norm_cache_ver = version
        return norm

    @torch.no_grad()
    def search(
        self,
        queries: torch.Tensor,   # (B, D)
        k: int = TOP_K_NBR,
        q_chunk: int = 1024,     # chunk Q
        m_chunk: int = 4096,     # chunk M
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Top-k cosine-similarity search.

        Two fast-paths short-circuit the streaming-topk machinery, which
        otherwise dominates runtime when called many times per training
        step (the memory_transformer calls search per layer per chunk):

          - m_slots <= m_chunk  → one-shot topk over the full bank.
          - B <= q_chunk        → no outer Q loop.

        slots_norm is cached on the bank so we don't re-normalize the
        full (M, D) tensor on every call.
        """
        if queries.dim() == 1:
            queries = queries.unsqueeze(0)
        slots_norm = self._get_slots_norm(queries.dtype)
        q_norm = F.normalize(queries.to(slots_norm.dtype), dim=-1)
        B = q_norm.shape[0]
        k = min(k, self.m_slots)

        # fast path: single M-chunk, single Q-chunk
        if self.m_slots <= m_chunk and B <= q_chunk:
            sims = q_norm @ slots_norm.T          # (B, M)
            top_v, top_i = sims.topk(k, dim=-1)   # (B, k)
            return top_v, top_i

        # fast path: single M-chunk, multiple Q-chunks
        if self.m_slots <= m_chunk:
            all_v = torch.empty(B, k, device=self.device, dtype=q_norm.dtype)
            all_i = torch.empty(B, k, device=self.device, dtype=torch.long)
            for qs in range(0, B, q_chunk):
                qe = min(qs + q_chunk, B)
                sims = q_norm[qs:qe] @ slots_norm.T
                tv, ti = sims.topk(k, dim=-1)
                all_v[qs:qe] = tv
                all_i[qs:qe] = ti
            return all_v, all_i

        # general path: streaming-topk over both dims
        all_top_sims = torch.empty(
            B, k, device=self.device, dtype=q_norm.dtype)
        all_top_idx = torch.empty(
            B, k, device=self.device, dtype=torch.long)

        for qs in range(0, B, q_chunk):
            qe = min(qs + q_chunk, B)
            q_part = q_norm[qs:qe]
            running_sims = torch.full(
                (qe - qs, k), -1.0,
                device=self.device, dtype=q_norm.dtype)
            running_idx = torch.zeros(
                (qe - qs, k), device=self.device, dtype=torch.long)
            for ms in range(0, self.m_slots, m_chunk):
                me = min(ms + m_chunk, self.m_slots)
                sims = q_part @ slots_norm[ms:me].T
                combined_sims = torch.cat([running_sims, sims], dim=-1)
                combined_idx = torch.cat([
                    running_idx,
                    torch.arange(ms, me, device=self.device)
                         .unsqueeze(0).expand(qe - qs, -1)
                ], dim=-1)
                top_v, top_i = combined_sims.topk(k, dim=-1)
                running_sims = top_v
                running_idx = torch.gather(combined_idx, -1, top_i)
            all_top_sims[qs:qe] = running_sims
            all_top_idx[qs:qe] = running_idx

        return all_top_sims, all_top_idx

    def get_top_active(self, k: int = 50) -> list[tuple[int, float]]:
        scores = self.activation_count.float()
        top_v, top_i = scores.topk(min(k, self.m_slots))
        return [(int(i), float(v)) for i, v in zip(top_i, top_v)]

    def get_field_centroid(self, weights: torch.Tensor = None) -> torch.Tensor:
        if weights is None:
            counts = self.activation_count.float()
            weights = F.softmax(counts, dim=0)
        slots = self.slots.float()
        c = (weights.unsqueeze(-1) * slots).sum(0)
        return F.normalize(c, dim=-1)

    def save(self, path: str):
        torch.save({
            'trained_slots': self.trained_slots.data.cpu(),
            'experience_slots': self.experience_slots.cpu(),
            'alpha_logit': self.alpha_logit.data.cpu(),
            'activation_count': self.activation_count.cpu(),
            'last_written': self.last_written.cpu(),
            'surprise_at_write': self.surprise_at_write.cpu(),
            'affect_traces': self.affect_traces.cpu(),
            'n_written': self.n_written.cpu(),
            'm_slots': self.m_slots, 'd_rep': self.d_rep,
        }, path)

    @classmethod
    def load(cls, path: str, device=None) -> 'PersistentMemoryBank':
        d = torch.load(path, map_location='cpu')
        bank = cls(m_slots=d['m_slots'], d_rep=d['d_rep'], device=device)
        with torch.no_grad():
            bank.trained_slots.data = d['trained_slots'].to(bank.device)
            bank.experience_slots = d['experience_slots'].to(bank.device)
            bank.alpha_logit.data = d['alpha_logit'].to(bank.device)
            bank.activation_count = d['activation_count'].to(bank.device)
            bank.last_written = d['last_written'].to(bank.device)
            bank.surprise_at_write = d['surprise_at_write'].to(bank.device)
            bank.affect_traces = d['affect_traces'].to(bank.device)
            bank.n_written = d['n_written'].to(bank.device)
        return bank
