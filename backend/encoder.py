"""MultiModalEncoder: perceiver-style encoder.

Cross-attention DIRECTION (CORRECTION 2):
  - input tokens (L) query
  - top-K=256 active memory slots key/value
  - output: per-input-token contextualized representations (L, D)
  - we then scatter-update the K active slots from input tokens that attended to them

This is the right way around. Previous direction (M queries -> L keys)
produced (65536, 64) attention matrices, OOM on M1.
"""
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.memory_bank import PersistentMemoryBank
from backend.unified_config import (
    D_REP,
    N_ENC_LAYERS,
    N_HEADS,
    TOP_K_ACTIVE,
    VOCAB_SIZE,
)

log = logging.getLogger('encoder')


class CurriculumTokenizer:
    PAD, BOS, EOS, UNK = 0, 1, 2, 3

    def __init__(self, path: str = None):
        self._tok = None
        if path:
            self.load(path)

    def load(self, path: str):
        from tokenizers import Tokenizer
        self._tok = Tokenizer.from_file(path)
        self.vocab_size = self._tok.get_vocab_size()

    @classmethod
    def from_path(cls, path: str):
        obj = cls()
        obj.load(path)
        return obj

    def encode(self, text: str) -> list[int]:
        if self._tok is None:
            return [self.BOS] + [ord(c) % 256 for c in text[:256]] + [self.EOS]
        ids = self._tok.encode(text).ids
        return [self.BOS] + ids + [self.EOS]

    def decode(self, ids: list[int]) -> str:
        if self._tok is None:
            return ''.join(chr(i) for i in ids if 32 <= i < 127)
        keep = [i for i in ids if i not in {self.PAD, self.BOS, self.EOS}]
        return self._tok.decode(keep)


class InputCrossAttn(nn.Module):
    """Input attends to top-K memory slots. One PC level."""

    def __init__(self, d_model=D_REP, n_heads=N_HEADS):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        # top-down predictor: predict input from memory pool
        self.predictor = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, d_model))
        self.n_in = nn.LayerNorm(d_model)
        self.n_mem = nn.LayerNorm(d_model)
        self.n_ff = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(),
            nn.Linear(d_model * 4, d_model))

    def forward(self, inputs: torch.Tensor, mem_active: torch.Tensor):
        """
        inputs: (L, D)
        mem_active: (K, D) top-K active memory slots
        returns:
          updated_inputs: (L, D)
          pc_error: (D,) prediction error magnitude (mean over input tokens)
        """
        L = inputs.shape[0]
        K = mem_active.shape[0]
        x = self.n_in(inputs)
        m = self.n_mem(mem_active)
        Q = self.q_proj(x).view(L, self.n_heads, self.d_head)
        Kp = self.k_proj(m).view(K, self.n_heads, self.d_head)
        V = self.v_proj(m).view(K, self.n_heads, self.d_head)
        # (L, H, d) x (K, H, d) -> (L, H, K)
        scores = torch.einsum('lhd,khd->lhk', Q, Kp) * self.scale
        attn = F.softmax(scores, dim=-1)
        out = torch.einsum('lhk,khd->lhd', attn, V).reshape(L, -1)
        out = self.out_proj(out)
        updated = inputs + out
        updated = updated + self.ff(self.n_ff(updated))

        # PC prediction error: top-down from memory pool predicts input
        mem_pool = mem_active.mean(0)              # (D,)
        predicted = self.predictor(mem_pool)       # (D,)
        actual = inputs.mean(0)                    # (D,)
        pc_error = (actual - predicted).float()
        return updated, pc_error


class MultiModalEncoder(nn.Module):
    def __init__(
        self,
        memory_bank: PersistentMemoryBank,
        vocab_size: int = VOCAB_SIZE,
        n_layers: int = N_ENC_LAYERS,
        d_model: int = D_REP,
        tokenizer: CurriculumTokenizer = None,
        top_k_active: int = TOP_K_ACTIVE,
    ):
        super().__init__()
        self.bank = memory_bank
        self.tokenizer = tokenizer
        self.top_k = top_k_active

        self.text_embedding = nn.Embedding(vocab_size, d_model)
        self.text_pos_embedding = nn.Embedding(2048, d_model)
        # Scale embedding init so LM-head logits have unit-ish std.
        # nn.Embedding default is N(0,1); lm_head is weight-tied so
        # logit std = norm(hidden) * sqrt(d_model) ≈ sqrt(d_model) at
        # init = 22 for d=512, which makes CE loss ~30× expected.
        # GPT-2 uses 0.02; we use 1/sqrt(d_model) which is similar
        # (1/sqrt(512) ≈ 0.044) and gives logit std ≈ 1 at init.
        import math
        nn.init.normal_(self.text_embedding.weight,
                        std=1.0 / math.sqrt(d_model))
        nn.init.normal_(self.text_pos_embedding.weight,
                        std=1.0 / math.sqrt(d_model))

        # vision/audio projections (frozen pretrained encoders are lazy-loaded
        # in encode_vision / encode_audio; stub for now)
        D_CLIP = 512
        D_WHISPER = 512
        self.vision_proj = nn.Linear(D_CLIP, d_model)
        self.audio_proj = nn.Linear(D_WHISPER, d_model)
        self._clip = None
        self._whisper = None

        self.layers = nn.ModuleList([InputCrossAttn() for _ in range(n_layers)])

    def encode_text(self, text: str) -> torch.Tensor:
        device = next(self.parameters()).device
        if self.tokenizer:
            ids = self.tokenizer.encode(text)
        else:
            ids = [1] + [ord(c) % 1024 for c in text[:256]] + [2]
        ids_t = torch.tensor(ids, device=device, dtype=torch.long)
        positions = torch.arange(len(ids_t), device=device)
        emb = (self.text_embedding(ids_t).to(torch.bfloat16)
               + self.text_pos_embedding(positions).to(torch.bfloat16))
        return emb  # (L, D)

    def encode_vision(self, image: torch.Tensor) -> torch.Tensor:
        device = next(self.parameters()).device
        if self._clip is None:
            try:
                import clip
                self._clip, _ = clip.load('ViT-B/32', device=device)
                for p in self._clip.parameters():
                    p.requires_grad_(False)
            except Exception:
                return torch.zeros(1, D_REP, device=device,
                                   dtype=torch.bfloat16)
        if image.dim() == 3:
            image = image.unsqueeze(0)
        with torch.no_grad():
            feat = self._clip.encode_image(image).to(torch.bfloat16)
        return self.vision_proj(feat)

    def encode_audio(self, audio: torch.Tensor) -> torch.Tensor:
        device = next(self.parameters()).device
        # Whisper integration deferred; return zero tokens
        return torch.zeros(1, D_REP, device=device, dtype=torch.bfloat16)

    def forward(
        self,
        text: str = None,
        image: torch.Tensor = None,
        audio: torch.Tensor = None,
    ) -> dict:
        tokens = []
        if text is not None:
            tokens.append(self.encode_text(text))
        if image is not None:
            tokens.append(self.encode_vision(image))
        if audio is not None:
            tokens.append(self.encode_audio(audio))
        if not tokens:
            raise ValueError("at least one modality required")
        inputs = torch.cat(tokens, dim=0)  # (L, D)

        # get top-K active memory slots
        counts = self.bank.activation_count.float()
        if (counts > 0).sum() < self.top_k:
            # not enough active slots yet — pad with first-N
            top_idx = torch.arange(min(self.top_k, self.bank.m_slots),
                                   device=self.bank.device)
        else:
            _, top_idx = counts.topk(self.top_k)
        mem_active = self.bank.slots[top_idx]  # (K, D)

        pc_errors = []
        x = inputs
        for layer in self.layers:
            x, err = layer(x, mem_active)
            pc_errors.append(err)

        # surprise = mean error magnitude across layers
        surprise = torch.stack([e.norm() for e in pc_errors]).mean()

        # memory_delta to scatter back: pool input contributions, project,
        # write proportional to attention. Simplification: aggregate input
        # rep, return it; the soft_write step lives in UnifiedMind.process().
        input_rep = x.mean(0)  # (D,)

        return {
            'updated_inputs': x,                 # (L, D)
            'input_rep': input_rep,              # (D,) for soft-write
            'active_slot_indices': top_idx,      # (K,)
            'mem_active': mem_active,            # (K, D)
            'pc_errors': pc_errors,
            'surprise': surprise,
        }
