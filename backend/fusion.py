"""
MultimodalFusion: combines sensory streams into unified representation.

For now: text only (vision and audio are zero-input stubs).
Architecture ready for full multimodal when encoders are wired.

The thalamus of the architecture. All sensory signals meet here.
Output: continuous 512-dim fused representation.

Loss: cross-modal contrastive
  same-event modalities pulled together
  different-event modalities pushed apart
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ModalityProjector(nn.Module):
    """Projects one modality into the common 512-dim space."""

    def __init__(self, in_dim: int, out_dim: int = 512):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class FusionTransformer(nn.Module):
    """Small transformer over modality tokens. 3 tokens (vision, audio, text)
    attend to each other. Output: unified representation."""

    def __init__(self, dim: int = 512, n_heads: int = 8):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (B, 3, 512) — vision, audio, text
        attn_out, _ = self.attn(tokens, tokens, tokens)
        tokens = self.norm1(tokens + attn_out)
        tokens = self.norm2(tokens + self.ff(tokens))
        return tokens.mean(dim=1)  # (B, 512)


class MultimodalFusion(nn.Module):
    """Fuses vision, audio, and text into one representation."""

    def __init__(self, common_dim: int = 512):
        super().__init__()
        self.vision_proj = ModalityProjector(512, common_dim)
        self.audio_proj  = ModalityProjector(512, common_dim)
        self.text_proj   = ModalityProjector(512, common_dim)
        self.fusion = FusionTransformer(common_dim)
        self.temperature = nn.Parameter(torch.tensor(0.07))
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        self.device = (
            torch.device('mps') if torch.backends.mps.is_available()
            else torch.device('cpu')
        )
        self.to(self.device)

    def fuse(self, text_rep: np.ndarray,
             vision_rep: Optional[np.ndarray] = None,
             audio_rep: Optional[np.ndarray] = None) -> np.ndarray:
        text = torch.tensor(text_rep, dtype=torch.float32,
                            device=self.device).unsqueeze(0)
        vision = (torch.tensor(vision_rep, dtype=torch.float32,
                               device=self.device).unsqueeze(0)
                  if vision_rep is not None
                  else torch.zeros(1, 512, device=self.device))
        audio  = (torch.tensor(audio_rep, dtype=torch.float32,
                               device=self.device).unsqueeze(0)
                  if audio_rep is not None
                  else torch.zeros(1, 512, device=self.device))
        v = self.vision_proj(vision)
        a = self.audio_proj(audio)
        t = self.text_proj(text)
        tokens = torch.stack([v, a, t], dim=1)
        fused = self.fusion(tokens)
        fused = F.normalize(fused, dim=-1)
        return fused.squeeze(0).detach().cpu().numpy()

    def contrastive_learn(self, text_rep_a: np.ndarray,
                          text_rep_b: np.ndarray,
                          same_event: bool) -> float:
        a = torch.tensor(text_rep_a, dtype=torch.float32,
                         device=self.device).unsqueeze(0)
        b = torch.tensor(text_rep_b, dtype=torch.float32,
                         device=self.device).unsqueeze(0)
        a_proj = F.normalize(self.text_proj(a), dim=-1)
        b_proj = F.normalize(self.text_proj(b), dim=-1)
        similarity = (a_proj * b_proj).sum() / self.temperature
        if same_event:
            loss = -torch.log(torch.sigmoid(similarity) + 1e-9)
        else:
            loss = -torch.log(torch.sigmoid(-similarity) + 1e-9)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss)
