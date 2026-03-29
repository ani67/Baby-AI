"""
Native vision encoder — a small ConvNet distilled from CLIP.

~520K params, ~2MB on disk. Runs on CPU with no external dependencies
beyond PyTorch and PIL. Produces 512-dim L2-normalized embeddings that
are drop-in compatible with the CLIP-backed ImageEncoder.
"""

import io
from pathlib import Path

import PIL.Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

_preprocess = transforms.Compose([
    transforms.Resize((64, 64), interpolation=transforms.InterpolationMode.LANCZOS),
    transforms.ToTensor(),                         # (3, 64, 64), float32, [0, 1]
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def _prepare(image: PIL.Image.Image) -> torch.Tensor:
    """PIL image → (1, 3, 64, 64) preprocessed tensor."""
    return _preprocess(image.convert("RGB")).unsqueeze(0)


def _prepare_batch(images: list[PIL.Image.Image]) -> torch.Tensor:
    """List of PIL images → (N, 3, 64, 64) preprocessed tensor."""
    return torch.stack([_preprocess(img.convert("RGB")) for img in images])


# ---------------------------------------------------------------------------
# ConvNet backbone
# ---------------------------------------------------------------------------

class _ConvNet(nn.Module):
    """
    4-layer ConvNet producing a 512-dim global vector and 64 spatial patches.

    Architecture:
        64×64×3  → 32×32×32  (conv1)
                 → 16×16×64  (conv2)
                 → 8×8×128   (conv3)  ← spatial patches tapped here
                 → 4×4×256   (conv4)
                 → 1×1×256   (pool)
                 → 512       (fc)
    """

    def __init__(self) -> None:
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, 512)
        self.patch_head = nn.Linear(128, 512)

    def forward(
        self, x: torch.Tensor, *, with_patches: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            x: (N, 3, 64, 64)
            with_patches: if True, also return spatial patch features.

        Returns:
            global_emb: (N, 512) L2-normalized
            patches:    (N, 64, 512) L2-normalized  — or None
        """
        x = self.conv1(x)
        x = self.conv2(x)
        feat3 = self.conv3(x)           # (N, 128, 8, 8)
        x = self.conv4(feat3)            # (N, 256, 4, 4)

        # Global embedding
        pooled = self.pool(x).flatten(1)  # (N, 256)
        global_emb = F.normalize(self.fc(pooled), dim=-1)

        # Spatial patches from layer-3 feature map
        patches = None
        if with_patches:
            n, c, h, w = feat3.shape      # (N, 128, 8, 8)
            flat = feat3.permute(0, 2, 3, 1).reshape(n, h * w, c)  # (N, 64, 128)
            patches = F.normalize(self.patch_head(flat), dim=-1)    # (N, 64, 512)

        return global_emb, patches


# ---------------------------------------------------------------------------
# Public encoder
# ---------------------------------------------------------------------------

class NativeVisionEncoder:
    """
    Drop-in replacement for the CLIP-backed ImageEncoder.

    Satisfies the ``ImageEncoder`` protocol from ``encoder.base``.
    """

    def __init__(self, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.net = _ConvNet().to(self.device)
        self.net.eval()
        self._optimizer: torch.optim.Adam | None = None

    # -- ImageEncoder protocol -----------------------------------------------

    def encode(self, image: PIL.Image.Image) -> torch.Tensor:
        """Single image -> (512,) L2-normalized."""
        with torch.no_grad():
            x = _prepare(image).to(self.device)
            emb, _ = self.net(x)
        return emb.squeeze(0).cpu()

    def encode_batch(self, images: list[PIL.Image.Image]) -> torch.Tensor:
        """Batch -> (N, 512) L2-normalized."""
        with torch.no_grad():
            x = _prepare_batch(images).to(self.device)
            emb, _ = self.net(x)
        return emb.cpu()

    def encode_bytes(self, image_bytes: bytes) -> torch.Tensor:
        """Raw bytes -> (512,)."""
        image = PIL.Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return self.encode(image)

    # -- Spatial patches -----------------------------------------------------

    def encode_with_patches(
        self, image: PIL.Image.Image,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (global_512, patches_64x512)."""
        with torch.no_grad():
            x = _prepare(image).to(self.device)
            emb, patches = self.net(x, with_patches=True)
        return emb.squeeze(0).cpu(), patches.squeeze(0).cpu()

    # -- Distillation --------------------------------------------------------

    def distill_step(
        self, images: list[PIL.Image.Image], clip_targets: torch.Tensor,
    ) -> float:
        """
        One gradient step: make the ConvNet output match CLIP embeddings.

        Args:
            images:       list of PIL images (batch)
            clip_targets: (N, 512) CLIP embeddings (already L2-normalized)

        Returns:
            Mean cosine-distance loss (1 - cosine_similarity).
        """
        if self._optimizer is None:
            self._optimizer = torch.optim.Adam(self.net.parameters(), lr=0.001)

        self.net.train()
        x = _prepare_batch(images).to(self.device)
        clip_targets = clip_targets.to(self.device)

        emb, _ = self.net(x)
        loss = (1.0 - F.cosine_similarity(emb, clip_targets)).mean()

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        self.net.eval()

        return loss.item()

    # -- Blend mode ----------------------------------------------------------

    def encode_blended(
        self,
        image: PIL.Image.Image,
        clip_vector: torch.Tensor,
        alpha: float,
    ) -> torch.Tensor:
        """alpha * native + (1 - alpha) * clip, L2-normalized."""
        native = self.encode(image)
        clip_vector = clip_vector.cpu()
        blended = alpha * native + (1.0 - alpha) * clip_vector
        return F.normalize(blended, dim=-1)

    # -- Serialization -------------------------------------------------------

    def state_dict(self) -> dict:
        return self.net.state_dict()

    def load_state_dict(self, d: dict) -> None:
        self.net.load_state_dict(d)
        self.net.eval()

    # -- Convenience ---------------------------------------------------------

    def save(self, path: str | Path) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str | Path) -> None:
        self.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))

    def param_count(self) -> int:
        return sum(p.numel() for p in self.net.parameters())

    def to(self, device: str) -> "NativeVisionEncoder":
        self.device = torch.device(device)
        self.net.to(self.device)
        return self
