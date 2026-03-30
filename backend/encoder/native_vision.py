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
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.LANCZOS),
    transforms.ToTensor(),                         # (3, 224, 224), float32, [0, 1]
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def _prepare(image: PIL.Image.Image) -> torch.Tensor:
    """PIL image → (1, 3, 224, 224) preprocessed tensor."""
    return _preprocess(image.convert("RGB")).unsqueeze(0)


def _prepare_batch(images: list[PIL.Image.Image]) -> torch.Tensor:
    """List of PIL images → (N, 3, 224, 224) preprocessed tensor."""
    return torch.stack([_preprocess(img.convert("RGB")) for img in images])


# ---------------------------------------------------------------------------
# ConvNet backbone
# ---------------------------------------------------------------------------

class _ConvNet(nn.Module):
    """
    5-layer residual ConvNet producing a 512-dim global vector and spatial patches.

    Architecture (224x224 input, adaptive pool makes it resolution-agnostic):
        224×224×3 → 96×96×32   (conv1 + skip via 1x1)
                  → 48×48×64   (conv2 + skip via 1x1)
                  → 24×24×128  (conv3 + skip via 1x1)  ← spatial patches tapped here
                  → 12×12×256  (conv4 + skip via 1x1)
                  → 6×6×512    (conv5 + skip via 1x1)
                  → 1×1×512    (adaptive pool)
                  → 512        (fc)

    Every block has a residual connection. Where channels change, a 1x1 conv
    projects the skip to match dimensions. Improves gradient flow without
    significant parameter overhead.
    """

    def __init__(self) -> None:
        super().__init__()

        # Conv blocks (no final ReLU — applied after residual addition)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2),
            nn.BatchNorm2d(32),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.BatchNorm2d(512),
        )

        # 1x1 skip projections (channel mismatch at every block)
        self.skip1 = nn.Sequential(nn.Conv2d(3, 32, 1, stride=2), nn.BatchNorm2d(32))
        self.skip2 = nn.Sequential(nn.Conv2d(32, 64, 1, stride=2), nn.BatchNorm2d(64))
        self.skip3 = nn.Sequential(nn.Conv2d(64, 128, 1, stride=2), nn.BatchNorm2d(128))
        self.skip4 = nn.Sequential(nn.Conv2d(128, 256, 1, stride=2), nn.BatchNorm2d(256))
        self.skip5 = nn.Sequential(nn.Conv2d(256, 512, 1, stride=2), nn.BatchNorm2d(512))

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, 512)
        self.patch_head = nn.Linear(128, 512)

    def extract_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Run through conv1-conv3 with residuals, return [f1, f2, f3]."""
        f1 = F.relu(self.conv1(x) + self.skip1(x))
        f2 = F.relu(self.conv2(f1) + self.skip2(f1))
        f3 = F.relu(self.conv3(f2) + self.skip3(f2))
        return [f1, f2, f3]

    def forward(
        self, x: torch.Tensor, *, with_patches: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            x: (N, 3, H, W) — any resolution, adaptive pool handles it.
            with_patches: if True, also return spatial patch features.

        Returns:
            global_emb: (N, 512) L2-normalized
            patches:    (N, P, 512) L2-normalized  — or None
        """
        x = F.relu(self.conv1(x) + self.skip1(x))
        x = F.relu(self.conv2(x) + self.skip2(x))
        feat3 = F.relu(self.conv3(x) + self.skip3(x))
        x = F.relu(self.conv4(feat3) + self.skip4(feat3))
        x = F.relu(self.conv5(x) + self.skip5(x))

        # Global embedding
        pooled = self.pool(x).flatten(1)  # (N, 512)
        global_emb = F.normalize(self.fc(pooled), dim=-1)

        # Spatial patches from layer-3 feature map
        patches = None
        if with_patches:
            n, c, h, w = feat3.shape
            flat = feat3.permute(0, 2, 3, 1).reshape(n, h * w, c)  # (N, h*w, 128)
            patches = F.normalize(self.patch_head(flat), dim=-1)    # (N, h*w, 512)

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
        from .positional import PositionalEncoding
        self._pos_enc = PositionalEncoding(max_positions=32, dim=512)
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
        """Returns (global_512, patches_Px512)."""
        with torch.no_grad():
            x = _prepare(image).to(self.device)
            emb, patches = self.net(x, with_patches=True)
        return emb.squeeze(0).cpu(), patches.squeeze(0).cpu()

    # -- Patch grid (sequential) -----------------------------------------------

    def encode_patches_grid(self, image: PIL.Image.Image) -> list[torch.Tensor]:
        """Image -> 16 patch vectors (4x4 grid) with positional encoding.

        Layer-3 features (h x w) are adaptive-avg-pooled to 4x4, projected
        to 512-dim, then positional encoding is added. Resolution-agnostic.
        """
        with torch.no_grad():
            x = _prepare(image).to(self.device)
            _, _, feat3 = self.net.extract_features(x)

        # feat3: (1, 128, H, W) — adaptive pool to 4x4
        pooled = F.adaptive_avg_pool2d(feat3, (4, 4))  # (1, 128, 4, 4)
        flat = pooled.squeeze(0).permute(1, 2, 0).reshape(16, 128)  # (16, 128)
        patches = F.normalize(self.net.patch_head(flat), dim=-1).cpu()  # (16, 512)

        # Add positional encoding (raster scan: top-left to bottom-right)
        if self._pos_enc is not None:
            return [self._pos_enc.encode(i, patches[i]) for i in range(16)]
        return [patches[i] for i in range(16)]

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
        d = {"net": self.net.state_dict()}
        if self._pos_enc is not None:
            d["pos_enc"] = self._pos_enc.state_dict()
        return d

    def load_state_dict(self, d: dict) -> None:
        # Backward compat: old checkpoints save net state directly
        net_state = d["net"] if "net" in d else d
        try:
            self.net.load_state_dict(net_state)
        except RuntimeError:
            # Old checkpoint missing skip connection weights — load what we can
            self.net.load_state_dict(net_state, strict=False)
            print("[native_vision] partial load (old checkpoint), skip connections freshly initialized", flush=True)
        self.net.eval()
        if "pos_enc" in d and self._pos_enc is not None:
            self._pos_enc.load_state_dict(d["pos_enc"])

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
