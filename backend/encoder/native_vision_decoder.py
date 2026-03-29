"""
Native vision decoder -- a small transposed ConvNet that generates images
from the brain's 512-dim output vectors.

Mirror of the NativeVisionEncoder architecture:

    512-dim  -> Linear -> (256, 4, 4)
             -> ConvT   -> (128, 8, 8)
             -> ConvT   -> (64, 16, 16)
             -> ConvT   -> (32, 32, 32)
             -> ConvT   -> (3, 64, 64)  + Sigmoid

~520K params, ~2MB on disk. Standard backprop (not Forward-Forward).
"""

from pathlib import Path

import PIL.Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


# ---------------------------------------------------------------------------
# Target preprocessing (resize to 64x64, tensor in [0, 1] -- no ImageNet norm)
# ---------------------------------------------------------------------------

_target_transform = transforms.Compose([
    transforms.Resize((64, 64), interpolation=transforms.InterpolationMode.LANCZOS),
    transforms.ToTensor(),  # (3, 64, 64), float32, [0, 1]
])


def _prepare_target(image: PIL.Image.Image) -> torch.Tensor:
    """PIL image -> (1, 3, 64, 64) in [0, 1]."""
    return _target_transform(image.convert("RGB")).unsqueeze(0)


def _prepare_target_batch(images: list[PIL.Image.Image]) -> torch.Tensor:
    """List of PIL images -> (N, 3, 64, 64) in [0, 1]."""
    return torch.stack([_target_transform(img.convert("RGB")) for img in images])


# ---------------------------------------------------------------------------
# Transposed ConvNet backbone
# ---------------------------------------------------------------------------

class _DeconvNet(nn.Module):
    """
    4-layer transposed ConvNet, mirror of the encoder's _ConvNet.

    Architecture:
        512       -> (256, 4, 4)   (fc + reshape)
        4x4x256   -> 8x8x128      (deconv1)
        8x8x128   -> 16x16x64     (deconv2)
        16x16x64  -> 32x32x32     (deconv3)
        32x32x32  -> 64x64x3      (deconv4) + Sigmoid
    """

    def __init__(self, dim: int = 512) -> None:
        super().__init__()

        self.fc = nn.Linear(dim, 256 * 4 * 4)

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, dim) input vectors.

        Returns:
            (N, 3, 64, 64) pixel values in [0, 1].
        """
        x = self.fc(x)                       # (N, 256*4*4)
        x = x.view(-1, 256, 4, 4)            # (N, 256, 4, 4)
        x = self.deconv1(x)                  # (N, 128, 8, 8)
        x = self.deconv2(x)                  # (N, 64, 16, 16)
        x = self.deconv3(x)                  # (N, 32, 32, 32)
        x = self.deconv4(x)                  # (N, 3, 64, 64)
        return x


# ---------------------------------------------------------------------------
# Public decoder
# ---------------------------------------------------------------------------

class VisionDecoder:
    """
    Generates 64x64 RGB images from 512-dim brain vectors.

    Mirrors the NativeVisionEncoder and is trained via standard backprop
    (MSE reconstruction loss).
    """

    def __init__(self, dim: int = 512, device: str = "cpu") -> None:
        self.dim = dim
        self.device = torch.device(device)
        self.net = _DeconvNet(dim).to(self.device)
        self.net.eval()
        self._optimizer: torch.optim.Adam | None = None

    # -- Decoding -----------------------------------------------------------

    def decode_tensor(self, vector: torch.Tensor) -> torch.Tensor:
        """512-dim vector -> (3, 64, 64) tensor, values in [0, 1]."""
        with torch.no_grad():
            x = vector.to(self.device)
            if x.dim() == 1:
                x = x.unsqueeze(0)
            out = self.net(x)
        return out.squeeze(0).cpu()

    def decode(self, vector: torch.Tensor) -> PIL.Image.Image:
        """512-dim vector -> 64x64 PIL Image."""
        tensor = self.decode_tensor(vector)  # (3, 64, 64) in [0, 1]
        pixels = (tensor * 255).clamp(0, 255).byte()
        arr = pixels.permute(1, 2, 0).numpy()  # (64, 64, 3) uint8
        return PIL.Image.fromarray(arr, mode="RGB")

    def decode_batch(self, vectors: torch.Tensor) -> list[PIL.Image.Image]:
        """(N, 512) -> list of N PIL Images."""
        with torch.no_grad():
            x = vectors.to(self.device)
            out = self.net(x)  # (N, 3, 64, 64)
        images = []
        for i in range(out.shape[0]):
            pixels = (out[i].cpu() * 255).clamp(0, 255).byte()
            arr = pixels.permute(1, 2, 0).numpy()
            images.append(PIL.Image.fromarray(arr, mode="RGB"))
        return images

    # -- Training -----------------------------------------------------------

    def train_step(
        self,
        vectors: torch.Tensor,
        target_images: list[PIL.Image.Image],
    ) -> float:
        """
        One gradient step: reconstruct images from their embeddings.

        Args:
            vectors:       (N, dim) embedding vectors.
            target_images: list of N PIL images (ground truth).

        Returns:
            MSE loss value.
        """
        if self._optimizer is None:
            self._optimizer = torch.optim.Adam(self.net.parameters(), lr=0.001)

        self.net.train()

        vectors = vectors.to(self.device)
        targets = _prepare_target_batch(target_images).to(self.device)

        reconstructed = self.net(vectors)
        loss = F.mse_loss(reconstructed, targets)

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        self.net.eval()
        return loss.item()

    # -- Serialization ------------------------------------------------------

    def state_dict(self) -> dict:
        return self.net.state_dict()

    def load_state_dict(self, d: dict) -> None:
        self.net.load_state_dict(d)
        self.net.eval()

    def save(self, path: str | Path) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str | Path) -> None:
        self.load_state_dict(
            torch.load(path, map_location=self.device, weights_only=True)
        )

    # -- Convenience --------------------------------------------------------

    def param_count(self) -> int:
        return sum(p.numel() for p in self.net.parameters())

    def to(self, device: str) -> "VisionDecoder":
        self.device = torch.device(device)
        self.net.to(self.device)
        return self
