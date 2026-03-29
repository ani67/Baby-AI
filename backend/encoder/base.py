"""
Encoder / Decoder protocols.

All encoders map raw sensory input to a fixed-dim embedding vector.
All decoders map an embedding vector back to raw output.
Concrete implementations can be CLIP-backed, native ConvNet, or anything else —
the brain and orchestrator only see these interfaces.
"""

from typing import Protocol, runtime_checkable

import PIL.Image
import torch


@runtime_checkable
class ImageEncoder(Protocol):
    def encode(self, image: PIL.Image.Image) -> torch.Tensor:
        """Single image → (dim,) L2-normalized vector."""
        ...

    def encode_batch(self, images: list[PIL.Image.Image]) -> torch.Tensor:
        """Batch of images → (N, dim) L2-normalized vectors."""
        ...

    def encode_bytes(self, image_bytes: bytes) -> torch.Tensor:
        """Raw bytes → (dim,) L2-normalized vector."""
        ...


@runtime_checkable
class TextEncoder(Protocol):
    def encode(self, text: str) -> torch.Tensor:
        """Single text → (dim,) L2-normalized vector."""
        ...

    def encode_batch(self, texts: list[str]) -> torch.Tensor:
        """Batch of texts → (N, dim) L2-normalized vectors."""
        ...


@runtime_checkable
class VideoEncoder(Protocol):
    frames_per_clip: int

    def encode(self, video_path: str) -> torch.Tensor:
        """Video file → (frames_per_clip, dim) L2-normalized per frame."""
        ...

    def encode_frames(self, frames: list[PIL.Image.Image]) -> torch.Tensor:
        """Pre-extracted frames → (frames_per_clip, dim)."""
        ...


@runtime_checkable
class TextDecoder(Protocol):
    def decode(self, vector: torch.Tensor, max_words: int = 4, model_step: int = 0) -> str:
        """Embedding vector → text string."""
        ...

    def train_step(self, output_vector: torch.Tensor, teacher_text: str) -> None:
        """Learn from a teacher description."""
        ...
