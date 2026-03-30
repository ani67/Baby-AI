"""
Image generation integration module.

Bridges the brain's 512-dim output vectors to visual output via the
VisionDecoder. Supports three modes:

    1. generate_from_vector  — raw 512-dim brain output to image
    2. generate_from_text    — text -> encode -> brain -> decode -> image
    3. reconstruct           — image -> encode -> decode (autoencoder test)
"""

import torch
import PIL.Image

from encoder.native_vision import NativeVisionEncoder
from encoder.native_vision_decoder import VisionDecoder


class ImageGenerator:
    """High-level interface for brain-to-image generation."""

    def __init__(
        self,
        encoder: NativeVisionEncoder,
        decoder: VisionDecoder,
    ) -> None:
        self.encoder = encoder
        self.decoder = decoder

    def generate_from_vector(self, vec_512dim: torch.Tensor) -> PIL.Image.Image:
        """Brain output (512-dim) -> PIL image."""
        return self.decoder.decode(vec_512dim)

    def generate_from_text(self, text: str, text_encoder, brain) -> PIL.Image.Image:
        """Text -> encode -> brain forward -> decode -> PIL image.

        Args:
            text:         input string
            text_encoder: anything with .encode(str) -> (512,) tensor
            brain:        brain module with .forward(tensor) -> (tensor, _)
        """
        input_vec = text_encoder.encode(text)
        output_vec, _ = brain.forward(input_vec)
        return self.decoder.decode(output_vec)

    def reconstruct(self, image: PIL.Image.Image) -> PIL.Image.Image:
        """Image -> encode -> decode (autoencoder round-trip test)."""
        vec = self.encoder.encode(image)
        return self.decoder.decode(vec)
