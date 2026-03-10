import io

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F

from .clip_mlx import CLIPWrapper


def normalize(v: torch.Tensor) -> torch.Tensor:
    return F.normalize(v, dim=-1)


def temporal_position_encoding(
    frame_index: int,
    total_frames: int,
    dim: int = 512,
) -> torch.Tensor:
    """
    Sinusoidal encoding of relative position (0.0 to 1.0).
    Same principle as transformer positional encoding
    but over temporal position rather than sequence position.
    """
    position = frame_index / max(total_frames - 1, 1)
    dims = torch.arange(0, dim, 2).float()
    encoding = torch.zeros(dim)
    encoding[0::2] = torch.sin(position * 10000 ** (-dims / dim))
    encoding[1::2] = torch.cos(position * 10000 ** (-dims / dim))
    return encoding * 0.1


class ImageEncoder:
    def __init__(self, clip_wrapper: CLIPWrapper):
        self._clip = clip_wrapper

    def encode(self, image: PIL.Image.Image) -> torch.Tensor:
        """Returns: tensor of shape (512,), L2-normalized, on CPU."""
        raw = self._clip.encode_images([image])
        v = torch.from_numpy(raw[0])
        return normalize(v)

    def encode_batch(self, images: list[PIL.Image.Image]) -> torch.Tensor:
        """Returns: tensor of shape (N, 512), each row L2-normalized."""
        batch_size = 16
        all_vecs = []
        for i in range(0, len(images), batch_size):
            chunk = images[i : i + batch_size]
            raw = self._clip.encode_images(chunk)
            all_vecs.append(torch.from_numpy(raw))
        v = torch.cat(all_vecs, dim=0)
        return normalize(v)

    def encode_bytes(self, image_bytes: bytes) -> torch.Tensor:
        """Convenience method — decodes bytes to PIL then encodes."""
        image = PIL.Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return self.encode(image)


class TextEncoder:
    def __init__(self, clip_wrapper: CLIPWrapper):
        self._clip = clip_wrapper

    def encode(self, text: str) -> torch.Tensor:
        """Returns: tensor of shape (512,), L2-normalized, on CPU."""
        raw = self._clip.encode_texts([text])
        v = torch.from_numpy(raw[0])
        return normalize(v)

    def encode_batch(self, texts: list[str]) -> torch.Tensor:
        """Returns: tensor of shape (N, 512), each row L2-normalized."""
        raw = self._clip.encode_texts(texts)
        v = torch.from_numpy(raw)
        return normalize(v)


class VideoEncoder:
    def __init__(
        self,
        image_encoder: ImageEncoder,
        frames_per_clip: int = 8,
        add_position_encoding: bool = True,
    ):
        self._image_encoder = image_encoder
        self.frames_per_clip = frames_per_clip
        self.add_position_encoding = add_position_encoding

    def encode(self, video_path: str) -> torch.Tensor:
        """
        Returns: tensor of shape (frames_per_clip, 512), L2-normalized per frame.
        """
        import cv2

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            raise ValueError(f"Cannot read video: {video_path}")

        indices = np.linspace(0, total_frames - 1, self.frames_per_clip, dtype=int)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(PIL.Image.fromarray(frame_rgb))
        cap.release()

        # Pad with last frame if some reads failed
        while len(frames) < self.frames_per_clip:
            frames.append(frames[-1] if frames else PIL.Image.new("RGB", (224, 224)))

        return self.encode_frames(frames)

    def encode_frames(self, frames: list[PIL.Image.Image]) -> torch.Tensor:
        """
        For when frames are already extracted.
        Returns: tensor of shape (frames_per_clip, 512).
        """
        # Sample frames_per_clip frames evenly
        if len(frames) != self.frames_per_clip:
            indices = np.linspace(0, len(frames) - 1, self.frames_per_clip, dtype=int)
            frames = [frames[i] for i in indices]

        # Encode all frames as a batch
        v = self._image_encoder.encode_batch(frames)  # (N, 512), already normalized

        if self.add_position_encoding:
            for i in range(self.frames_per_clip):
                pe = temporal_position_encoding(i, self.frames_per_clip)
                v[i] = v[i] + pe
            v = normalize(v)

        return v
