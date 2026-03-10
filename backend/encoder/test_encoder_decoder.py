import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import PIL.Image
import torch

from encoder.clip_mlx import CLIPWrapper
from encoder.encoder import ImageEncoder, TextEncoder, VideoEncoder
from encoder.decoder import TextDecoder
from encoder.vocab import Vocabulary


# Shared CLIP instance — loaded once for all tests
_clip = None


def get_clip():
    global _clip
    if _clip is None:
        _clip = CLIPWrapper()
    return _clip


# ── ImageEncoder ──


def test_image_encoder_output_shape():
    enc = ImageEncoder(get_clip())
    img = PIL.Image.new("RGB", (224, 224), color="red")
    v = enc.encode(img)
    assert v.shape == (512,)
    print("PASS: test_image_encoder_output_shape")


def test_image_encoder_output_normalized():
    enc = ImageEncoder(get_clip())
    img = PIL.Image.new("RGB", (224, 224))
    v = enc.encode(img)
    assert abs(torch.norm(v).item() - 1.0) < 1e-5
    print("PASS: test_image_encoder_output_normalized")


# ── TextEncoder ──


def test_text_encoder_output_shape():
    enc = TextEncoder(get_clip())
    v = enc.encode("a photo of a dog")
    assert v.shape == (512,)
    print("PASS: test_text_encoder_output_shape")


def test_text_encoder_output_normalized():
    enc = TextEncoder(get_clip())
    v = enc.encode("hello world")
    assert abs(torch.norm(v).item() - 1.0) < 1e-5
    print("PASS: test_text_encoder_output_normalized")


# ── CLIP alignment ──


def test_clip_alignment():
    """Image and matching text should be more similar than mismatched."""
    clip = get_clip()
    img_enc = ImageEncoder(clip)
    txt_enc = TextEncoder(clip)

    # Use a solid red image — "a solid red color" should match better than "a blue chair"
    red_image = PIL.Image.new("RGB", (224, 224), color=(255, 0, 0))
    v_red_img = img_enc.encode(red_image)
    v_red_txt = txt_enc.encode("a solid red color")
    v_chair_txt = txt_enc.encode("a photo of a blue chair")

    sim_correct = torch.dot(v_red_img, v_red_txt).item()
    sim_wrong = torch.dot(v_red_img, v_chair_txt).item()
    assert sim_correct > sim_wrong, (
        f"Alignment failed: sim_correct={sim_correct:.4f} <= sim_wrong={sim_wrong:.4f}"
    )
    print(f"PASS: test_clip_alignment (correct={sim_correct:.4f} > wrong={sim_wrong:.4f})")


# ── VideoEncoder ──


def test_video_encoder_output_shape():
    clip = get_clip()
    img_enc = ImageEncoder(clip)
    vid_enc = VideoEncoder(img_enc, frames_per_clip=8)
    frames = [PIL.Image.new("RGB", (224, 224)) for _ in range(16)]
    v = vid_enc.encode_frames(frames)
    assert v.shape == (8, 512), f"Expected (8, 512), got {v.shape}"
    print("PASS: test_video_encoder_output_shape")


def test_video_encoder_temporal_ordering():
    """Earlier frames should have different encoding than later frames."""
    clip = get_clip()
    img_enc = ImageEncoder(clip)
    vid_enc = VideoEncoder(img_enc, frames_per_clip=4)
    frame = PIL.Image.new("RGB", (224, 224), color="blue")
    frames = [frame] * 8
    v = vid_enc.encode_frames(frames)
    assert not torch.allclose(v[0], v[-1]), "Position encoding should make vectors differ"
    print("PASS: test_video_encoder_temporal_ordering")


# ── Vocabulary ──


def test_vocabulary_special_tokens():
    vocab = Vocabulary()
    assert vocab.encode("<START>") == [1]
    assert vocab.encode("<END>") == [2]
    print("PASS: test_vocabulary_special_tokens")


def test_vocabulary_grows():
    vocab = Vocabulary(max_size=100)
    for _ in range(3):
        vocab.add_word("serendipity")
    assert "serendipity" in vocab.word_to_id
    print("PASS: test_vocabulary_grows")


def test_vocabulary_max_size_respected():
    vocab = Vocabulary(max_size=10)
    for i in range(20):
        for _ in range(3):
            vocab.add_word(f"word_{i}")
    assert len(vocab.word_to_id) <= 10
    print("PASS: test_vocabulary_max_size_respected")


if __name__ == "__main__":
    # Run vocabulary tests first (no model load needed)
    test_vocabulary_special_tokens()
    test_vocabulary_grows()
    test_vocabulary_max_size_respected()

    # CLIP-dependent tests
    print("\nLoading CLIP model...")
    test_image_encoder_output_shape()
    test_image_encoder_output_normalized()
    test_text_encoder_output_shape()
    test_text_encoder_output_normalized()
    test_clip_alignment()
    test_video_encoder_output_shape()
    test_video_encoder_temporal_ordering()

    print("\nAll 10 tests passed.")
