"""
Train the NativeVisionEncoder + VisionDecoder as an autoencoder.

    image -> encoder -> 512-dim -> decoder -> reconstructed image
    Loss = MSE(original_pixels, reconstructed_pixels)

Backpropagates through both encoder and decoder jointly.

Usage:
    cd backend
    python -m scripts.train_autoencoder --epochs 50 --batch-size 16 --lr 0.001 --save-samples
"""

import argparse
import random
import sys
from pathlib import Path

import PIL.Image
import torch
import torch.nn.functional as F
from torchvision import transforms

# Ensure backend/ is on the path when run as a script
_backend = Path(__file__).resolve().parent.parent
if str(_backend) not in sys.path:
    sys.path.insert(0, str(_backend))

from encoder.native_vision import NativeVisionEncoder, _prepare_batch
from encoder.native_vision_decoder import VisionDecoder, _prepare_target_batch


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_images(data_dir: Path) -> list[Path]:
    """Recursively collect all JPEG/PNG image paths under data_dir."""
    exts = {".jpg", ".jpeg", ".png"}
    paths = sorted(
        p for p in data_dir.rglob("*")
        if p.suffix.lower() in exts
    )
    return paths


def load_pil_batch(paths: list[Path]) -> list[PIL.Image.Image]:
    """Load a list of file paths as PIL images."""
    images = []
    for p in paths:
        try:
            images.append(PIL.Image.open(p).convert("RGB"))
        except Exception:
            continue
    return images


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    data_dir: Path,
    encoder_ckpt: Path | None,
    decoder_ckpt: Path | None,
    epochs: int,
    batch_size: int,
    lr: float,
    save_samples: bool,
    sample_dir: Path,
    device: str,
) -> None:
    # Discover images
    image_paths = load_images(data_dir)
    if not image_paths:
        print(f"No images found in {data_dir}")
        sys.exit(1)
    print(f"Found {len(image_paths)} images in {data_dir}")

    # Build models
    encoder = NativeVisionEncoder(device=device)
    decoder = VisionDecoder(dim=512, device=device)

    if encoder_ckpt and encoder_ckpt.exists():
        encoder.load(encoder_ckpt)
        print(f"Loaded encoder checkpoint: {encoder_ckpt}")

    if decoder_ckpt and decoder_ckpt.exists():
        decoder.load(decoder_ckpt)
        print(f"Loaded decoder checkpoint: {decoder_ckpt}")

    print(f"Encoder params: {encoder.param_count():,}")
    print(f"Decoder params: {decoder.param_count():,}")

    # Joint optimizer over both networks
    all_params = list(encoder.net.parameters()) + list(decoder.net.parameters())
    optimizer = torch.optim.Adam(all_params, lr=lr)

    # Pixel-space target transform (64x64, [0,1], no ImageNet norm)
    target_transform = transforms.Compose([
        transforms.Resize(
            (64, 64), interpolation=transforms.InterpolationMode.LANCZOS,
        ),
        transforms.ToTensor(),
    ])

    dev = torch.device(device)

    if save_samples:
        sample_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0

    for epoch in range(1, epochs + 1):
        random.shuffle(image_paths)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]
            pil_images = load_pil_batch(batch_paths)
            if len(pil_images) < 2:
                continue

            # Encoder input: ImageNet-normalized (same as NativeVisionEncoder)
            enc_input = _prepare_batch(pil_images).to(dev)

            # Decoder target: raw pixels in [0, 1]
            pixel_targets = torch.stack(
                [target_transform(img) for img in pil_images]
            ).to(dev)

            # Forward
            encoder.net.train()
            decoder.net.train()

            embeddings, _ = encoder.net(enc_input)       # (N, 512)
            reconstructed = decoder.net(embeddings)       # (N, 3, 64, 64)

            loss = F.mse_loss(reconstructed, pixel_targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            encoder.net.eval()
            decoder.net.eval()

            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1

            # Save sample reconstructions
            if save_samples and global_step % 100 == 0:
                _save_samples(
                    pil_images[:4],
                    reconstructed[:4].detach().cpu(),
                    sample_dir,
                    global_step,
                )

        avg = epoch_loss / max(n_batches, 1)
        print(f"Epoch {epoch}/{epochs}  loss={avg:.6f}  steps={global_step}")

    # Save checkpoints
    ckpt_dir = _backend / "data" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    enc_path = ckpt_dir / "native_vision.pt"
    dec_path = ckpt_dir / "native_vision_decoder.pt"
    encoder.save(enc_path)
    decoder.save(dec_path)
    print(f"Saved encoder -> {enc_path}")
    print(f"Saved decoder -> {dec_path}")


# ---------------------------------------------------------------------------
# Sample saving
# ---------------------------------------------------------------------------

def _save_samples(
    originals: list[PIL.Image.Image],
    reconstructed: torch.Tensor,
    out_dir: Path,
    step: int,
) -> None:
    """Side-by-side: original | reconstruction, saved as a single image."""
    n = min(len(originals), reconstructed.shape[0])
    pairs = []
    for i in range(n):
        orig = originals[i].convert("RGB").resize((64, 64), PIL.Image.LANCZOS)
        rec_pixels = (reconstructed[i] * 255).clamp(0, 255).byte()
        rec_arr = rec_pixels.permute(1, 2, 0).numpy()
        rec_img = PIL.Image.fromarray(rec_arr, mode="RGB")
        # Concatenate horizontally
        pair = PIL.Image.new("RGB", (128, 64))
        pair.paste(orig, (0, 0))
        pair.paste(rec_img, (64, 0))
        pairs.append(pair)

    # Stack vertically
    grid = PIL.Image.new("RGB", (128, 64 * n))
    for i, pair in enumerate(pairs):
        grid.paste(pair, (0, 64 * i))

    path = out_dir / f"step_{step:06d}.png"
    grid.save(path)
    print(f"  Saved sample -> {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train encoder + decoder as an autoencoder",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=_backend / "data" / "stage0",
        help="Directory containing training images (default: backend/data/stage0/)",
    )
    parser.add_argument(
        "--encoder-ckpt",
        type=Path,
        default=_backend / "data" / "checkpoints" / "native_vision.pt",
        help="Path to encoder checkpoint (loaded if exists)",
    )
    parser.add_argument(
        "--decoder-ckpt",
        type=Path,
        default=_backend / "data" / "checkpoints" / "native_vision_decoder.pt",
        help="Path to decoder checkpoint (loaded if exists)",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument(
        "--save-samples",
        action="store_true",
        help="Save sample reconstructions every 100 steps",
    )
    parser.add_argument(
        "--sample-dir",
        type=Path,
        default=_backend / "data" / "reconstructions",
    )
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        encoder_ckpt=args.encoder_ckpt,
        decoder_ckpt=args.decoder_ckpt,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_samples=args.save_samples,
        sample_dir=args.sample_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
