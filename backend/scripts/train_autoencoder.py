"""
Train the NativeVisionEncoder + VisionDecoder as an autoencoder.

    image -> encoder -> 512-dim -> decoder -> reconstructed image
    Loss = MSE(original_pixels, reconstructed_pixels) + 0.5 * perceptual_loss

Backpropagates through both encoder and decoder jointly.
Perceptual loss compares intermediate conv features (conv1-conv3) between
original and reconstructed images through a frozen encoder copy.

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
# Data augmentation (training only)
# ---------------------------------------------------------------------------

_train_augment = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomRotation(degrees=10),
])


def augment_pil(images: list[PIL.Image.Image]) -> list[PIL.Image.Image]:
    """Apply training augmentations to a list of PIL images."""
    return [_train_augment(img) for img in images]


# ---------------------------------------------------------------------------
# Perceptual loss (frozen encoder features)
# ---------------------------------------------------------------------------

class PerceptualLoss:
    """Extract intermediate features from conv1-conv3 of a frozen encoder.

    Compares feature maps between original and reconstructed images to
    encourage structural similarity beyond pixel-level MSE.
    """

    def __init__(self, encoder: NativeVisionEncoder, device: torch.device) -> None:
        self.net = encoder.net
        self.net.eval()
        # Freeze all encoder parameters — no gradient flow
        for p in self.net.parameters():
            p.requires_grad = False
        self.device = device
        # Resize reconstructed 64x64 -> 192x192 for encoder input
        self._resize = transforms.Resize(
            (192, 192), interpolation=transforms.InterpolationMode.BILINEAR,
        )
        self._normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def _extract_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Run through conv1-conv3 with residuals and return intermediate feature maps."""
        return self.net.extract_features(x)

    def _prepare_for_encoder(self, pixels: torch.Tensor) -> torch.Tensor:
        """Convert [0,1] 64x64 pixels to encoder input format (192x192, ImageNet-normalized)."""
        # pixels: (N, 3, 64, 64) in [0, 1]
        upsampled = F.interpolate(pixels, size=(192, 192), mode="bilinear", align_corners=False)
        # Apply ImageNet normalization per-sample
        return self._normalize(upsampled)

    @torch.no_grad()
    def compute(
        self,
        original_enc_input: torch.Tensor,
        reconstructed_pixels: torch.Tensor,
    ) -> torch.Tensor:
        """Perceptual loss between original (already encoder-prepped) and
        reconstructed pixels (64x64, [0,1]).

        Returns scalar loss (mean MSE across conv1-conv3 features).
        """
        recon_input = self._prepare_for_encoder(reconstructed_pixels)
        orig_feats = self._extract_features(original_enc_input)
        recon_feats = self._extract_features(recon_input)

        loss = torch.tensor(0.0, device=self.device)
        for of, rf in zip(orig_feats, recon_feats):
            loss = loss + F.mse_loss(rf, of)
        return loss / len(orig_feats)


# ---------------------------------------------------------------------------
# Nearest-neighbor baseline
# ---------------------------------------------------------------------------

@torch.no_grad()
def find_nearest_neighbors(
    query_embeddings: torch.Tensor,
    all_embeddings: torch.Tensor,
    all_paths: list[Path],
    k: int = 1,
) -> list[list[Path]]:
    """For each query embedding, find the k nearest training images by cosine similarity.

    Args:
        query_embeddings: (N, 512) L2-normalized embeddings of sample images.
        all_embeddings:   (M, 512) L2-normalized embeddings of all training images.
        all_paths:        list of M file paths corresponding to all_embeddings.
        k:                number of neighbors to return.

    Returns:
        List of N lists, each containing k file paths.
    """
    # Cosine similarity: (N, M)
    sims = query_embeddings @ all_embeddings.T
    topk = sims.topk(k, dim=1)
    results = []
    for i in range(query_embeddings.shape[0]):
        neighbors = [all_paths[idx] for idx in topk.indices[i].tolist()]
        results.append(neighbors)
    return results


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
        print(f"[autoencoder] No images found in {data_dir}")
        sys.exit(1)
    print(f"[autoencoder] Found {len(image_paths)} images in {data_dir}")

    # Split into train/val (90/10)
    random.shuffle(image_paths)
    split = max(1, int(len(image_paths) * 0.9))
    train_paths = image_paths[:split]
    val_paths = image_paths[split:] if split < len(image_paths) else image_paths[-4:]
    print(f"[autoencoder] Train: {len(train_paths)}, Val: {len(val_paths)}")

    # Build models
    encoder = NativeVisionEncoder(device=device)
    decoder = VisionDecoder(dim=512, device=device)

    if encoder_ckpt and encoder_ckpt.exists():
        encoder.load(encoder_ckpt)
        print(f"[autoencoder] Loaded encoder checkpoint: {encoder_ckpt}")

    if decoder_ckpt and decoder_ckpt.exists():
        decoder.load(decoder_ckpt)
        print(f"[autoencoder] Loaded decoder checkpoint: {decoder_ckpt}")

    print(f"[autoencoder] Encoder params: {encoder.param_count():,}")
    print(f"[autoencoder] Decoder params: {decoder.param_count():,}")

    # Frozen encoder copy for perceptual loss
    percept_encoder = NativeVisionEncoder(device=device)
    if encoder_ckpt and encoder_ckpt.exists():
        percept_encoder.load(encoder_ckpt)
    perceptual = PerceptualLoss(percept_encoder, torch.device(device))
    print("[autoencoder] Perceptual loss encoder frozen and ready.")

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

    # Pre-compute all training embeddings for nearest-neighbor baseline
    print("[autoencoder] Pre-computing training embeddings for NN baseline...")
    all_train_embeddings = []
    embed_batch_size = 32
    for i in range(0, len(train_paths), embed_batch_size):
        batch = load_pil_batch(train_paths[i:i + embed_batch_size])
        if batch:
            embs = encoder.encode_batch(batch)
            all_train_embeddings.append(embs)
    if all_train_embeddings:
        all_train_embeddings = torch.cat(all_train_embeddings, dim=0)
        print(f"[autoencoder] Cached {all_train_embeddings.shape[0]} training embeddings.")
    else:
        all_train_embeddings = torch.zeros(0, 512)

    global_step = 0

    for epoch in range(1, epochs + 1):
        random.shuffle(train_paths)
        epoch_loss = 0.0
        epoch_mse = 0.0
        epoch_percept = 0.0
        n_batches = 0

        for i in range(0, len(train_paths), batch_size):
            batch_paths = train_paths[i : i + batch_size]
            pil_images = load_pil_batch(batch_paths)
            if len(pil_images) < 2:
                continue

            # Apply data augmentation (training only)
            aug_images = augment_pil(pil_images)

            # Encoder input: ImageNet-normalized (same as NativeVisionEncoder)
            enc_input = _prepare_batch(aug_images).to(dev)

            # Decoder target: raw pixels in [0, 1] (from augmented images)
            pixel_targets = torch.stack(
                [target_transform(img) for img in aug_images]
            ).to(dev)

            # Forward
            encoder.net.train()
            decoder.net.train()

            embeddings, _ = encoder.net(enc_input)       # (N, 512)
            reconstructed = decoder.net(embeddings)       # (N, 3, 64, 64)

            mse_loss = F.mse_loss(reconstructed, pixel_targets)

            # Perceptual loss: compare intermediate features
            # Use non-augmented originals for perceptual reference (stable target)
            orig_enc_input = _prepare_batch(pil_images).to(dev)
            # Recompute embeddings from originals for perceptual comparison
            with torch.no_grad():
                orig_embeddings, _ = encoder.net(orig_enc_input)
            orig_recon = decoder.net(orig_embeddings.detach())
            p_loss = perceptual.compute(orig_enc_input, orig_recon)

            loss = mse_loss + 0.5 * p_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            encoder.net.eval()
            decoder.net.eval()

            epoch_loss += loss.item()
            epoch_mse += mse_loss.item()
            epoch_percept += p_loss.item()
            n_batches += 1
            global_step += 1

            # Progress every 50 steps
            if global_step % 50 == 0:
                print(
                    f"  step {global_step}  "
                    f"loss={loss.item():.5f}  "
                    f"mse={mse_loss.item():.5f}  "
                    f"percept={p_loss.item():.5f}",
                    flush=True,
                )

            # Save sample reconstructions
            if save_samples and global_step % 100 == 0:
                _save_samples(
                    pil_images[:4],
                    reconstructed[:4].detach().cpu(),
                    sample_dir,
                    global_step,
                )

        avg = epoch_loss / max(n_batches, 1)
        avg_mse = epoch_mse / max(n_batches, 1)
        avg_p = epoch_percept / max(n_batches, 1)
        print(
            f"[autoencoder] Epoch {epoch}/{epochs}  "
            f"loss={avg:.6f}  mse={avg_mse:.6f}  percept={avg_p:.6f}  "
            f"steps={global_step}",
            flush=True,
        )

        # Nearest-neighbor baseline at end of each epoch
        _run_nn_baseline(
            encoder, decoder, val_paths[:4], train_paths,
            all_train_embeddings, target_transform,
            sample_dir if save_samples else None, epoch, dev,
        )

    # Save checkpoints
    ckpt_dir = _backend / "data" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    enc_path = ckpt_dir / "native_vision.pt"
    dec_path = ckpt_dir / "native_vision_decoder.pt"
    encoder.save(enc_path)
    decoder.save(dec_path)
    print(f"[autoencoder] Saved encoder -> {enc_path}")
    print(f"[autoencoder] Saved decoder -> {dec_path}")


# ---------------------------------------------------------------------------
# Nearest-neighbor baseline
# ---------------------------------------------------------------------------

def _run_nn_baseline(
    encoder: NativeVisionEncoder,
    decoder: VisionDecoder,
    sample_paths: list[Path],
    train_paths: list[Path],
    all_train_embeddings: torch.Tensor,
    target_transform,
    sample_dir: Path | None,
    epoch: int,
    dev: torch.device,
) -> None:
    """For a few validation samples, compare decoder output vs nearest training image."""
    sample_images = load_pil_batch(sample_paths[:4])
    if not sample_images or all_train_embeddings.shape[0] == 0:
        return

    sample_embs = encoder.encode_batch(sample_images)  # (N, 512)
    neighbors = find_nearest_neighbors(sample_embs, all_train_embeddings, train_paths, k=1)

    # Compute reconstruction MSE vs nearest-neighbor MSE
    for i, (img, nn_paths) in enumerate(zip(sample_images, neighbors)):
        target_t = target_transform(img).to(dev)  # (3, 64, 64)
        emb = sample_embs[i:i+1].to(dev)

        # Decoder reconstruction
        with torch.no_grad():
            recon = decoder.net(emb).squeeze(0).cpu()  # (3, 64, 64)
        recon_mse = F.mse_loss(recon, target_t.cpu()).item()

        # Nearest neighbor
        nn_img = PIL.Image.open(nn_paths[0]).convert("RGB")
        nn_t = target_transform(nn_img)  # (3, 64, 64)
        nn_mse = F.mse_loss(nn_t, target_t.cpu()).item()

        print(
            f"  [NN baseline] sample {i}: "
            f"decoder_mse={recon_mse:.5f}  "
            f"nearest_mse={nn_mse:.5f}  "
            f"{'DECODER WINS' if recon_mse < nn_mse else 'NN wins'}",
            flush=True,
        )

    # Save comparison grid: original | reconstruction | nearest neighbor
    if sample_dir is not None:
        _save_nn_comparison(
            sample_images, sample_embs, neighbors,
            decoder, target_transform, sample_dir, epoch, dev,
        )


def _save_nn_comparison(
    originals: list[PIL.Image.Image],
    embeddings: torch.Tensor,
    neighbors: list[list[Path]],
    decoder: VisionDecoder,
    target_transform,
    out_dir: Path,
    epoch: int,
    dev: torch.device,
) -> None:
    """Save a grid: original | decoder output | nearest neighbor."""
    n = min(len(originals), 4)
    rows = []
    for i in range(n):
        orig = originals[i].convert("RGB").resize((64, 64), PIL.Image.LANCZOS)

        # Decoder reconstruction
        with torch.no_grad():
            emb = embeddings[i:i+1].to(dev)
            rec_t = decoder.net(emb).squeeze(0).cpu()
        rec_pixels = (rec_t * 255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
        rec_img = PIL.Image.fromarray(rec_pixels, mode="RGB")

        # Nearest neighbor
        nn_img = PIL.Image.open(neighbors[i][0]).convert("RGB").resize(
            (64, 64), PIL.Image.LANCZOS,
        )

        # Concatenate: original | reconstruction | nearest neighbor
        row = PIL.Image.new("RGB", (192, 64))
        row.paste(orig, (0, 0))
        row.paste(rec_img, (64, 0))
        row.paste(nn_img, (128, 0))
        rows.append(row)

    grid = PIL.Image.new("RGB", (192, 64 * n))
    for i, row in enumerate(rows):
        grid.paste(row, (0, 64 * i))

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"nn_baseline_epoch_{epoch:03d}.png"
    grid.save(path)
    print(f"  [NN baseline] Saved comparison -> {path}", flush=True)


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
    print(f"  [autoencoder] Saved sample -> {path}", flush=True)


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
