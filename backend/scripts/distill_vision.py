#!/usr/bin/env python3
"""
Offline distillation: train NativeVisionEncoder to match CLIP embeddings.

Reads CLIP embeddings from the SQLite embedding_cache table, downloads
the corresponding raw images from COCO, and trains the ConvNet to match.

Usage:
    cd /Users/ani/Frameo/Baby/backend
    python -m scripts.distill_vision [--epochs 5] [--batch-size 32] [--lr 0.001]
"""

import argparse
import io
import sqlite3
import struct
import sys
import time
from pathlib import Path

import numpy as np
import PIL.Image
import urllib.request
import torch

# Allow running as `python -m scripts.distill_vision` from backend/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from encoder.native_vision import NativeVisionEncoder

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "dev.db"
CHECKPOINT_DIR = Path(__file__).resolve().parents[1] / "data" / "checkpoints"
CHECKPOINT_PATH = CHECKPOINT_DIR / "native_vision.pt"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _blob_to_tensor(blob: bytes) -> torch.Tensor:
    """Decode a raw-float32 BLOB into a 1-D tensor."""
    n_floats = len(blob) // 4
    values = struct.unpack(f"{n_floats}f", blob)
    return torch.tensor(values, dtype=torch.float32)


def load_dataset(db_path: Path) -> list[tuple[str, torch.Tensor]]:
    """Returns list of (image_url, clip_embedding) from the cache."""
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute(
        "SELECT image_url, image_emb FROM embedding_cache WHERE image_url IS NOT NULL"
    ).fetchall()
    conn.close()

    dataset = []
    for url, emb_blob in rows:
        if url is None:
            continue
        clip_vec = _blob_to_tensor(emb_blob)
        dataset.append((url, clip_vec))

    return dataset


def download_image(url: str, timeout: float = 10.0) -> PIL.Image.Image | None:
    """Download an image from a URL. Returns None on failure."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        return PIL.Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 0.001,
    resume: bool = True,
) -> None:
    print(f"Database:   {DB_PATH}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Config:     epochs={epochs}  batch_size={batch_size}  lr={lr}")
    print()

    # Load dataset
    print("Loading embedding cache...")
    dataset = load_dataset(DB_PATH)
    print(f"  {len(dataset)} entries with image URLs")
    if not dataset:
        print("No data — run download_coco.py first.")
        return

    # Encoder
    encoder = NativeVisionEncoder()
    print(f"  ConvNet params: {encoder.param_count():,}")

    if resume and CHECKPOINT_PATH.exists():
        encoder.load(CHECKPOINT_PATH)
        print(f"  Resumed from {CHECKPOINT_PATH}")

    # Override LR if specified
    encoder._optimizer = torch.optim.Adam(encoder.net.parameters(), lr=lr)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # Training
    global_step = 0
    for epoch in range(1, epochs + 1):
        # Shuffle each epoch
        indices = np.random.permutation(len(dataset))
        epoch_losses: list[float] = []
        skipped = 0
        t0 = time.time()

        for batch_start in range(0, len(dataset), batch_size):
            batch_idx = indices[batch_start : batch_start + batch_size]
            batch_entries = [dataset[i] for i in batch_idx]

            # Download images for this batch
            images: list[PIL.Image.Image] = []
            targets: list[torch.Tensor] = []
            for url, clip_vec in batch_entries:
                img = download_image(url)
                if img is not None:
                    images.append(img)
                    targets.append(clip_vec)
                else:
                    skipped += 1

            if len(images) < 2:
                continue

            clip_targets = torch.stack(targets)
            loss = encoder.distill_step(images, clip_targets)
            epoch_losses.append(loss)
            global_step += 1

            if global_step % 100 == 0:
                _report(encoder, images, clip_targets, global_step, epoch_losses)

        elapsed = time.time() - t0
        avg_loss = np.mean(epoch_losses) if epoch_losses else float("nan")
        print(
            f"\n  Epoch {epoch}/{epochs}  "
            f"avg_loss={avg_loss:.4f}  "
            f"steps={len(epoch_losses)}  "
            f"skipped={skipped}  "
            f"time={elapsed:.1f}s"
        )

        # Save checkpoint after each epoch
        encoder.save(CHECKPOINT_PATH)
        print(f"  Saved checkpoint → {CHECKPOINT_PATH}")

    print("\nDone.")


def _report(
    encoder: NativeVisionEncoder,
    images: list[PIL.Image.Image],
    clip_targets: torch.Tensor,
    step: int,
    recent_losses: list[float],
) -> None:
    """Print cosine similarity metrics every 100 steps."""
    with torch.no_grad():
        native_embs = encoder.encode_batch(images)
        cos_sim = torch.nn.functional.cosine_similarity(native_embs, clip_targets.cpu())

    avg_loss = np.mean(recent_losses[-100:])
    print(
        f"  step {step:>5d}  "
        f"loss={avg_loss:.4f}  "
        f"cos_sim  mean={cos_sim.mean():.4f}  "
        f"min={cos_sim.min():.4f}  "
        f"max={cos_sim.max():.4f}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Distill CLIP → NativeVisionEncoder")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
