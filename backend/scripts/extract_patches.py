"""
Extract CLIP ViT-B/32 patch-level features from COCO images.

Saves a .pt file mapping image_id → (49, 512) tensor.
Each image produces 49 patch embeddings (7×7 grid from 224×224 input).

Usage:
    python -m scripts.extract_patches                     # val2017 (5K images, ~500MB)
    python -m scripts.extract_patches --split train2017   # train2017 (118K images, ~12GB)
    python -m scripts.extract_patches --limit 1000        # first 1000 images only
"""

import argparse
import io
import os
import sqlite3
import time
import zipfile

import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image


SPLITS = {
    "val2017": {
        "images_url": "http://images.cocodataset.org/zips/val2017.zip",
        "images_zip": "val2017.zip",
        "images_dir": "val2017",
    },
    "train2017": {
        "images_url": "http://images.cocodataset.org/zips/train2017.zip",
        "images_zip": "train2017.zip",
        "images_dir": "train2017",
    },
}


def main():
    parser = argparse.ArgumentParser(description="Extract CLIP patch features from COCO images")
    _default_db = str(Path(__file__).resolve().parent.parent / "data" / "dev.db")
    _default_out = str(Path(__file__).resolve().parent.parent / "data" / "patch_features.pt")
    parser.add_argument("--db-path", default=_default_db, help="SQLite db with embedding_cache")
    parser.add_argument("--output", default=_default_out, help="Output .pt file")
    parser.add_argument("--split", default="val2017", choices=list(SPLITS.keys()))
    parser.add_argument("--data-dir", default=None, help="Directory with COCO zips")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    split = SPLITS[args.split]
    import tempfile
    data_dir = args.data_dir or os.path.join(tempfile.gettempdir(), "coco_cache")

    # Get image IDs from existing embedding_cache
    conn = sqlite3.connect(args.db_path)
    image_ids = [r[0] for r in conn.execute("SELECT image_id FROM embedding_cache ORDER BY image_id").fetchall()]
    conn.close()

    if args.limit:
        image_ids = image_ids[:args.limit]
    print(f"[patches] {len(image_ids)} images to process from {args.split}")

    # Load existing patches to skip
    existing = {}
    if os.path.exists(args.output):
        existing = torch.load(args.output, weights_only=False)
        print(f"[patches] {len(existing)} already extracted, resuming")
    remaining = [iid for iid in image_ids if iid not in existing]
    if not remaining:
        print("[patches] all done!")
        return

    # Open image zip
    images_zip = os.path.join(data_dir, split["images_zip"])
    if not os.path.exists(images_zip):
        print(f"[patches] images zip not found at {images_zip}")
        print(f"[patches] run: python -m scripts.download_coco --split {args.split} first")
        return

    # We need image_id → filename mapping from the zip
    # Read filenames from the zip directly
    zf = zipfile.ZipFile(images_zip)
    images_dir = split["images_dir"]
    # Build filename → image_id mapping (COCO filenames are zero-padded IDs)
    id_to_file = {}
    for name in zf.namelist():
        if name.startswith(images_dir + "/") and name.endswith(".jpg"):
            fname = os.path.basename(name)
            # COCO filename format: 000000000001.jpg → image_id = 1
            try:
                iid = int(fname.replace(".jpg", ""))
                id_to_file[iid] = name
            except ValueError:
                continue

    remaining = [iid for iid in remaining if iid in id_to_file]
    print(f"[patches] {len(remaining)} images remaining")

    # Load CLIP
    print("[patches] loading CLIP ViT-B/32...")
    from transformers import CLIPModel, CLIPProcessor
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.eval()
    visual_proj = clip_model.visual_projection  # (768 → 512)
    print("[patches] CLIP loaded")

    # Process in batches
    patches_dict = dict(existing)
    t0 = time.time()
    done = 0
    errors = 0

    for batch_start in range(0, len(remaining), args.batch_size):
        batch_ids = remaining[batch_start:batch_start + args.batch_size]
        images = []
        valid_ids = []

        for iid in batch_ids:
            try:
                with zf.open(id_to_file[iid]) as f:
                    img = Image.open(io.BytesIO(f.read())).convert("RGB")
                images.append(img)
                valid_ids.append(iid)
            except Exception as e:
                errors += 1
                continue

        if not images:
            continue

        try:
            inputs = clip_proc(images=images, return_tensors="pt")
            with torch.no_grad():
                # Get vision model hidden states (includes CLS + 49 patches)
                vision_out = clip_model.vision_model(pixel_values=inputs["pixel_values"])
                # last_hidden_state: (batch, 50, 768) — CLS token + 49 patches
                hidden = vision_out.last_hidden_state[:, 1:]  # skip CLS → (batch, 49, 768)
                # Project to CLIP embedding space
                projected = visual_proj(hidden)  # (batch, 49, 512)
                # L2 normalize each patch
                projected = F.normalize(projected, dim=-1)

            for i, iid in enumerate(valid_ids):
                patches_dict[iid] = projected[i].cpu()  # (49, 512)
                done += 1

        except Exception as e:
            print(f"  [skip] batch error: {e}")
            errors += 1
            continue

        if done % 100 == 0 and done > 0:
            elapsed = time.time() - t0
            rate = done / elapsed
            eta = (len(remaining) - done) / rate if rate > 0 else 0
            print(f"  {done}/{len(remaining)}  {rate:.1f} img/s  ETA {eta:.0f}s  errors={errors}")

        # Save periodically
        if done % 500 == 0 and done > 0:
            torch.save(patches_dict, args.output)

    zf.close()

    # Final save
    torch.save(patches_dict, args.output)
    elapsed = time.time() - t0
    size_mb = os.path.getsize(args.output) / 1e6
    print(f"[patches] done: {done} images, {errors} errors, {elapsed:.1f}s, {size_mb:.0f}MB → {args.output}")


if __name__ == "__main__":
    main()
