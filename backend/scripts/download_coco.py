#!/usr/bin/env python3
"""
Download MS-COCO 2017 images + captions, compute CLIP ViT-B/32 embeddings,
and store (image_id, image_emb, caption_emb, caption_text, image_url) in
the embedding_cache table of the existing SQLite database.

Supports val2017 (5K images, ~1GB) and train2017 (118K images, ~18GB).

Usage:
    cd backend && python3 -m scripts.download_coco                         # val2017 (default)
    cd backend && python3 -m scripts.download_coco --split train2017       # train2017, first 20K
    cd backend && python3 -m scripts.download_coco --split train2017 --limit 50000
    cd backend && python3 -m scripts.download_coco --db-path /path/to/dev.db
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sqlite3
import struct
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# ---------------------------------------------------------------------------
# Split configuration
# ---------------------------------------------------------------------------

SPLITS = {
    "val2017": {
        "images_url": "http://images.cocodataset.org/zips/val2017.zip",
        "images_zip": "val2017.zip",
        "images_dir": "val2017",                        # path inside zip
        "annots_url": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        "annots_zip": "annotations_trainval2017.zip",
        "annots_json": "annotations/captions_val2017.json",
        "image_url_prefix": "http://images.cocodataset.org/val2017/",
        "default_limit": 0,                             # 0 = all (5K)
    },
    "train2017": {
        "images_url": "http://images.cocodataset.org/zips/train2017.zip",
        "images_zip": "train2017.zip",
        "images_dir": "train2017",
        "annots_url": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        "annots_zip": "annotations_trainval2017.zip",
        "annots_json": "annotations/captions_train2017.json",
        "image_url_prefix": "http://images.cocodataset.org/train2017/",
        "default_limit": 20000,                         # 18GB is big — default to 20K
    },
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _download(url: str, dest: str) -> None:
    """Download with a simple progress indicator."""
    if os.path.exists(dest):
        print(f"  already exists: {dest}")
        return
    print(f"  downloading {url}")
    t0 = time.time()

    def _progress(count, block_size, total_size):
        pct = count * block_size * 100 / total_size if total_size > 0 else 0
        mb = count * block_size / (1024 * 1024)
        elapsed = time.time() - t0
        speed = mb / elapsed if elapsed > 0 else 0
        print(f"\r  {pct:5.1f}%  {mb:.0f}MB  {speed:.1f}MB/s", end="", flush=True)

    urlretrieve(url, dest, reporthook=_progress)
    print()


def _emb_to_bytes(v: np.ndarray) -> bytes:
    """Pack a float32 vector into raw bytes for SQLite BLOB storage."""
    return struct.pack(f"{len(v)}f", *v.tolist())


def _pick_best_caption(captions: list[str], clip_model, clip_proc, image: Image.Image) -> tuple[str, np.ndarray, np.ndarray]:
    """
    Pick the caption with the highest CLIP similarity to the image.
    Returns (best_caption_text, image_emb, caption_emb).
    """
    img_inputs = clip_proc(images=[image], return_tensors="pt")
    with torch.no_grad():
        img_feat = clip_model.get_image_features(**img_inputs)
        if hasattr(img_feat, "pooler_output"):
            img_feat = img_feat.pooler_output
    img_emb = F.normalize(img_feat, dim=-1).cpu()

    txt_inputs = clip_proc(text=captions, return_tensors="pt",
                           padding=True, truncation=True, max_length=77)
    with torch.no_grad():
        txt_feat = clip_model.get_text_features(**txt_inputs)
        if hasattr(txt_feat, "pooler_output"):
            txt_feat = txt_feat.pooler_output
    txt_embs = F.normalize(txt_feat, dim=-1).cpu()

    sims = (img_emb @ txt_embs.T).squeeze(0)
    best_idx = sims.argmax().item()

    return (
        captions[best_idx],
        img_emb.squeeze(0).numpy(),
        txt_embs[best_idx].numpy(),
    )


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS embedding_cache (
    image_id     INTEGER PRIMARY KEY,
    image_emb    BLOB    NOT NULL,
    caption_emb  BLOB    NOT NULL,
    caption_text TEXT    NOT NULL,
    image_url    TEXT
);
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Download COCO 2017 & build embedding cache")
    _default_db = str(Path(__file__).resolve().parent.parent / "data" / "dev.db")
    parser.add_argument("--db-path", default=_default_db,
                        help="Path to SQLite database")
    parser.add_argument("--split", default="val2017", choices=list(SPLITS.keys()),
                        help="COCO split to download (default: val2017)")
    parser.add_argument("--data-dir", default=None,
                        help="Directory for COCO downloads (default: temp dir)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Images per CLIP batch")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process first N images (default: split-dependent)")
    args = parser.parse_args()

    split = SPLITS[args.split]
    limit = args.limit if args.limit is not None else split["default_limit"]

    db_path = args.db_path
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)

    data_dir = args.data_dir or os.path.join(tempfile.gettempdir(), "coco_cache")
    os.makedirs(data_dir, exist_ok=True)

    print(f"[coco] split: {args.split}")
    print(f"[coco] db: {db_path}")
    print(f"[coco] cache: {data_dir}")
    if limit > 0:
        print(f"[coco] limit: {limit} images")

    # ── 1. Download ──
    images_zip = os.path.join(data_dir, split["images_zip"])
    annots_zip = os.path.join(data_dir, split["annots_zip"])

    print("[coco] step 1/4: downloading images...")
    _download(split["images_url"], images_zip)
    print("[coco] step 2/4: downloading annotations...")
    _download(split["annots_url"], annots_zip)

    # ── 2. Parse annotations ──
    print("[coco] step 3/4: parsing annotations...")
    with zipfile.ZipFile(annots_zip) as zf:
        with zf.open(split["annots_json"]) as f:
            annots = json.load(f)

    # Build image_id → list of captions
    id_to_captions: dict[int, list[str]] = {}
    for ann in annots["annotations"]:
        iid = ann["image_id"]
        id_to_captions.setdefault(iid, []).append(ann["caption"])

    # Build image_id → filename
    id_to_file: dict[int, str] = {}
    for img_info in annots["images"]:
        id_to_file[img_info["id"]] = img_info["file_name"]

    image_ids = sorted(id_to_captions.keys())
    if limit > 0:
        image_ids = image_ids[:limit]
    total = len(image_ids)
    print(f"[coco] {total} images with captions")

    # ── 3. Load CLIP ──
    print("[coco] loading CLIP ViT-B/32...")
    from transformers import CLIPModel, CLIPProcessor
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.eval()
    print("[coco] CLIP loaded")

    # ── 4. Process & store ──
    print("[coco] step 4/4: encoding embeddings...")
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)

    # Migrate: add image_url column if missing (for pre-existing tables)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(embedding_cache)").fetchall()}
    if "image_url" not in cols:
        conn.execute("ALTER TABLE embedding_cache ADD COLUMN image_url TEXT")
        conn.commit()
        print("[coco] migrated: added image_url column", flush=True)

    # Check how many already done
    existing = set(
        r[0] for r in conn.execute("SELECT image_id FROM embedding_cache").fetchall()
    )
    remaining = [iid for iid in image_ids if iid not in existing]
    print(f"[coco] {len(existing)} already cached, {len(remaining)} remaining")

    if not remaining:
        print("[coco] all done!")
        conn.close()
        return

    images_dir = split["images_dir"]
    image_url_prefix = split["image_url_prefix"]

    zf = zipfile.ZipFile(images_zip)
    t0 = time.time()
    done = 0
    errors = 0

    for i, iid in enumerate(remaining):
        fname = id_to_file.get(iid)
        if fname is None:
            errors += 1
            continue

        try:
            with zf.open(f"{images_dir}/{fname}") as f:
                img = Image.open(io.BytesIO(f.read())).convert("RGB")
        except Exception as e:
            print(f"  [skip] image {iid}: {e}")
            errors += 1
            continue

        captions = id_to_captions[iid]
        try:
            best_cap, img_emb, cap_emb = _pick_best_caption(
                captions, clip_model, clip_proc, img
            )
        except Exception as e:
            print(f"  [skip] embed {iid}: {e}")
            errors += 1
            continue

        image_url = image_url_prefix + fname
        conn.execute(
            "INSERT OR REPLACE INTO embedding_cache (image_id, image_emb, caption_emb, caption_text, image_url) VALUES (?, ?, ?, ?, ?)",
            (iid, _emb_to_bytes(img_emb), _emb_to_bytes(cap_emb), best_cap, image_url),
        )

        done += 1
        if done % 100 == 0:
            conn.commit()
            elapsed = time.time() - t0
            rate = done / elapsed
            eta = (len(remaining) - done) / rate if rate > 0 else 0
            print(
                f"  {done}/{len(remaining)} ({done*100/len(remaining):.0f}%) "
                f"{rate:.1f} img/s  ETA {eta/60:.0f}m",
                flush=True,
            )

    conn.commit()
    zf.close()
    conn.close()

    elapsed = time.time() - t0
    print(f"[coco] done: {done} images in {elapsed:.0f}s ({errors} errors)")
    print(f"[coco] total cached: {len(existing) + done}")


if __name__ == "__main__":
    main()
