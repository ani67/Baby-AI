"""
Download COCO images from URLs stored in the embedding_cache table.

Usage:
    cd backend && python -m scripts.download_images --limit 500
    cd backend && python -m scripts.download_images          # all 5000
"""

import argparse
import json
import os
import sqlite3
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(ROOT, "data", "dev.db")
OUT_DIR = os.path.join(ROOT, "data", "coco_images")
MANIFEST_PATH = os.path.join(OUT_DIR, "manifest.json")
STAGE0_DIR = os.path.join(ROOT, "data", "stage0")
CAPTIONS_PATH = os.path.join(ROOT, "data", "image_captions.json")


# ── Download COCO images ─────────────────────────────────────────────


def fetch_rows(limit: int | None = None) -> list[tuple[int, str, str]]:
    """Read image URLs from the embedding_cache table."""
    conn = sqlite3.connect(DB_PATH)
    query = (
        "SELECT image_id, image_url, caption_text "
        "FROM embedding_cache WHERE image_url IS NOT NULL "
        "ORDER BY image_id"
    )
    if limit:
        query += f" LIMIT {limit}"
    rows = conn.execute(query).fetchall()
    conn.close()
    return rows


def download_images(limit: int | None = None) -> dict[str, dict]:
    """Download images and return manifest dict."""
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load existing manifest so we can resume
    manifest: dict[str, dict] = {}
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH) as f:
            manifest = json.load(f)

    rows = fetch_rows(limit)
    total = len(rows)
    downloaded = 0
    skipped = 0
    failed = 0

    print(f"[download] {total} image URLs to process, saving to {OUT_DIR}")

    for i, (image_id, url, caption) in enumerate(rows, 1):
        filename = f"{image_id}.jpg"
        filepath = os.path.join(OUT_DIR, filename)
        key = str(image_id)

        # Skip if already downloaded
        if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            manifest[key] = {"path": f"data/coco_images/{filename}", "caption": caption}
            skipped += 1
            if i % 50 == 0:
                _progress(i, total, downloaded, skipped, failed)
            continue

        # Download
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "BabyAI/1.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = resp.read()
            with open(filepath, "wb") as f:
                f.write(data)
            manifest[key] = {"path": f"data/coco_images/{filename}", "caption": caption}
            downloaded += 1
        except (urllib.error.URLError, urllib.error.HTTPError, OSError, TimeoutError) as e:
            failed += 1
            if failed <= 5:
                print(f"  [skip] {image_id}: {e}")
            elif failed == 6:
                print("  [skip] suppressing further error messages...")

        if i % 50 == 0:
            _progress(i, total, downloaded, skipped, failed)
            # Save manifest periodically for resumability
            _save_manifest(manifest)

    _progress(total, total, downloaded, skipped, failed)
    _save_manifest(manifest)
    print(f"[download] manifest saved to {MANIFEST_PATH}")
    return manifest


def _progress(i: int, total: int, dl: int, sk: int, fa: int):
    print(f"  [{i}/{total}] downloaded={dl}  skipped={sk}  failed={fa}")


def _save_manifest(manifest: dict):
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)


# ── Build unified captions file ──────────────────────────────────────


def build_captions(manifest: dict[str, dict]):
    """Merge stage0 images and COCO downloads into a single captions file."""
    entries: list[dict] = []

    # Stage0: derive caption from category folder name
    if os.path.isdir(STAGE0_DIR):
        for category in sorted(os.listdir(STAGE0_DIR)):
            cat_dir = os.path.join(STAGE0_DIR, category)
            if not os.path.isdir(cat_dir):
                continue
            for fname in sorted(os.listdir(cat_dir)):
                if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                entries.append({
                    "image_path": f"data/stage0/{category}/{fname}",
                    "caption": f"a photo of a {category}",
                    "category": category,
                })

    # COCO images from manifest
    for image_id, info in sorted(manifest.items(), key=lambda x: int(x[0])):
        caption = info["caption"]
        # Derive a rough category from the caption (first noun-like word)
        category = _guess_category(caption)
        entries.append({
            "image_path": info["path"],
            "caption": caption,
            "category": category,
        })

    with open(CAPTIONS_PATH, "w") as f:
        json.dump(entries, f, indent=2)

    print(f"[captions] {len(entries)} entries written to {CAPTIONS_PATH}")
    print(f"  stage0: {sum(1 for e in entries if e['image_path'].startswith('data/stage0'))}")
    print(f"  coco:   {sum(1 for e in entries if e['image_path'].startswith('data/coco'))}")


def _guess_category(caption: str) -> str:
    """Extract a rough category from a COCO caption.

    Simple heuristic: look for known object nouns in the caption,
    falling back to 'scene' for captions that don't match.
    """
    caption_lower = caption.lower()
    # Common COCO categories sorted roughly by specificity
    coco_categories = [
        "airplane", "apple", "backpack", "banana", "baseball", "basketball",
        "bear", "bed", "bench", "bicycle", "bird", "boat", "book", "bottle",
        "bowl", "broccoli", "bus", "cake", "car", "carrot", "cat", "chair",
        "clock", "computer", "couch", "cow", "cup", "desk", "dog", "donut",
        "elephant", "fire hydrant", "fork", "frisbee", "giraffe", "guitar",
        "handbag", "horse", "hot dog", "keyboard", "kite", "knife", "laptop",
        "microwave", "motorcycle", "mouse", "orange", "oven", "parking meter",
        "person", "phone", "piano", "pizza", "potted plant", "refrigerator",
        "sandwich", "scissors", "sheep", "sink", "skateboard", "ski", "snowboard",
        "sofa", "spoon", "sports", "stop sign", "suitcase", "surfboard",
        "table", "teddy bear", "television", "tennis", "tie", "toaster",
        "toilet", "toothbrush", "traffic light", "train", "truck", "umbrella",
        "vase", "wine glass", "zebra",
    ]
    for cat in coco_categories:
        if cat in caption_lower:
            return cat
    return "scene"


# ── CLI ───────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Download COCO images from dev.db")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Max images to download (default: all)",
    )
    parser.add_argument(
        "--captions-only", action="store_true",
        help="Only rebuild the captions file (no downloads)",
    )
    args = parser.parse_args()

    if args.captions_only:
        manifest = {}
        if os.path.exists(MANIFEST_PATH):
            with open(MANIFEST_PATH) as f:
                manifest = json.load(f)
        build_captions(manifest)
        return

    manifest = download_images(args.limit)
    build_captions(manifest)


if __name__ == "__main__":
    main()
