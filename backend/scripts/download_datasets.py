"""
Download curated training datasets for Baby AI via HuggingFace datasets library.

Stores everything in backend/data/datasets/ as JSON files ready for the curriculum.

Usage:
    python scripts/download_datasets.py              # download all
    python scripts/download_datasets.py --only gsm8k arc  # specific ones
    python scripts/download_datasets.py --list       # list available
    python scripts/download_datasets.py --skip-coco  # skip large image download
"""

import argparse
import json
import os
import time

from datasets import load_dataset

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "datasets")
os.makedirs(DATA_DIR, exist_ok=True)


def _save(name, items):
    path = os.path.join(DATA_DIR, f"{name}.json")
    with open(path, "w") as f:
        json.dump({"items": items, "count": len(items), "source": name}, f)
    print(f"  [{name}] saved {len(items):,} items to {path}")
    return path


# ── 1. GSM8K — Math Reasoning ──

def download_gsm8k():
    """8.8K grade school math with step-by-step solutions."""
    print("\n=== GSM8K (Math Reasoning) ===")
    ds = load_dataset("openai/gsm8k", "main", trust_remote_code=True)
    items = []
    for split in ds:
        for row in ds[split]:
            q, a = row["question"], row["answer"]
            final = a.split("####")[-1].strip() if "####" in a else ""
            items.append({
                "text": f"Q: {q}\nA: {a}",
                "question": q, "answer": a, "final_answer": final,
                "category": "math", "level": 2,
            })
    return _save("gsm8k", items)


# ── 2. ARC — Science Reasoning ──

def download_arc():
    """7.8K science multiple-choice questions."""
    print("\n=== ARC (Science Reasoning) ===")
    items = []
    for subset in ["ARC-Easy", "ARC-Challenge"]:
        ds = load_dataset("allenai/ai2_arc", subset, trust_remote_code=True)
        lvl = 3 if "Challenge" in subset else 2
        for split in ds:
            for row in ds[split]:
                stem = row["question"]
                choices = row["choices"]
                labels = choices["label"]
                texts = choices["text"]
                answer_key = row["answerKey"]
                choices_str = " ".join(f"({l}) {t}" for l, t in zip(labels, texts))
                answer_text = next((t for l, t in zip(labels, texts) if l == answer_key), "")
                items.append({
                    "text": f"Q: {stem} {choices_str}\nA: {answer_text}",
                    "question": stem, "answer": answer_text,
                    "category": "science_reasoning", "level": lvl,
                })
    return _save("arc", items)


# ── 3. CommonsenseQA ──

def download_commonsenseqa():
    """12K common sense reasoning questions."""
    print("\n=== CommonsenseQA ===")
    ds = load_dataset("tau/commonsense_qa", trust_remote_code=True)
    items = []
    for split in ds:
        for row in ds[split]:
            stem = row["question"]
            choices = row["choices"]
            labels = choices["label"]
            texts = choices["text"]
            answer_key = row["answerKey"]
            choices_str = " ".join(f"({l}) {t}" for l, t in zip(labels, texts))
            answer_text = next((t for l, t in zip(labels, texts) if l == answer_key), "")
            items.append({
                "text": f"Q: {stem} {choices_str}\nA: {answer_text}",
                "question": stem, "answer": answer_text,
                "category": "commonsense_reasoning", "level": 2,
            })
    return _save("commonsenseqa", items)


# ── 4. BoolQ — Yes/No Reasoning ──

def download_boolq():
    """15K yes/no reasoning with passages."""
    print("\n=== BoolQ (Yes/No Reasoning) ===")
    ds = load_dataset("google/boolq", trust_remote_code=True)
    items = []
    for split in ds:
        for row in ds[split]:
            q = row["question"]
            passage = row["passage"][:300]
            answer = "yes" if row["answer"] else "no"
            items.append({
                "text": f"Passage: {passage}\nQ: {q}\nA: {answer}",
                "question": q, "passage": passage, "answer": answer,
                "category": "reasoning", "level": 2,
            })
    return _save("boolq", items)


# ── 5. Code Alpaca — Coding ──

def download_code_alpaca():
    """20K coding instruction-response pairs."""
    print("\n=== Code Alpaca (Coding) ===")
    ds = load_dataset("sahil2801/CodeAlpaca-20k", trust_remote_code=True)
    items = []
    for split in ds:
        for row in ds[split]:
            instruction = row.get("instruction", "")
            inp = row.get("input", "")
            output = row.get("output", "")
            text = f"Q: {instruction}"
            if inp:
                text += f"\nInput: {inp}"
            text += f"\nA: {output}"
            items.append({
                "text": text, "question": instruction,
                "input": inp, "answer": output,
                "category": "coding", "level": 3,
            })
    return _save("code_alpaca", items)


# ── 6. Simple Wikipedia — Facts & Grammar ──

def download_simple_wikipedia():
    """Clean factual sentences from Simple English Wikipedia."""
    print("\n=== Simple Wikipedia ===")
    ds = load_dataset("wikipedia", "20220301.simple", trust_remote_code=True)
    items = []
    for row in ds["train"]:
        text = row["text"]
        # Split into sentences, take clean ones
        for sent in text.split(". "):
            sent = sent.strip()
            if len(sent) > 30 and len(sent) < 200 and not sent.startswith("{{"):
                if not sent.endswith("."):
                    sent += "."
                items.append({"text": sent, "category": "facts", "level": 1})
                if len(items) >= 100000:
                    break
        if len(items) >= 100000:
            break
    return _save("simple_wikipedia", items)


# ── 7. HellaSwag — Commonsense Completion ──

def download_hellaswag():
    """10K commonsense sentence completion."""
    print("\n=== HellaSwag (Commonsense Completion) ===")
    ds = load_dataset("Rowan/hellaswag", trust_remote_code=True)
    items = []
    for split in ds:
        for row in ds[split]:
            ctx = row["ctx"]
            endings = row["endings"]
            label = int(row["label"]) if row["label"] != "" else 0
            answer = endings[label] if label < len(endings) else endings[0]
            items.append({
                "text": f"{ctx} {answer}",
                "question": ctx, "answer": answer,
                "category": "commonsense_completion", "level": 2,
            })
    return _save("hellaswag", items)


# ── 8. MATH — Competition Math ──

def download_math():
    """12.5K competition math problems with solutions."""
    print("\n=== MATH (Competition Math) ===")
    ds = load_dataset("lighteval/MATH", "all", trust_remote_code=True)
    items = []
    for split in ds:
        for row in ds[split]:
            items.append({
                "text": f"Q: {row['problem']}\nA: {row['solution']}",
                "question": row["problem"], "answer": row["solution"],
                "category": "math_competition",
                "level": min(int(row.get("level", "3").replace("Level ", "")), 5) if isinstance(row.get("level"), str) else 3,
            })
    return _save("math_competition", items)


# ── 9. WinoGrande — Pronoun Reasoning ──

def download_winogrande():
    """44K pronoun resolution (reasoning about who/what)."""
    print("\n=== WinoGrande (Pronoun Reasoning) ===")
    ds = load_dataset("allenai/winogrande", "winogrande_xl", trust_remote_code=True)
    items = []
    for split in ds:
        for row in ds[split]:
            sentence = row["sentence"]
            opt1, opt2 = row["option1"], row["option2"]
            answer = opt1 if row["answer"] == "1" else opt2
            filled = sentence.replace("_", answer)
            items.append({
                "text": filled,
                "question": sentence, "answer": answer,
                "options": [opt1, opt2],
                "category": "pronoun_reasoning", "level": 2,
            })
    return _save("winogrande", items)


# ── 10. COCO Captions — Image + Text Pairs ──

def download_coco():
    """50K COCO image-caption pairs."""
    print("\n=== COCO Captions (Images + Text) ===")
    import urllib.request

    coco_dir = os.path.join(DATA_DIR, "coco")
    images_dir = os.path.join(coco_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    # Use HF datasets for annotations (more reliable than direct download)
    print("  Loading COCO captions from HuggingFace...")
    ds = load_dataset("HuggingFaceM4/COCO", trust_remote_code=True, split="train[:50000]")

    items = []
    downloaded = 0
    skipped = 0
    for i, row in enumerate(ds):
        try:
            # Save image
            img = row["image"]
            filename = f"coco_{i:06d}.jpg"
            img_path = os.path.join(images_dir, filename)
            if not os.path.exists(img_path):
                img.save(img_path)
                downloaded += 1

            captions = row.get("sentences", {}).get("raw", [])
            if not captions:
                captions = [row.get("caption", "")]

            items.append({
                "image_path": img_path,
                "captions": captions,
                "text": captions[0] if captions else "",
                "category": "coco",
            })
        except Exception:
            skipped += 1
            continue

        if (i + 1) % 5000 == 0:
            print(f"  Progress: {i+1}/50000 ({downloaded} saved, {skipped} skipped)")

    print(f"  Done: {downloaded} images saved, {skipped} skipped")
    return _save("coco_captions", items)


# ── Main ──

DATASETS = {
    "gsm8k": ("GSM8K — 8.8K math reasoning", download_gsm8k),
    "arc": ("ARC — 7.8K science reasoning", download_arc),
    "commonsenseqa": ("CommonsenseQA — 12K common sense", download_commonsenseqa),
    "boolq": ("BoolQ — 15K yes/no reasoning", download_boolq),
    "code_alpaca": ("Code Alpaca — 20K coding", download_code_alpaca),
    "wikipedia": ("Simple Wikipedia — 100K factual sentences", download_simple_wikipedia),
    "hellaswag": ("HellaSwag — 10K commonsense completion", download_hellaswag),
    "math": ("MATH — 12.5K competition math", download_math),
    "winogrande": ("WinoGrande — 44K pronoun reasoning", download_winogrande),
    "coco": ("COCO — 50K image-caption pairs", download_coco),
}


def main():
    parser = argparse.ArgumentParser(description="Download training datasets for Baby AI")
    parser.add_argument("--only", nargs="+", help="Download only specific datasets")
    parser.add_argument("--list", action="store_true", help="List available datasets")
    parser.add_argument("--skip-coco", action="store_true", help="Skip COCO (large)")
    args = parser.parse_args()

    if args.list:
        print("Available datasets:")
        for name, (desc, _) in DATASETS.items():
            print(f"  {name:20s} — {desc}")
        return

    targets = args.only if args.only else list(DATASETS.keys())
    if args.skip_coco and "coco" in targets:
        targets.remove("coco")

    print(f"Downloading {len(targets)} datasets to {DATA_DIR}")
    print("=" * 60)

    results = {}
    for name in targets:
        if name not in DATASETS:
            print(f"Unknown dataset: {name}")
            continue
        desc, func = DATASETS[name]
        t0 = time.time()
        try:
            path = func()
            results[name] = {"status": "ok", "time": time.time() - t0}
        except Exception as e:
            print(f"  FAILED: {e}")
            results[name] = {"status": "error", "error": str(e)}

    print("\n" + "=" * 60)
    print("SUMMARY:")
    for name, r in results.items():
        if r["status"] == "ok":
            print(f"  ✓ {name:20s} — {r['time']:.1f}s")
        else:
            print(f"  ✗ {name:20s} — {r['error'][:80]}")


if __name__ == "__main__":
    main()
