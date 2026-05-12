#!/usr/bin/env python3
"""
Agent 9 — Curriculum v2 Pipeline & Integration
================================================

Pipeline order (run top to bottom):

  1. COLLECT     glob data/curriculum_v2/**/raw/*.jsonl, stream records,
                 assign stable id = sha1(text)[:16].
  2. DEDUPE      MinHash LSH, threshold=0.8, num_perm=128; first-occurrence
                 wins. Words lowercased, pure-punctuation tokens dropped.
  3. QUALITY     re-compute quality_score with a calibrated formula
                 (spec baseline + expanded reasoning vocabulary).
                 Soft scoring, no rejection.
  4. BALANCE     enforce TARGET_DISTRIBUTION ratios across present domains,
                 recomputing proportions over the domains that actually
                 have data. Order each domain by quality desc and cap at
                 its share of the collected token pool.
  5. TOKENIZER   train 32K BPE on the balanced corpus (ByteLevel
                 pre-tokenizer, min_frequency=3, the 5 special tokens).
  6. SPLIT       90/5/5 train/val/test, grouped by doc_id = source+'_'+title
                 so chunks of one document stay together. seed=42.
  7. WRITE       train.jsonl, val.jsonl, test.jsonl, tokenizer.json,
                 stats.json, manifest.json under data/curriculum_v2/final/.

Spec reference: doc/curriculum/COLLECTION_SPEC.md, section
"Agent 9 — Pipeline & Integration".

This script is the only artefact committed; all generated data lives
under data/curriculum_v2/final/ which is gitignored.
"""

from __future__ import annotations

import datetime as _dt
import glob
import hashlib
import json
import os
import random
import re
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Tuple


# ---------------------------------------------------------------------------
# Paths — all absolute so the script runs from anywhere.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data" / "curriculum_v2"
FINAL_DIR = DATA_ROOT / "final"
RAW_GLOB = str(DATA_ROOT / "**" / "raw" / "*.jsonl")
SPEC_PATH_REL = "doc/curriculum/COLLECTION_SPEC.md"

# ---------------------------------------------------------------------------
# Constants from the spec.
# ---------------------------------------------------------------------------
TARGET_DISTRIBUTION: Dict[str, float] = {
    "foundations/mathematics": 0.10,
    "foundations/logic":       0.05,
    "foundations/language":    0.05,
    "philosophy":              0.08,
    "science":                 0.07,
    "ai":                      0.05,
    "code":                    0.05,
    "literature":              0.07,
    "dialogue":                0.25,
    "design":                  0.03,
    "finance":                 0.03,
    "media":                   0.02,
    "history":                 0.08,
    "culture":                 0.07,
}

SPLIT_TRAIN = 0.90
SPLIT_VAL = 0.05
SPLIT_TEST = 0.05
SPLIT_SEED = 42

DEDUP_THRESHOLD = 0.8
DEDUP_NUM_PERM = 128

TOKENIZER_VOCAB_SIZE = 32768
TOKENIZER_MIN_FREQ = 3
SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>", "<mask>"]

# Spec baseline + the calibrated additions called out in the brief.
REASONING_WORDS = {
    # spec baseline
    "therefore", "because", "however", "although", "implies",
    "suggests", "shows", "demonstrates", "furthermore",
    "nevertheless", "consequently",
    # calibrated additions for science / code / math contexts
    "thus", "hence", "observe", "note", "recall", "consider",
    "specifically", "particular", "analogously",
}

# "note that" / "in particular" — handle bigrams as a side channel.
REASONING_BIGRAMS = {"note that", "in particular"}

PUNCT_TOKEN_RE = re.compile(r"^[^\w]+$", flags=re.UNICODE)

LOG_EVERY = 100


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def log(msg: str) -> None:
    print(f"[agent9] {msg}", flush=True)


def stream_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for ln, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                log(f"  WARN bad json {path.name}:{ln} — {exc}")


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> int:
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def stable_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def tokens_for_minhash(text: str) -> List[str]:
    out: List[str] = []
    for tok in text.lower().split():
        if PUNCT_TOKEN_RE.match(tok):
            continue
        out.append(tok)
    return out


# ---------------------------------------------------------------------------
# Step 1 — Collect
# ---------------------------------------------------------------------------
def collect() -> List[Dict[str, Any]]:
    files = sorted(glob.glob(RAW_GLOB, recursive=True))
    log(f"step 1 collect: scanning {len(files)} raw jsonl files")
    records: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()
    skipped_no_text = 0
    for f in files:
        p = Path(f)
        size = p.stat().st_size
        if size == 0:
            log(f"  skip empty file: {p.relative_to(REPO_ROOT)}")
            continue
        n_before = len(records)
        for rec in stream_jsonl(p):
            text = rec.get("text")
            if not text or not isinstance(text, str):
                skipped_no_text += 1
                continue
            rid = stable_id(text)
            if rid in seen_ids:
                # Identical text already counted from another file — skip.
                continue
            seen_ids.add(rid)
            rec["id"] = rid
            # Defensive defaults for optional fields.
            rec.setdefault("source", "unknown")
            rec.setdefault("domain", "unknown")
            rec.setdefault("subdomain", "")
            rec.setdefault("author", "")
            rec.setdefault("title", "")
            rec.setdefault("language", "en")
            rec.setdefault("dialogue", False)
            rec.setdefault("quality_score", 0.5)
            if "tokens" not in rec or not isinstance(rec["tokens"], int):
                rec["tokens"] = len(text.split())
            records.append(rec)
        log(f"  +{len(records) - n_before:>4} from {p.relative_to(REPO_ROOT)}")
    log(f"step 1 collect: {len(records)} unique records "
        f"(skipped {skipped_no_text} with no text)")
    return records


# ---------------------------------------------------------------------------
# Step 2 — Deduplicate (MinHash LSH)
# ---------------------------------------------------------------------------
def deduplicate(records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    from datasketch import MinHash, MinHashLSH

    log(f"step 2 dedupe: MinHash LSH t={DEDUP_THRESHOLD} "
        f"num_perm={DEDUP_NUM_PERM} on {len(records)} records")
    lsh = MinHashLSH(threshold=DEDUP_THRESHOLD, num_perm=DEDUP_NUM_PERM)
    kept: List[Dict[str, Any]] = []
    dup_count = 0
    for idx, rec in enumerate(records, 1):
        mh = MinHash(num_perm=DEDUP_NUM_PERM)
        toks = tokens_for_minhash(rec["text"])
        if not toks:
            # Degenerate text — drop silently.
            continue
        for w in toks:
            mh.update(w.encode("utf-8"))
        # query returns list of ids similar above threshold
        if lsh.query(mh):
            dup_count += 1
        else:
            lsh.insert(rec["id"], mh)
            kept.append(rec)
        if idx % LOG_EVERY == 0:
            log(f"  dedup progress {idx}/{len(records)} "
                f"(kept={len(kept)} dup={dup_count})")
    log(f"step 2 dedupe: kept {len(kept)} | removed {dup_count}")
    return kept, dup_count


# ---------------------------------------------------------------------------
# Step 3 — Quality score (calibrated)
# ---------------------------------------------------------------------------
def compute_quality_score(record: Dict[str, Any]) -> float:
    """Calibrated re-score. Soft, no rejection.

    Honours short, dense scientific / programming content (arXiv abstracts,
    Stack Exchange answers) instead of penalising it for being short.
    """
    text: str = record["text"]
    words = text.split()
    n = len(words)
    if n == 0:
        return 0.0
    score = 0.5

    # Length sweet-spot — broaden the lower bound so short technical
    # content isn't penalised. Spec was 200..2000.
    if 80 <= n <= 2000:
        score += 0.10
    elif n > 2000:
        score += 0.05
    # Below 80 words: no length bonus, but also no hard penalty.

    unique = set(words)
    unique_ratio = len(unique) / n
    score += unique_ratio * 0.2

    last_char = text.rstrip()[-1:] if text.rstrip() else ""
    if last_char in ".!?":
        score += 0.05

    # Reasoning vocabulary (lowercase-word set intersection).
    lower_unique = {w.lower().strip(".,;:!?\"'()[]") for w in words}
    reason_count = len(lower_unique & REASONING_WORDS)
    # Bigram check — cheap pass over the lowercase text.
    text_lc = text.lower()
    for bigram in REASONING_BIGRAMS:
        if bigram in text_lc:
            reason_count += 1
    score += min(reason_count * 0.02, 0.1)

    # Repetition penalty (kept from spec).
    if unique_ratio < 0.4:
        score -= 0.2

    return max(0.0, min(1.0, score))


def rescore_all(records: List[Dict[str, Any]]) -> None:
    log(f"step 3 quality: re-scoring {len(records)} records (calibrated)")
    for r in records:
        r["quality_score"] = round(compute_quality_score(r), 4)
    if records:
        vals = [r["quality_score"] for r in records]
        log(f"  quality range min={min(vals):.3f} max={max(vals):.3f} "
            f"mean={sum(vals)/len(vals):.3f}")


# ---------------------------------------------------------------------------
# Step 4 — Domain balance
# ---------------------------------------------------------------------------
def balance(records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Enforce TARGET_DISTRIBUTION ratios proportionally over the data
    actually present. target_tokens = total_collected_tokens so we use
    everything; the cap just prevents any single domain from exceeding
    its proportional share.
    """
    by_domain: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_domain[r["domain"]].append(r)

    total_tokens = sum(r["tokens"] for r in records)
    log(f"step 4 balance: total_tokens={total_tokens} "
        f"across {len(by_domain)} domains")

    # Recompute proportions over present domains. Domains not in spec keep
    # their actual share (no target shrink) but participate in normalisation.
    present_spec_ratios = {d: TARGET_DISTRIBUTION[d]
                           for d in TARGET_DISTRIBUTION
                           if d in by_domain}
    present_sum = sum(present_spec_ratios.values())
    if present_sum <= 0:
        # No domains in spec — fall back to uniform.
        present_spec_ratios = {d: 1.0 / len(by_domain) for d in by_domain}
        present_sum = 1.0

    normalised = {d: ratio / present_sum
                  for d, ratio in present_spec_ratios.items()}

    # Domains present in data but absent from spec: keep all their tokens.
    extra_domains = [d for d in by_domain if d not in TARGET_DISTRIBUTION]

    balanced: List[Dict[str, Any]] = []
    per_domain_report: Dict[str, Any] = {}

    for domain, recs in by_domain.items():
        recs_sorted = sorted(recs, key=lambda r: -r["quality_score"])
        if domain in normalised:
            cap = int(total_tokens * normalised[domain])
        else:
            cap = sum(r["tokens"] for r in recs_sorted)  # take all
        current = 0
        kept_here: List[Dict[str, Any]] = []
        for r in recs_sorted:
            if current >= cap and kept_here:  # always keep at least 1
                break
            kept_here.append(r)
            current += r["tokens"]
        balanced.extend(kept_here)
        per_domain_report[domain] = {
            "collected_tokens": sum(r["tokens"] for r in recs),
            "kept_tokens": current,
            "record_count_in": len(recs),
            "record_count_out": len(kept_here),
            "target_ratio": TARGET_DISTRIBUTION.get(domain),
            "normalised_target_ratio": normalised.get(domain),
            "mean_quality_score": round(
                sum(r["quality_score"] for r in recs) / len(recs), 4
            ) if recs else 0.0,
        }
        log(f"  domain={domain:<28} in={len(recs):>4} "
            f"out={len(kept_here):>4} tokens={current:>8} "
            f"(cap={cap})")

    if extra_domains:
        log(f"  note: domains not in spec kept fully: {extra_domains}")

    log(f"step 4 balance: kept {len(balanced)} records "
        f"({sum(r['tokens'] for r in balanced)} tokens)")
    return balanced, per_domain_report


# ---------------------------------------------------------------------------
# Step 5 — Tokenizer
# ---------------------------------------------------------------------------
def train_tokenizer(records: List[Dict[str, Any]], save_path: Path) -> int:
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers

    log(f"step 5 tokenizer: training BPE vocab={TOKENIZER_VOCAB_SIZE} "
        f"on {len(records)} documents")
    # Write one document per line to a temp file (tokenizers trains on files).
    with tempfile.NamedTemporaryFile("w", suffix=".txt", encoding="utf-8",
                                     delete=False) as tmp:
        tmp_path = tmp.name
        for r in records:
            # Replace newlines so each document occupies exactly one line.
            tmp.write(r["text"].replace("\n", " ") + "\n")

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=TOKENIZER_VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=TOKENIZER_MIN_FREQ,
        show_progress=False,
    )
    try:
        tokenizer.train([tmp_path], trainer)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(save_path))
        vocab_size = tokenizer.get_vocab_size()
        log(f"step 5 tokenizer: vocab_size={vocab_size} saved to "
            f"{save_path.relative_to(REPO_ROOT)}")
        return vocab_size
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Step 6 — Split by document
# ---------------------------------------------------------------------------
def split_by_doc(records: List[Dict[str, Any]]) -> Tuple[
    List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]
]:
    log(f"step 6 split: 90/5/5 by doc_id, seed={SPLIT_SEED}")
    by_doc: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        doc_id = f"{r.get('source','unknown')}_{r.get('title','')}"
        by_doc[doc_id].append(r)
    docs = list(by_doc.keys())
    rng = random.Random(SPLIT_SEED)
    rng.shuffle(docs)
    n = len(docs)
    cut_train = int(n * SPLIT_TRAIN)
    cut_val = int(n * (SPLIT_TRAIN + SPLIT_VAL))
    train_docs = set(docs[:cut_train])
    val_docs = set(docs[cut_train:cut_val])
    test_docs = set(docs[cut_val:])

    train, val, test = [], [], []
    for doc_id, recs in by_doc.items():
        bucket = (train if doc_id in train_docs
                  else val if doc_id in val_docs
                  else test)
        bucket.extend(recs)

    log(f"  docs total={n} train={len(train_docs)} "
        f"val={len(val_docs)} test={len(test_docs)}")
    log(f"  recs train={len(train)} val={len(val)} test={len(test)}")
    return train, val, test


# ---------------------------------------------------------------------------
# Step 7 — Write outputs
# ---------------------------------------------------------------------------
def write_outputs(
    train: List[Dict[str, Any]],
    val: List[Dict[str, Any]],
    test: List[Dict[str, Any]],
    domain_report: Dict[str, Any],
    pre_dedup_count: int,
    dedup_removed: int,
    vocab_size: int,
    all_records_for_manifest: List[Dict[str, Any]],
) -> None:
    FINAL_DIR.mkdir(parents=True, exist_ok=True)

    n_train = write_jsonl(FINAL_DIR / "train.jsonl", train)
    n_val = write_jsonl(FINAL_DIR / "val.jsonl", val)
    n_test = write_jsonl(FINAL_DIR / "test.jsonl", test)

    train_tokens = sum(r["tokens"] for r in train)
    val_tokens = sum(r["tokens"] for r in val)
    test_tokens = sum(r["tokens"] for r in test)

    stats = {
        "global": {
            "total_records_in": pre_dedup_count,
            "dedup_removed": dedup_removed,
            "total_records_after_dedup": pre_dedup_count - dedup_removed,
            "total_records_final": n_train + n_val + n_test,
            "total_tokens_final": train_tokens + val_tokens + test_tokens,
            "tokenizer_vocab_size": vocab_size,
            "train_records": n_train,
            "val_records": n_val,
            "test_records": n_test,
            "train_tokens": train_tokens,
            "val_tokens": val_tokens,
            "test_tokens": test_tokens,
            "target_distribution": TARGET_DISTRIBUTION,
        },
        "per_domain": domain_report,
    }
    with (FINAL_DIR / "stats.json").open("w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2, ensure_ascii=False)

    sources = sorted({r.get("source", "unknown")
                      for r in all_records_for_manifest})
    # License hints taken from spec (Licenses section). Map by source.
    license_table = {
        "gutenberg": "public domain (pre-1927)",
        "arxiv": "arXiv license (non-commercial OK)",
        "mit_ocw": "CC BY-NC-SA",
        "stack_overflow": "CC BY-SA 4.0",
        "stack_exchange": "CC BY-SA 4.0",
        "stackexchange": "CC BY-SA 4.0",
        "reddit": "user-generated, research use",
        "wikipedia": "CC BY-SA 3.0",
        "wikipedia_fa": "CC BY-SA 3.0",
        "imsdb": "fair use for research",
        "ted": "CC BY-NC-ND",
        "berkshire": "public (posted freely)",
        "oaktree": "public (posted freely)",
        "philosophy_se": "CC BY-SA 4.0",
        "python_docs": "PSF license (Python docs)",
        "stanford_encyclopedia_of_philosophy":
            "SEP — academic fair use (public-access scholarly reference)",
    }
    licenses = []
    for s in sources:
        licenses.append({"source": s,
                         "license": license_table.get(s, "unknown / see source")})

    manifest = {
        "collection_date": _dt.datetime.now(_dt.timezone.utc)
                              .isoformat().replace("+00:00", "Z"),
        "sources": sources,
        "licenses": licenses,
        "collection_spec_path": SPEC_PATH_REL,
        "notes": "Validation slice, not full 600M target. "
                 "Ratios enforced proportionally over present domains.",
    }
    with (FINAL_DIR / "manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)

    log(f"step 7 write: train={n_train} val={n_val} test={n_test} "
        f"(tokens train={train_tokens} val={val_tokens} test={test_tokens})")
    log(f"  wrote {FINAL_DIR.relative_to(REPO_ROOT)}/{{train,val,test}}.jsonl, "
        f"stats.json, manifest.json")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    log(f"repo={REPO_ROOT}")
    log(f"data_root={DATA_ROOT}")
    log(f"final_dir={FINAL_DIR}")

    # Step 1
    records = collect()
    if not records:
        log("FATAL: no records collected")
        return 1
    pre_dedup_count = len(records)

    # Step 2
    records, dedup_removed = deduplicate(records)

    # Step 3
    rescore_all(records)

    # Step 4
    balanced, domain_report = balance(records)

    # Step 5
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    vocab_size = train_tokenizer(balanced, FINAL_DIR / "tokenizer.json")

    # Step 6
    train, val, test = split_by_doc(balanced)

    # Step 7
    write_outputs(train, val, test,
                  domain_report=domain_report,
                  pre_dedup_count=pre_dedup_count,
                  dedup_removed=dedup_removed,
                  vocab_size=vocab_size,
                  all_records_for_manifest=balanced)

    log("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
