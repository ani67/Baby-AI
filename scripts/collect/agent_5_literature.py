"""
Agent 5 — Literature & Narrative collection.

Fetches canonical literature from Project Gutenberg, chunks it into
JSONL records, and writes them to data/curriculum_v2/literature/raw/.

Validation mode (default): 5 hand-picked books spanning ancient /
19th-century / 20th-century, including at least one poetry and one
essay collection. Writes to agent_5_validation.jsonl.

Full mode (--full): full Agent 5 source list. Not used in this
validation session; the catalog below is a stub the next pass will
expand.

Spec: doc/curriculum/COLLECTION_SPEC.md, section "Agent 5 —
Literature & Narrative".
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    import requests
except ImportError:  # pragma: no cover
    sys.stderr.write(
        "requests not installed. Run: pip install requests --break-system-packages -q\n"
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Source catalog
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Book:
    gid: int
    title: str
    author: str
    subdomain: str  # "narrative" | "poetry" | "essay"


# Validation set: 5 books, ancient + 19th + 20th, >= 1 poetry, >= 1 essay.
VALIDATION_BOOKS: tuple[Book, ...] = (
    Book(1727,  "The Odyssey",           "Homer (Butler trans.)", "narrative"),  # ancient
    Book(1342,  "Pride and Prejudice",   "Jane Austen",           "narrative"),  # 19th c.
    Book(2814,  "Dubliners",             "James Joyce",           "narrative"),  # 20th c.
    Book(1322,  "Leaves of Grass",       "Walt Whitman",          "poetry"),     # 19th c.
    Book(16328, "Essays",                "Francis Bacon",         "essay"),      # classical
)

# Full source list — placeholder for the --full run. Today this just
# expands VALIDATION_BOOKS with a few more canonical IDs so --full
# still does something sensible, but the real curation pass will
# populate this from the spec's complete Agent 5 catalog.
FULL_BOOKS: tuple[Book, ...] = VALIDATION_BOOKS + (
    Book(2701,  "Moby Dick",                "Herman Melville",   "narrative"),
    Book(1399,  "Anna Karenina",            "Leo Tolstoy",       "narrative"),
    Book(28054, "The Brothers Karamazov",   "Fyodor Dostoevsky", "narrative"),
    Book(100,   "Complete Works",           "William Shakespeare","narrative"),
    Book(12242, "Complete Poems",           "Emily Dickinson",   "poetry"),
    Book(20,    "Paradise Lost",            "John Milton",       "poetry"),
    Book(6130,  "The Iliad",                "Homer",             "poetry"),
    Book(16643, "Essays: First Series",     "Ralph Waldo Emerson","essay"),
    Book(205,   "Walden",                   "Henry David Thoreau","essay"),
)


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------

GUTENBERG_URLS = (
    "https://www.gutenberg.org/files/{gid}/{gid}-0.txt",
    "https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt",
    "https://www.gutenberg.org/files/{gid}/{gid}.txt",
)

USER_AGENT = "curriculum-v2-agent5/0.1 (research; contact via repo)"


def fetch_book(gid: int, timeout: float = 30.0) -> str | None:
    """Try the known Gutenberg URL patterns until one returns text."""
    for tmpl in GUTENBERG_URLS:
        url = tmpl.format(gid=gid)
        try:
            r = requests.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
        except requests.RequestException as exc:
            print(f"  fetch error for {url}: {exc}", file=sys.stderr)
            continue
        if r.status_code == 200 and len(r.text) > 5000:
            # Gutenberg serves utf-8 with a BOM for some books; requests
            # decodes it but we strip the leading marker just in case.
            return r.text.lstrip("﻿")
    return None


# ---------------------------------------------------------------------------
# Header / footer / boilerplate stripping
# ---------------------------------------------------------------------------

_START_RE = re.compile(
    r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
    re.IGNORECASE,
)
_END_RE = re.compile(
    r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
    re.IGNORECASE,
)


def strip_gutenberg_boilerplate(raw: str) -> str:
    """Cut Gutenberg headers and footers; leave the body intact."""
    m = _START_RE.search(raw)
    if m:
        raw = raw[m.end():]
    m = _END_RE.search(raw)
    if m:
        raw = raw[: m.start()]
    return raw.strip()


# Chunk-level non-body filter. We only reject obvious headings /
# tables of contents / colophons here; the proper quality gate is
# quality_filter_literature.
_NON_BODY_PATTERNS = (
    re.compile(r"^\s*chapter\s+[ivxlcdm0-9]+\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*contents\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"copyright", re.IGNORECASE),
    re.compile(r"all rights reserved", re.IGNORECASE),
    re.compile(r"printed in", re.IGNORECASE),
    re.compile(r"transcriber'?s note", re.IGNORECASE),
)


def looks_like_non_body(chunk: str) -> bool:
    """True if the chunk is dominated by headings/TOC/colophon lines."""
    head = chunk[:200]
    for pat in _NON_BODY_PATTERNS:
        if pat.search(head):
            # A real chapter body that starts with "Chapter I" is
            # still mostly body, so only reject if the chunk is small.
            if len(chunk.split()) < 250:
                return True
    # Page-of-roman-numerals TOC: many short lines, almost no
    # sentence-ending punctuation.
    lines = [ln for ln in chunk.splitlines() if ln.strip()]
    if lines and sum(1 for ln in lines if len(ln) < 60) / len(lines) > 0.85:
        if chunk.count(".") + chunk.count("?") + chunk.count("!") < 5:
            return True
    return False


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def _split_oversized_block(block: str, target_words: int, max_words: int) -> list[str]:
    """A single block bigger than max_words (e.g. one Book of Paradise
    Lost). Split on line breaks, then greedy-group lines to target."""
    lines = block.splitlines()
    out: list[str] = []
    buf: list[str] = []
    buf_words = 0
    for line in lines:
        lw = len(line.split())
        if buf_words + lw > max_words and buf:
            out.append("\n".join(buf).strip())
            buf = [line]
            buf_words = lw
            continue
        buf.append(line)
        buf_words += lw
        if buf_words >= target_words:
            out.append("\n".join(buf).strip())
            buf = []
            buf_words = 0
    if buf:
        tail = "\n".join(buf).strip()
        if tail:
            out.append(tail)
    return [c for c in out if c]


def _group_to_targets(blocks: list[str], target_words: int, max_words: int) -> list[str]:
    """Greedy paragraph-grouper. Oversized single blocks get split on
    line breaks so no record exceeds max_words."""
    chunks: list[str] = []
    buf: list[str] = []
    buf_words = 0
    for block in blocks:
        bw = len(block.split())
        if bw == 0:
            continue
        if bw > max_words:
            # Flush whatever we have, then split the giant block.
            if buf:
                chunks.append("\n\n".join(buf).strip())
                buf = []
                buf_words = 0
            chunks.extend(_split_oversized_block(block, target_words, max_words))
            continue
        if buf_words + bw > max_words and buf:
            chunks.append("\n\n".join(buf).strip())
            buf = [block]
            buf_words = bw
            continue
        buf.append(block)
        buf_words += bw
        if buf_words >= target_words:
            chunks.append("\n\n".join(buf).strip())
            buf = []
            buf_words = 0
    if buf:
        chunks.append("\n\n".join(buf).strip())
    return chunks


def chunk_prose(body: str) -> list[str]:
    """Narrative / essay chunks: ~1000 words each, hard cap 2000."""
    blocks = [b for b in re.split(r"\n\s*\n", body) if b.strip()]
    return _group_to_targets(blocks, target_words=1000, max_words=2000)


def chunk_poetry(body: str) -> list[str]:
    """Poetry chunks: ~300 words, hard cap 500. Prefer poem breaks
    (triple newline); fall back to stanza breaks (double newline)."""
    by_poem = [b for b in re.split(r"\n\s*\n\s*\n", body) if b.strip()]
    if len(by_poem) < 2:
        # Single-poem book (e.g. Paradise Lost): use stanza breaks.
        by_poem = [b for b in re.split(r"\n\s*\n", body) if b.strip()]
    return _group_to_targets(by_poem, target_words=300, max_words=500)


# ---------------------------------------------------------------------------
# Quality filter (verbatim from spec, kept self-contained here)
# ---------------------------------------------------------------------------

def quality_filter_literature(record: dict) -> bool:
    text = record["text"]
    if len(text.split()) < 200:
        return False
    suspicious = [
        "chapter i", "contents", "copyright",
        "all rights reserved", "printed in",
    ]
    if any(s in text.lower()[:100] for s in suspicious):
        return False
    return True


# ---------------------------------------------------------------------------
# Quality score
# ---------------------------------------------------------------------------

def quality_score(text: str) -> float:
    """Composite of length / sentence structure / vocabulary richness."""
    words = text.split()
    n = len(words)
    if n == 0:
        return 0.0

    score = 0.5

    # length band
    if 300 <= n <= 1500:
        score += 0.1
    elif n > 1500:
        score += 0.05

    # vocabulary richness (type-token ratio)
    unique_ratio = len(set(w.lower() for w in words)) / n
    score += unique_ratio * 0.2

    # sentence structure: terminal punctuation present
    if text.rstrip()[-1:] in ".!?":
        score += 0.05
    sentence_terms = text.count(".") + text.count("?") + text.count("!")
    if sentence_terms >= 5:
        score += 0.05

    # low-diversity penalty (lists, repeated words)
    if unique_ratio < 0.35:
        score -= 0.2

    return round(max(0.0, min(1.0, score)), 4)


# ---------------------------------------------------------------------------
# Token count — rough proxy. Real BPE happens in Agent 9.
# ---------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """~4 chars / token. Good enough for budgeting; Agent 9 retokenizes."""
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Record builder
# ---------------------------------------------------------------------------

def build_record(text: str, book: Book) -> dict:
    return {
        "text": text,
        "source": "gutenberg",
        "domain": "literature",
        "subdomain": book.subdomain,
        "quality_score": quality_score(text),
        "dialogue": False,
        "author": book.author,
        "title": book.title,
        "tokens": estimate_tokens(text),
        "language": "en",
    }


# ---------------------------------------------------------------------------
# Per-book collection
# ---------------------------------------------------------------------------

@dataclass
class BookStats:
    book: Book
    chunks_emitted: int
    chunks_rejected: int
    tokens_emitted: int
    fetched: bool


def collect_book(book: Book) -> tuple[list[dict], BookStats]:
    print(f"[fetch] {book.gid} — {book.author} — {book.title}")
    raw = fetch_book(book.gid)
    if raw is None:
        print(f"  FAILED to fetch {book.gid}", file=sys.stderr)
        return [], BookStats(book, 0, 0, 0, fetched=False)

    body = strip_gutenberg_boilerplate(raw)
    chunks = chunk_poetry(body) if book.subdomain == "poetry" else chunk_prose(body)

    records: list[dict] = []
    rejected = 0
    for chunk in chunks:
        if looks_like_non_body(chunk):
            rejected += 1
            continue
        rec = build_record(chunk, book)
        if not quality_filter_literature(rec):
            rejected += 1
            continue
        records.append(rec)

    tokens = sum(r["tokens"] for r in records)
    print(f"  → {len(records)} records kept, {rejected} rejected, ~{tokens:,} tokens")
    return records, BookStats(book, len(records), rejected, tokens, fetched=True)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "data" / "curriculum_v2" / "literature" / "raw"


def write_jsonl(records: Iterable[dict], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False))
            f.write("\n")
            n += 1
    return n


def run(books: Iterable[Book], out_path: Path, sleep_between: float = 0.5) -> None:
    all_records: list[dict] = []
    stats: list[BookStats] = []

    books_list = list(books)
    for i, book in enumerate(books_list):
        try:
            records, st = collect_book(book)
        except Exception as exc:  # noqa: BLE001
            print(f"  EXCEPTION on {book.gid}: {exc}", file=sys.stderr)
            stats.append(BookStats(book, 0, 0, 0, fetched=False))
            continue
        all_records.extend(records)
        stats.append(st)
        if i < len(books_list) - 1:
            time.sleep(sleep_between)

    n_written = write_jsonl(all_records, out_path)

    # Summary
    print()
    print("=" * 72)
    print(f"Wrote {n_written} records to {out_path}")
    total_tokens = sum(s.tokens_emitted for s in stats)
    total_rejected = sum(s.chunks_rejected for s in stats)
    print(f"Total ~tokens: {total_tokens:,}")
    print(f"Total chunks rejected: {total_rejected}")
    print()
    print("Per-book:")
    for s in stats:
        status = "OK " if s.fetched else "FAIL"
        print(
            f"  [{status}] gid={s.book.gid:<6} "
            f"sub={s.book.subdomain:<9} "
            f"kept={s.chunks_emitted:<4} "
            f"rejected={s.chunks_rejected:<3} "
            f"tokens~{s.tokens_emitted:<8,} "
            f"{s.book.author} — {s.book.title}"
        )
    failed = [s for s in stats if not s.fetched]
    if failed:
        print()
        print(f"FAILED FETCHES: {len(failed)}")
        for s in failed:
            print(f"  gid={s.book.gid}  {s.book.author} — {s.book.title}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--full",
        action="store_true",
        help="Run the full Agent 5 source list (not the 5-book validation set).",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Override output path. Defaults depend on --full.",
    )
    args = ap.parse_args()

    if args.full:
        books = FULL_BOOKS
        out = args.out or (OUTPUT_DIR / "agent_5_full.jsonl")
    else:
        books = VALIDATION_BOOKS
        out = args.out or (OUTPUT_DIR / "agent_5_validation.jsonl")

    run(books, out)


if __name__ == "__main__":
    main()
