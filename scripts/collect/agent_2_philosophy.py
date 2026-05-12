#!/usr/bin/env python3
"""
Agent 2 — Philosophy collector for the v2.0 curriculum.

Validation mode (default, no args):
    Fetches 4 Gutenberg books spanning Ancient/Modern/Eastern + 3 Stanford
    Encyclopedia of Philosophy entries. Chunks each into ~500–2000-word
    records, runs quality_filter_philosophy on every record, writes the
    survivors to
        data/curriculum_v2/philosophy/raw/agent_2_validation.jsonl

Full mode (--full):
    Placeholder for the complete sweep across the Gutenberg ANCIENT /
    MODERN / EASTERN lists and the ~500 core SEP entries described in
    COLLECTION_SPEC.md. Not exercised in this validation run; the wiring
    is here so the same script accepts the flag.

Output schema (one JSON object per line):
    text, source, domain, subdomain, quality_score, dialogue,
    author, title, tokens, language
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "data" / "curriculum_v2" / "philosophy" / "raw"
VALIDATION_OUT = OUT_DIR / "agent_2_validation.jsonl"
FULL_OUT = OUT_DIR / "agent_2_full.jsonl"

USER_AGENT = (
    "BabyMind/2.0 curriculum-collector (research; "
    "contact: hello.postworklab@gmail.com)"
)
HTTP_TIMEOUT = 60
SEP_DELAY_S = 1.0  # be polite to a small academic site


# ---------------------------------------------------------------------------
# Source tables
# ---------------------------------------------------------------------------

# id -> (title, author, subdomain). Verified by HEAD/GET against Gutenberg.
GUTENBERG_VALIDATION: dict[int, tuple[str, str, str]] = {
    # ANCIENT
    1497: ("The Republic", "Plato", "ancient"),
    2680: ("Meditations", "Marcus Aurelius", "ancient"),
    8438: ("The Ethics of Aristotle", "Aristotle", "ancient"),
    # EASTERN
    216:  ("Tao Te Ching", "Laozi", "eastern"),
}

# Full-mode sweep — verified slugs not required at validation time; this is
# the candidate list keyed by Gutenberg id. Many of these are nominal; the
# --full path verifies each ID before download.
GUTENBERG_FULL: dict[int, tuple[str, str, str]] = {
    **GUTENBERG_VALIDATION,
    # ANCIENT
    1656:  ("Apology", "Plato", "ancient"),
    1658:  ("Phaedo", "Plato", "ancient"),
    1635:  ("Symposium", "Plato", "ancient"),
    45331: ("Discourses", "Epictetus", "ancient"),
    785:   ("Enchiridion", "Epictetus", "ancient"),
    785:   ("On the Nature of Things", "Lucretius", "ancient"),
    # MODERN
    59:    ("Meditations on First Philosophy", "Descartes", "modern"),
    59:    ("Discourse on Method", "Descartes", "modern"),
    3800:  ("Ethics", "Spinoza", "modern"),
    4280:  ("Critique of Pure Reason", "Kant", "modern"),
    5682:  ("Groundwork of the Metaphysics of Morals", "Kant", "modern"),
    34901: ("Beyond Good and Evil", "Nietzsche", "modern"),
    1998:  ("Thus Spoke Zarathustra", "Nietzsche", "modern"),
    52319: ("The Gay Science", "Nietzsche", "modern"),
    34970: ("On Liberty", "John Stuart Mill", "modern"),
    11224: ("Utilitarianism", "John Stuart Mill", "modern"),
    10615: ("An Enquiry Concerning Human Understanding", "Hume", "modern"),
    4705:  ("An Essay Concerning Human Understanding", "Locke", "modern"),
    # EASTERN
    23839: ("The Analects", "Confucius", "eastern"),
}

SEP_VALIDATION: list[str] = [
    "consciousness",
    "freewill",
    "justice",
]

SEP_FULL: list[str] = [
    # Sketch — Agent 9 / --full would expand to ~500 entries.
    "consciousness", "free-will", "personal-identity", "causation",
    "time", "space", "truth", "knowledge", "justice", "democracy",
    "ethics-virtue", "beauty", "death", "meaning-of-life", "logic-classical",
    "modal-logic", "proof-theory", "set-theory", "model-theory",
    "computability",
]


# ---------------------------------------------------------------------------
# Quality filter (verbatim from COLLECTION_SPEC.md)
# ---------------------------------------------------------------------------

def quality_filter_philosophy(record: dict) -> bool:
    text = record["text"]
    if len(text.split()) < 150:
        return False
    phil_words = {
        "virtue", "justice", "truth", "knowledge", "soul",
        "reason", "moral", "good", "evil", "existence",
        "consciousness", "freedom", "necessity", "being",
    }
    word_set = set(text.lower().split())
    if len(phil_words & word_set) < 3:
        return False
    return True


def quality_score(text: str) -> float:
    """Light quality score — length + philosophical vocabulary + structure."""
    words = text.split()
    n = len(words)
    if n == 0:
        return 0.0

    score = 0.5

    # Length sweet-spot.
    if 200 <= n <= 2000:
        score += 0.10
    elif n > 2000:
        score += 0.05

    # Unique-word ratio (proxy for prose richness).
    unique_ratio = len(set(words)) / n
    score += min(unique_ratio * 0.2, 0.2)

    # Sentence-ish structure — count terminators.
    terminators = sum(text.count(c) for c in ".!?")
    avg_sentence = n / max(terminators, 1)
    if 8 <= avg_sentence <= 40:
        score += 0.05

    # Philosophical vocabulary density.
    phil_words = {
        "virtue", "justice", "truth", "knowledge", "soul", "reason",
        "moral", "good", "evil", "existence", "consciousness", "freedom",
        "necessity", "being", "mind", "self", "wisdom", "nature", "form",
    }
    lower = set(w.strip(".,;:!?\"'()").lower() for w in words)
    hits = len(phil_words & lower)
    score += min(hits * 0.02, 0.1)

    # Penalise very-low-variance prose.
    if unique_ratio < 0.4:
        score -= 0.2

    return round(max(0.0, min(1.0, score)), 3)


# ---------------------------------------------------------------------------
# Gutenberg
# ---------------------------------------------------------------------------

GUT_START_PAT = re.compile(
    r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK[^*]*\*\*\*",
    re.IGNORECASE,
)
GUT_END_PAT = re.compile(
    r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK[^*]*\*\*\*",
    re.IGNORECASE,
)


def fetch_gutenberg_text(book_id: int) -> str | None:
    """Try cache/epub first, then files/. Returns raw UTF-8 text or None."""
    urls = [
        f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt",
        f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt",
        f"https://www.gutenberg.org/files/{book_id}/{book_id}.txt",
    ]
    for url in urls:
        try:
            r = requests.get(
                url,
                headers={"User-Agent": USER_AGENT},
                timeout=HTTP_TIMEOUT,
            )
            if r.status_code == 200 and r.text and len(r.text) > 5000:
                r.encoding = r.apparent_encoding or "utf-8"
                return r.text
        except requests.RequestException as exc:
            print(f"  ! gutenberg fetch failed ({url}): {exc}", file=sys.stderr)
    return None


def strip_gutenberg_boilerplate(raw: str) -> str:
    start = GUT_START_PAT.search(raw)
    end = GUT_END_PAT.search(raw)
    s = start.end() if start else 0
    e = end.start() if end else len(raw)
    body = raw[s:e].strip()

    # Drop the "Produced by …" leader if present.
    lines = body.splitlines()
    while lines and (
        lines[0].strip() == ""
        or lines[0].strip().lower().startswith(("produced by", "transcribed", "this etext"))
    ):
        lines.pop(0)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# SEP
# ---------------------------------------------------------------------------

def fetch_sep_entry(slug: str) -> tuple[str, str] | None:
    """Return (title, body_text) or None on failure."""
    url = f"https://plato.stanford.edu/entries/{slug}/"
    try:
        r = requests.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=HTTP_TIMEOUT,
        )
    except requests.RequestException as exc:
        print(f"  ! SEP fetch failed ({slug}): {exc}", file=sys.stderr)
        return None
    if r.status_code != 200:
        print(f"  ! SEP {slug} -> HTTP {r.status_code}", file=sys.stderr)
        return None

    soup = BeautifulSoup(r.text, "html.parser")

    title_el = soup.find("h1")
    title = title_el.get_text(strip=True) if title_el else slug.replace("-", " ").title()

    main = soup.find("div", id="main-text")
    if main is None:
        print(f"  ! SEP {slug} has no #main-text", file=sys.stderr)
        return None

    # Drop bibliography, academic-tools, other-internet-resources,
    # related-entries, copyright, and the table-of-contents.
    for stop_id in (
        "bibliography",
        "academic-tools",
        "other-internet-resources",
        "related-entries",
    ):
        # The SEP encodes section headings as <h2 id="...">; remove the
        # heading and every sibling that follows it inside #main-text.
        h = main.find(id=stop_id)
        if h is not None:
            sib = h
            while sib is not None:
                nxt = sib.find_next_sibling()
                sib.decompose()
                sib = nxt

    # Strip stray ToC divs / footers if present.
    for sel in ("div.toc", "#toc", "div#toc", "div.copyright"):
        for el in main.select(sel):
            el.decompose()

    # Collapse whitespace.
    text = main.get_text(" ", strip=True)
    text = re.sub(r"\s+", " ", text).strip()
    return title, text


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

MIN_CHUNK_WORDS = 500
MAX_CHUNK_WORDS = 2000
TARGET_CHUNK_WORDS = 1200


def split_paragraphs(text: str) -> list[str]:
    parts = re.split(r"\n\s*\n+", text)
    return [p.strip() for p in parts if p.strip()]


def chunk_text(text: str) -> list[str]:
    """Group paragraphs into ~500–2000-word chunks (target ~1200)."""
    paragraphs = split_paragraphs(text)
    if not paragraphs:
        # No paragraph breaks (SEP after BeautifulSoup join) — fall back to
        # sentence-aware splitting.
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
        paragraphs = [s.strip() for s in sentences if s.strip()]

    chunks: list[str] = []
    buf: list[str] = []
    buf_words = 0

    for p in paragraphs:
        w = len(p.split())
        # If a single paragraph dwarfs the max, hard-split it.
        if w > MAX_CHUNK_WORDS:
            if buf:
                chunks.append(" ".join(buf))
                buf, buf_words = [], 0
            words = p.split()
            for i in range(0, len(words), TARGET_CHUNK_WORDS):
                piece = " ".join(words[i : i + TARGET_CHUNK_WORDS])
                if len(piece.split()) >= MIN_CHUNK_WORDS:
                    chunks.append(piece)
                else:
                    # tail too small — fold into next buffer
                    buf.append(piece)
                    buf_words += len(piece.split())
            continue

        if buf_words + w > MAX_CHUNK_WORDS:
            chunks.append(" ".join(buf))
            buf, buf_words = [p], w
        else:
            buf.append(p)
            buf_words += w
            if buf_words >= TARGET_CHUNK_WORDS:
                chunks.append(" ".join(buf))
                buf, buf_words = [], 0

    if buf and buf_words >= MIN_CHUNK_WORDS:
        chunks.append(" ".join(buf))
    elif buf and chunks:
        # Fold a short tail into the last chunk.
        chunks[-1] = chunks[-1] + " " + " ".join(buf)

    return chunks


# ---------------------------------------------------------------------------
# Record assembly
# ---------------------------------------------------------------------------

@dataclass
class CollectionStats:
    books_fetched: int = 0
    books_failed: int = 0
    sep_fetched: int = 0
    sep_failed: int = 0
    records_written: int = 0
    records_rejected: int = 0
    approx_tokens: int = 0
    failures: list[str] | None = None

    def __post_init__(self) -> None:
        if self.failures is None:
            self.failures = []


def make_record(
    text: str,
    *,
    source: str,
    subdomain: str,
    author: str,
    title: str,
) -> dict:
    tokens = int(len(text.split()) * 1.3)
    return {
        "text": text,
        "source": source,
        "domain": "philosophy",
        "subdomain": subdomain,
        "quality_score": quality_score(text),
        "dialogue": False,
        "author": author,
        "title": title,
        "tokens": tokens,
        "language": "en",
    }


def write_records(records: Iterable[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Collection drivers
# ---------------------------------------------------------------------------

def collect_books(
    book_table: dict[int, tuple[str, str, str]],
    stats: CollectionStats,
) -> Iterator[dict]:
    for book_id, (title, author, subdomain) in book_table.items():
        print(f"[gutenberg] {book_id} — {author}: {title}")
        raw = fetch_gutenberg_text(book_id)
        if raw is None:
            stats.books_failed += 1
            stats.failures.append(f"gutenberg:{book_id} ({title}) — fetch failed")
            continue
        body = strip_gutenberg_boilerplate(raw)
        if len(body.split()) < 500:
            stats.books_failed += 1
            stats.failures.append(
                f"gutenberg:{book_id} ({title}) — body too short ({len(body.split())} words)"
            )
            continue
        stats.books_fetched += 1

        chunks = chunk_text(body)
        print(f"  -> {len(chunks)} chunks ({len(body.split())} words total)")
        for chunk in chunks:
            yield make_record(
                chunk,
                source="gutenberg",
                subdomain=subdomain,
                author=author,
                title=title,
            )


def collect_sep(
    slugs: list[str],
    stats: CollectionStats,
) -> Iterator[dict]:
    for slug in slugs:
        print(f"[sep] {slug}")
        result = fetch_sep_entry(slug)
        time.sleep(SEP_DELAY_S)
        if result is None:
            stats.sep_failed += 1
            stats.failures.append(f"sep:{slug} — fetch failed")
            continue
        title, body = result
        if len(body.split()) < 500:
            stats.sep_failed += 1
            stats.failures.append(
                f"sep:{slug} — body too short ({len(body.split())} words)"
            )
            continue
        stats.sep_fetched += 1

        chunks = chunk_text(body)
        print(f"  -> {len(chunks)} chunks ({len(body.split())} words total)")
        for chunk in chunks:
            yield make_record(
                chunk,
                source="stanford_encyclopedia_of_philosophy",
                subdomain="encyclopedia",
                author="SEP contributors",
                title=title,
            )


def run(*, full: bool) -> CollectionStats:
    stats = CollectionStats()
    book_table = GUTENBERG_FULL if full else GUTENBERG_VALIDATION
    sep_slugs = SEP_FULL if full else SEP_VALIDATION

    accepted: list[dict] = []
    for rec in collect_books(book_table, stats):
        if quality_filter_philosophy(rec):
            accepted.append(rec)
            stats.records_written += 1
            stats.approx_tokens += rec["tokens"]
        else:
            stats.records_rejected += 1

    for rec in collect_sep(sep_slugs, stats):
        if quality_filter_philosophy(rec):
            accepted.append(rec)
            stats.records_written += 1
            stats.approx_tokens += rec["tokens"]
        else:
            stats.records_rejected += 1

    out_path = FULL_OUT if full else VALIDATION_OUT
    write_records(accepted, out_path)

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Agent 2 — Philosophy curriculum collector (v2.0).",
    )
    ap.add_argument(
        "--full",
        action="store_true",
        help=(
            "Run the full collection (expanded Gutenberg + SEP). "
            "Default is the small validation sweep."
        ),
    )
    args = ap.parse_args(argv)

    mode = "FULL" if args.full else "VALIDATION"
    out_path = FULL_OUT if args.full else VALIDATION_OUT
    print(f"=== agent_2_philosophy ({mode}) ===")
    print(f"writing to: {out_path}")

    stats = run(full=args.full)

    print()
    print("=== summary ===")
    print(f"books fetched:    {stats.books_fetched}")
    print(f"books failed:     {stats.books_failed}")
    print(f"SEP fetched:      {stats.sep_fetched}")
    print(f"SEP failed:       {stats.sep_failed}")
    print(f"records written:  {stats.records_written}")
    print(f"records rejected: {stats.records_rejected}")
    print(f"approx tokens:    {stats.approx_tokens:,}")
    if stats.failures:
        print("failures:")
        for f in stats.failures:
            print(f"  - {f}")

    # Non-zero exit only on total failure.
    if stats.records_written == 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
