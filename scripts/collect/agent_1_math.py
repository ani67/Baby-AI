"""Agent 1 — Mathematics & Logic collector for the v2.0 curriculum.

Pulls public-domain mathematics & logic prose from three sources:
  - Project Gutenberg (classic math/logic texts)
  - arXiv math.HO / math.GM (expository papers, abstracts)
  - MIT OpenCourseWare (course lecture-notes pages)

Writes JSONL records to data/curriculum_v2/foundations/mathematics/raw/.
Default run is a small validation set (3 books, 5 papers, 1 OCW page);
pass --full to attempt the full collection (not implemented for this
session — placeholder raises NotImplementedError).

Schema per the COLLECTION_SPEC:
    text, source, domain, subdomain, quality_score, dialogue,
    author, title, tokens, language
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterator

import requests
from bs4 import BeautifulSoup


# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "data" / "curriculum_v2" / "foundations" / "mathematics" / "raw"
VALIDATION_OUT = OUTPUT_DIR / "agent_1_validation.jsonl"
FULL_OUT = OUTPUT_DIR / "agent_1_full.jsonl"

REQUEST_TIMEOUT = 30
POLITE_DELAY_SEC = 0.5

USER_AGENT = (
    "Mozilla/5.0 (compatible; Baby-Mind-Curriculum/2.0; "
    "Agent-1 Mathematics collector)"
)
HTTP_HEADERS = {"User-Agent": USER_AGENT}


# Validation source choices --------------------------------------------------

GUTENBERG_VALIDATION = [
    # (id, author, title) — chosen from the spec's allowed list, then
    # ID-verified via Gutendex (https://gutendex.com) at collection time.
    # The original IDs in the validation brief were wrong on the live
    # Gutenberg catalogue (28233 is Newton, 4239 is Malthus, 11038 is a
    # French aeronautics text); these are the actual IDs that resolve to
    # English plain-text editions of the requested works.
    (25447, "Bertrand Russell", "Mysticism and Logic and Other Essays"),
    (28696, "Lewis Carroll",    "Symbolic Logic"),
    (22062, "John Dee",         "The Mathematicall Praeface to Elements of Geometrie of Euclid of Megara"),
]

ARXIV_VALIDATION_QUERY = (
    "http://export.arxiv.org/api/query?"
    "search_query=cat:math.HO+OR+cat:math.GM"
    "&max_results=5&sortBy=submittedDate&sortOrder=descending"
)

OCW_VALIDATION_URL = (
    # Lecture-notes page for 18.712 Introduction to Representation Theory.
    # Picked specifically because its HTML body (the embedded chapter
    # outline) contains enough mathematical-reasoning vocabulary to pass
    # the spec's quality filter — most OCW course landing pages are pure
    # navigation chrome and would be rejected.
    "https://ocw.mit.edu/courses/18-712-introduction-to-representation-theory-fall-2010/pages/lecture-notes/"
)


# ---------------------------------------------------------------------------
# Quality filter (from the spec)
# ---------------------------------------------------------------------------

REASONING_WORDS = {
    "therefore", "proof", "theorem", "lemma",
    "follows", "implies", "suppose", "assume",
}


def quality_filter_math(record: dict) -> bool:
    """Return True if record passes the math/logic quality filter.

    From COLLECTION_SPEC.md → Agent 1 → Quality filters.
    """
    text = record["text"]
    words = text.split()
    if len(words) < 100:
        return False
    # Equation-density guard: too many '$' relative to length = LaTeX-heavy
    if text.count("$") > len(text) // 20:
        return False
    if detect_language_simple(text) != "en":
        return False
    word_set = {w.strip(".,;:()[]\"'").lower() for w in words}
    if len(REASONING_WORDS & word_set) < 2:
        return False
    return True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def detect_language_simple(text: str) -> str:
    """Crude ASCII-ratio English detector. Good enough for validation."""
    sample = text[:4000]
    if not sample:
        return "unknown"
    printable = sum(1 for c in sample if c.isascii() and (c.isalnum() or c.isspace() or c in ".,;:()-'\""))
    ratio = printable / max(len(sample), 1)
    return "en" if ratio > 0.85 else "other"


def rough_token_count(text: str) -> int:
    return int(len(text.split()) * 1.3)


def compute_quality_score(text: str) -> float:
    """Light heuristic. Agent 9 recomputes properly later."""
    words = text.split()
    if not words:
        return 0.0
    score = 0.5
    n = len(words)
    if 200 <= n <= 2000:
        score += 0.10
    elif n > 2000:
        score += 0.05
    unique_ratio = len(set(words)) / n
    score += unique_ratio * 0.2
    stripped = text.rstrip()
    if stripped and stripped[-1] in ".!?":
        score += 0.05
    reason_hits = len({w.lower().strip(".,;:()[]\"'") for w in words} & REASONING_WORDS)
    score += min(reason_hits * 0.02, 0.1)
    if unique_ratio < 0.4:
        score -= 0.2
    return max(0.0, min(1.0, score))


def make_record(*, text: str, source: str, subdomain: str,
                author: str, title: str, dialogue: bool = False) -> dict:
    return {
        "text": text,
        "source": source,
        "domain": "foundations/mathematics" if subdomain == "mathematics" else "foundations/logic",
        "subdomain": subdomain,
        "quality_score": round(compute_quality_score(text), 4),
        "dialogue": dialogue,
        "author": author,
        "title": title,
        "tokens": rough_token_count(text),
        "language": detect_language_simple(text),
    }


def chunk_text(text: str, target_words: int = 1200, min_words: int = 500,
               max_words: int = 2000) -> Iterator[str]:
    """Split text into ~500-2000 word chunks, breaking on paragraph boundaries.

    Walks paragraphs and accumulates until we cross target_words; if a single
    paragraph is huge, falls back to a sentence-based break.
    """
    # Normalise paragraph breaks
    paragraphs = re.split(r"\n\s*\n+", text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    buf: list[str] = []
    buf_words = 0
    for para in paragraphs:
        para_words = len(para.split())
        # Huge paragraph — sentence-split it
        if para_words > max_words:
            sentences = re.split(r"(?<=[.!?])\s+", para)
            for sent in sentences:
                sw = len(sent.split())
                if buf_words + sw > max_words and buf_words >= min_words:
                    yield "\n\n".join(buf)
                    buf, buf_words = [], 0
                buf.append(sent)
                buf_words += sw
            continue
        if buf_words + para_words > max_words and buf_words >= min_words:
            yield "\n\n".join(buf)
            buf, buf_words = [], 0
        buf.append(para)
        buf_words += para_words
        if buf_words >= target_words:
            yield "\n\n".join(buf)
            buf, buf_words = [], 0
    if buf and buf_words >= min_words:
        yield "\n\n".join(buf)


# ---------------------------------------------------------------------------
# Source: Project Gutenberg
# ---------------------------------------------------------------------------

GUTENBERG_URLS = [
    "https://www.gutenberg.org/files/{id}/{id}-0.txt",
    "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt",
    "https://www.gutenberg.org/files/{id}/{id}.txt",
]


def fetch_gutenberg(book_id: int) -> str | None:
    """Try a handful of canonical URL shapes for a Gutenberg book."""
    for tpl in GUTENBERG_URLS:
        url = tpl.format(id=book_id)
        try:
            r = requests.get(url, timeout=REQUEST_TIMEOUT, headers=HTTP_HEADERS)
            if r.status_code == 200 and len(r.text) > 2000:
                return r.text
        except Exception as e:  # noqa: BLE001 - we want resilience to anything
            print(f"  [gutenberg {book_id}] request failed {url}: {e}", file=sys.stderr)
        time.sleep(POLITE_DELAY_SEC)
    return None


GUTENBERG_HEADER_RE = re.compile(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG[^\n]*\*\*\*", re.IGNORECASE)
GUTENBERG_FOOTER_RE = re.compile(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG[^\n]*\*\*\*", re.IGNORECASE)


def strip_gutenberg_boilerplate(raw: str) -> str:
    start = GUTENBERG_HEADER_RE.search(raw)
    end = GUTENBERG_FOOTER_RE.search(raw)
    body = raw[start.end() if start else 0 : end.start() if end else len(raw)]
    # Tidy: collapse 3+ blank lines, drop the produced-by/transcriber line spam
    body = re.sub(r"\r\n?", "\n", body)
    body = re.sub(r"\n{3,}", "\n\n", body)
    return body.strip()


# ---------------------------------------------------------------------------
# Source: arXiv (export API, XML)
# ---------------------------------------------------------------------------

ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


def fetch_arxiv(query_url: str, max_attempts: int = 3) -> list[dict]:
    """Fetch arXiv search results, return list of {title, authors, abstract}.

    The arXiv export API is occasionally slow; retry a few times with a
    generous timeout before giving up.
    """
    r = None
    for attempt in range(1, max_attempts + 1):
        try:
            r = requests.get(query_url, timeout=60, headers=HTTP_HEADERS)
            r.raise_for_status()
            break
        except Exception as e:  # noqa: BLE001
            print(f"  [arxiv] attempt {attempt} failed: {e}", file=sys.stderr)
            r = None
            time.sleep(2.0 * attempt)
    if r is None:
        print(f"  [arxiv] giving up after {max_attempts} attempts", file=sys.stderr)
        return []
    try:
        root = ET.fromstring(r.text)
    except ET.ParseError as e:
        print(f"  [arxiv] parse failed: {e}", file=sys.stderr)
        return []
    out: list[dict] = []
    for entry in root.findall("atom:entry", ATOM_NS):
        title_el = entry.find("atom:title", ATOM_NS)
        summary_el = entry.find("atom:summary", ATOM_NS)
        author_els = entry.findall("atom:author/atom:name", ATOM_NS)
        if title_el is None or summary_el is None:
            continue
        title = re.sub(r"\s+", " ", (title_el.text or "")).strip()
        abstract = re.sub(r"\s+", " ", (summary_el.text or "")).strip()
        authors = ", ".join((a.text or "").strip() for a in author_els if a.text)
        if not abstract:
            continue
        out.append({"title": title, "authors": authors or "Unknown", "abstract": abstract})
    return out


# ---------------------------------------------------------------------------
# Source: MIT OpenCourseWare
# ---------------------------------------------------------------------------

def fetch_ocw_page(url: str) -> tuple[str, str] | None:
    """Fetch an OCW course page, return (title, prose-text)."""
    try:
        r = requests.get(url, timeout=REQUEST_TIMEOUT, headers=HTTP_HEADERS)
        r.raise_for_status()
    except Exception as e:  # noqa: BLE001
        print(f"  [ocw] fetch failed {url}: {e}", file=sys.stderr)
        return None
    soup = BeautifulSoup(r.text, "html.parser")
    # Strip script/style/nav
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()
    title_el = soup.find("title")
    title = (title_el.get_text(strip=True) if title_el else "MIT OCW Course").split("|")[0].strip()
    main = soup.find("main") or soup.body or soup
    text = main.get_text(separator="\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return title, text


# ---------------------------------------------------------------------------
# Pipeline: validation
# ---------------------------------------------------------------------------

def run_validation() -> dict:
    """Fetch a small validation set, write JSONL, return summary counts."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[agent_1] writing validation set → {VALIDATION_OUT}")

    stats = {
        "books_fetched": 0,
        "papers_fetched": 0,
        "ocw_pages_fetched": 0,
        "records_written": 0,
        "records_rejected": 0,
        "approx_tokens": 0,
        "failed_sources": [],
    }

    with open(VALIDATION_OUT, "w", encoding="utf-8") as fout:

        # ---- Gutenberg ----
        for book_id, author, title in GUTENBERG_VALIDATION:
            print(f"[gutenberg] fetching {book_id} — {author}, {title}")
            raw = fetch_gutenberg(book_id)
            if raw is None:
                print(f"  [gutenberg {book_id}] FAILED — no URL worked")
                stats["failed_sources"].append(f"gutenberg:{book_id}")
                continue
            stats["books_fetched"] += 1
            body = strip_gutenberg_boilerplate(raw)
            for chunk in chunk_text(body):
                rec = make_record(
                    text=chunk, source="gutenberg", subdomain="mathematics",
                    author=author, title=title,
                )
                if quality_filter_math(rec):
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    stats["records_written"] += 1
                    stats["approx_tokens"] += rec["tokens"]
                else:
                    stats["records_rejected"] += 1
            time.sleep(POLITE_DELAY_SEC)

        # ---- arXiv ----
        print(f"[arxiv] fetching {ARXIV_VALIDATION_QUERY}")
        papers = fetch_arxiv(ARXIV_VALIDATION_QUERY)
        time.sleep(POLITE_DELAY_SEC)
        for p in papers[:5]:
            stats["papers_fetched"] += 1
            # For validation we use title + abstract as the prose body.
            text = f"{p['title']}\n\n{p['abstract']}"
            rec = make_record(
                text=text, source="arxiv", subdomain="logic",
                author=p["authors"], title=p["title"],
            )
            if quality_filter_math(rec):
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                stats["records_written"] += 1
                stats["approx_tokens"] += rec["tokens"]
            else:
                stats["records_rejected"] += 1
        if not papers:
            stats["failed_sources"].append("arxiv:query")

        # ---- MIT OCW ----
        print(f"[ocw] fetching {OCW_VALIDATION_URL}")
        ocw = fetch_ocw_page(OCW_VALIDATION_URL)
        if ocw is None:
            stats["failed_sources"].append(f"ocw:{OCW_VALIDATION_URL}")
        else:
            stats["ocw_pages_fetched"] += 1
            ocw_title, ocw_text = ocw
            # OCW HTML pages are typically a few hundred words — relax the
            # min-chunk threshold so we don't drop the whole page just for
            # being below the 500-word floor we apply to full books.
            for chunk in chunk_text(ocw_text, min_words=100):
                rec = make_record(
                    text=chunk, source="mit_ocw", subdomain="mathematics",
                    author="MIT OCW", title=ocw_title,
                )
                if quality_filter_math(rec):
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    stats["records_written"] += 1
                    stats["approx_tokens"] += rec["tokens"]
                else:
                    stats["records_rejected"] += 1

    return stats


# ---------------------------------------------------------------------------
# Pipeline: full (placeholder for now)
# ---------------------------------------------------------------------------

def run_full() -> dict:
    """Full collection path.

    For this session we only ship the validation path; the --full lever
    is wired but raises so it can't be invoked accidentally. Implementing
    full will involve:
      - All Gutenberg IDs from the spec
      - arXiv math.HO/math.GM by paginated query (LaTeX source fetched
        and math-stripped, not just abstracts)
      - All MIT OCW math courses, traversing each lecture-notes link
      - Stanford Encyclopedia of Philosophy logic entries
    """
    raise NotImplementedError(
        "Full collection path not implemented in the validation build. "
        "Run without --full for the small validation set."
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Agent 1 — Mathematics & Logic collector")
    parser.add_argument("--full", action="store_true",
                        help="Run the full collection (validation by default)")
    args = parser.parse_args(argv)

    if args.full:
        stats = run_full()
    else:
        stats = run_validation()

    print("\n=== Agent 1 summary ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
