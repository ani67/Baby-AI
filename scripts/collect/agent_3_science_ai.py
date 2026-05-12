#!/usr/bin/env python3
"""
Agent 3 — Science & AI collector for the v2.0 curriculum.

Validation mode (default, no args):
    Fetches a small validation set across three sources:
      1. Project Gutenberg — 2-3 classical science texts (Darwin, Einstein,
         Faraday). Each Gutenberg ID is verified live against the gutendex
         catalog before fetch; mismatched IDs are skipped.
      2. arXiv — 5 foundational AI/ML paper abstracts (Attention/BERT/
         Scaling Laws/VAE/GAN) via the export API. Per-id queries with
         exponential backoff on 429 — accept 0 with a warning if arXiv
         rate-limits the whole session.
      3. MIT OCW — 6.034 AI lecture-notes page. Body must contain >300
         words of real prose after stripping nav/chrome; otherwise skip.

    Chunks each long source into 500-2000-word records, runs
    quality_filter_science() from the spec on every record, writes the
    survivors to:
        data/curriculum_v2/{science|ai}/raw/agent_3_validation.jsonl

    Note: classical science records get domain="science", arXiv ML
    abstracts get domain="ai", per the spec.

Full mode (--full):
    Sketch only — a function whose body comments enumerate what the full
    sweep would do (all Gutenberg science texts, paginated arXiv survey
    queries on cs.LG/cs.AI, every MIT OCW science+AI course, Nature/
    Science open-access articles). Implementation deferred.

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
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterator

import requests
from bs4 import BeautifulSoup


# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
SCIENCE_OUT_DIR = REPO_ROOT / "data" / "curriculum_v2" / "science" / "raw"
AI_OUT_DIR = REPO_ROOT / "data" / "curriculum_v2" / "ai" / "raw"
SCIENCE_VALIDATION_OUT = SCIENCE_OUT_DIR / "agent_3_validation.jsonl"
AI_VALIDATION_OUT = AI_OUT_DIR / "agent_3_validation.jsonl"

USER_AGENT = (
    "BabyMind/2.0 curriculum-collector (research; "
    "contact: hello.postworklab@gmail.com)"
)
HTTP_HEADERS = {"User-Agent": USER_AGENT}
HTTP_TIMEOUT = 60

GUTENBERG_DELAY_S = 0.5
ARXIV_DELAY_S = 3.0
ARXIV_MAX_ATTEMPTS = 3


# ---------------------------------------------------------------------------
# Validation source tables
# ---------------------------------------------------------------------------

# Gutenberg IDs verified live against gutendex.com on 2026-05-12:
#   1228 → Darwin, "On the Origin of Species" (en, text/plain available)
#   30155 → Einstein, "Relativity: The Special and General Theory" (en, txt)
#   14986 → Faraday, "Experimental Researches in Electricity, Vol 1" (en, txt)
# Any ID that fails verification at runtime is skipped with a warning;
# no silent substitution.
GUTENBERG_VALIDATION: list[tuple[int, str, str]] = [
    (1228,  "Charles Darwin",  "On the Origin of Species"),
    (30155, "Albert Einstein", "Relativity: The Special and General Theory"),
    (14986, "Michael Faraday", "Experimental Researches in Electricity, Volume 1"),
]

# Five mandatory papers from the spec's "Agent 3 → arXiv MANDATORY" list.
# Per the wave-1 lesson, we query one ID at a time (multi-id queries 429 fast)
# and use the abstract endpoint directly. Each tuple = (arxiv_id, expected
# short-title — used only to log which paper is missing if the title in the
# response looks wrong; never to override the response).
ARXIV_VALIDATION: list[tuple[str, str]] = [
    ("1706.03762", "Attention Is All You Need"),
    ("1810.04805", "BERT"),
    ("2001.08361", "Scaling Laws"),
    ("1312.6114", "VAE"),
    ("1406.2661", "GAN"),
]

# MIT OCW page — 6.034 AI course root. Wave-1 lesson: OCW course landing
# pages are mostly navigation chrome; we require >300 words of real prose
# after stripping before we keep anything. The pages/lecture-notes/ subpath
# infinite-redirects on the current OCW deployment, so we hit the course
# root instead (which the brief lists as the canonical URL).
OCW_VALIDATION_URL = (
    "https://ocw.mit.edu/courses/6-034-artificial-intelligence-fall-2010/"
)
OCW_PROSE_MIN_WORDS = 300


# ---------------------------------------------------------------------------
# Quality filter (verbatim from COLLECTION_SPEC.md → Agent 3)
# ---------------------------------------------------------------------------

PROSE_MARKERS = (
    "because", "therefore", "suggests", "implies",
    "shows", "demonstrates", "indicates",
)


def quality_filter_science(record: dict) -> bool:
    """Return True if record passes the science/AI quality filter.

    From COLLECTION_SPEC.md:
        - >= 200 words
        - reject pure wet-lab protocol pages (heavy 'ml '/'μl ' usage)
        - must contain at least one prose marker word
    """
    text = record["text"]
    if len(text.split()) < 200:
        return False
    lower = text.lower()
    if lower.count("ml ") + lower.count("μl ") > 10:
        return False
    return any(w in lower for w in PROSE_MARKERS)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def detect_language_simple(text: str) -> str:
    """Crude ASCII-ratio English detector — good enough for validation."""
    sample = text[:4000]
    if not sample:
        return "unknown"
    keep_chars = set(".,;:()-'\"")
    ok = sum(1 for c in sample
             if c.isascii() and (c.isalnum() or c.isspace() or c in keep_chars))
    ratio = ok / max(len(sample), 1)
    return "en" if ratio > 0.85 else "other"


def rough_token_count(text: str) -> int:
    # ~1.3 tokens per whitespace-split word for BPE-ish vocab.
    return int(len(text.split()) * 1.3)


REASONING_WORDS = {
    "therefore", "because", "however", "although", "implies", "suggests",
    "shows", "demonstrates", "furthermore", "nevertheless", "consequently",
}


def compute_quality_score(text: str) -> float:
    """Light heuristic. Agent 9 will recompute properly during pipeline."""
    words = text.split()
    if not words:
        return 0.0
    n = len(words)
    score = 0.5
    if 200 <= n <= 2000:
        score += 0.10
    elif n > 2000:
        score += 0.05
    unique_ratio = len(set(words)) / n
    score += unique_ratio * 0.2
    stripped = text.rstrip()
    if stripped and stripped[-1] in ".!?":
        score += 0.05
    word_set = {w.lower().strip(".,;:()[]\"'") for w in words}
    reason_hits = len(word_set & REASONING_WORDS)
    score += min(reason_hits * 0.02, 0.1)
    if unique_ratio < 0.4:
        score -= 0.2
    return max(0.0, min(1.0, score))


def make_record(*, text: str, source: str, domain: str, subdomain: str,
                author: str, title: str, dialogue: bool = False) -> dict:
    return {
        "text": text,
        "source": source,
        "domain": domain,
        "subdomain": subdomain,
        "quality_score": round(compute_quality_score(text), 4),
        "dialogue": dialogue,
        "author": author,
        "title": title,
        "tokens": rough_token_count(text),
        "language": detect_language_simple(text),
    }


def chunk_text(text: str, target_words: int = 1200,
               min_words: int = 500, max_words: int = 2000) -> Iterator[str]:
    """Break text into ~500-2000 word records along paragraph boundaries."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    buf: list[str] = []
    buf_words = 0
    for para in paragraphs:
        pw = len(para.split())
        if pw > max_words:
            sentences = re.split(r"(?<=[.!?])\s+", para)
            for sent in sentences:
                sw = len(sent.split())
                if buf_words + sw > max_words and buf_words >= min_words:
                    yield "\n\n".join(buf)
                    buf, buf_words = [], 0
                buf.append(sent)
                buf_words += sw
            continue
        if buf_words + pw > max_words and buf_words >= min_words:
            yield "\n\n".join(buf)
            buf, buf_words = [], 0
        buf.append(para)
        buf_words += pw
        if buf_words >= target_words:
            yield "\n\n".join(buf)
            buf, buf_words = [], 0
    if buf and buf_words >= min_words:
        yield "\n\n".join(buf)


# ---------------------------------------------------------------------------
# Source: Project Gutenberg (with gutendex verification)
# ---------------------------------------------------------------------------

GUTENBERG_URL_PATTERNS = [
    "https://www.gutenberg.org/files/{id}/{id}-0.txt",
    "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt",
    "https://www.gutenberg.org/files/{id}/{id}.txt",
]

GUTENBERG_HEADER_RE = re.compile(
    r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG[^\n]*\*\*\*",
    re.IGNORECASE,
)
GUTENBERG_FOOTER_RE = re.compile(
    r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG[^\n]*\*\*\*",
    re.IGNORECASE,
)


def verify_gutenberg_id(book_id: int, expected_author: str,
                        expected_title: str) -> tuple[bool, str]:
    """Hit gutendex to confirm the ID still points to the expected book.

    Returns (ok, message). Match is loose — we just require the expected
    author surname and the first content word of the expected title both
    appear (case-insensitive) in the catalog response.
    """
    url = f"https://gutendex.com/books/{book_id}"
    try:
        r = requests.get(url, timeout=20, headers=HTTP_HEADERS)
        if r.status_code != 200:
            return False, f"gutendex HTTP {r.status_code}"
        data = r.json()
    except Exception as e:  # noqa: BLE001
        return False, f"gutendex error: {e}"
    actual_title = (data.get("title") or "").lower()
    actual_authors = ", ".join(
        (a.get("name") or "") for a in (data.get("authors") or [])
    ).lower()
    # Surname check: take the last whitespace-token of expected_author
    surname = expected_author.lower().split()[-1] if expected_author else ""
    title_keyword = next(
        (w for w in expected_title.lower().split() if len(w) > 3),
        expected_title.lower(),
    )
    if surname and surname not in actual_authors:
        return False, (
            f"author mismatch: expected '{expected_author}', "
            f"catalog says '{actual_authors}'"
        )
    if title_keyword and title_keyword not in actual_title:
        return False, (
            f"title mismatch: expected '{expected_title}', "
            f"catalog says '{actual_title}'"
        )
    return True, f"verified: {actual_title} | {actual_authors}"


def fetch_gutenberg(book_id: int) -> str | None:
    """Try canonical URL shapes for a Gutenberg book's plain-text edition."""
    for tpl in GUTENBERG_URL_PATTERNS:
        url = tpl.format(id=book_id)
        try:
            r = requests.get(url, timeout=HTTP_TIMEOUT, headers=HTTP_HEADERS)
            if r.status_code == 200 and len(r.text) > 2000:
                return r.text
        except Exception as e:  # noqa: BLE001
            print(f"  [gutenberg {book_id}] {url} failed: {e}", file=sys.stderr)
        time.sleep(GUTENBERG_DELAY_S)
    return None


def strip_gutenberg_boilerplate(raw: str) -> str:
    start = GUTENBERG_HEADER_RE.search(raw)
    end = GUTENBERG_FOOTER_RE.search(raw)
    body = raw[start.end() if start else 0: end.start() if end else len(raw)]
    body = re.sub(r"\r\n?", "\n", body)
    body = re.sub(r"\n{3,}", "\n\n", body)
    return body.strip()


# ---------------------------------------------------------------------------
# Source: arXiv (export API, Atom XML, per-id with exponential backoff)
# ---------------------------------------------------------------------------

ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


def fetch_arxiv_single(arxiv_id: str) -> dict | None:
    """Fetch one arXiv paper's metadata. Returns {title,authors,abstract}.

    Per the wave-1 lesson, arXiv 429s aggressively. We try up to
    ARXIV_MAX_ATTEMPTS with exponential backoff (3s → 9s → 27s); if it
    still 429s, we give up on this ID and return None — the caller logs
    a warning and continues.
    """
    url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
    backoff = 3.0
    for attempt in range(1, ARXIV_MAX_ATTEMPTS + 1):
        try:
            r = requests.get(url, timeout=HTTP_TIMEOUT, headers=HTTP_HEADERS)
        except Exception as e:  # noqa: BLE001
            print(f"  [arxiv {arxiv_id}] attempt {attempt} request failed: {e}",
                  file=sys.stderr)
            time.sleep(backoff)
            backoff *= 3.0
            continue
        if r.status_code == 429:
            print(f"  [arxiv {arxiv_id}] attempt {attempt} got 429; "
                  f"sleeping {backoff}s", file=sys.stderr)
            time.sleep(backoff)
            backoff *= 3.0
            continue
        if r.status_code != 200:
            print(f"  [arxiv {arxiv_id}] HTTP {r.status_code}",
                  file=sys.stderr)
            return None
        try:
            root = ET.fromstring(r.text)
        except ET.ParseError as e:
            print(f"  [arxiv {arxiv_id}] parse error: {e}", file=sys.stderr)
            return None
        entry = root.find("atom:entry", ATOM_NS)
        if entry is None:
            print(f"  [arxiv {arxiv_id}] no <entry> in response",
                  file=sys.stderr)
            return None
        title_el = entry.find("atom:title", ATOM_NS)
        summary_el = entry.find("atom:summary", ATOM_NS)
        if title_el is None or summary_el is None:
            return None
        title = re.sub(r"\s+", " ", (title_el.text or "")).strip()
        abstract = re.sub(r"\s+", " ", (summary_el.text or "")).strip()
        authors = ", ".join(
            (a.text or "").strip()
            for a in entry.findall("atom:author/atom:name", ATOM_NS)
            if a.text
        )
        if not abstract:
            return None
        return {
            "title": title,
            "authors": authors or "Unknown",
            "abstract": abstract,
        }
    print(f"  [arxiv {arxiv_id}] giving up after "
          f"{ARXIV_MAX_ATTEMPTS} attempts (rate-limited)", file=sys.stderr)
    return None


# ---------------------------------------------------------------------------
# Source: MIT OCW
# ---------------------------------------------------------------------------

def fetch_ocw_page(url: str) -> tuple[str, str] | None:
    """Fetch one OCW page; return (title, prose_text) or None on failure.

    We cap redirects at 5 — some OCW subpaths (e.g. pages/lecture-notes/)
    have an infinite-redirect bug on the live site, which would otherwise
    raise 'Exceeded 30 redirects' instead of returning cleanly.
    """
    session = requests.Session()
    session.max_redirects = 5
    try:
        r = session.get(url, timeout=HTTP_TIMEOUT, headers=HTTP_HEADERS)
        r.raise_for_status()
    except Exception as e:  # noqa: BLE001
        print(f"  [ocw] fetch failed {url}: {e}", file=sys.stderr)
        return None
    soup = BeautifulSoup(r.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header",
                     "noscript", "form", "button"]):
        tag.decompose()
    title_el = soup.find("title")
    title = (title_el.get_text(strip=True) if title_el
             else "MIT OCW Course").split("|")[0].strip()
    main = soup.find("main") or soup.body or soup
    text = main.get_text(separator="\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return title, text


# ---------------------------------------------------------------------------
# Validation pipeline
# ---------------------------------------------------------------------------

def run_validation() -> dict:
    """Run the validation set; write JSONL; return summary counts."""
    SCIENCE_OUT_DIR.mkdir(parents=True, exist_ok=True)
    AI_OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[agent_3] science → {SCIENCE_VALIDATION_OUT}")
    print(f"[agent_3] ai      → {AI_VALIDATION_OUT}")

    stats: dict = {
        "books_verified": 0,
        "books_fetched": 0,
        "books_failed_verification": [],
        "arxiv_papers_fetched": 0,
        "arxiv_papers_failed": [],
        "ocw_pages_fetched": 0,
        "ocw_pages_skipped": [],
        "science_records_written": 0,
        "ai_records_written": 0,
        "records_rejected": 0,
        "approx_tokens": 0,
        "failed_sources": [],
    }

    science_fout = open(SCIENCE_VALIDATION_OUT, "w", encoding="utf-8")
    ai_fout = open(AI_VALIDATION_OUT, "w", encoding="utf-8")
    try:
        # ---- Project Gutenberg (classical science) ----
        for book_id, author, title in GUTENBERG_VALIDATION:
            print(f"[gutenberg] verifying {book_id} — {author}, {title}")
            ok, msg = verify_gutenberg_id(book_id, author, title)
            time.sleep(GUTENBERG_DELAY_S)
            if not ok:
                print(f"  [gutenberg {book_id}] verification FAILED — {msg}")
                stats["books_failed_verification"].append(
                    f"{book_id}:{msg}"
                )
                stats["failed_sources"].append(f"gutenberg:{book_id}")
                continue
            stats["books_verified"] += 1
            print(f"  [gutenberg {book_id}] OK — {msg}")
            raw = fetch_gutenberg(book_id)
            if raw is None:
                print(f"  [gutenberg {book_id}] download FAILED")
                stats["failed_sources"].append(
                    f"gutenberg:{book_id}:download"
                )
                continue
            stats["books_fetched"] += 1
            body = strip_gutenberg_boilerplate(raw)
            for chunk in chunk_text(body):
                rec = make_record(
                    text=chunk, source="gutenberg",
                    domain="science", subdomain="classical_science",
                    author=author, title=title,
                )
                if quality_filter_science(rec):
                    science_fout.write(
                        json.dumps(rec, ensure_ascii=False) + "\n"
                    )
                    stats["science_records_written"] += 1
                    stats["approx_tokens"] += rec["tokens"]
                else:
                    stats["records_rejected"] += 1
            time.sleep(GUTENBERG_DELAY_S)

        # ---- arXiv ML/AI papers (abstracts) ----
        for arxiv_id, expected_title in ARXIV_VALIDATION:
            print(f"[arxiv] fetching {arxiv_id} ({expected_title})")
            paper = fetch_arxiv_single(arxiv_id)
            time.sleep(ARXIV_DELAY_S)
            if paper is None:
                stats["arxiv_papers_failed"].append(arxiv_id)
                stats["failed_sources"].append(f"arxiv:{arxiv_id}")
                continue
            stats["arxiv_papers_fetched"] += 1
            # Abstracts are 200-400 words — keep as one record, don't chunk.
            text = f"{paper['title']}\n\n{paper['abstract']}"
            rec = make_record(
                text=text, source="arxiv",
                domain="ai", subdomain="ai",
                author=paper["authors"], title=paper["title"],
            )
            if quality_filter_science(rec):
                ai_fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                stats["ai_records_written"] += 1
                stats["approx_tokens"] += rec["tokens"]
            else:
                stats["records_rejected"] += 1

        if stats["arxiv_papers_fetched"] == 0:
            print("  [arxiv] WARNING — zero papers fetched "
                  "(likely rate-limited); continuing")

        # ---- MIT OCW (6.034 AI) ----
        print(f"[ocw] fetching {OCW_VALIDATION_URL}")
        ocw = fetch_ocw_page(OCW_VALIDATION_URL)
        if ocw is None:
            stats["failed_sources"].append(f"ocw:{OCW_VALIDATION_URL}")
        else:
            ocw_title, ocw_text = ocw
            wc = len(ocw_text.split())
            if wc < OCW_PROSE_MIN_WORDS:
                print(f"  [ocw] only {wc} words of prose; "
                      f"below {OCW_PROSE_MIN_WORDS} threshold — skipping")
                stats["ocw_pages_skipped"].append(
                    f"{OCW_VALIDATION_URL}:{wc}w"
                )
            else:
                stats["ocw_pages_fetched"] += 1
                for chunk in chunk_text(ocw_text, min_words=200):
                    rec = make_record(
                        text=chunk, source="mit_ocw",
                        domain="ai", subdomain="ai",
                        author="MIT OCW", title=ocw_title,
                    )
                    if quality_filter_science(rec):
                        ai_fout.write(
                            json.dumps(rec, ensure_ascii=False) + "\n"
                        )
                        stats["ai_records_written"] += 1
                        stats["approx_tokens"] += rec["tokens"]
                    else:
                        stats["records_rejected"] += 1
    finally:
        science_fout.close()
        ai_fout.close()

    return stats


# ---------------------------------------------------------------------------
# Full pipeline (sketch — implementation deferred)
# ---------------------------------------------------------------------------

def run_full() -> dict:
    """Full collection path — sketch.

    Implementing the --full sweep would involve, roughly:

      1. Classical science — sweep all Gutenberg IDs from the spec's
         Agent 3 source list (Darwin x4, Faraday, Maxwell, Einstein,
         Feynman where public-domain, Schrödinger, Heisenberg, Poincaré,
         Curie). For each: verify via gutendex first (wave-1 lesson),
         then download plain-text, strip boilerplate, chunk to 500-2000
         words, quality-filter, write to data/curriculum_v2/science/raw/.

      2. arXiv AI/ML — the mandatory list (Attention, BERT, GPT-2,
         Scaling Laws, NTM, Dropout, BatchNorm, ResNet, GAN, VAE, World
         Models, JEPA, Predictive Coding, FEP) PLUS paginated survey
         queries (ti:"survey" OR ti:"review" AND cat:cs.LG, citations
         >500, 2015-2024, target ~200 papers). Per the wave-1 lesson:
         one ID per query, 3s sleep between, exponential backoff 3-9-27s
         on 429, then skip. For full text use the LaTeX source endpoint
         (export.arxiv.org/e-print/{id}) and strip math; abstracts are
         only good enough for validation.

      3. MIT OCW — every science+AI course on the spec list (6.034,
         6.036, 6.867, 9.641, 8.01, 7.012). For each: walk the
         lecture-notes page, follow every transcript-link, fetch each
         page, require >300 prose-words, chunk + filter + write to
         data/curriculum_v2/{ai or science}/raw/.

      4. Nature + Science open-access — only properly OA URLs
         (nature.com/articles/{slug} with the CC-BY tag). NOT a general
         scrape; we resolve through the OpenAlex or Unpaywall APIs for
         each candidate DOI, then download the publisher's open PDF
         (where one exists) and extract via pdfminer. Deferred because
         it requires significant per-publisher tuning and OA-license
         verification.

    Output target ~50M tokens split roughly 60% science / 40% ai per
    the COLLECTION_SPEC distribution.
    """
    raise NotImplementedError(
        "Full collection path is a sketch only. "
        "Run without --full for the small validation set."
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Agent 3 — Science & AI collector"
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Run the full collection (validation set by default)",
    )
    args = parser.parse_args(argv)

    if args.full:
        stats = run_full()
    else:
        stats = run_validation()

    print("\n=== Agent 3 summary ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
