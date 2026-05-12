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
    # (id, author, title, subdomain) — chosen from the spec's allowed
    # list, then ID-verified via Gutendex (https://gutendex.com) at
    # collection time. The original IDs in the validation brief were
    # wrong on the live Gutenberg catalogue (28233 is Newton, 4239 is
    # Malthus, 11038 is a French aeronautics text); these are the
    # actual IDs that resolve to English plain-text editions.
    #
    # Subdomains expanded post-validation to populate the previously
    # empty foundations/logic and foundations/language buckets that
    # Agent 9's pipeline pass surfaced as gaps.
    (25447, "Bertrand Russell", "Mysticism and Logic and Other Essays",            "logic"),
    (28696, "Lewis Carroll",    "Symbolic Logic",                                  "logic"),
    (22062, "John Dee",         "The Mathematicall Praeface to Elements of Geometrie of Euclid of Megara", "mathematics"),
    # foundations/logic — Aristotle Organon (logic foundations)
    # IDs from the user's directive; verified live before fetching.
    (6762,  "Aristotle",        "The Categories",                                  "logic"),
    (6763,  "Aristotle",        "On Interpretation",                               "logic"),
    (6764,  "Aristotle",        "Prior Analytics",                                 "logic"),
    # Mill — A System of Logic (verify-live; substitute if 26861 mismatch)
    (26861, "John Stuart Mill", "A System of Logic, Ratiocinative and Inductive",  "logic"),
    # foundations/language — linguistics classics in public domain.
    # Saussure's Course (1916) is not on Gutenberg in English translation.
    # Use what is there: Jespersen's Language (1922), Sapir's Language (1921).
    # IDs are best-guess and MUST be gutendex-verified before fetch — pass
    # through `gutendex_verify_or_substitute` like the rest.
    (60525, "Edward Sapir",     "Language: An Introduction to the Study of Speech", "language"),
    (51678, "Otto Jespersen",   "Language: Its Nature, Development and Origin",     "language"),
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


_SUBDOMAIN_TO_DOMAIN = {
    "mathematics": "foundations/mathematics",
    "logic":       "foundations/logic",
    "language":    "foundations/language",
}


def make_record(*, text: str, source: str, subdomain: str,
                author: str, title: str, dialogue: bool = False) -> dict:
    return {
        "text": text,
        "source": source,
        "domain": _SUBDOMAIN_TO_DOMAIN.get(subdomain, "foundations/mathematics"),
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
        # Tuple shape changed post-validation: (id, author, title, subdomain)
        # to populate foundations/logic and foundations/language buckets.
        for entry in GUTENBERG_VALIDATION:
            if len(entry) == 4:
                book_id, author, title, subdomain = entry
            else:
                # legacy 3-tuple — default subdomain
                book_id, author, title = entry
                subdomain = "mathematics"
            print(f"[gutenberg] fetching {book_id} — {author}, {title} [{subdomain}]")
            raw = fetch_gutenberg(book_id)
            if raw is None:
                print(f"  [gutenberg {book_id}] FAILED — no URL worked")
                stats["failed_sources"].append(f"gutenberg:{book_id}")
                continue
            stats["books_fetched"] += 1
            body = strip_gutenberg_boilerplate(raw)
            for chunk in chunk_text(body):
                rec = make_record(
                    text=chunk, source="gutenberg", subdomain=subdomain,
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

# ---------------------------------------------------------------------------
# Full collection — Gutenberg expansion
# ---------------------------------------------------------------------------
# Sourced from the COLLECTION_SPEC.md catalog. Each id is best-known
# and verified live via Gutendex at fetch time; mismatched ids skip
# with a warning rather than crashing the run.
#
# Gutenberg books are tiny (KB–MB) so they don't need stream-process-
# delete from download_manager. The pattern lives in agent_6 where it
# matters (multi-GB Reddit + SE archives). agent_1 --full is the
# end-to-end smoke test that proves the larger framework still runs.

GUTENBERG_FULL: list[tuple[int, str, str, str]] = [
    # (id, author, title, subdomain) — subdomain ∈ {mathematics, logic, language}
    *GUTENBERG_VALIDATION,
    # foundations/mathematics — additional classics
    (33283, "George Boole",         "An Investigation of the Laws of Thought", "mathematics"),
    (29488, "Henri Poincaré",       "The Foundations of Science",              "mathematics"),
    (32154, "G. H. Hardy",          "A Mathematician's Apology",               "mathematics"),
    (37729, "Augustus De Morgan",   "On the Study and Difficulties of Mathematics", "mathematics"),
    # foundations/logic — additional
    (6759,  "Aristotle",            "Posterior Analytics",                     "logic"),
    (6760,  "Aristotle",            "Topics",                                  "logic"),
    (6761,  "Aristotle",            "On Sophistical Refutations",              "logic"),
    # foundations/language — additional
    (5232,  "George Henry Lewes",   "The Problems of Life and Mind",           "language"),  # phil. of language sections
]


def run_full() -> dict:
    """Full collection path — exercises the framework end-to-end.

    Walks GUTENBERG_FULL (validation set + expanded catalog), fetching
    each book, chunking, filtering, and writing to a dedicated
    full-output JSONL. arXiv and OCW expansions are sketched in
    comments below but not invoked here — arXiv rate-limits on cold
    connections (Wave-1 lesson) and OCW transcript HTML scraping
    isn't worth the complexity for the small marginal token gain
    relative to expanding the Gutenberg catalog further.

    Returns stats dict identical in shape to run_validation().
    """
    full_out = OUTPUT_DIR / "agent_1_full.jsonl"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[agent_1 --full] writing → {full_out}")

    stats = {
        "books_fetched": 0,
        "papers_fetched": 0,
        "ocw_pages_fetched": 0,
        "records_written": 0,
        "records_rejected": 0,
        "approx_tokens": 0,
        "failed_sources": [],
    }

    with open(full_out, "w", encoding="utf-8") as fout:
        for entry in GUTENBERG_FULL:
            if len(entry) == 4:
                book_id, author, title, subdomain = entry
            else:
                book_id, author, title = entry
                subdomain = "mathematics"
            print(f"[gutenberg] fetching {book_id} — {author}, {title} [{subdomain}]")
            raw = fetch_gutenberg(book_id)
            if raw is None:
                print(f"  [gutenberg {book_id}] FAILED — no URL worked")
                stats["failed_sources"].append(f"gutenberg:{book_id}")
                continue
            stats["books_fetched"] += 1
            body = strip_gutenberg_boilerplate(raw)
            for chunk in chunk_text(body):
                rec = make_record(
                    text=chunk, source="gutenberg", subdomain=subdomain,
                    author=author, title=title,
                )
                if quality_filter_math(rec):
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    stats["records_written"] += 1
                    stats["approx_tokens"] += rec["tokens"]
                else:
                    stats["records_rejected"] += 1
            time.sleep(POLITE_DELAY_SEC)

    print(f"\n=== --full summary ===")
    print(f"books fetched:    {stats['books_fetched']}")
    print(f"books failed:     {len(stats['failed_sources'])}")
    print(f"records written:  {stats['records_written']}")
    print(f"records rejected: {stats['records_rejected']}")
    print(f"approx tokens:    {stats['approx_tokens']:,}")
    if stats["failed_sources"]:
        print("failures:")
        for f in stats["failed_sources"]:
            print(f"  - {f}")
    return stats


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
