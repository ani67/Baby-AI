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
    Fetches the full Gutenberg classical-science catalog (~18 books:
    Darwin x4, Faraday, Maxwell, Einstein, Huxley x2, Tyndall x4,
    Poincaré x2, Whewell, Helmholtz, Curie) and the mandatory arXiv
    AI/ML abstract list plus a small survey-paper expansion (~18
    papers total). Each Gutenberg ID is verified via gutendex before
    fetch — stale IDs are substituted by gutendex search at constant-
    table compile time, runtime ones get skip-and-warn. arXiv uses
    single-ID queries with 3s sleep and 3/9/27s exponential backoff
    on 429.

    MIT OCW is intentionally skipped in --full because wave-1
    validation showed the lecture-notes pages are almost entirely
    navigation chrome — not worth the per-page filter overhead until
    a dedicated OCW transcript collector exists.

    Nature/Science open access is also out of scope here — it
    requires per-publisher OA license verification (OpenAlex /
    Unpaywall) and PDF extraction, which is its own collection
    agent.

    Writes JSONL in append mode to:
        data/curriculum_v2/science/raw/agent_3_full.jsonl   (Gutenberg)
        data/curriculum_v2/ai/raw/agent_3_full.jsonl        (arXiv)

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
SCIENCE_FULL_OUT = SCIENCE_OUT_DIR / "agent_3_full.jsonl"
AI_FULL_OUT = AI_OUT_DIR / "agent_3_full.jsonl"

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
# Full collection source tables
# ---------------------------------------------------------------------------

# Full classical-science Gutenberg list. Each ID was verified live against
# gutendex on 2026-05-12 (see commit log for verification trace). Stale
# spec IDs (12959, 35067, 53884) were substituted via gutendex search:
#   - 12959 (Faraday Vol 2) → not in gutendex; dropped
#   - 35067 (Tyndall heat)  → 54969 "Sound" + 24527 "Fragments of Science"
#                             + 14000 "Six Lectures on Light" + 1225
#                             "Faraday as a Discoverer"
#   - 53884 (Poincaré)      → 39713 "Foundations of Science" (bundle of
#                             Science and Hypothesis / Value of Science /
#                             Science and Method) + 37157 "Science and
#                             Hypothesis"
# verify_gutenberg_id() still runs at fetch time — if any of these drift,
# the run will skip+warn rather than fetch the wrong text.
GUTENBERG_FULL: list[tuple[int, str, str]] = [
    # Darwin
    (1228,  "Charles Darwin",       "On the Origin of Species"),
    (2300,  "Charles Darwin",       "The Descent of Man"),
    (944,   "Charles Darwin",       "The Voyage of the Beagle"),
    (1227,  "Charles Darwin",       "The Expression of the Emotions in Man and Animals"),
    # Faraday + electricity
    (14986, "Michael Faraday",      "Experimental Researches in Electricity, Volume 1"),
    (69914, "James Clerk Maxwell",  "An Elementary Treatise on Electricity"),
    # Einstein
    (30155, "Albert Einstein",      "Relativity: The Special and General Theory"),
    # Huxley
    (16474, "Thomas Henry Huxley",  "Lectures and Essays"),
    (6414,  "Thomas Henry Huxley",  "Lectures and Essays (collection)"),
    # Tyndall (subbing for the stale 35067 — same scientist, comparable corpus)
    (54969, "John Tyndall",         "Sound"),
    (24527, "John Tyndall",         "Fragments of Science"),
    (14000, "John Tyndall",         "Six Lectures on Light"),
    (1225,  "John Tyndall",         "Faraday as a Discoverer"),
    # Poincaré (subbing for the stale 53884 — Foundations of Science is the
    # canonical English bundle that includes Science and Method)
    (39713, "Henri Poincaré",       "The Foundations of Science"),
    (37157, "Henri Poincaré",       "Science and Hypothesis"),
    # Whewell
    (68693, "William Whewell",      "History of the Inductive Sciences"),
    # Helmholtz
    (77725, "Hermann von Helmholtz", "Popular Lectures on Scientific Subjects"),
    # Curie
    (69617, "Marie Curie",          "Pierre Curie"),
]

# arXiv full list — the MANDATORY spec IDs that exist on arXiv plus a
# small survey/tutorial expansion (cs.LG / cs.AI, ti:"survey"). Per
# wave-1: per-id queries, 3s sleep, 3/9/27s backoff, skip-and-warn.
# Note: GPT-2 (Radford 2019), JEPA (LeCun 2022 position paper),
# Predictive Coding (Rao & Ballard 1999), and Free Energy Principle
# (Friston) are NOT on arXiv as primary host — skipped here.
ARXIV_FULL_MANDATORY: list[tuple[str, str]] = [
    ("1706.03762", "Attention Is All You Need"),
    ("1810.04805", "BERT"),
    ("2001.08361", "Scaling Laws for Neural Language Models"),
    ("1410.5401", "Neural Turing Machines"),
    ("1207.0580", "Improving neural networks by preventing co-adaptation (Dropout)"),
    ("1502.03167", "Batch Normalization"),
    ("1512.03385", "Deep Residual Learning (ResNet)"),
    ("1406.2661", "Generative Adversarial Networks"),
    ("1312.6114", "Auto-Encoding Variational Bayes (VAE)"),
    ("1803.10122", "World Models"),
]

# Survey/tutorial expansion — sourced from the arXiv API on 2026-05-12
# via `cat:cs.LG AND ti:survey`. Selected for breadth across modern ML.
ARXIV_FULL_SURVEYS: list[tuple[str, str]] = [
    ("2106.04554", "A Survey of Transformers"),
    ("2408.01129", "A Survey of Mamba"),
    ("2003.07278", "A Survey on Contextual Embeddings"),
    ("2010.13166", "A Survey on Curriculum Learning"),
    ("1810.03548", "Meta-Learning: A Survey"),
    ("2101.01169", "Transformers in Vision: A Survey"),
    ("2202.12040", "Self-Training: A Survey"),
    ("2410.15042", "Adversarial Training: A Survey"),
]


# ---------------------------------------------------------------------------
# Quality filter (verbatim from COLLECTION_SPEC.md → Agent 3)
# ---------------------------------------------------------------------------

PROSE_MARKERS = (
    # general explanatory prose
    "because", "therefore", "suggests", "implies",
    "shows", "demonstrates", "indicates",
)

# Abstract markers — academic abstracts use a distinct register from
# long-form science prose. Wave-3 validation surfaced this: all 5 arXiv
# abstracts (~150 words each) were rejected by the prose-only filter
# even though they're load-bearing for the corpus.
ABSTRACT_MARKERS = (
    "we propose", "we show", "we demonstrate", "we present",
    "we introduce", "we evaluate", "we find",
    "our approach", "our method", "our model",
    "this paper", "this work", "in this work",
    "results show", "results demonstrate",
)


def quality_filter_science(record: dict) -> bool:
    """Return True if record passes the science/AI quality filter.

    From COLLECTION_SPEC.md, with one validation-driven adjustment:
        - >= 100 words minimum (was 200; arXiv abstracts are 125–170 words)
        - reject pure wet-lab protocol pages (heavy 'ml '/'μl ' usage)
        - pass if:
            • contains an explanatory prose marker (long-form prose), OR
            • contains an abstract marker (academic abstract register), OR
            • text is long (>= 200 words — full papers pass on length alone)
    """
    text = record["text"]
    n_words = len(text.split())
    if n_words < 100:
        return False
    lower = text.lower()
    if lower.count("ml ") + lower.count("μl ") > 10:
        return False
    if any(w in lower for w in PROSE_MARKERS):
        return True
    if any(w in lower for w in ABSTRACT_MARKERS):
        return True
    return n_words > 200


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
    """Real full collection — Gutenberg classical science + arXiv AI/ML.

    See module docstring for scope. MIT OCW and Nature/Science OA are
    intentionally out of scope here (see docstring rationale).

    Output files are APPENDED so re-runs accumulate rather than
    clobber. Callers who want a clean run should delete the target
    files first.
    """
    SCIENCE_OUT_DIR.mkdir(parents=True, exist_ok=True)
    AI_OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[agent_3:full] science → {SCIENCE_FULL_OUT}")
    print(f"[agent_3:full] ai      → {AI_FULL_OUT}")

    stats: dict = {
        "books_verified": 0,
        "books_fetched": 0,
        "books_fetched_titles": [],
        "books_failed_verification": [],
        "books_failed_download": [],
        "arxiv_papers_fetched": 0,
        "arxiv_papers_fetched_ids": [],
        "arxiv_papers_failed": [],
        "science_records_written": 0,
        "ai_records_written": 0,
        "records_rejected": 0,
        "approx_tokens": 0,
        "failed_sources": [],
    }

    # APPEND mode — re-runs add to existing output rather than clobbering.
    science_fout = open(SCIENCE_FULL_OUT, "a", encoding="utf-8")
    ai_fout = open(AI_FULL_OUT, "a", encoding="utf-8")
    try:
        # ---- Project Gutenberg (classical science) ----
        for book_id, author, title in GUTENBERG_FULL:
            try:
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
                    stats["books_failed_download"].append(book_id)
                    stats["failed_sources"].append(
                        f"gutenberg:{book_id}:download"
                    )
                    continue
                stats["books_fetched"] += 1
                stats["books_fetched_titles"].append(f"{book_id}:{title}")
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
                science_fout.flush()
                time.sleep(GUTENBERG_DELAY_S)
            except Exception as e:  # noqa: BLE001
                # Per-source try/except — one bad book must not crash the run.
                print(f"  [gutenberg {book_id}] UNEXPECTED ERROR: {e}",
                      file=sys.stderr)
                stats["failed_sources"].append(
                    f"gutenberg:{book_id}:exception:{type(e).__name__}"
                )

        # ---- arXiv AI/ML papers (abstracts) ----
        arxiv_all = ARXIV_FULL_MANDATORY + ARXIV_FULL_SURVEYS
        for arxiv_id, expected_title in arxiv_all:
            try:
                print(f"[arxiv] fetching {arxiv_id} ({expected_title})")
                paper = fetch_arxiv_single(arxiv_id)
                time.sleep(ARXIV_DELAY_S)
                if paper is None:
                    stats["arxiv_papers_failed"].append(arxiv_id)
                    stats["failed_sources"].append(f"arxiv:{arxiv_id}")
                    continue
                stats["arxiv_papers_fetched"] += 1
                stats["arxiv_papers_fetched_ids"].append(arxiv_id)
                text = f"{paper['title']}\n\n{paper['abstract']}"
                rec = make_record(
                    text=text, source="arxiv",
                    domain="ai", subdomain="ml_paper",
                    author=paper["authors"], title=paper["title"],
                )
                if quality_filter_science(rec):
                    ai_fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    stats["ai_records_written"] += 1
                    stats["approx_tokens"] += rec["tokens"]
                else:
                    stats["records_rejected"] += 1
                ai_fout.flush()
            except Exception as e:  # noqa: BLE001
                print(f"  [arxiv {arxiv_id}] UNEXPECTED ERROR: {e}",
                      file=sys.stderr)
                stats["failed_sources"].append(
                    f"arxiv:{arxiv_id}:exception:{type(e).__name__}"
                )

        if stats["arxiv_papers_fetched"] == 0:
            print("  [arxiv] WARNING — zero papers fetched "
                  "(likely rate-limited); continuing")

        # ---- MIT OCW: SKIPPED in --full ----
        # Wave-1 validation showed OCW lecture-notes pages are mostly
        # navigation chrome and the few that have transcripts are an
        # entirely different ingestion pattern (multi-page walks,
        # PDFs, redirect bugs). Out of scope for this collector; will
        # be a dedicated OCW agent later. Not a failure — just not
        # part of this surface.
    finally:
        science_fout.close()
        ai_fout.close()

    return stats


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
