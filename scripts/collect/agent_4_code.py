#!/usr/bin/env python3
"""Agent 4 — Code & Systems collector for the v2.0 curriculum.

Validation mode (default, no args):
    Pulls a small, directly-fetchable mix from three public sources:
      1. Stack Exchange 2.3 API (Stack Overflow questions+answers, no auth)
      2. Python official tutorial / how-to prose pages (PSF License, CC-ish)
      3. One Gutenberg CS-history text (Babbage's autobiography, id 57532)
    Records get chunked into 200–2000-word slices, filtered with
    quality_filter_code, and written to
        data/curriculum_v2/code/raw/agent_4_validation.jsonl

Full mode (--full):
    SKETCH only. Prints the source plan + URL list and bails. Implementing
    the full sweep requires processing the Stack Exchange 90 GB dump, the
    Hacker News PushShift dumps, a GitHub README crawl, and the classic CS
    texts — all dump-scale / rate-limited work that doesn't belong in the
    validation script.

Output schema (one JSON object per line):
    text, source, domain, subdomain, quality_score, dialogue,
    author, title, tokens, language
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

import requests
from bs4 import BeautifulSoup


# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "data" / "curriculum_v2" / "code" / "raw"
VALIDATION_OUT = OUTPUT_DIR / "agent_4_validation.jsonl"

USER_AGENT = "baby-mind-curriculum-collector/2.0"
HTTP_HEADERS = {"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"}
REQUEST_TIMEOUT = 30
POLITE_DELAY_SEC = 0.5
SE_QUOTA_FLOOR = 30  # stop early if remaining quota drops below this


# Stack Exchange — small validation set. Tags chosen from the spec
# ("python, ... algorithms, ... machine-learning, ... design-patterns").
# Each tag yields a top-voted question; we then fetch its single
# top-voted answer. With 5 tags and 4 questions each = ~20 question +
# answer fetches → ~10 API requests (questions are batched; answers are
# per-question), comfortable inside the 300/day free quota.
SE_TAGS = ("python", "algorithm", "machine-learning", "design-patterns", "data-structures")
SE_QUESTIONS_PER_TAG = 4

# Python tutorial / how-to pages with explanatory prose (not API refs).
# All published by the PSF under the Python documentation license (BSD-
# style / CC-ish, redistributable for any use with attribution).
PYTHON_DOC_URLS = (
    ("https://docs.python.org/3/tutorial/datastructures.html",
     "Python Tutorial: Data Structures"),
    ("https://docs.python.org/3/tutorial/classes.html",
     "Python Tutorial: Classes"),
    ("https://docs.python.org/3/howto/functional.html",
     "Python HOWTO: Functional Programming"),
)

# Gutenberg CS-history text — Babbage. Verified via gutendex on
# build-day:  curl -sL https://gutendex.com/books/57532 returned
# {"title": "Passages from the Life of a Philosopher",
#  "authors": [{"name": "Babbage, Charles", ...}]}
GUTENBERG_CS_TEXT = (57532, "Charles Babbage", "Passages from the Life of a Philosopher")


# ---------------------------------------------------------------------------
# Quality filter (from the spec — verbatim)
# ---------------------------------------------------------------------------

EXPLANATION_PHRASES = (
    "because", "this means", "in other words",
    "the reason", "which allows", "this ensures",
)


def quality_filter_code(record: dict) -> bool:
    """Return True if the record passes Agent 4's quality gate.

    Lifted verbatim from COLLECTION_SPEC.md → Agent 4 → Quality filters.
    code_ratio counts indented lines (`'\\n    '`) over total lines; >0.6
    rejects records that are 60%+ code blocks.
    """
    text = record["text"]
    lines = max(len(text.split("\n")), 1)
    code_ratio = text.count("\n    ") / lines
    if code_ratio > 0.6:
        return False
    if len(text.split()) < 100:
        return False
    explains = any(w in text.lower() for w in EXPLANATION_PHRASES)
    return explains


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def detect_language_simple(text: str) -> str:
    """Crude ASCII-ratio English heuristic — good enough for validation."""
    sample = text[:4000]
    if not sample:
        return "unknown"
    printable = sum(1 for c in sample
                    if c.isascii() and (c.isalnum() or c.isspace()
                                        or c in ".,;:()-'\"!?[]{}/<>"))
    return "en" if printable / max(len(sample), 1) > 0.85 else "other"


def rough_token_count(text: str) -> int:
    return int(len(text.split()) * 1.3)


REASONING_WORDS = {
    "therefore", "because", "however", "although",
    "implies", "suggests", "shows", "demonstrates",
    "furthermore", "nevertheless", "consequently",
}


def compute_quality_score(text: str) -> float:
    """Light heuristic matching Agent 9's scorer; Agent 9 recomputes properly."""
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
    lowered = {w.lower().strip(".,;:()[]\"'") for w in words}
    score += min(len(lowered & REASONING_WORDS) * 0.02, 0.1)
    if unique_ratio < 0.4:
        score -= 0.2
    return max(0.0, min(1.0, score))


def make_record(*, text: str, source: str, subdomain: str,
                author: str, title: str, dialogue: bool = False) -> dict:
    return {
        "text": text,
        "source": source,
        "domain": "code",
        "subdomain": subdomain,
        "quality_score": round(compute_quality_score(text), 4),
        "dialogue": dialogue,
        "author": author,
        "title": title,
        "tokens": rough_token_count(text),
        "language": detect_language_simple(text),
    }


def chunk_text(text: str, target_words: int = 1000, min_words: int = 200,
               max_words: int = 2000) -> Iterator[str]:
    """Greedy paragraph-merging chunker. Yields ~target_words slices, never
    smaller than min_words (so we don't ship one-paragraph fragments) and
    never larger than max_words.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    buf: list[str] = []
    buf_words = 0
    for para in paragraphs:
        pw = len(para.split())
        # Single oversize paragraph — sentence-split it.
        if pw > max_words:
            for sent in re.split(r"(?<=[.!?])\s+", para):
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
# Stats container
# ---------------------------------------------------------------------------

@dataclass
class Stats:
    se_questions_fetched: int = 0
    se_answers_fetched: int = 0
    se_api_calls: int = 0
    se_quota_remaining: int | None = None
    python_doc_pages: int = 0
    gutenberg_books: int = 0
    records_written: int = 0
    records_rejected: int = 0
    rejected_short: int = 0
    rejected_no_explain: int = 0
    rejected_code_ratio: int = 0
    approx_tokens: int = 0
    failed_sources: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Source 1 — Stack Exchange (Stack Overflow) live API
# ---------------------------------------------------------------------------

SE_API = "https://api.stackexchange.com/2.3"


def _se_get(path: str, params: dict, stats: Stats) -> dict | None:
    """One throttled SE API call. Updates stats.se_quota_remaining."""
    url = f"{SE_API}{path}"
    try:
        r = requests.get(url, params=params, headers=HTTP_HEADERS,
                         timeout=REQUEST_TIMEOUT)
        stats.se_api_calls += 1
        if r.status_code != 200:
            print(f"  [se] {path} status={r.status_code}", file=sys.stderr)
            return None
        data = r.json()
        if "quota_remaining" in data:
            stats.se_quota_remaining = data["quota_remaining"]
        return data
    except Exception as exc:  # noqa: BLE001 — defensive
        print(f"  [se] {path} request failed: {exc}", file=sys.stderr)
        return None
    finally:
        time.sleep(POLITE_DELAY_SEC)


def _html_to_prose(body_html: str) -> str:
    """SE bodies are HTML; pull out the text with code blocks elided.

    Code blocks > 50 lines get replaced with a placeholder so they don't
    dominate the chunk (per the spec — code_ratio > 0.6 rejects).
    Inline code stays inline.
    """
    soup = BeautifulSoup(body_html, "html.parser")
    for pre in soup.find_all("pre"):
        code_text = pre.get_text("\n")
        if code_text.count("\n") > 50:
            pre.replace_with(soup.new_string("[long code block elided]"))
        # short blocks stay; get_text below preserves them
    text = soup.get_text("\n")
    text = html.unescape(text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def collect_stack_exchange(stats: Stats) -> Iterator[dict]:
    """Yield records from the Stack Exchange API. One Q&A pair per record."""
    for tag in SE_TAGS:
        if (stats.se_quota_remaining is not None
                and stats.se_quota_remaining < SE_QUOTA_FLOOR):
            print(f"  [se] quota low ({stats.se_quota_remaining}); stopping early")
            stats.failed_sources.append(f"se:quota_floor@{tag}")
            return
        print(f"[se] fetching top {SE_QUESTIONS_PER_TAG} questions tagged {tag!r}")
        data = _se_get(
            "/questions",
            {"site": "stackoverflow",
             "order": "desc",
             "sort": "votes",
             "tagged": tag,
             "filter": "withbody",
             "pagesize": SE_QUESTIONS_PER_TAG},
            stats,
        )
        if not data or not data.get("items"):
            stats.failed_sources.append(f"se:questions:{tag}")
            continue
        for q in data["items"]:
            stats.se_questions_fetched += 1
            qid = q.get("question_id")
            title = q.get("title", "").strip() or f"Stack Overflow {qid}"
            q_body_html = q.get("body", "")
            q_prose = _html_to_prose(q_body_html)
            if not qid or not q_prose:
                continue
            if (stats.se_quota_remaining is not None
                    and stats.se_quota_remaining < SE_QUOTA_FLOOR):
                print(f"  [se] quota low ({stats.se_quota_remaining}); skipping answers")
                stats.failed_sources.append(f"se:quota_floor@answer:{qid}")
                return
            ans = _se_get(
                f"/questions/{qid}/answers",
                {"site": "stackoverflow",
                 "order": "desc",
                 "sort": "votes",
                 "filter": "withbody",
                 "pagesize": 1},
                stats,
            )
            if not ans or not ans.get("items"):
                continue
            a_body_html = ans["items"][0].get("body", "")
            a_prose = _html_to_prose(a_body_html)
            if not a_prose:
                continue
            stats.se_answers_fetched += 1
            text = (f"Question: {html.unescape(title)}\n"
                    f"{q_prose}\n\n"
                    f"Answer: {a_prose}")
            yield make_record(
                text=text,
                source="stack_overflow",
                subdomain=tag,
                author="Stack Overflow contributors",
                title=html.unescape(title),
                dialogue=True,  # Q/A pair is dialogue-shaped
            )


# ---------------------------------------------------------------------------
# Source 2 — Python official docs
# ---------------------------------------------------------------------------

def _fetch_python_doc(url: str) -> str | None:
    """Pull the article body out of a docs.python.org page.

    The Sphinx layout puts the content in <div class="body"> /
    <section> / <article>. Strips code blocks longer than 50 lines so the
    code_ratio filter doesn't reject the page wholesale.
    """
    try:
        r = requests.get(url, headers=HTTP_HEADERS, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        print(f"  [pydocs] {url}: {exc}", file=sys.stderr)
        return None
    soup = BeautifulSoup(r.text, "html.parser")
    # Drop chrome.
    for tag in soup(["script", "style", "nav", "header", "footer", "noscript"]):
        tag.decompose()
    # Sphinx puts the main page content here:
    main = (soup.find("article") or soup.find("div", class_="body")
            or soup.find("section") or soup.body or soup)
    # Long code blocks → placeholder.
    for pre in main.find_all(["pre"]):
        if pre.get_text("\n").count("\n") > 50:
            pre.replace_with(soup.new_string("[long code block elided]"))
    # Drop the table-of-contents sidebars and "deprecated since" admonitions
    # that add noise.
    for tag in main.find_all(["table"], class_="docutils"):
        tag.decompose()
    text = main.get_text("\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def collect_python_docs(stats: Stats) -> Iterator[dict]:
    for url, title in PYTHON_DOC_URLS:
        print(f"[pydocs] fetching {url}")
        body = _fetch_python_doc(url)
        if not body or len(body.split()) < 200:
            stats.failed_sources.append(f"pydocs:{url}")
            continue
        stats.python_doc_pages += 1
        for chunk in chunk_text(body):
            yield make_record(
                text=chunk,
                source="python_docs",
                subdomain="python",
                author="Python Software Foundation",
                title=title,
            )
        time.sleep(POLITE_DELAY_SEC)


# ---------------------------------------------------------------------------
# Source 3 — Gutenberg CS-history text (Babbage)
# ---------------------------------------------------------------------------

GUTENBERG_URLS = (
    "https://www.gutenberg.org/files/{gid}/{gid}-0.txt",
    "https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt",
    "https://www.gutenberg.org/files/{gid}/{gid}.txt",
)
GUTENBERG_START_RE = re.compile(
    r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG[^\n]*\*\*\*",
    re.IGNORECASE,
)
GUTENBERG_END_RE = re.compile(
    r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG[^\n]*\*\*\*",
    re.IGNORECASE,
)


def _verify_gutenberg(gid: int, expected_author: str, expected_title: str) -> bool:
    """Hit gutendex to confirm the ID still resolves to the expected work."""
    try:
        r = requests.get(f"https://gutendex.com/books/{gid}",
                         headers=HTTP_HEADERS, timeout=REQUEST_TIMEOUT,
                         allow_redirects=True)
        if r.status_code != 200:
            print(f"  [gutendex] {gid} status={r.status_code}", file=sys.stderr)
            return False
        info = r.json()
        title = (info.get("title") or "").lower()
        authors = ", ".join(a.get("name", "") for a in info.get("authors", []))
        exp_last = expected_author.split()[-1].lower()
        if exp_last not in authors.lower():
            print(f"  [gutendex] {gid}: author mismatch (got {authors!r})",
                  file=sys.stderr)
            return False
        # Loose title match — Gutenberg titles often include subtitle.
        if expected_title.lower().split()[0] not in title:
            print(f"  [gutendex] {gid}: title mismatch (got {title!r})",
                  file=sys.stderr)
            return False
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  [gutendex] verify {gid} failed: {exc}", file=sys.stderr)
        return False


def _fetch_gutenberg(gid: int) -> str | None:
    for tmpl in GUTENBERG_URLS:
        url = tmpl.format(gid=gid)
        try:
            r = requests.get(url, headers=HTTP_HEADERS, timeout=REQUEST_TIMEOUT)
            if r.status_code == 200 and len(r.text) > 2000:
                return r.text.lstrip("﻿")
        except Exception as exc:  # noqa: BLE001
            print(f"  [gutenberg] {url}: {exc}", file=sys.stderr)
        time.sleep(POLITE_DELAY_SEC)
    return None


def _strip_gutenberg_boilerplate(raw: str) -> str:
    m = GUTENBERG_START_RE.search(raw)
    if m:
        raw = raw[m.end():]
    m = GUTENBERG_END_RE.search(raw)
    if m:
        raw = raw[: m.start()]
    raw = re.sub(r"\r\n?", "\n", raw)
    raw = re.sub(r"\n{3,}", "\n\n", raw)
    return raw.strip()


def collect_gutenberg(stats: Stats) -> Iterator[dict]:
    gid, author, title = GUTENBERG_CS_TEXT
    print(f"[gutenberg] verifying id={gid} ({author}, {title!r})")
    if not _verify_gutenberg(gid, author, title):
        stats.failed_sources.append(f"gutenberg:verify:{gid}")
        print(f"  [gutenberg] skipping {gid} — ID did not verify")
        return
    print(f"[gutenberg] fetching {gid}")
    raw = _fetch_gutenberg(gid)
    if raw is None:
        stats.failed_sources.append(f"gutenberg:fetch:{gid}")
        return
    body = _strip_gutenberg_boilerplate(raw)
    stats.gutenberg_books += 1
    for chunk in chunk_text(body):
        yield make_record(
            text=chunk,
            source="gutenberg",
            subdomain="cs_history",
            author=author,
            title=title,
        )


# ---------------------------------------------------------------------------
# Validation pipeline
# ---------------------------------------------------------------------------

def _filter_and_record(rec: dict, fout, stats: Stats) -> None:
    """Apply quality_filter_code and either write or count the rejection."""
    text = rec["text"]
    words = text.split()
    if len(words) < 100:
        stats.records_rejected += 1
        stats.rejected_short += 1
        return
    lines = max(len(text.split("\n")), 1)
    if text.count("\n    ") / lines > 0.6:
        stats.records_rejected += 1
        stats.rejected_code_ratio += 1
        return
    if not any(w in text.lower() for w in EXPLANATION_PHRASES):
        stats.records_rejected += 1
        stats.rejected_no_explain += 1
        return
    # Final sanity (should always be True after the above):
    if not quality_filter_code(rec):
        stats.records_rejected += 1
        return
    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
    stats.records_written += 1
    stats.approx_tokens += rec["tokens"]


def run_validation() -> Stats:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"=== agent_4_code (VALIDATION) ===")
    print(f"writing to: {VALIDATION_OUT}")
    stats = Stats()

    with open(VALIDATION_OUT, "w", encoding="utf-8") as fout:
        # 1) Stack Exchange
        try:
            for rec in collect_stack_exchange(stats):
                _filter_and_record(rec, fout, stats)
        except Exception as exc:  # noqa: BLE001
            print(f"  [se] aborted: {exc}", file=sys.stderr)
            stats.failed_sources.append(f"se:exception:{exc}")

        # 2) Python docs
        try:
            for rec in collect_python_docs(stats):
                _filter_and_record(rec, fout, stats)
        except Exception as exc:  # noqa: BLE001
            print(f"  [pydocs] aborted: {exc}", file=sys.stderr)
            stats.failed_sources.append(f"pydocs:exception:{exc}")

        # 3) Gutenberg CS-history text
        try:
            for rec in collect_gutenberg(stats):
                _filter_and_record(rec, fout, stats)
        except Exception as exc:  # noqa: BLE001
            print(f"  [gutenberg] aborted: {exc}", file=sys.stderr)
            stats.failed_sources.append(f"gutenberg:exception:{exc}")

    return stats


# ---------------------------------------------------------------------------
# Full mode — SKETCH only
# ---------------------------------------------------------------------------

FULL_SOURCES_SKETCH = """\
Agent 4 — Code & Systems — FULL collection sketch
================================================

Target: ~30M tokens across four pipelines. None of these are runnable
inside this validation script — each is a multi-day, dump-scale job
that belongs in its own worker.

1. Stack Overflow data dump  (~10M tokens)
   - Archive:  https://archive.org/details/stackexchange
   - File:     stackoverflow.com-Posts.7z   (~90 GB compressed)
   - Filter:   IsAcceptedAnswer=True AND Score>=20 AND tag in
               {python, rust, javascript, algorithms,
                data-structures, machine-learning, linux,
                design-patterns, architecture}
   - Strip:    HTML body; elide code blocks > 200 lines.
   - Format:   "Question: {title}\\n{question_body}\\n"
               "Answer: {answer_body}"
   - Process:  py7zr decompression -> XML stream parse with
               lxml.etree.iterparse -> join Posts.xml on ParentId.

2. GitHub READMEs and docstrings  (~5M tokens — docs not raw code)
   - API:      https://api.github.com/search/repositories?
                  q=stars:>1000+language:python   (and rust, ts)
               followed by /contents/README.md per repo.
   - Auth:     anonymous = 60 req/h; with a PAT = 5000 req/h.
   - Filter:   README length > 500 words; exclude template repos;
               keep CHANGELOG, ARCHITECTURE.md, docs/*.md as well.
   - Specific: numpy, pandas, pytorch, sklearn docs (sphinx);
               Linux kernel Documentation/ (Plain text);
               Python official docs (already covered);
               Rust Book (MIT, github.com/rust-lang/book);
               MDN Web Docs (CC BY-SA 2.5).

3. Classic CS texts  (~5M tokens)
   - SICP:     https://mitpress.mit.edu/sicp/  (MIT license)
   - K&R:      NOT public domain — skip.
   - TAOCP:    NOT public domain — skip.
   - Babbage:  Gutenberg id 57532 (covered in validation).
   - SICP HTML source: github.com/sarabander/sicp-pdf or
                       https://web.mit.edu/alexmv/6.037/sicp.pdf.
   - GoF Design Patterns summary: NOT public domain — synthesise
     from open-source pattern catalogs (refactoring.guru is CC BY-SA).

4. Hacker News PushShift  (~3M tokens)
   - Dump:     https://files.pushshift.io/hackernews/
   - File:     HN_2006-{year}-{month}.json.bz2
   - Filter:   score >= 100, top-level comments, length > 200 words,
               topic-classifier rejects politics/drama/meta.
   - Process:  stream-decompress -> json-lines parse -> rank, filter.

Pipeline notes:
   - All four pipelines emit the same JSONL schema (see make_record).
   - Quality filter (quality_filter_code) runs on every chunked record.
   - Output paths: data/curriculum_v2/code/raw/agent_4_full_{stackoverflow,
     github,classics,hackernews}.jsonl ; Agent 9 merges + dedupes.
"""


def run_full() -> Stats:
    """Print the SKETCH; do not attempt the dump-scale work."""
    print(FULL_SOURCES_SKETCH)
    print("[agent_4] --full is a SKETCH only — no records written.")
    print("[agent_4] run without --full for the validation sweep.")
    return Stats()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Agent 4 — Code & Systems collector (v2.0).")
    ap.add_argument(
        "--full",
        action="store_true",
        help="Print the FULL-collection sketch instead of running validation.",
    )
    args = ap.parse_args(argv)

    if args.full:
        run_full()
        return 0

    stats = run_validation()

    print()
    print("=== summary ===")
    print(f"  SE questions fetched:  {stats.se_questions_fetched}")
    print(f"  SE answers fetched:    {stats.se_answers_fetched}")
    print(f"  SE API calls:          {stats.se_api_calls}")
    print(f"  SE quota remaining:    {stats.se_quota_remaining}")
    print(f"  Python doc pages:      {stats.python_doc_pages}")
    print(f"  Gutenberg books:       {stats.gutenberg_books}")
    print(f"  Records written:       {stats.records_written}")
    print(f"  Records rejected:      {stats.records_rejected}")
    print(f"     - short (<100 wd):  {stats.rejected_short}")
    print(f"     - code_ratio >0.6:  {stats.rejected_code_ratio}")
    print(f"     - no-explain:       {stats.rejected_no_explain}")
    print(f"  Approx tokens:         {stats.approx_tokens:,}")
    if stats.failed_sources:
        print("  Failed sources:")
        for f in stats.failed_sources:
            print(f"    - {f}")

    return 0 if stats.records_written > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
