"""
Agent 7 — Design, Finance & Media collection.

Fetches three small, public corpora — one per domain — and writes
them as JSONL chunks under data/curriculum_v2/{domain}/raw/.

Validation mode (default). Sources:

  finance / annual_letter   Berkshire Hathaway annual letters
                            (priority anchor — Warren Buffett, 60 years of
                            plain-English reasoning).
  finance / classic         One Gutenberg classic (Adam Smith or Veblen).
  design  / architecture    One Gutenberg architecture text (Vitruvius
                            or Ruskin).
  media   / theory          Wikipedia REST API pages on media topics.

Full mode (--full) is the real expansion pass. APPENDs to:
  data/curriculum_v2/finance/raw/agent_7_full.jsonl
  data/curriculum_v2/design/raw/agent_7_full.jsonl
  data/curriculum_v2/media/raw/agent_7_full.jsonl

  finance / annual_letter   All Berkshire HTML letters 1977..2001
                            (priority anchor — ~25 letters, target
                            ~500K tokens). PDF-only letters
                            (2002..present) are out of scope and
                            documented as a follow-up pass.
  finance / classic         Adam Smith, Veblen, Mill, Smith essays,
                            Smith Theory of Moral Sentiments.
  design  / architecture    Vitruvius, Ruskin Seven Lamps, Ruskin
                            Stones of Venice vols 1-3.
  media   / theory          ~15 Wikipedia REST API pages spanning
                            cinema, photography, music theory,
                            painting, sculpture, animation, etc.

Spec: doc/curriculum/COLLECTION_SPEC.md → "Agent 7 — Design, Finance
& Media", with the Berkshire callout flagged as the priority anchor.

Note from prior waves: the spec's Berkshire HTML/PDF cutover was
inverted. Reality is 1977–2001 = HTML, 2002–present = PDF.
BERKSHIRE_HTML_LETTERS_FULL_SKETCH below has the correct HTML set.

Schema (per record) follows COLLECTION_SPEC.md "Output format":
  text, source, domain, subdomain, quality_score, dialogue, author,
  title, tokens, language.
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

try:
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover
    sys.stderr.write(
        "beautifulsoup4 not installed. "
        "Run: pip install beautifulsoup4 --break-system-packages -q\n"
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

USER_AGENT = "baby-mind-curriculum-collector/2.0 (research)"
POLITE_DELAY = 0.7  # seconds between requests, per spec


def http_get(
    url: str,
    *,
    timeout: float = 30.0,
    max_retries: int = 3,
    backoff: float = 3.0,
) -> requests.Response | None:
    """GET with exponential backoff. Returns None on definitive failure.

    Treats 200/3xx as success, 4xx as definitive (no retry), and 5xx /
    network errors as retryable with exponential backoff."""
    delay = backoff
    for attempt in range(max_retries):
        try:
            r = requests.get(
                url,
                timeout=timeout,
                headers={"User-Agent": USER_AGENT},
            )
        except requests.RequestException as exc:
            print(f"  http error ({attempt + 1}/{max_retries}) {url}: {exc}",
                  file=sys.stderr)
            time.sleep(delay)
            delay *= 2
            continue
        if r.status_code == 200:
            return r
        if 400 <= r.status_code < 500:
            print(f"  http {r.status_code} (no retry) {url}", file=sys.stderr)
            return None
        # 5xx: retry
        print(f"  http {r.status_code} ({attempt + 1}/{max_retries}) {url}",
              file=sys.stderr)
        time.sleep(delay)
        delay *= 2
    return None


# ---------------------------------------------------------------------------
# Quality score (shared formula, copied from COLLECTION_SPEC.md §Agent 9)
# ---------------------------------------------------------------------------

REASONING_WORDS = frozenset({
    "therefore", "because", "however", "although", "implies",
    "suggests", "shows", "demonstrates", "furthermore",
    "nevertheless", "consequently",
})


def compute_quality_score(text: str) -> float:
    words = text.split()
    n = len(words)
    if n == 0:
        return 0.0
    score = 0.5

    if 200 <= n <= 2000:
        score += 0.1
    elif n > 2000:
        score += 0.05

    unique_ratio = len(set(words)) / n
    score += unique_ratio * 0.2

    if text.rstrip()[-1:] in ".!?":
        score += 0.05

    reason_count = len(set(words) & REASONING_WORDS)
    score += min(reason_count * 0.02, 0.1)

    if unique_ratio < 0.4:
        score -= 0.2

    return round(max(0.0, min(1.0, score)), 4)


def quality_filter(record: dict) -> bool:
    """Shared filter for design/finance/media: minimum 150 words.

    No per-domain filter is defined in the spec for these three
    domains; we rely on compute_quality_score plus a length floor.
    Agent 9 retunes the floor / score cutoff downstream."""
    return len(record["text"].split()) >= 150


def estimate_tokens(text: str) -> int:
    """~4 chars / token. Real tokenization happens in Agent 9."""
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Chunking — generic prose 500..2000 words
# ---------------------------------------------------------------------------

CHUNK_TARGET = 1000
CHUNK_MAX = 2000
CHUNK_MIN = 500


def _split_oversized_block(
    block: str, target_words: int, max_words: int
) -> list[str]:
    """Split one paragraph that's larger than max_words on line breaks."""
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


def chunk_prose(
    body: str,
    target_words: int = CHUNK_TARGET,
    max_words: int = CHUNK_MAX,
    min_words: int = CHUNK_MIN,
) -> list[str]:
    """Greedy paragraph-grouper for prose. Records land in
    [min_words, max_words] when possible; final tail may be smaller
    and is kept (the length filter will sort it out)."""
    blocks = [b.strip() for b in re.split(r"\n\s*\n", body) if b.strip()]
    chunks: list[str] = []
    buf: list[str] = []
    buf_words = 0
    for block in blocks:
        bw = len(block.split())
        if bw == 0:
            continue
        if bw > max_words:
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


# ---------------------------------------------------------------------------
# Record builder
# ---------------------------------------------------------------------------

def build_record(
    text: str,
    *,
    source: str,
    domain: str,
    subdomain: str,
    author: str,
    title: str,
) -> dict:
    return {
        "text": text,
        "source": source,
        "domain": domain,
        "subdomain": subdomain,
        "quality_score": compute_quality_score(text),
        "dialogue": False,
        "author": author,
        "title": title,
        "tokens": estimate_tokens(text),
        "language": "en",
    }


# ---------------------------------------------------------------------------
# Berkshire Hathaway letters (PRIORITY)
# ---------------------------------------------------------------------------

# The Berkshire index at /letters/letters.html exposes:
#   1977..1997     /letters/{year}.html              real HTML, prose-only
#   1998..1999     /letters/{year}htm.html           ({year}.html is a
#                                                    redirector pointing
#                                                    at the PDF + htm)
#   2000..2001     /{year}ar/{year}letter.html       HTML in the AR sub-tree
#   2002..2003     /letters/{year}.html              redirector-only;
#                                                    PDF-only in practice
#   2004..present  /letters/{year}ltr.pdf            PDF-only
#
# For validation we fetch 3 of the most recent HTML letters — 2001,
# 2000, 1999. Spec asked for "≈2000–present", and these are the
# newest letters that exist as HTML. Earlier years (1977–1998) are
# also pure HTML and would be the natural --full extension.
# The 2024 Buffett "final letter" announcement is also PDF-only.

BERKSHIRE_HTML_LETTERS_VALIDATION: tuple[tuple[int, str], ...] = (
    (2001, "https://www.berkshirehathaway.com/2001ar/2001letter.html"),
    (2000, "https://www.berkshirehathaway.com/2000ar/2000letter.html"),
    (1999, "https://www.berkshirehathaway.com/letters/1999htm.html"),
)

# For --full sketch (commented out): the rest of the HTML-available
# letters, oldest first. The list was extracted from the index page at
# https://www.berkshirehathaway.com/letters/letters.html .
BERKSHIRE_HTML_LETTERS_FULL_SKETCH: tuple[tuple[int, str], ...] = (
    # 1977..1997 are pure HTML chairman's letters
    *((y, f"https://www.berkshirehathaway.com/letters/{y}.html")
      for y in range(1977, 1998)),
    # 1998..1999 use the {year}htm.html variant (the {year}.html page
    # is just an "important note" redirector)
    (1998, "https://www.berkshirehathaway.com/letters/1998htm.html"),
    (1999, "https://www.berkshirehathaway.com/letters/1999htm.html"),
    # 2000..2001 live under the annual-report sub-tree
    (2000, "https://www.berkshirehathaway.com/2000ar/2000letter.html"),
    (2001, "https://www.berkshirehathaway.com/2001ar/2001letter.html"),
    # 2002..2003 letters and 2004..present letters are PDF-only:
    #   https://www.berkshirehathaway.com/letters/{year}ltr.pdf
    # extracting them requires pdfminer.six or pypdf — out of scope
    # for the validation script. Add a follow-up agent or a thin
    # PDF-extract pass when expanding past the HTML window.
    # 1965..1976 letters are not posted publicly on the BH site; they
    # are in the printed annual reports only.
)


_BERKSHIRE_HEADER_RE = re.compile(
    r"^[\s\S]*?(BERKSHIRE\s+HATHAWAY\s+INC\.?)",
    re.IGNORECASE,
)
# Pre-letter table-of-results lead-in used on 2000–2001 pages.
_BERKSHIRE_AR_NOTE_RE = re.compile(
    r"^\s*Note:\s*The following table appears.*?(?=To the Shareholders)",
    re.IGNORECASE | re.DOTALL,
)


def extract_berkshire_text(html: str) -> str:
    """Strip <script>/<style>, take <body>, decode entities.

    Preserves paragraph structure. Does NOT aggressively strip
    whitespace inside paragraphs — the spec calls out that the year
    header and section structure are load-bearing."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    body = soup.body
    if body is None:
        return ""
    # get_text preserves paragraph breaks because the upstream HTML
    # uses <P> tags; we ask bs to insert "\n" at boundaries.
    text = body.get_text("\n", strip=False)
    # Collapse 3+ blank lines but keep paragraph spacing.
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Cut the SEC / boilerplate ahead of "BERKSHIRE HATHAWAY INC." if
    # there is any. The header line itself stays.
    m = _BERKSHIRE_HEADER_RE.search(text)
    if m:
        text = text[m.start(1):]
    # The 2000/2001 AR variant prepends a "Note: The following table
    # appears in the printed Annual Report..." preamble before the
    # actual letter. Skip past it to the salutation.
    text = _BERKSHIRE_AR_NOTE_RE.sub("", text)
    return text.strip()


def collect_berkshire(
    letters: Iterable[tuple[int, str]],
) -> tuple[list[dict], list["SourceStats"]]:
    records: list[dict] = []
    stats: list[SourceStats] = []
    for year, url in letters:
        print(f"[berkshire] {year}  {url}")
        r = http_get(url)
        if r is None:
            stats.append(SourceStats(
                label=f"berkshire/{year}",
                fetched=False,
            ))
            print(f"  FAILED to fetch", file=sys.stderr)
            time.sleep(POLITE_DELAY)
            continue
        body = extract_berkshire_text(r.text)
        if not body or len(body.split()) < 500:
            stats.append(SourceStats(
                label=f"berkshire/{year}",
                fetched=True,
                empty=True,
            ))
            print(f"  empty/skeletal body ({len(body.split())} words), skipping",
                  file=sys.stderr)
            time.sleep(POLITE_DELAY)
            continue

        title = f"Berkshire Hathaway Letter {year}"
        chunks = chunk_prose(body)
        kept, rejected = 0, 0
        for chunk in chunks:
            rec = build_record(
                chunk,
                source="berkshire",
                domain="finance",
                subdomain="annual_letter",
                author="Warren Buffett",
                title=title,
            )
            if not quality_filter(rec):
                rejected += 1
                continue
            records.append(rec)
            kept += 1
        tokens = sum(r["tokens"] for r in records[-kept:]) if kept else 0
        print(f"  -> {kept} chunks kept, {rejected} rejected, ~{tokens:,} tokens")
        stats.append(SourceStats(
            label=f"berkshire/{year}",
            fetched=True,
            kept=kept,
            rejected=rejected,
            tokens=tokens,
        ))
        time.sleep(POLITE_DELAY)
    return records, stats


# ---------------------------------------------------------------------------
# Gutenberg (Adam Smith / Veblen for finance; Vitruvius / Ruskin for design)
# ---------------------------------------------------------------------------

GUTENBERG_TXT_URLS = (
    "https://www.gutenberg.org/files/{gid}/{gid}-0.txt",
    "https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt",
    "https://www.gutenberg.org/files/{gid}/{gid}.txt",
)
GUTENDEX_URL = "https://gutendex.com/books/{gid}"

_GB_START_RE = re.compile(
    r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
    re.IGNORECASE,
)
_GB_END_RE = re.compile(
    r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
    re.IGNORECASE,
)


def verify_gutenberg(gid: int) -> dict | None:
    """Hit gutendex to confirm a Gutenberg ID hasn't been retired or
    renumbered. Returns the parsed JSON record or None."""
    r = http_get(GUTENDEX_URL.format(gid=gid), max_retries=2)
    if r is None:
        return None
    try:
        return r.json()
    except ValueError:
        return None


def fetch_gutenberg_text(gid: int) -> str | None:
    for tmpl in GUTENBERG_TXT_URLS:
        url = tmpl.format(gid=gid)
        r = http_get(url, max_retries=2)
        if r is not None and len(r.text) > 5000:
            return r.text.lstrip("﻿")
    return None


def strip_gutenberg_boilerplate(raw: str) -> str:
    m = _GB_START_RE.search(raw)
    if m:
        raw = raw[m.end():]
    m = _GB_END_RE.search(raw)
    if m:
        raw = raw[: m.start()]
    return raw.strip()


@dataclass(frozen=True)
class GutenbergSource:
    gid: int
    fallback_title: str
    fallback_author: str
    domain: str
    subdomain: str


# Validation Gutenberg picks. The script tries them in order and stops
# after the first success per domain (one per spec instructions).
FINANCE_GUTENBERG_CANDIDATES: tuple[GutenbergSource, ...] = (
    GutenbergSource(
        gid=3300,
        fallback_title="The Wealth of Nations",
        fallback_author="Adam Smith",
        domain="finance",
        subdomain="classic",
    ),
    GutenbergSource(
        gid=833,
        fallback_title="The Theory of the Leisure Class",
        fallback_author="Thorstein Veblen",
        domain="finance",
        subdomain="classic",
    ),
)

DESIGN_GUTENBERG_CANDIDATES: tuple[GutenbergSource, ...] = (
    GutenbergSource(
        gid=20239,
        fallback_title="The Ten Books on Architecture",
        fallback_author="Vitruvius",
        domain="design",
        subdomain="architecture",
    ),
    GutenbergSource(
        gid=35898,
        fallback_title="The Seven Lamps of Architecture",
        fallback_author="John Ruskin",
        domain="design",
        subdomain="architecture",
    ),
)


# --- --full expansions ---------------------------------------------------
#
# Finance: validation already covers 3300 (Smith - Wealth of Nations).
# Add Veblen, Mill, Smith's other two public-domain works. Walras's
# Elements of Pure Economics is not on Gutenberg (verified via
# gutendex search) — documented and skipped. Pugin's Contrasts is
# also not on Gutenberg — documented and skipped.
#
# Verified IDs (all confirmed via gutendex at build time):
#   3300   Smith     - Wealth of Nations
#   833    Veblen    - Theory of the Leisure Class
#   30107  Mill      - Principles of Political Economy (abridged)
#   67363  Smith     - Theory of Moral Sentiments
#   58559  Smith     - Essays of Adam Smith
#
# Design (architecture):
#   20239  Vitruvius - Ten Books on Architecture
#   35898  Ruskin    - Seven Lamps of Architecture
#   30754  Ruskin    - Stones of Venice, Vol 1
#   30755  Ruskin    - Stones of Venice, Vol 2
#   30756  Ruskin    - Stones of Venice, Vol 3

FINANCE_GUTENBERG_FULL: tuple[GutenbergSource, ...] = (
    GutenbergSource(
        gid=3300,
        fallback_title="The Wealth of Nations",
        fallback_author="Adam Smith",
        domain="finance",
        subdomain="classic",
    ),
    GutenbergSource(
        gid=833,
        fallback_title="The Theory of the Leisure Class",
        fallback_author="Thorstein Veblen",
        domain="finance",
        subdomain="classic",
    ),
    GutenbergSource(
        gid=30107,
        fallback_title="Principles of Political Economy",
        fallback_author="John Stuart Mill",
        domain="finance",
        subdomain="classic",
    ),
    GutenbergSource(
        gid=67363,
        fallback_title="The Theory of Moral Sentiments",
        fallback_author="Adam Smith",
        domain="finance",
        subdomain="classic",
    ),
    GutenbergSource(
        gid=58559,
        fallback_title="The Essays of Adam Smith",
        fallback_author="Adam Smith",
        domain="finance",
        subdomain="classic",
    ),
)

DESIGN_GUTENBERG_FULL: tuple[GutenbergSource, ...] = (
    GutenbergSource(
        gid=20239,
        fallback_title="The Ten Books on Architecture",
        fallback_author="Vitruvius",
        domain="design",
        subdomain="architecture",
    ),
    GutenbergSource(
        gid=35898,
        fallback_title="The Seven Lamps of Architecture",
        fallback_author="John Ruskin",
        domain="design",
        subdomain="architecture",
    ),
    GutenbergSource(
        gid=30754,
        fallback_title="The Stones of Venice, Volume 1",
        fallback_author="John Ruskin",
        domain="design",
        subdomain="architecture",
    ),
    GutenbergSource(
        gid=30755,
        fallback_title="The Stones of Venice, Volume 2",
        fallback_author="John Ruskin",
        domain="design",
        subdomain="architecture",
    ),
    GutenbergSource(
        gid=30756,
        fallback_title="The Stones of Venice, Volume 3",
        fallback_author="John Ruskin",
        domain="design",
        subdomain="architecture",
    ),
)


def _collect_one_gutenberg(
    src: GutenbergSource,
) -> tuple[list[dict], "SourceStats"]:
    """Verify, fetch, chunk, filter one Gutenberg source. Returns
    (records, stats). On any failure returns ([], stats with
    fetched=False)."""
    label = f"gutenberg/{src.domain}/{src.gid}"
    print(f"[gutenberg] {src.gid}  ({src.fallback_author} - {src.fallback_title})")
    meta = verify_gutenberg(src.gid)
    if meta is None:
        print(f"  gutendex lookup failed", file=sys.stderr)
        time.sleep(POLITE_DELAY)
        return [], SourceStats(label=label, fetched=False)
    title = meta.get("title") or src.fallback_title
    authors = meta.get("authors") or []
    author = authors[0]["name"] if authors else src.fallback_author
    print(f"  verified: {title} - {author}")
    raw = fetch_gutenberg_text(src.gid)
    if raw is None:
        print(f"  text fetch failed", file=sys.stderr)
        time.sleep(POLITE_DELAY)
        return [], SourceStats(label=label, fetched=False)
    body = strip_gutenberg_boilerplate(raw)
    chunks = chunk_prose(body)
    records: list[dict] = []
    rejected = 0
    for chunk in chunks:
        rec = build_record(
            chunk,
            source="gutenberg",
            domain=src.domain,
            subdomain=src.subdomain,
            author=author,
            title=title,
        )
        if not quality_filter(rec):
            rejected += 1
            continue
        records.append(rec)
    tokens = sum(r["tokens"] for r in records)
    print(f"  -> {len(records)} chunks kept, {rejected} rejected, "
          f"~{tokens:,} tokens")
    time.sleep(POLITE_DELAY)
    return records, SourceStats(
        label=label,
        fetched=True,
        kept=len(records),
        rejected=rejected,
        tokens=tokens,
    )


def collect_first_gutenberg(
    candidates: Iterable[GutenbergSource],
) -> tuple[list[dict], "SourceStats"]:
    """Try candidates in order, return chunks from the first one that
    verifies via Gutendex and fetches text successfully."""
    last_stat = SourceStats(label="gutenberg/none", fetched=False)
    for src in candidates:
        records, stats = _collect_one_gutenberg(src)
        last_stat = stats
        if records:
            return records, stats
    return [], last_stat


def collect_all_gutenberg(
    candidates: Iterable[GutenbergSource],
) -> tuple[list[dict], list["SourceStats"]]:
    """Walk all candidates, keep every one that fetches. Used by
    --full to expand beyond a single text per domain."""
    all_records: list[dict] = []
    all_stats: list[SourceStats] = []
    for src in candidates:
        records, stats = _collect_one_gutenberg(src)
        all_records.extend(records)
        all_stats.append(stats)
    return all_records, all_stats


# ---------------------------------------------------------------------------
# Wikipedia REST (media theory validation corpus)
# ---------------------------------------------------------------------------

WIKIPEDIA_HTML_URL = "https://en.wikipedia.org/api/rest_v1/page/html/{title}"

WIKIPEDIA_MEDIA_TOPICS: tuple[str, ...] = (
    "Cinema_of_the_United_States",
    "Photography",
    "Film_theory",
)

# --full Wikipedia expansion. Validation's 3 are first, then 12 more
# spanning music, painting, sculpture, animation, photojournalism,
# cinematography, modern art, architecture, plus film-theory sub-topics
# (mise-en-scene, auteur, documentary, cinéma vérité, history of
# photography). Target ~15 articles total.
WIKIPEDIA_MEDIA_TOPICS_FULL: tuple[str, ...] = (
    "Cinema_of_the_United_States",
    "Photography",
    "Film_theory",
    "Music_theory",
    "Architecture",
    "Modern_art",
    "Photojournalism",
    "Cinematography",
    "Sculpture",
    "Painting",
    "Animation",
    "Mise-en-scène",
    "Auteur",
    "Documentary_film",
    "Cinéma_vérité",
    "History_of_photography",
)


def extract_wikipedia_text(html: str) -> str:
    """Strip script/style/nav/infobox/reference clutter, keep prose."""
    soup = BeautifulSoup(html, "html.parser")

    # Discard non-prose: navigation, infobox tables, references,
    # navboxes, edit-section links, math/svg, audio, image captions.
    for tag in soup([
        "script", "style", "noscript", "table", "sup",
        "figure", "img", "audio", "video", "svg",
    ]):
        tag.decompose()

    # Class-based: infoboxes, navboxes, references, hatnotes, toc.
    junk_class_re = re.compile(
        r"\b("
        r"infobox|navbox|reference|references|reflist|hatnote|"
        r"mw-editsection|toc|metadata|sistersitebox|noprint|"
        r"thumb|gallery|portal|sidebar|ambox"
        r")\b",
        re.IGNORECASE,
    )
    for tag in soup.find_all(class_=junk_class_re):
        tag.decompose()

    # IDs we never want.
    for bad_id in ("References", "External_links", "Further_reading",
                   "See_also", "Notes"):
        h = soup.find(id=bad_id)
        if h is not None:
            # Drop the heading and everything after — these sections
            # are almost always bibliographic.
            for sibling in list(h.find_all_next()):
                sibling.decompose()
            h.decompose()

    text = soup.get_text("\n", strip=False)
    # Collapse 3+ blank lines, drop empty bracketed citations like "[1]".
    text = re.sub(r"\[\s*\d+\s*\]", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse runs of whitespace within a line.
    text = "\n".join(re.sub(r"[ \t]+", " ", ln).rstrip()
                     for ln in text.splitlines())
    return text.strip()


def collect_wikipedia(
    topics: Iterable[str],
) -> tuple[list[dict], list["SourceStats"]]:
    records: list[dict] = []
    stats: list[SourceStats] = []
    for title in topics:
        url = WIKIPEDIA_HTML_URL.format(title=title)
        print(f"[wikipedia] {title}")
        r = http_get(url)
        if r is None:
            stats.append(SourceStats(label=f"wikipedia/{title}", fetched=False))
            time.sleep(POLITE_DELAY)
            continue
        body = extract_wikipedia_text(r.text)
        if not body or len(body.split()) < 300:
            print(f"  empty/skeletal body ({len(body.split())} words)",
                  file=sys.stderr)
            stats.append(SourceStats(
                label=f"wikipedia/{title}",
                fetched=True,
                empty=True,
            ))
            time.sleep(POLITE_DELAY)
            continue
        chunks = chunk_prose(body)
        kept_records: list[dict] = []
        rejected = 0
        display_title = title.replace("_", " ")
        for chunk in chunks:
            rec = build_record(
                chunk,
                source="wikipedia",
                domain="media",
                subdomain="theory",
                author="Wikipedia contributors",
                title=display_title,
            )
            if not quality_filter(rec):
                rejected += 1
                continue
            kept_records.append(rec)
        tokens = sum(r["tokens"] for r in kept_records)
        records.extend(kept_records)
        print(f"  -> {len(kept_records)} chunks kept, {rejected} rejected, "
              f"~{tokens:,} tokens")
        stats.append(SourceStats(
            label=f"wikipedia/{title}",
            fetched=True,
            kept=len(kept_records),
            rejected=rejected,
            tokens=tokens,
        ))
        time.sleep(POLITE_DELAY)
    return records, stats


# ---------------------------------------------------------------------------
# Stats + driver
# ---------------------------------------------------------------------------

@dataclass
class SourceStats:
    label: str
    fetched: bool = False
    empty: bool = False
    kept: int = 0
    rejected: int = 0
    tokens: int = 0


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = REPO_ROOT / "data" / "curriculum_v2"


def output_path(domain: str, full: bool) -> Path:
    fname = "agent_7_full.jsonl" if full else "agent_7_validation.jsonl"
    return OUTPUT_ROOT / domain / "raw" / fname


def write_jsonl(records: list[dict], path: Path, *, append: bool = False) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False))
            f.write("\n")
    return len(records)


def summarize(label: str, records: list[dict], stats: list[SourceStats]) -> None:
    print()
    print("-" * 72)
    print(f"{label}: {len(records)} records, "
          f"~{sum(r['tokens'] for r in records):,} tokens")
    for s in stats:
        flag = "OK " if s.fetched and not s.empty else (
            "EMPTY" if s.empty else "FAIL"
        )
        print(f"  [{flag}] {s.label:<32} kept={s.kept:<4} "
              f"rejected={s.rejected:<3} tokens~{s.tokens:,}")


def run_validation() -> None:
    print("=" * 72)
    print("Agent 7 — validation run (design, finance, media)")
    print("=" * 72)

    # --- finance: Berkshire (priority) + 1 Gutenberg classic
    print("\n[FINANCE]")
    berk_records, berk_stats = collect_berkshire(BERKSHIRE_HTML_LETTERS_VALIDATION)
    fin_gut_records, fin_gut_stats = collect_first_gutenberg(
        FINANCE_GUTENBERG_CANDIDATES
    )
    finance_records = berk_records + fin_gut_records
    finance_stats = berk_stats + [fin_gut_stats]
    n = write_jsonl(finance_records, output_path("finance", full=False))
    summarize(f"finance ({n} written)", finance_records, finance_stats)

    # --- design: 1 Gutenberg architecture text
    print("\n[DESIGN]")
    design_records, design_stat = collect_first_gutenberg(
        DESIGN_GUTENBERG_CANDIDATES
    )
    n = write_jsonl(design_records, output_path("design", full=False))
    summarize(f"design ({n} written)", design_records, [design_stat])

    # --- media: Wikipedia media-theory pages
    print("\n[MEDIA]")
    media_records, media_stats = collect_wikipedia(WIKIPEDIA_MEDIA_TOPICS)
    n = write_jsonl(media_records, output_path("media", full=False))
    summarize(f"media ({n} written)", media_records, media_stats)

    # Final report
    print()
    print("=" * 72)
    print("AGENT 7 — VALIDATION SUMMARY")
    print("=" * 72)
    total = len(finance_records) + len(design_records) + len(media_records)
    total_tokens = (
        sum(r["tokens"] for r in finance_records)
        + sum(r["tokens"] for r in design_records)
        + sum(r["tokens"] for r in media_records)
    )
    print(f"finance : {len(finance_records):>4} records, "
          f"~{sum(r['tokens'] for r in finance_records):,} tokens "
          f"-> {output_path('finance', False)}")
    print(f"design  : {len(design_records):>4} records, "
          f"~{sum(r['tokens'] for r in design_records):,} tokens "
          f"-> {output_path('design', False)}")
    print(f"media   : {len(media_records):>4} records, "
          f"~{sum(r['tokens'] for r in media_records):,} tokens "
          f"-> {output_path('media', False)}")
    print(f"TOTAL   : {total} records, ~{total_tokens:,} tokens")

    # Berkshire roll-up — the priority anchor, called out explicitly
    print()
    print("Berkshire anchor:")
    for s in berk_stats:
        flag = "OK" if s.fetched and not s.empty else (
            "EMPTY" if s.empty else "FAIL"
        )
        print(f"  [{flag}] {s.label:<24} kept={s.kept:<4} tokens~{s.tokens:,}")


def run_full() -> None:
    """Full expansion pass. APPENDS records to per-domain
    agent_7_full.jsonl files. Sources:

      finance  Berkshire HTML letters 1977..2001 (priority anchor) +
               Gutenberg classics (Smith x3, Veblen, Mill).
      design   Gutenberg architecture (Vitruvius, Ruskin Seven Lamps,
               Ruskin Stones of Venice vols 1-3).
      media    Wikipedia REST API pages on media-theory topics,
               ~15 articles.

    Out of scope, documented and skipped:
      - PDF Berkshire letters (2002..present) — need pdfminer.six.
      - Pre-1977 Berkshire letters — not posted publicly.
      - Walras's Elements of Pure Economics — not on Gutenberg
        (verified via gutendex search).
      - Pugin's Contrasts — not on Gutenberg (verified)."""
    print("=" * 72)
    print("Agent 7 — --full run (design, finance, media)")
    print("=" * 72)

    # --- FINANCE -----------------------------------------------------
    print("\n[FINANCE / berkshire — priority anchor]")
    # Dedupe by year (the sketch tuple has the 1998/1999 htm variants
    # and the 2000/2001 AR variants alongside the 1977..1997 range —
    # but no overlapping years, so a dict by year is just defensive).
    berk_letters = list({y: url for y, url in
                         BERKSHIRE_HTML_LETTERS_FULL_SKETCH}.items())
    berk_letters.sort()  # oldest first
    berk_records, berk_stats = collect_berkshire(berk_letters)

    print("\n[FINANCE / gutenberg classics]")
    fin_gut_records, fin_gut_stats = collect_all_gutenberg(
        FINANCE_GUTENBERG_FULL
    )

    finance_records = berk_records + fin_gut_records
    finance_stats = berk_stats + fin_gut_stats
    n_fin = write_jsonl(
        finance_records, output_path("finance", full=True), append=True
    )
    summarize(f"finance ({n_fin} appended)", finance_records, finance_stats)

    # --- DESIGN ------------------------------------------------------
    print("\n[DESIGN / gutenberg architecture]")
    design_records, design_stats = collect_all_gutenberg(
        DESIGN_GUTENBERG_FULL
    )
    n_des = write_jsonl(
        design_records, output_path("design", full=True), append=True
    )
    summarize(f"design ({n_des} appended)", design_records, design_stats)

    # --- MEDIA -------------------------------------------------------
    print("\n[MEDIA / wikipedia]")
    media_records, media_stats = collect_wikipedia(
        WIKIPEDIA_MEDIA_TOPICS_FULL
    )
    n_med = write_jsonl(
        media_records, output_path("media", full=True), append=True
    )
    summarize(f"media ({n_med} appended)", media_records, media_stats)

    # --- Final report ------------------------------------------------
    print()
    print("=" * 72)
    print("AGENT 7 — FULL SUMMARY")
    print("=" * 72)
    total = len(finance_records) + len(design_records) + len(media_records)
    fin_tokens = sum(r["tokens"] for r in finance_records)
    des_tokens = sum(r["tokens"] for r in design_records)
    med_tokens = sum(r["tokens"] for r in media_records)
    total_tokens = fin_tokens + des_tokens + med_tokens
    print(f"finance : {len(finance_records):>4} records, "
          f"~{fin_tokens:,} tokens "
          f"-> {output_path('finance', True)}")
    print(f"design  : {len(design_records):>4} records, "
          f"~{des_tokens:,} tokens "
          f"-> {output_path('design', True)}")
    print(f"media   : {len(media_records):>4} records, "
          f"~{med_tokens:,} tokens "
          f"-> {output_path('media', True)}")
    print(f"TOTAL   : {total} records, ~{total_tokens:,} tokens")

    # Berkshire roll-up — the priority anchor, called out explicitly.
    print()
    print("Berkshire anchor (HTML letters 1977..2001):")
    ok_years = []
    fail_years = []
    for s in berk_stats:
        flag = "OK" if s.fetched and not s.empty else (
            "EMPTY" if s.empty else "FAIL"
        )
        year = s.label.rsplit("/", 1)[-1]
        if flag == "OK":
            ok_years.append(year)
        else:
            fail_years.append(year)
        print(f"  [{flag}] {s.label:<24} kept={s.kept:<4} tokens~{s.tokens:,}")
    print(f"  -- OK years ({len(ok_years)}): {', '.join(ok_years)}")
    if fail_years:
        print(f"  -- FAILED/EMPTY years ({len(fail_years)}): "
              f"{', '.join(fail_years)}")

    # Out-of-scope summary
    print()
    print("Documented and skipped (out of scope for --full):")
    print("  - Berkshire 2002..present (PDF-only, needs pdfminer.six)")
    print("  - Berkshire 1965..1976 (not posted on BH site)")
    print("  - Walras, Elements of Pure Economics (not on Gutenberg)")
    print("  - Pugin, Contrasts (not on Gutenberg)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--full",
        action="store_true",
        help="Run the full expansion pass (Berkshire HTML letters + "
             "expanded Gutenberg + expanded Wikipedia).",
    )
    args = ap.parse_args()
    if args.full:
        run_full()
    else:
        run_validation()


if __name__ == "__main__":
    main()
