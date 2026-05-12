#!/usr/bin/env python3
"""
Agent 8 — World Knowledge collector for the v2.0 curriculum.

Covers three sibling domains: ``history``, ``culture``, ``science_general``.
The spec for this agent lists Wikipedia Featured/Good Articles (highest
priority), the public-domain Britannica 11th edition, classical history
texts on Project Gutenberg, and biographies/memoirs.

Validation mode (default, no args):
    * Wikipedia Featured Articles via the REST HTML endpoint:
        - Roman_Empire        -> domain=history,         subdomain=roman_empire
        - World_War_II        -> domain=history,         subdomain=world_war_ii
        - Mathematics         -> domain=science_general, subdomain=mathematics
        - Albert_Einstein     -> domain=science_general, subdomain=albert_einstein
        - History_of_China    -> domain=history,         subdomain=history_of_china
    * Gutenberg classical history (IDs verified via gutendex):
        - 7142 Thucydides, History of the Peloponnesian War
        - 2707 Herodotus,  Histories (Vol. 1)
        - 731  Gibbon,     Decline and Fall (Vol. 1)
    * Gutenberg biography/memoir:
        - 148  Franklin,   Autobiography
        - 23   Douglass,   Narrative of the Life

    Writes three files (one per domain):
        data/curriculum_v2/history/raw/agent_8_validation.jsonl
        data/curriculum_v2/culture/raw/agent_8_validation.jsonl
        data/curriculum_v2/science_general/raw/agent_8_validation.jsonl

Full mode (--full):
    Sketch only. Lists the in-scope follow-up sources (the rest of the
    Wikipedia FA + GA set, Britannica 1911 scans on archive.org, the full
    classical-history Gutenberg list, and the rest of the biography list)
    as URLs and notes. The real --full sweep is deferred to a later wave.

Output schema (per spec, one JSON object per line):
    text, source, domain, subdomain, quality_score, dialogue,
    author, title, tokens, language

Libraries: requests, beautifulsoup4. Install with
    pip install requests beautifulsoup4 --break-system-packages -q
if missing.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

try:
    import requests
except ImportError:  # pragma: no cover
    sys.stderr.write(
        "requests not installed. Run: "
        "pip install requests --break-system-packages -q\n"
    )
    sys.exit(1)

try:
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover
    sys.stderr.write(
        "beautifulsoup4 not installed. Run: "
        "pip install beautifulsoup4 --break-system-packages -q\n"
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data" / "curriculum_v2"

VALIDATION_FILENAME = "agent_8_validation.jsonl"
FULL_FILENAME = "agent_8_full.jsonl"

DOMAINS = ("history", "culture", "science_general")


def output_path(domain: str, *, full: bool) -> Path:
    name = FULL_FILENAME if full else VALIDATION_FILENAME
    return DATA_ROOT / domain / "raw" / name


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

# Wikipedia explicitly asks for a contact in the User-Agent.
USER_AGENT = (
    "baby-mind-curriculum-collector/2.0 "
    "(research; hello.postworklab@gmail.com)"
)
HTTP_TIMEOUT = 60
POLITE_DELAY_S = 0.75  # between requests to the same host


# ---------------------------------------------------------------------------
# Source tables
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WikiArticle:
    slug: str           # URL slug
    domain: str         # history | culture | science_general
    subdomain: str


@dataclass(frozen=True)
class GutenbergBook:
    gid: int
    title: str
    author: str
    domain: str
    subdomain: str


# Five known stable Featured Articles. Domain assigned per-article per spec.
WIKI_VALIDATION: tuple[WikiArticle, ...] = (
    WikiArticle("Roman_Empire",     "history",         "roman_empire"),
    WikiArticle("World_War_II",     "history",         "world_war_ii"),
    WikiArticle("Mathematics",      "science_general", "mathematics"),
    WikiArticle("Albert_Einstein",  "science_general", "albert_einstein"),
    WikiArticle("History_of_China", "history",         "history_of_china"),
)

# Gutenberg classical history. IDs verified against gutendex 2026-05-12.
GUTENBERG_HISTORY_VALIDATION: tuple[GutenbergBook, ...] = (
    GutenbergBook(
        gid=7142,
        title="The History of the Peloponnesian War",
        author="Thucydides",
        domain="history",
        subdomain="ancient_history",
    ),
    GutenbergBook(
        gid=2707,
        title="The History of Herodotus — Volume 1",
        author="Herodotus",
        domain="history",
        subdomain="ancient_history",
    ),
    GutenbergBook(
        gid=731,
        title="History of the Decline and Fall of the Roman Empire — Volume 1",
        author="Edward Gibbon",
        domain="history",
        subdomain="roman_history",
    ),
)

# Biography/memoir. Single record needed per spec; we try both for resilience.
GUTENBERG_BIOGRAPHY_VALIDATION: tuple[GutenbergBook, ...] = (
    GutenbergBook(
        gid=148,
        title="The Autobiography of Benjamin Franklin",
        author="Benjamin Franklin",
        domain="culture",
        subdomain="biography",
    ),
    GutenbergBook(
        gid=23,
        title="Narrative of the Life of Frederick Douglass, an American Slave",
        author="Frederick Douglass",
        domain="culture",
        subdomain="biography",
    ),
)


# ---------------------------------------------------------------------------
# Gutendex verification
# ---------------------------------------------------------------------------

def gutendex_metadata(gid: int) -> dict | None:
    """Look up Gutenberg metadata via gutendex. Returns dict or None.

    Wave-1/2 lesson: some IDs in the spec are stale; verify before fetch.
    """
    url = f"https://gutendex.com/books/{gid}"
    try:
        r = requests.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=HTTP_TIMEOUT,
            allow_redirects=True,
        )
    except requests.RequestException as exc:
        print(f"  ! gutendex lookup failed (id={gid}): {exc}", file=sys.stderr)
        return None
    if r.status_code != 200:
        print(
            f"  ! gutendex id={gid} -> HTTP {r.status_code}",
            file=sys.stderr,
        )
        return None
    try:
        return r.json()
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Gutenberg fetch + boilerplate strip
# ---------------------------------------------------------------------------

GUT_START_PAT = re.compile(
    r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK[^*]*\*\*\*",
    re.IGNORECASE,
)
GUT_END_PAT = re.compile(
    r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK[^*]*\*\*\*",
    re.IGNORECASE,
)


def fetch_gutenberg_text(gid: int) -> str | None:
    """Try the known plain-text URL patterns until one works."""
    urls = [
        f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt",
        f"https://www.gutenberg.org/files/{gid}/{gid}-0.txt",
        f"https://www.gutenberg.org/files/{gid}/{gid}.txt",
    ]
    for url in urls:
        try:
            r = requests.get(
                url,
                headers={"User-Agent": USER_AGENT},
                timeout=HTTP_TIMEOUT,
            )
        except requests.RequestException as exc:
            print(f"  ! gutenberg fetch failed ({url}): {exc}", file=sys.stderr)
            continue
        if r.status_code == 200 and r.text and len(r.text) > 5000:
            r.encoding = r.apparent_encoding or "utf-8"
            return r.text
    return None


def strip_gutenberg_boilerplate(raw: str) -> str:
    start = GUT_START_PAT.search(raw)
    end = GUT_END_PAT.search(raw)
    s = start.end() if start else 0
    e = end.start() if end else len(raw)
    body = raw[s:e].strip()

    # Strip "Produced by ..." / "Transcribed by ..." leader if present.
    lines = body.splitlines()
    while lines and (
        lines[0].strip() == ""
        or lines[0].strip().lower().startswith(
            ("produced by", "transcribed", "this etext", "e-text prepared")
        )
    ):
        lines.pop(0)
    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# Wikipedia fetch + strip
# ---------------------------------------------------------------------------

WIKI_REST_URL = "https://en.wikipedia.org/api/rest_v1/page/html/{slug}"

# Sections we cut from the end of the article (everything from the first
# matching <h2> onward is dropped).
WIKI_STOP_SECTION_IDS = (
    "See_also",
    "Notes",
    "References",
    "Bibliography",
    "Further_reading",
    "External_links",
    "Citations",
    "Sources",
)


def fetch_wikipedia_html(slug: str) -> str | None:
    url = WIKI_REST_URL.format(slug=slug)
    try:
        r = requests.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=HTTP_TIMEOUT,
        )
    except requests.RequestException as exc:
        print(f"  ! wikipedia fetch failed ({slug}): {exc}", file=sys.stderr)
        return None
    if r.status_code != 200:
        print(f"  ! wikipedia {slug} -> HTTP {r.status_code}", file=sys.stderr)
        return None
    if not r.text or len(r.text) < 5000:
        print(f"  ! wikipedia {slug} -> response too short", file=sys.stderr)
        return None
    return r.text


def extract_wikipedia_body(html: str) -> str:
    """Pull prose body text from a Wikipedia REST HTML page.

    Strips infobox tables, reference superscripts, navboxes, edit-section
    markers, and everything from the first reference/see-also section
    onward. Joins remaining <p> elements with blank lines so the chunker
    can split on paragraphs.
    """
    soup = BeautifulSoup(html, "html.parser")

    # 1) Drop the first appearance of a stop-section <h2> and everything
    #    after it within the article body. (REST returns a flat sequence
    #    of section/h2/p/... elements at the top level of <body>.)
    body = soup.body if soup.body else soup
    stop_node = None
    for h2 in body.find_all("h2"):
        if h2.get("id") in WIKI_STOP_SECTION_IDS:
            stop_node = h2
            break
    if stop_node is not None:
        # Remove the stop heading and every following sibling at the same
        # level. REST HTML uses <section> wrappers in newer renderings; if
        # the h2 sits inside a <section>, climb to it.
        anchor = stop_node
        if anchor.parent and anchor.parent.name == "section":
            anchor = anchor.parent
        cursor = anchor
        while cursor is not None:
            nxt = cursor.find_next_sibling()
            cursor.decompose()
            cursor = nxt

    # 2) Strip non-prose elements that may be sprinkled through the article.
    selectors_to_kill = (
        "table.infobox",
        "table.sidebar",
        "table.navbox",
        "table.metadata",
        "table.ambox",
        "table.vertical-navbox",
        "sup.reference",
        "sup.noprint",
        "ol.references",
        "div.reflist",
        "div.thumb",
        "div.hatnote",
        "div.navbox",
        "div.mw-editsection",
        "div.mw-cite-backlink",
        "span.mw-editsection",
        "figure",
        "style",
        "script",
    )
    for sel in selectors_to_kill:
        for el in body.select(sel):
            el.decompose()

    # 3) Collect prose paragraphs. Keep <p> text only; that's the article
    #    body. Skip empty or boilerplate paragraphs.
    paragraphs: list[str] = []
    for p in body.find_all("p"):
        text = p.get_text(" ", strip=True)
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            continue
        if len(text.split()) < 10:
            # Wikipedia leaves stub "p" elements; skip them.
            continue
        paragraphs.append(text)

    return "\n\n".join(paragraphs)


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
    """Group paragraphs into 500–2000-word chunks (target ~1200)."""
    paragraphs = split_paragraphs(text)
    if not paragraphs:
        # No paragraph breaks at all — fall back to sentence-aware split
        # so single-blob Wikipedia/Gutenberg input still produces records.
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
        paragraphs = [s.strip() for s in sentences if s.strip()]

    chunks: list[str] = []
    buf: list[str] = []
    buf_words = 0

    for p in paragraphs:
        w = len(p.split())
        # Single paragraph bigger than the cap: hard-split by words.
        if w > MAX_CHUNK_WORDS:
            if buf:
                chunks.append("\n\n".join(buf))
                buf, buf_words = [], 0
            words = p.split()
            for i in range(0, len(words), TARGET_CHUNK_WORDS):
                piece = " ".join(words[i : i + TARGET_CHUNK_WORDS])
                if len(piece.split()) >= MIN_CHUNK_WORDS:
                    chunks.append(piece)
                else:
                    # Tail too small — fold into the next buffer.
                    buf.append(piece)
                    buf_words += len(piece.split())
            continue

        if buf_words + w > MAX_CHUNK_WORDS:
            chunks.append("\n\n".join(buf))
            buf, buf_words = [p], w
        else:
            buf.append(p)
            buf_words += w
            if buf_words >= TARGET_CHUNK_WORDS:
                chunks.append("\n\n".join(buf))
                buf, buf_words = [], 0

    if buf and buf_words >= MIN_CHUNK_WORDS:
        chunks.append("\n\n".join(buf))
    elif buf and chunks:
        # Short tail — fold into the previous chunk.
        chunks[-1] = chunks[-1] + "\n\n" + "\n\n".join(buf)

    return chunks


# ---------------------------------------------------------------------------
# Quality
# ---------------------------------------------------------------------------

REASONING_WORDS = frozenset({
    "therefore", "because", "however", "although",
    "implies", "suggests", "shows", "demonstrates",
    "furthermore", "nevertheless", "consequently",
    "thus", "hence", "while", "whereas", "indicates",
})


def compute_quality_score(text: str) -> float:
    """Shared generic quality score (mirrors COLLECTION_SPEC Agent 9).

    Length + unique-word ratio + terminal punctuation + reasoning vocab +
    low-diversity penalty. No domain-specific keyword filter — Agent 8 has
    no per-domain word list defined in the spec.
    """
    words = text.split()
    n = len(words)
    if n == 0:
        return 0.0

    score = 0.5

    # Length sweet-spot
    if 200 <= n <= 2000:
        score += 0.10
    elif n > 2000:
        score += 0.05

    # Vocabulary richness (type-token ratio)
    unique_ratio = len(set(w.lower() for w in words)) / n
    score += min(unique_ratio * 0.20, 0.20)

    # Terminal punctuation present (sentence-ish ending)
    stripped = text.rstrip()
    if stripped and stripped[-1] in ".!?":
        score += 0.05

    # Sentence structure (multiple sentence terminators)
    terms = text.count(".") + text.count("?") + text.count("!")
    if terms >= 5:
        score += 0.05

    # Reasoning vocabulary bonus
    lower = {w.strip(".,;:!?\"'()").lower() for w in words}
    reason_hits = len(REASONING_WORDS & lower)
    score += min(reason_hits * 0.02, 0.10)

    # Low-diversity penalty (lists, repeated rote text)
    if unique_ratio < 0.40:
        score -= 0.20

    return round(max(0.0, min(1.0, score)), 4)


def quality_filter(record: dict) -> bool:
    """Generic length-based gate per spec: >= 200 words."""
    text = record["text"]
    if len(text.split()) < 200:
        return False
    return True


# ---------------------------------------------------------------------------
# Record assembly
# ---------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """Rough word-based proxy. Agent 9 retokenizes with the real BPE."""
    return int(len(text.split()) * 1.3)


def make_record(
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


def write_records(records: list[dict], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@dataclass
class SourceStats:
    label: str
    fetched: bool
    chunks_written: int = 0
    chunks_rejected: int = 0
    tokens: int = 0
    note: str = ""


@dataclass
class CollectionStats:
    sources: list[SourceStats] = field(default_factory=list)
    by_domain_records: dict[str, int] = field(default_factory=dict)
    by_domain_tokens: dict[str, int] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Per-source collectors
# ---------------------------------------------------------------------------

def collect_wikipedia(
    articles: Iterable[WikiArticle],
    stats: CollectionStats,
) -> Iterator[dict]:
    items = list(articles)
    for i, art in enumerate(items):
        print(f"[wikipedia] {art.slug}  ({art.domain})")
        html = fetch_wikipedia_html(art.slug)
        if html is None:
            stats.failures.append(f"wikipedia:{art.slug} — fetch failed")
            stats.sources.append(
                SourceStats(label=f"wiki:{art.slug}", fetched=False, note="fetch failed")
            )
            if i < len(items) - 1:
                time.sleep(POLITE_DELAY_S)
            continue

        body = extract_wikipedia_body(html)
        words = len(body.split())
        if words < 500:
            stats.failures.append(
                f"wikipedia:{art.slug} — body too short ({words} words)"
            )
            stats.sources.append(
                SourceStats(
                    label=f"wiki:{art.slug}",
                    fetched=False,
                    note=f"body too short ({words} w)",
                )
            )
            if i < len(items) - 1:
                time.sleep(POLITE_DELAY_S)
            continue

        chunks = chunk_text(body)
        src_stats = SourceStats(label=f"wiki:{art.slug}", fetched=True)
        print(f"  -> {len(chunks)} chunks from {words} body words")
        title = art.slug.replace("_", " ")

        for chunk in chunks:
            rec = make_record(
                chunk,
                source="wikipedia_fa",
                domain=art.domain,
                subdomain=art.slug.lower(),
                author="Wikipedia contributors",
                title=title,
            )
            if quality_filter(rec):
                src_stats.chunks_written += 1
                src_stats.tokens += rec["tokens"]
                yield rec
            else:
                src_stats.chunks_rejected += 1
        stats.sources.append(src_stats)

        if i < len(items) - 1:
            time.sleep(POLITE_DELAY_S)


def collect_gutenberg(
    books: Iterable[GutenbergBook],
    stats: CollectionStats,
    *,
    stop_after_first_success: bool = False,
) -> Iterator[dict]:
    """Fetch Gutenberg books. Each ID is verified via gutendex first.

    When stop_after_first_success is True the loop ends after the first
    book that produces records — useful for "fetch 1 biography" semantics.
    """
    items = list(books)
    successes = 0
    for i, book in enumerate(items):
        print(f"[gutenberg] {book.gid} — {book.author}: {book.title}")
        meta = gutendex_metadata(book.gid)
        if meta is None:
            stats.failures.append(
                f"gutenberg:{book.gid} ({book.title}) — gutendex lookup failed"
            )
            stats.sources.append(
                SourceStats(
                    label=f"gut:{book.gid}",
                    fetched=False,
                    note="gutendex lookup failed",
                )
            )
            if i < len(items) - 1:
                time.sleep(POLITE_DELAY_S)
            continue

        # Light sanity check: the title we have should look right.
        gx_title = meta.get("title", "")
        if gx_title:
            print(f"  gutendex says: {gx_title}")

        raw = fetch_gutenberg_text(book.gid)
        if raw is None:
            stats.failures.append(
                f"gutenberg:{book.gid} ({book.title}) — fetch failed"
            )
            stats.sources.append(
                SourceStats(label=f"gut:{book.gid}", fetched=False, note="fetch failed")
            )
            if i < len(items) - 1:
                time.sleep(POLITE_DELAY_S)
            continue

        body = strip_gutenberg_boilerplate(raw)
        body_words = len(body.split())
        if body_words < 500:
            stats.failures.append(
                f"gutenberg:{book.gid} ({book.title}) — body too short ({body_words} w)"
            )
            stats.sources.append(
                SourceStats(
                    label=f"gut:{book.gid}",
                    fetched=False,
                    note=f"body too short ({body_words} w)",
                )
            )
            if i < len(items) - 1:
                time.sleep(POLITE_DELAY_S)
            continue

        chunks = chunk_text(body)
        print(f"  -> {len(chunks)} chunks from {body_words} body words")
        src_stats = SourceStats(label=f"gut:{book.gid}", fetched=True)

        for chunk in chunks:
            rec = make_record(
                chunk,
                source="gutenberg",
                domain=book.domain,
                subdomain=book.subdomain,
                author=book.author,
                title=book.title,
            )
            if quality_filter(rec):
                src_stats.chunks_written += 1
                src_stats.tokens += rec["tokens"]
                yield rec
            else:
                src_stats.chunks_rejected += 1
        stats.sources.append(src_stats)

        if src_stats.chunks_written > 0:
            successes += 1
            if stop_after_first_success:
                break

        if i < len(items) - 1:
            time.sleep(POLITE_DELAY_S)


# ---------------------------------------------------------------------------
# Validation driver
# ---------------------------------------------------------------------------

def run_validation() -> CollectionStats:
    stats = CollectionStats()

    accepted: list[dict] = []

    # 1) Wikipedia Featured Articles (highest priority per spec)
    accepted.extend(collect_wikipedia(WIKI_VALIDATION, stats))

    # 2) Gutenberg classical history (target: 2 successful fetches)
    accepted.extend(collect_gutenberg(GUTENBERG_HISTORY_VALIDATION, stats))

    # 3) Gutenberg biography/memoir (target: 1 successful fetch)
    accepted.extend(
        collect_gutenberg(
            GUTENBERG_BIOGRAPHY_VALIDATION,
            stats,
            stop_after_first_success=True,
        )
    )

    # Split by domain and write three files.
    by_domain: dict[str, list[dict]] = {d: [] for d in DOMAINS}
    for rec in accepted:
        d = rec["domain"]
        if d not in by_domain:
            # Defensive: never silently drop. Surface as failure.
            stats.failures.append(f"unexpected domain emitted: {d}")
            continue
        by_domain[d].append(rec)

    for domain in DOMAINS:
        records = by_domain[domain]
        path = output_path(domain, full=False)
        n = write_records(records, path)
        stats.by_domain_records[domain] = n
        stats.by_domain_tokens[domain] = sum(r["tokens"] for r in records)
        print(f"[write] {path} -> {n} records, "
              f"~{stats.by_domain_tokens[domain]:,} tokens")

    return stats


# ---------------------------------------------------------------------------
# Full mode — SKETCH ONLY (per wave 1+2 convention)
# ---------------------------------------------------------------------------

def run_full() -> CollectionStats:
    """SKETCH. This function does not perform a real --full sweep.

    The follow-up wave should expand the source tables below into actual
    fetch loops. For now we just enumerate the URLs/IDs that would be
    pulled, write an empty per-domain output file so downstream code
    doesn't blow up, and return a stats object reflecting that nothing
    was collected.
    """
    print("=== agent_8 --full SKETCH ONLY ===")
    print("This run intentionally collects nothing.")
    print("It enumerates the planned follow-up sources for a later wave.")
    print()

    # ---- Wikipedia FA + GA (full set) ----------------------------------
    # The spec calls for the curated FA + GA list (~50K articles) filtered
    # to history / biography / science / mathematics / philosophy / arts
    # / technology / geography. The full sweep would:
    #   1. Pull https://en.wikipedia.org/wiki/Wikipedia:Featured_articles
    #      and follow the topic-category links to enumerate FA slugs.
    #   2. Pull https://en.wikipedia.org/wiki/Wikipedia:Good_articles
    #      similarly for GA slugs.
    #   3. For each slug, fetch via the REST endpoint already used here,
    #      reuse extract_wikipedia_body / chunk_text, and emit records.
    # NOT downloading the full enwiki XML dump (multi-GB; out of scope
    # per spec).
    wikipedia_planned = [
        "https://en.wikipedia.org/wiki/Wikipedia:Featured_articles",
        "https://en.wikipedia.org/wiki/Wikipedia:Good_articles",
        "https://en.wikipedia.org/api/rest_v1/page/html/<slug>",
    ]

    # ---- Britannica 1911 (public domain) -------------------------------
    # archive.org has full scans; quality varies because OCR. A real
    # --full pass would prefer the Project Gutenberg subset where
    # available (cleaner text) and fall back to archive.org plain-text
    # exports for missing entries.
    britannica_planned = [
        "https://archive.org/details/EncyclopaediaBritannica1911HQ",
        # Project Gutenberg has many EB1911 volumes; e.g. id 35753.
        "https://gutendex.com/books/?search=Encyclopaedia+Britannica",
    ]

    # ---- Gutenberg classical history (rest of the spec list) ----------
    # IDs to verify via gutendex before fetch (lesson from waves 1+2).
    gutenberg_history_planned = {
        # Already in validation: 7142 Thucydides, 2707 Herodotus,
        # 731 Gibbon Vol 1.
        # Livy — History of Rome (selected).
        # Tacitus — Annals (16904), Germania.
        # Carlyle — The French Revolution.
        # Macaulay — History of England (selected).
        # Prescott — Conquest of Mexico, Conquest of Peru.
        # Parkman — Montcalm and Wolfe.
        # Wells — The Outline of History.
        # Plutarch — Parallel Lives (full, multiple volumes: 14033, 14140, …).
        "tacitus_annals": 16904,
        "plutarch_lives_vol1": 674,
        "plutarch_lives_vol2": 14114,
        "plutarch_lives_vol3": 14140,
        "wells_outline_of_history": 45368,
    }

    # ---- Biography / memoir (rest of the spec list) -------------------
    gutenberg_biography_planned = {
        # Already in validation: 148 Franklin, 23 Douglass.
        # Autobiography of Charles Darwin.
        # Harriet Martineau — Autobiography.
        # John Stuart Mill — Autobiography.
        # Bertrand Russell — Autobiography Vol 1.
        # Gandhi — Autobiography.
        "darwin_autobiography": 2087,
        "mill_autobiography": 10378,
        "gandhi_autobiography": 8089,
    }

    # The sketch writes empty placeholder files so the downstream Agent 9
    # tooling can still glob the directory without a KeyError.
    stats = CollectionStats()
    for domain in DOMAINS:
        path = output_path(domain, full=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
        stats.by_domain_records[domain] = 0
        stats.by_domain_tokens[domain] = 0
        print(f"[write] {path} -> 0 records (sketch placeholder)")

    print()
    print("Planned Wikipedia entry points:")
    for u in wikipedia_planned:
        print(f"  - {u}")
    print("Planned Britannica entry points:")
    for u in britannica_planned:
        print(f"  - {u}")
    print("Planned Gutenberg history IDs:")
    for k, v in gutenberg_history_planned.items():
        print(f"  - {v} ({k})")
    print("Planned Gutenberg biography IDs:")
    for k, v in gutenberg_biography_planned.items():
        print(f"  - {v} ({k})")
    print()
    print("(NotImplementedError intentionally NOT raised — per wave 1+2 lesson,")
    print(" --full is a sketch with URL lists, not a hard failure.)")

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_summary(stats: CollectionStats, *, full: bool) -> None:
    print()
    print("=== summary ===")
    total_records = sum(stats.by_domain_records.values())
    total_tokens = sum(stats.by_domain_tokens.values())
    for domain in DOMAINS:
        n = stats.by_domain_records.get(domain, 0)
        t = stats.by_domain_tokens.get(domain, 0)
        print(f"  {domain:<16} {n:>4} records  ~{t:>10,} tokens")
    print(f"  {'TOTAL':<16} {total_records:>4} records  ~{total_tokens:>10,} tokens")
    print()

    if not full and stats.sources:
        print("per-source:")
        for s in stats.sources:
            tag = "OK  " if s.fetched else "FAIL"
            note = f" ({s.note})" if s.note else ""
            print(
                f"  [{tag}] {s.label:<32} "
                f"written={s.chunks_written:<4} "
                f"rejected={s.chunks_rejected:<3} "
                f"tokens~{s.tokens:>8,}{note}"
            )

    if stats.failures:
        print()
        print(f"failures ({len(stats.failures)}):")
        for f in stats.failures:
            print(f"  - {f}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Agent 8 — World Knowledge curriculum collector (v2.0).",
    )
    ap.add_argument(
        "--full",
        action="store_true",
        help=(
            "Run the full collection sketch. The default (no --full) runs "
            "the small validation sweep over 5 Wikipedia FAs + 3 classical "
            "history books + 1 biography. --full is a SKETCH with URL "
            "lists only; the real full sweep is deferred to a later wave."
        ),
    )
    args = ap.parse_args(argv)

    mode = "FULL (sketch)" if args.full else "VALIDATION"
    print(f"=== agent_8_world_knowledge ({mode}) ===")
    for domain in DOMAINS:
        print(f"  output: {output_path(domain, full=args.full)}")
    print()

    if args.full:
        stats = run_full()
    else:
        stats = run_validation()

    _print_summary(stats, full=args.full)

    # Non-zero exit only on total failure in validation mode.
    if not args.full and sum(stats.by_domain_records.values()) == 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
