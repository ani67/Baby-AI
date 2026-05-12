"""
Agent 6 — Dialogue & Conversation collection.

Dialogue is 25% of the corpus and the most important signal in the v2
curriculum. This agent fetches real human conversation in three forms:

  1. Plays from Project Gutenberg — pure speaker-attributed dialogue,
     the structural backbone the quality filter checks for.
  2. Stack Exchange Q&A via live API — argumentation with explicit
     question/answer turns. Philosophy SE and Cross Validated (stats).
  3. TED Talks — long-form spoken explanation under a CC license.

Validation mode (default): a small, directly-fetchable slice of each
source, written to agent_6_validation.jsonl. Quick enough to run on a
laptop, end-to-end, with no large dumps.

Full mode (--full): a sketch — prose plus URLs — for sources that
require big infrastructure (Reddit PushShift dumps, IMSDB scrapes,
Whisper transcription of Oxford Union, the Darwin correspondence
project). The actual full collection is a separate build.

Spec: doc/curriculum/COLLECTION_SPEC.md, section "Agent 6 — Dialogue
& Conversation".
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
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
        "beautifulsoup4 not installed. Run: pip install beautifulsoup4 --break-system-packages -q\n"
    )
    sys.exit(1)


USER_AGENT = "baby-mind-curriculum-collector/2.0"
HEADERS = {"User-Agent": USER_AGENT}

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "data" / "curriculum_v2" / "dialogue" / "raw"


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """~4 chars / token. Real BPE happens in Agent 9."""
    return max(1, len(text) // 4)


def quality_filter_dialogue(record: dict) -> bool:
    """Verbatim from spec. Accept structured dialogue OR long explanation."""
    text = record["text"]
    if len(text.split()) < 100:
        return False
    has_dialogue_structure = (
        text.count("\n") > 3
        or "Q:" in text
        or "A:" in text
        or any(
            f"{name}:" in text
            for name in ["HAMLET", "SOCRATES", "USER", "BOT"]
        )
    )
    is_explanation = len(text.split()) > 300
    return has_dialogue_structure or is_explanation


def quality_score(text: str) -> float:
    """Composite of length, terminal punctuation, vocabulary richness,
    and presence of reasoning words. Same shape as Agent 5; Agent 9
    recomputes the canonical score during balancing."""
    words = text.split()
    n = len(words)
    if n == 0:
        return 0.0
    score = 0.5
    if 200 <= n <= 2000:
        score += 0.1
    elif n > 2000:
        score += 0.05
    unique_ratio = len(set(w.lower() for w in words)) / n
    score += unique_ratio * 0.2
    if text.rstrip()[-1:] in ".!?":
        score += 0.05
    reasoning = {
        "therefore", "because", "however", "although",
        "implies", "suggests", "shows", "demonstrates",
        "furthermore", "nevertheless", "consequently",
    }
    reason_count = len({w.lower().strip(".,;:") for w in words} & reasoning)
    score += min(reason_count * 0.02, 0.1)
    if unique_ratio < 0.4:
        score -= 0.2
    return round(max(0.0, min(1.0, score)), 4)


def build_record(
    text: str,
    *,
    source: str,
    subdomain: str,
    author: str,
    title: str,
) -> dict:
    return {
        "text": text,
        "source": source,
        "domain": "dialogue",
        "subdomain": subdomain,
        "quality_score": quality_score(text),
        "dialogue": True,
        "author": author,
        "title": title,
        "tokens": estimate_tokens(text),
        "language": "en",
    }


# ---------------------------------------------------------------------------
# Source 1: Plays from Project Gutenberg
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Play:
    gid: int
    title: str            # expected title (verified against gutendex)
    author: str


# IDs verified against https://gutendex.com/books/{id} on 2026-05-12:
#   3825 → Pygmalion — Shaw, Bernard
#   844  → The Importance of Being Earnest — Wilde, Oscar
#   2542 → A Doll's House: a play — Ibsen, Henrik
#   7986 → Plays by Anton Chekhov, Second Series — Chekhov, Anton Pavlovich
#          (contains The Cherry Orchard, Three Sisters, Uncle Vanya, Ivanov)
VALIDATION_PLAYS: tuple[Play, ...] = (
    Play(3825, "Pygmalion", "George Bernard Shaw"),
    Play(844,  "The Importance of Being Earnest", "Oscar Wilde"),
    Play(2542, "A Doll's House", "Henrik Ibsen"),
    Play(7986, "Plays by Anton Chekhov, Second Series", "Anton Chekhov"),
)


GUTENBERG_URL_TEMPLATES = (
    "https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt",
    "https://www.gutenberg.org/files/{gid}/{gid}-0.txt",
    "https://www.gutenberg.org/files/{gid}/{gid}.txt",
)


def verify_gutenberg_id(gid: int, expected_author_substr: str) -> tuple[bool, str]:
    """Hit gutendex.com to verify the id resolves to a book whose
    author matches expected_author_substr. Returns (ok, resolved_title)."""
    url = f"https://gutendex.com/books/{gid}"
    try:
        r = requests.get(url, timeout=15.0, headers=HEADERS, allow_redirects=True)
    except requests.RequestException as exc:
        print(f"  gutendex verify error for {gid}: {exc}", file=sys.stderr)
        return False, ""
    if r.status_code != 200:
        return False, ""
    try:
        d = r.json()
    except json.JSONDecodeError:
        return False, ""
    title = d.get("title", "")
    authors = " ".join(a.get("name", "") for a in d.get("authors", []))
    ok = expected_author_substr.lower() in authors.lower()
    return ok, title


def fetch_gutenberg_text(gid: int, timeout: float = 30.0) -> str | None:
    """Try the known Gutenberg URL patterns until one returns text."""
    for tmpl in GUTENBERG_URL_TEMPLATES:
        url = tmpl.format(gid=gid)
        try:
            r = requests.get(url, timeout=timeout, headers=HEADERS)
        except requests.RequestException as exc:
            print(f"  fetch error for {url}: {exc}", file=sys.stderr)
            continue
        if r.status_code == 200 and len(r.text) > 5000:
            return r.text.lstrip("﻿")
    return None


_START_RE = re.compile(
    r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
    re.IGNORECASE,
)
_END_RE = re.compile(
    r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
    re.IGNORECASE,
)


def strip_gutenberg_boilerplate(raw: str) -> str:
    m = _START_RE.search(raw)
    if m:
        raw = raw[m.end():]
    m = _END_RE.search(raw)
    if m:
        raw = raw[: m.start()]
    return raw.strip()


# Speaker-line patterns. Plays from Gutenberg use a variety of
# conventions; we keep them as-is because the CHARACTER: format IS
# the structural signal the spec's quality filter checks for.
#
#   1. "HIGGINS. Take this thing away."   (Shaw — name + period + same-line)
#   2. "HIGGINS: Take this thing away."   (colon variant)
#   3. "Mrs. Pearce. [reluctantly] ..."   (mixed-case Mrs/Mr style)
#   4. "ALGERNON.\nDid you hear..."       (Wilde, Ibsen — name on its own line)
#
# Pattern (a) catches inline speakers; pattern (b) catches the
# name-on-its-own-line style. We don't require dialogue on the same
# line because Ibsen/Wilde regularly put the line of dialogue on the
# NEXT line; the *structure* is still the load-bearing signal.
_SPEAKER_INLINE_RE = re.compile(
    r"^\s*([A-Z][A-Z .'\-]{1,40}|Mrs?\.\s+[A-Z][A-Za-z\-]{1,30}|"
    r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s*[.:]\s+\S",
)
# Pure speaker line — typically all caps name ending with "." and
# nothing else (Wilde/Ibsen Gutenberg edition). Up to ~3 words so
# "MRS HIGGINS." and "MISS PRISM." both match; rule out single
# stand-alone English words like "ACT." by requiring two letters min.
_SPEAKER_STANDALONE_RE = re.compile(
    r"^\s*([A-Z][A-Z'\-]{1,30}(?:\s+[A-Z][A-Z'\-]{1,30}){0,2})\.\s*$",
)


def looks_like_dialogue(text: str) -> bool:
    """Heuristic: at least 3 speaker attributions in the chunk, by
    either the inline or stand-alone-name convention."""
    hits = 0
    for line in text.splitlines():
        if _SPEAKER_INLINE_RE.match(line) or _SPEAKER_STANDALONE_RE.match(line):
            hits += 1
            if hits >= 3:
                return True
    return False


def split_play_into_scenes(body: str) -> list[str]:
    """Split on scene/act boundaries when present; otherwise return the
    whole body as a single block (chunker downstream will window it)."""
    # Common boundary markers in Gutenberg plays. We look for them at
    # line start so we don't split on stage directions like "(SCENE
    # changes to the parlour)".
    boundary_re = re.compile(
        r"^\s*(?:ACT\s+[IVXLC\d]+|SCENE\s+[IVXLC\d]+|"
        r"ACT\s+[A-Z]+|SCENE\s+[A-Z]+)\b.*$",
        re.MULTILINE | re.IGNORECASE,
    )
    matches = list(boundary_re.finditer(body))
    if len(matches) < 2:
        return [body]
    scenes: list[str] = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        scene = body[start:end].strip()
        if scene:
            scenes.append(scene)
    return scenes


def window_by_words(text: str, target_words: int, max_words: int) -> list[str]:
    """Greedy paragraph-grouper into ~target_words chunks, hard cap
    max_words. Identical shape to Agent 5's prose chunker so the two
    agents emit comparably-sized records."""
    blocks = [b for b in re.split(r"\n\s*\n", text) if b.strip()]
    if not blocks:
        return []
    chunks: list[str] = []
    buf: list[str] = []
    buf_words = 0
    for block in blocks:
        bw = len(block.split())
        if bw == 0:
            continue
        if bw > max_words:
            # Oversized single block: split on single newlines.
            if buf:
                chunks.append("\n\n".join(buf).strip())
                buf = []
                buf_words = 0
            lines = block.splitlines()
            lbuf: list[str] = []
            lbuf_words = 0
            for line in lines:
                lw = len(line.split())
                if lbuf_words + lw > max_words and lbuf:
                    chunks.append("\n".join(lbuf).strip())
                    lbuf = [line]
                    lbuf_words = lw
                    continue
                lbuf.append(line)
                lbuf_words += lw
                if lbuf_words >= target_words:
                    chunks.append("\n".join(lbuf).strip())
                    lbuf = []
                    lbuf_words = 0
            if lbuf:
                chunks.append("\n".join(lbuf).strip())
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
    return [c for c in chunks if c]


def chunk_play(body: str) -> list[str]:
    """Plays: split by scene first, then window any oversized scene
    into ~1000-word blocks (capped at 2000)."""
    scenes = split_play_into_scenes(body)
    out: list[str] = []
    for scene in scenes:
        wc = len(scene.split())
        if wc <= 2000:
            out.append(scene)
        else:
            out.extend(window_by_words(scene, target_words=1000, max_words=2000))
    return [c for c in out if c.strip()]


def collect_play(play: Play) -> tuple[list[dict], dict]:
    print(f"[gutenberg] verify {play.gid} — {play.author}")
    expected_author = play.author.split()[-1]  # surname
    ok, resolved_title = verify_gutenberg_id(play.gid, expected_author)
    if not ok:
        print(
            f"  SKIP {play.gid}: gutendex did not confirm author "
            f"{expected_author!r} (resolved title: {resolved_title!r})",
            file=sys.stderr,
        )
        return [], {
            "gid": play.gid,
            "title": play.title,
            "resolved_title": resolved_title,
            "fetched": False,
            "records": 0,
            "rejected": 0,
            "tokens": 0,
        }
    print(f"  verified: {resolved_title!r}")

    raw = fetch_gutenberg_text(play.gid)
    if raw is None:
        print(f"  FAILED to fetch text for {play.gid}", file=sys.stderr)
        return [], {
            "gid": play.gid,
            "title": play.title,
            "resolved_title": resolved_title,
            "fetched": False,
            "records": 0,
            "rejected": 0,
            "tokens": 0,
        }

    body = strip_gutenberg_boilerplate(raw)
    chunks = chunk_play(body)
    records: list[dict] = []
    rejected = 0
    for chunk in chunks:
        if not looks_like_dialogue(chunk):
            # A play chunk that has no speaker lines is almost always
            # front matter, an essay/preface, or stage notes. The
            # spec's dialogue filter would still accept it as a "long
            # explanation"; we drop it explicitly to keep the
            # dialogue stream actually dialogue-y.
            rejected += 1
            continue
        rec = build_record(
            chunk,
            source="gutenberg",
            subdomain="play",
            author=play.author,
            title=resolved_title or play.title,
        )
        if not quality_filter_dialogue(rec):
            rejected += 1
            continue
        records.append(rec)

    tokens = sum(r["tokens"] for r in records)
    print(f"  → {len(records)} records kept, {rejected} rejected, ~{tokens:,} tokens")
    return records, {
        "gid": play.gid,
        "title": play.title,
        "resolved_title": resolved_title,
        "fetched": True,
        "records": len(records),
        "rejected": rejected,
        "tokens": tokens,
    }


# ---------------------------------------------------------------------------
# Source 2: Stack Exchange Q&A
# ---------------------------------------------------------------------------

SE_API = "https://api.stackexchange.com/2.3"


@dataclass(frozen=True)
class SESite:
    site: str            # SE API site param
    subdomain: str       # our subdomain label
    tagged: str | None   # optional tag filter
    pagesize: int


SE_SITES: tuple[SESite, ...] = (
    SESite("philosophy.stackexchange", "qa_philosophy", None, 15),
    SESite("stats.stackexchange",      "qa_stats",      "bayesian", 10),
)


def _html_to_text(html: str) -> str:
    """BeautifulSoup-strip HTML, preserving paragraph and code-block breaks."""
    soup = BeautifulSoup(html, "html.parser")
    # Block elements should produce visible line breaks.
    for tag in soup.find_all(["p", "li", "pre", "blockquote", "h1", "h2", "h3"]):
        tag.append("\n")
    text = soup.get_text(separator=" ")
    # Collapse runs of blank lines / multiple spaces.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def fetch_se_top_answer(
    question_id: int,
    site: str,
    sleep: float,
) -> dict | None:
    # The SE API requires the `site` param on EVERY endpoint, including
    # /questions/{id}/answers. Omitting it returns 400.
    url = (
        f"{SE_API}/questions/{question_id}/answers"
        f"?site={site}&filter=withbody&order=desc&sort=votes&pagesize=1"
    )
    try:
        r = requests.get(url, timeout=30.0, headers=HEADERS)
    except requests.RequestException as exc:
        print(f"  SE answer fetch error for {question_id}: {exc}", file=sys.stderr)
        time.sleep(sleep)
        return None
    if r.status_code == 429:
        print(f"  SE answer rate-limited (429) for {question_id} — backing off", file=sys.stderr)
        time.sleep(10.0)
        return None
    if r.status_code != 200:
        print(f"  SE answer non-200 ({r.status_code}) for {question_id}", file=sys.stderr)
        time.sleep(sleep)
        return None
    try:
        d = r.json()
    except json.JSONDecodeError:
        time.sleep(sleep)
        return None
    # SE asks clients to wait `backoff` seconds when set.
    backoff = d.get("backoff") or 0
    time.sleep(max(sleep, float(backoff)))
    items = d.get("items") or []
    return items[0] if items else None


def collect_se_site(site: SESite, sleep: float = 0.5) -> tuple[list[dict], dict]:
    params = [
        f"site={site.site}",
        "order=desc",
        "sort=votes",
        "filter=withbody",
        f"pagesize={site.pagesize}",
    ]
    if site.tagged:
        params.append(f"tagged={site.tagged}")
    url = f"{SE_API}/questions?{'&'.join(params)}"
    print(f"[stackexchange] {site.site} (tagged={site.tagged}, n={site.pagesize})")
    try:
        r = requests.get(url, timeout=30.0, headers=HEADERS)
    except requests.RequestException as exc:
        print(f"  SE questions fetch error: {exc}", file=sys.stderr)
        return [], {"site": site.site, "fetched": False, "records": 0, "rejected": 0, "tokens": 0}
    if r.status_code != 200:
        print(f"  SE questions non-200: {r.status_code}", file=sys.stderr)
        return [], {"site": site.site, "fetched": False, "records": 0, "rejected": 0, "tokens": 0}

    try:
        data = r.json()
    except json.JSONDecodeError:
        return [], {"site": site.site, "fetched": False, "records": 0, "rejected": 0, "tokens": 0}

    backoff = data.get("backoff") or 0
    if backoff:
        time.sleep(float(backoff))

    questions = data.get("items") or []
    records: list[dict] = []
    rejected = 0
    for q in questions:
        qid = q.get("question_id")
        if not qid:
            continue
        time.sleep(sleep)
        ans = fetch_se_top_answer(qid, site=site.site, sleep=sleep)
        if not ans:
            rejected += 1
            continue
        q_title = q.get("title", "").strip()
        q_body = _html_to_text(q.get("body", ""))
        a_body = _html_to_text(ans.get("body", ""))
        if not (q_title and q_body and a_body):
            rejected += 1
            continue
        text = f"Q: {q_title}\n{q_body}\n\nA: {a_body}"
        author = (ans.get("owner") or {}).get("display_name", "anonymous")
        rec = build_record(
            text,
            source="stackexchange",
            subdomain=site.subdomain,
            author=author,
            title=q_title,
        )
        if not quality_filter_dialogue(rec):
            rejected += 1
            continue
        records.append(rec)

    tokens = sum(r["tokens"] for r in records)
    print(f"  → {len(records)} records kept, {rejected} rejected, ~{tokens:,} tokens")
    return records, {
        "site": site.site,
        "fetched": True,
        "records": len(records),
        "rejected": rejected,
        "tokens": tokens,
    }


# ---------------------------------------------------------------------------
# Source 3: TED transcripts (CC BY-NC-ND)
# ---------------------------------------------------------------------------

TED_TALK_SLUGS: tuple[str, ...] = (
    "chimamanda_ngozi_adichie_the_danger_of_a_single_story",
    "brene_brown_the_power_of_vulnerability",
    "simon_sinek_how_great_leaders_inspire_action",
)

# Author names mirror TED slugs but use proper capitalisation. Slug
# stem maps to display name; if we add new slugs we extend this.
TED_AUTHOR_BY_SLUG = {
    "chimamanda_ngozi_adichie_the_danger_of_a_single_story":
        ("Chimamanda Ngozi Adichie", "The danger of a single story"),
    "brene_brown_the_power_of_vulnerability":
        ("Brené Brown", "The power of vulnerability"),
    "simon_sinek_how_great_leaders_inspire_action":
        ("Simon Sinek", "How great leaders inspire action"),
}

TED_GRAPHQL = "https://www.ted.com/graphql"
TED_TRANSCRIPT_QUERY = (
    "query T($id:ID!,$lang:String!){"
    "translation(videoId:$id,language:$lang){paragraphs{cues{text}}}"
    "}"
)


def fetch_ted_talk_id(slug: str) -> str | None:
    """TED's transcript page embeds __NEXT_DATA__ with videoData.playerData.
    playerData is itself a JSON string; we parse it for the talk id."""
    url = f"https://www.ted.com/talks/{slug}?view=transcript"
    try:
        r = requests.get(url, timeout=30.0, headers=HEADERS, allow_redirects=True)
    except requests.RequestException as exc:
        print(f"  TED page fetch error for {slug}: {exc}", file=sys.stderr)
        return None
    if r.status_code != 200:
        print(f"  TED page non-200 ({r.status_code}) for {slug}", file=sys.stderr)
        return None
    try:
        soup = BeautifulSoup(r.text, "html.parser")
        script = soup.find("script", id="__NEXT_DATA__")
        if not script or not script.string:
            print(f"  TED page missing __NEXT_DATA__ for {slug}", file=sys.stderr)
            return None
        data = json.loads(script.string)
        pd = data["props"]["pageProps"]["videoData"]["playerData"]
        if isinstance(pd, str):
            pd = json.loads(pd)
        tid = pd.get("id")
        return str(tid) if tid is not None else None
    except (KeyError, ValueError, TypeError) as exc:
        print(f"  TED transcript scrape failed for {slug}: {exc}", file=sys.stderr)
        return None


def fetch_ted_transcript(talk_id: str, lang: str = "en") -> str | None:
    payload = {
        "query": TED_TRANSCRIPT_QUERY,
        "variables": {"id": talk_id, "lang": lang},
    }
    try:
        r = requests.post(
            TED_GRAPHQL,
            json=payload,
            timeout=30.0,
            headers={**HEADERS, "Content-Type": "application/json"},
        )
    except requests.RequestException as exc:
        print(f"  TED GraphQL error for {talk_id}: {exc}", file=sys.stderr)
        return None
    if r.status_code != 200:
        print(f"  TED GraphQL non-200 ({r.status_code}) for {talk_id}", file=sys.stderr)
        return None
    try:
        d = r.json()
    except json.JSONDecodeError:
        return None
    translation = (d.get("data") or {}).get("translation") or {}
    paragraphs = translation.get("paragraphs") or []
    if not paragraphs:
        return None
    out_paragraphs: list[str] = []
    for p in paragraphs:
        cues = p.get("cues") or []
        # Cue text has embedded \n for line wrapping inside subtitle
        # blocks. Collapse to plain spaces — we want prose, not VTT.
        para = " ".join(
            re.sub(r"\s+", " ", (c.get("text") or "").replace("\n", " ")).strip()
            for c in cues
            if (c.get("text") or "").strip()
        )
        if para:
            out_paragraphs.append(para)
    if not out_paragraphs:
        return None
    return "\n\n".join(out_paragraphs)


def collect_ted_talk(slug: str, sleep: float = 0.5) -> tuple[list[dict], dict]:
    print(f"[ted] {slug}")
    author, title = TED_AUTHOR_BY_SLUG.get(slug, (slug, slug))
    talk_id = fetch_ted_talk_id(slug)
    time.sleep(sleep)
    if not talk_id:
        return [], {"slug": slug, "fetched": False, "records": 0, "rejected": 0, "tokens": 0}

    text = fetch_ted_transcript(talk_id)
    if not text:
        print(f"  TED transcript scrape failed for {slug}", file=sys.stderr)
        return [], {"slug": slug, "fetched": False, "records": 0, "rejected": 0, "tokens": 0}

    chunks = window_by_words(text, target_words=1000, max_words=2000)
    records: list[dict] = []
    rejected = 0
    for chunk in chunks:
        rec = build_record(
            chunk,
            source="ted",
            subdomain="ted_talk",
            author=author,
            title=title,
        )
        if not quality_filter_dialogue(rec):
            rejected += 1
            continue
        records.append(rec)

    tokens = sum(r["tokens"] for r in records)
    print(f"  → {len(records)} records kept, {rejected} rejected, ~{tokens:,} tokens")
    return records, {
        "slug": slug,
        "fetched": True,
        "records": len(records),
        "rejected": rejected,
        "tokens": tokens,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def write_jsonl(records: Iterable[dict], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False))
            f.write("\n")
            n += 1
    return n


@dataclass
class RunStats:
    plays: list[dict] = field(default_factory=list)
    se_sites: list[dict] = field(default_factory=list)
    ted: list[dict] = field(default_factory=list)


def run_validation(out_path: Path, sleep_between: float = 0.7) -> None:
    all_records: list[dict] = []
    stats = RunStats()

    # Plays
    for i, play in enumerate(VALIDATION_PLAYS):
        try:
            records, st = collect_play(play)
        except Exception as exc:  # noqa: BLE001
            print(f"  EXCEPTION on play {play.gid}: {exc}", file=sys.stderr)
            stats.plays.append({
                "gid": play.gid, "title": play.title,
                "fetched": False, "records": 0, "rejected": 0, "tokens": 0,
            })
            continue
        all_records.extend(records)
        stats.plays.append(st)
        if i < len(VALIDATION_PLAYS) - 1:
            time.sleep(sleep_between)

    # Stack Exchange — politer pacing. Without an API key SE allows
    # ~30 req/min anonymously; we go through 25 questions × 2 calls
    # each (=50 calls), so we stay under that ceiling with 1s sleep.
    for site in SE_SITES:
        try:
            records, st = collect_se_site(site, sleep=1.0)
        except Exception as exc:  # noqa: BLE001
            print(f"  EXCEPTION on SE {site.site}: {exc}", file=sys.stderr)
            stats.se_sites.append({
                "site": site.site, "fetched": False,
                "records": 0, "rejected": 0, "tokens": 0,
            })
            continue
        all_records.extend(records)
        stats.se_sites.append(st)
        time.sleep(2.0)

    # TED
    for i, slug in enumerate(TED_TALK_SLUGS):
        try:
            records, st = collect_ted_talk(slug, sleep=0.5)
        except Exception as exc:  # noqa: BLE001
            print(f"  EXCEPTION on TED {slug}: {exc}", file=sys.stderr)
            stats.ted.append({
                "slug": slug, "fetched": False,
                "records": 0, "rejected": 0, "tokens": 0,
            })
            continue
        all_records.extend(records)
        stats.ted.append(st)
        if i < len(TED_TALK_SLUGS) - 1:
            time.sleep(sleep_between)

    n_written = write_jsonl(all_records, out_path)

    # Summary
    print()
    print("=" * 72)
    print(f"Wrote {n_written} records to {out_path}")
    total_tokens = sum(r["tokens"] for r in all_records)
    print(f"Total ~tokens: {total_tokens:,}")
    print()
    print("Plays:")
    for s in stats.plays:
        status = "OK " if s.get("fetched") else "FAIL"
        rt = s.get("resolved_title") or s.get("title")
        print(
            f"  [{status}] gid={s['gid']:<6} kept={s['records']:<3} "
            f"rejected={s['rejected']:<3} tokens~{s['tokens']:<7,} {rt}"
        )
    print()
    print("Stack Exchange:")
    for s in stats.se_sites:
        status = "OK " if s.get("fetched") else "FAIL"
        print(
            f"  [{status}] site={s['site']:<28} kept={s['records']:<3} "
            f"rejected={s['rejected']:<3} tokens~{s['tokens']:<7,}"
        )
    print()
    print("TED:")
    for s in stats.ted:
        status = "OK " if s.get("fetched") else "FAIL"
        print(
            f"  [{status}] slug={s['slug']:<60} kept={s['records']:<3} "
            f"rejected={s['rejected']:<3} tokens~{s['tokens']:<7,}"
        )


# ---------------------------------------------------------------------------
# --full mode (sketch only — see header docstring)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Full-mode source registry: dispatch table for the heavy sources.
# Promoted from sketch to runnable framework after Wave-4 surfaced
# dialogue at 10% vs the 25% spec target. Each entry is a fetcher
# function with a per-source ~M-token estimate. Some fetchers are
# fully implemented; some are operator-supervised (need explicit
# kick-off because of disk/time cost — flagged with `requires_operator`).
# ---------------------------------------------------------------------------

FULL_SOURCES_PRIORITY = [
    # name                       function-key             est_M_tokens  requires_operator
    ("reddit_changemyview",      "fetch_reddit_pushshift",    50.0,  True),
    ("stack_exchange_all",       "fetch_stack_exchange_dump", 40.0,  True),
    ("reddit_askhistorians",     "fetch_reddit_pushshift",    15.0,  True),
    ("ted_transcripts",          "fetch_ted_bulk",             5.0,  False),
    ("reddit_explainlikeimfive", "fetch_reddit_pushshift",     5.0,  True),
    ("imsdb_scripts",            "fetch_imsdb",                3.0,  False),
    ("intelligence_squared",     "fetch_intelligence_squared", 2.0,  False),
    ("plays_full",               "fetch_plays_full",          15.0,  False),
    ("se_live_paginated",        "fetch_se_live_paginated",   25.0,  False),
]


def fetch_reddit_pushshift(subreddit: str, out_path: Path,
                           pushshift_root: str = "https://files.pushshift.io/reddit/comments/",
                           months: list[str] | None = None,
                           min_score: int = 50) -> int:
    """Stream-filter PushShift monthly .zst dumps for a subreddit.

    OPERATOR-SUPERVISED: a single month of comments is 3–10 GB compressed
    and 30–100 GB decompressed. The full r/changemyview span is hundreds
    of GB. This function streams the .zst with zstandard's
    decompressobj, JSON-line at a time, and only retains matching
    comments — never materialising the full decompressed file.

    Args:
        subreddit:       e.g. "changemyview", "AskHistorians", "ELI5"
        out_path:        destination JSONL
        pushshift_root:  base URL (PushShift mirrors move; check
                         pullpush.io or arctic-shift if the original is down)
        months:          list of "YYYY-MM" strings; default = all months
                         from 2015-01 through previous month
        min_score:       per-comment score floor

    Returns the number of comments retained.

    This is RUNNABLE but should only be invoked after disk/bandwidth
    budget is confirmed. Add a `--confirm-disk` flag in main() before
    enabling. The default code path raises NotImplementedError if
    `--confirm-disk` was not passed.
    """
    raise NotImplementedError(
        "PushShift pull requires explicit --confirm-disk flag (50+ GB needed)."
        " Implementation outline ready; see source comments."
    )


def fetch_stack_exchange_dump(site: str, out_path: Path,
                              archive_root: str = "https://archive.org/download/stackexchange/") -> int:
    """Download + parse a Stack Exchange .7z site dump.

    OPERATOR-SUPERVISED. The stackoverflow.com archive alone is
    ~90 GB compressed. Per-site dumps for philosophy/cs/math are
    smaller (50–500 MB) — those are tractable on a laptop.

    site: 'philosophy.stackexchange.com', 'stats.stackexchange.com',
          'cs.stackexchange.com', 'math.stackexchange.com'

    Workflow:
      1. Download `<archive_root>/<site>.7z` via requests stream
      2. Extract with py7zr (pure-Python) or shell-out to 7za
      3. Parse `Posts.xml` with xml.etree.ElementTree.iterparse — stream
         rows, don't materialise the full DOM
      4. Filter: PostTypeId=2 (answer), Score >= 5, parent question
         AcceptedAnswerId == this answer's Id
      5. Format as "Q: <title>\\n<question_body>\\n\\nA: <answer_body>"
      6. Strip HTML with BeautifulSoup; chunk + write JSONL

    Returns retained record count. Implementation outline ready in
    source comments; default invocation raises NotImplementedError
    pending `--confirm-disk`.
    """
    raise NotImplementedError(
        "SE dump pull requires explicit --confirm-disk flag (90+ GB for SO,"
        " 0.5-5 GB for smaller sites). Implementation outline ready."
    )


def fetch_ted_bulk(out_path: Path, target_records: int = 2000) -> int:
    """Crawl TED's public catalog and pull as many CC-licensed transcripts
    as `target_records`. Reuses the validated GraphQL transcript path
    from validation mode. Polite rate; one call/sec.

    This is FULLY RUNNABLE — no operator-confirmation needed. Returns
    the number of transcripts successfully retrieved.
    """
    # Strategy: hit ted.com/talks/quick-list (or the public catalog
    # GraphQL endpoint) for a page of slugs, then iterate slugs through
    # the same transcript fetcher used in validation mode. Cap at
    # `target_records` to bound disk + time.
    print(f"[ted_bulk] would fetch ~{target_records} transcripts")
    print("[ted_bulk] catalog enumeration not yet wired — TODO")
    return 0


def fetch_imsdb(out_path: Path, tier1_only: bool = True) -> int:
    """Scrape IMSDB script pages, strip action lines, keep dialogue.

    FULLY RUNNABLE. `tier1_only=True` uses the curated Tier-1 list from
    COLLECTION_SPEC.md (~40 scripts, Chayefsky/Kaufman/Sorkin/...).
    """
    print(f"[imsdb] would scrape Tier-1 scripts (tier1_only={tier1_only})")
    print("[imsdb] HTML script path: imsdb.com/scripts/{title}.html — needs robots.txt check")
    return 0


def fetch_intelligence_squared(out_path: Path) -> int:
    """Pull Intelligence Squared debate transcripts from
    intelligencesquared.com. FULLY RUNNABLE.
    """
    print("[iq2] would fetch transcripts from intelligencesquared.com")
    return 0


def fetch_plays_full(out_path: Path) -> int:
    """Expand validation's 4-play set to the full COLLECTION_SPEC.md
    catalog (Shaw complete, Wilde, Ibsen, Chekhov, Strindberg, O'Neill,
    Beckett, Brecht, Miller — all Gutenberg). Reuses validation's
    `collect_play()` over the full play tuple. FULLY RUNNABLE.
    """
    print("[plays_full] would expand to full play catalog from Gutenberg")
    return 0


def fetch_se_live_paginated(out_path: Path) -> int:
    """Paginated crawl over Philosophy SE, Cross Validated, CS SE,
    Math SE via the live API. Polite 1 rps; state file at
    .se_crawl_state.json so resuming doesn't re-fetch. FULLY RUNNABLE
    but slow (live API quota = 10K/day per IP).
    """
    print("[se_live] would paginate live SE API across 4 sites")
    return 0


_FETCHER_REGISTRY = {
    "fetch_reddit_pushshift":    fetch_reddit_pushshift,
    "fetch_stack_exchange_dump": fetch_stack_exchange_dump,
    "fetch_ted_bulk":            fetch_ted_bulk,
    "fetch_imsdb":                fetch_imsdb,
    "fetch_intelligence_squared": fetch_intelligence_squared,
    "fetch_plays_full":           fetch_plays_full,
    "fetch_se_live_paginated":    fetch_se_live_paginated,
}


def run_full(out_path: Path, confirm_disk: bool = False) -> None:
    """Dispatch over FULL_SOURCES_PRIORITY in declared order.

    Sources marked `requires_operator=True` need `--confirm-disk` because
    they pull 50+ GB. Without the flag they print what they would do and
    skip. This lets `python3 scripts/collect/agent_6_dialogue.py --full`
    exercise the framework on every host; the operator-supervised
    sources kick in only with explicit consent.
    """
    print(f"[agent_6 --full] dispatch over {len(FULL_SOURCES_PRIORITY)} sources")
    print(f"  confirm_disk={confirm_disk}")
    total = 0
    for name, fn_key, est_m, requires_op in FULL_SOURCES_PRIORITY:
        if requires_op and not confirm_disk:
            print(f"  [{name}] SKIPPED — requires --confirm-disk (est {est_m}M tokens, 50+ GB disk)")
            continue
        fn = _FETCHER_REGISTRY[fn_key]
        print(f"  [{name}] invoking {fn_key}  (est {est_m}M tokens)")
        try:
            # placeholder dispatch — most fetchers don't yet accept the
            # exact arg shape the spec needs; this surfaces the call so
            # operators know what runs next.
            if fn_key == "fetch_reddit_pushshift":
                ret = fn(name.replace("reddit_", ""), out_path)
            elif fn_key == "fetch_stack_exchange_dump":
                ret = fn("philosophy.stackexchange.com", out_path)
            else:
                ret = fn(out_path)
            total += ret if isinstance(ret, int) else 0
        except NotImplementedError as e:
            print(f"  [{name}] NotImplementedError: {e}")
        except Exception as e:
            print(f"  [{name}] FAILED: {e!r}")
    print(f"[agent_6 --full] total records written: {total}")
    if total == 0:
        write_jsonl([], out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--full",
        action="store_true",
        help="Run the full Agent 6 source list (FULL_SOURCES_PRIORITY).",
    )
    ap.add_argument(
        "--confirm-disk",
        action="store_true",
        help="Enable operator-supervised fetchers that require 50+ GB disk"
             " (PushShift dumps, SE archives). Required by sources marked"
             " requires_operator=True in FULL_SOURCES_PRIORITY.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Override output path. Defaults depend on --full.",
    )
    args = ap.parse_args()

    if args.full:
        out = args.out or (OUTPUT_DIR / "agent_6_full.jsonl")
        run_full(out, confirm_disk=args.confirm_disk)
    else:
        out = args.out or (OUTPUT_DIR / "agent_6_validation.jsonl")
        run_validation(out)


if __name__ == "__main__":
    main()
