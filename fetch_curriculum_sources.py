"""Fetch every source referenced by curriculum.json that lives at a known
remote endpoint. Two protocols:

  * Project Gutenberg books — `gutenberg_id` field, fetched as plain text
    from https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt

  * Wikipedia categories — `wiki_category` field, expanded into the top
    `LIMIT` (default 100) article extracts via the MediaWiki API. The
    extracts are concatenated and written to one .txt file matching the
    curriculum's `source` path.

Resumable: anything already on disk is skipped (size and rough sentence
estimate are still printed so a resume run gives the same summary as a
fresh one).

Usage:
    python3 fetch_curriculum_sources.py
    python3 fetch_curriculum_sources.py --only alice_in_wonderland,aesops_fables
    python3 fetch_curriculum_sources.py --no-wiki        # skip wiki categories
    python3 fetch_curriculum_sources.py --wiki-limit 50  # smaller wiki pulls
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request


GUTENBERG_URL_TEMPLATE = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"
WIKI_API_URL = "https://en.wikipedia.org/w/api.php"

DEFAULT_WIKI_LIMIT = 100
WIKI_REQUEST_DELAY_S = 1.0           # base delay (Wikipedia rate-limits aggressively)
WIKI_MAX_RETRIES = 5


# ============================================================
# Generic HTTP fetch with a sane User-Agent + 429 backoff
# ============================================================

def _http_get(url: str, timeout: float = 60.0) -> bytes:
    """Fetch with exponential-backoff retries on HTTP 429 (rate limit).
    The Wikipedia REST API gives 429 freely under burst load — sleeping
    1, 2, 4, 8, 16 s between attempts is sufficient on a single client.
    """
    delay = 2.0
    for attempt in range(WIKI_MAX_RETRIES):
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "the-mind/0.3 (curriculum fetcher)"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < WIKI_MAX_RETRIES - 1:
                print(f"    [429 rate-limit, sleeping {delay:.0f}s, retry {attempt + 1}]")
                time.sleep(delay)
                delay *= 2
                continue
            raise
    # Final attempt without retry catch.
    req = urllib.request.Request(url, headers={"User-Agent": "the-mind/0.3 (curriculum fetcher)"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


# ============================================================
# Project Gutenberg
# ============================================================

def fetch_gutenberg(book_id: int, out_path: str) -> tuple[int, str]:
    if os.path.exists(out_path):
        return os.path.getsize(out_path), "skipped"
    url = GUTENBERG_URL_TEMPLATE.format(id=book_id)
    data = _http_get(url)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(data)
    return len(data), "downloaded"


# ============================================================
# Wikipedia category → concatenated extracts
# ============================================================

def _wiki_api_get(params: dict) -> dict:
    qs = urllib.parse.urlencode(params)
    url = f"{WIKI_API_URL}?{qs}"
    raw = _http_get(url, timeout=30.0)
    return json.loads(raw.decode("utf-8"))


def _wiki_category_to_title(name: str) -> str:
    """`philosophy_of_mind` → `Philosophy of mind` (first letter only;
    Wikipedia keeps connectives lowercase)."""
    s = name.replace("_", " ").strip()
    if not s:
        return s
    return s[0].upper() + s[1:]


def _wiki_list_category_pages(category_title: str, limit: int) -> list[str]:
    titles: list[str] = []
    cmcontinue: str | None = None
    while len(titles) < limit:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": f"Category:{category_title}",
            "cmtype": "page",
            "cmlimit": min(50, limit - len(titles)),
            "format": "json",
        }
        if cmcontinue:
            params["cmcontinue"] = cmcontinue
        data = _wiki_api_get(params)
        for m in data.get("query", {}).get("categorymembers", []):
            titles.append(m["title"])
            if len(titles) >= limit:
                break
        cmcontinue = data.get("continue", {}).get("cmcontinue")
        if not cmcontinue:
            break
        time.sleep(WIKI_REQUEST_DELAY_S)
    return titles


def _wiki_fetch_extracts(titles: list[str]) -> list[str]:
    extracts: list[str] = []
    # The MediaWiki API allows up to 20 titles per `extracts` request when
    # `exintro=false`. We keep batches small to avoid 414 URI Too Long.
    for i in range(0, len(titles), 20):
        chunk = titles[i:i + 20]
        params = {
            "action": "query",
            "prop": "extracts",
            "explaintext": 1,
            "titles": "|".join(chunk),
            "format": "json",
            "redirects": 1,
        }
        data = _wiki_api_get(params)
        for page in data.get("query", {}).get("pages", {}).values():
            extract = page.get("extract") or ""
            if extract:
                extracts.append(extract)
        time.sleep(WIKI_REQUEST_DELAY_S)
    return extracts


def fetch_wiki_category(
    category: str,
    out_path: str,
    limit: int = DEFAULT_WIKI_LIMIT,
) -> tuple[int, str]:
    if os.path.exists(out_path):
        return os.path.getsize(out_path), "skipped"

    title = _wiki_category_to_title(category)
    titles = _wiki_list_category_pages(title, limit=limit)
    if not titles:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("")
        return 0, "empty"

    extracts = _wiki_fetch_extracts(titles)
    body = "\n\n".join(extracts)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(body)
    return len(body.encode("utf-8")), "downloaded"


# ============================================================
# Driver
# ============================================================

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--curriculum", default="curriculum.json")
    ap.add_argument(
        "--only",
        nargs="+",
        help="step names to fetch (space- or comma-separated). default: all book steps",
    )
    ap.add_argument("--no-wiki", action="store_true")
    ap.add_argument("--no-gutenberg", action="store_true")
    ap.add_argument("--wiki-limit", type=int, default=DEFAULT_WIKI_LIMIT)
    args = ap.parse_args()

    with open(args.curriculum, "r", encoding="utf-8") as f:
        curriculum = json.load(f)

    # Two supported schemas:
    #   curriculum.json            → top-level "sequence", entries tagged type:"book"
    #   curriculum_interleaved.json → top-level "sources", no "type" field
    if "sequence" in curriculum:
        steps = [s for s in curriculum["sequence"] if s.get("type") == "book"]
    elif "sources" in curriculum:
        steps = list(curriculum["sources"])
    else:
        print(f"  [error] {args.curriculum} has neither 'sequence' nor 'sources' key")
        return 2

    if args.only:
        # Accept space- or comma-separated tokens.
        wanted: set[str] = set()
        for tok in args.only:
            for n in tok.split(","):
                n = n.strip()
                if n:
                    wanted.add(n)
        steps = [s for s in steps if s["name"] in wanted]

    n_done = n_skip = n_fail = 0
    total_bytes = 0

    for step in steps:
        name = step["name"]
        out = step["source"]
        domain = step.get("domain", "?")

        try:
            if "gutenberg_id" in step:
                if args.no_gutenberg:
                    print(f"  [skip-gutenberg] {name}")
                    continue
                size, status = fetch_gutenberg(step["gutenberg_id"], out)
            elif "wiki_category" in step:
                if args.no_wiki:
                    print(f"  [skip-wiki] {name}")
                    continue
                size, status = fetch_wiki_category(
                    step["wiki_category"],
                    out,
                    limit=args.wiki_limit,
                )
            else:
                if os.path.exists(out):
                    size, status = os.path.getsize(out), "present"
                else:
                    size, status = 0, "missing"
        except Exception as exc:
            print(f"  [FAIL] {name}: {exc}")
            n_fail += 1
            continue

        # Sentence estimate: ~100 chars per kept sentence (conservative).
        est = size // 100
        kb = size / 1024
        print(f"  {name:32s} [{domain:12s}] {kb:8.1f} KB  ~{est:>6,d} sentences  [{status}]")
        total_bytes += size
        if status == "downloaded":
            n_done += 1
        elif status in ("skipped", "present"):
            n_skip += 1
        else:
            n_fail += 1

    print()
    print(f"  totals: {n_done} downloaded · {n_skip} already present · {n_fail} failed")
    print(f"  total size on disk: {total_bytes / 1024 / 1024:.1f} MB")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
