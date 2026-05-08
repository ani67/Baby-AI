"""Encode multiple corpus files in parallel into data/encoded_corpus.db.

Usage
-----
    python3 encode_corpus.py path/to/file1.txt:domain1 path/to/file2.txt:domain2

Each argument is `source:domain`. Workers run in separate processes so
GIL doesn't bottleneck encoding. SQLite WAL mode lets them write
concurrently with serialized commits.

Output
------
    data/encoded_corpus.db                    (SQLite, schema in preencoder.py)

Idempotent: skips sources already marked complete in `encoding_progress`.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time

# Allow running from repo root.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.preencoder import (    # noqa: E402
    DB_PATH,
    encode_source,
    init_db,
    is_source_encoded,
    progress_for,
)


def _encode_one(payload: tuple[str, str, str]) -> tuple[str, str, int, float]:
    """Worker entrypoint. payload = (source_file, domain, db_path).
    Returns (source, domain, n_written, seconds)."""
    source, domain, db_path = payload
    n, secs = encode_source(source, domain, db_path)
    return source, domain, n, secs


def parse_source_arg(arg: str) -> tuple[str, str]:
    if ":" in arg:
        path, domain = arg.rsplit(":", 1)
    else:
        path, domain = arg, "unknown"
    return path, domain


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("sources", nargs="*",
                    help="`path:domain` pairs. If omitted, reads --curriculum and "
                         "encodes every book step listed there.")
    ap.add_argument("--curriculum", default="curriculum.json",
                    help="Used when `sources` is empty.")
    ap.add_argument("--db", default=DB_PATH)
    ap.add_argument("--workers", type=int, default=None,
                    help="Default: min(N_SOURCES, cpu_count())")
    args = ap.parse_args()

    init_db(args.db)

    if args.sources:
        requested = [parse_source_arg(s) for s in args.sources]
    else:
        # Auto-discover from curriculum.json — every step with type=="book"
        # gets encoded under its declared domain.
        import json as _json
        with open(args.curriculum, "r", encoding="utf-8") as f:
            curr = _json.load(f)
        requested = [
            (s["source"], s.get("domain", "unknown"))
            for s in curr.get("sequence", [])
            if s.get("type") == "book"
        ]
        print(f"  curriculum {args.curriculum}: {len(requested)} book sources")

    pending: list[tuple[str, str, str]] = []
    for src, domain in requested:
        if not os.path.exists(src):
            print(f"  [skip] missing file: {src}")
            continue
        if is_source_encoded(src, args.db):
            p = progress_for(src, args.db) or {}
            print(f"  [skip] already encoded: {src} "
                  f"({p.get('encoded_sentences', '?'):,} sentences, domain={p.get('domain')})")
            continue
        pending.append((src, domain, args.db))

    if not pending:
        print("\nnothing to do.")
        return 0

    workers = args.workers or min(len(pending), mp.cpu_count())
    print(f"\nencoding {len(pending)} source(s) with {workers} worker(s)…")
    print(f"  db: {args.db}\n")

    t0 = time.perf_counter()
    if workers <= 1 or len(pending) == 1:
        results = [_encode_one(p) for p in pending]
    else:
        with mp.Pool(workers) as pool:
            results = pool.map(_encode_one, pending)
    duration = time.perf_counter() - t0

    total = sum(r[2] for r in results)
    print(f"\nencoded {total:,} sentences across {len(results)} source(s) in {duration:.1f}s")
    for source, domain, n, secs in results:
        rate = n / max(secs, 1e-6)
        print(f"  {source} [{domain}]: {n:,} sentences in {secs:.1f}s ({rate:,.0f}/s)")

    print()
    print(f"  total: {total:,} sentences across {len(results)} domain(s)")
    print(f"  estimated ingestion time at 5,000/s: {total / 5000:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
