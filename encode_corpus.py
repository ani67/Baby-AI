"""Encode multiple corpus files in parallel into data/encoded_corpus.db.

Usage
-----
    python3 encode_corpus.py path/to/file1.txt:domain1 path/to/file2.txt:domain2
    python3 encode_corpus.py --multilevel       # also encode word/phrase/paragraph items

Each argument is `source:domain`. Workers run in separate processes so
GIL doesn't bottleneck encoding. SQLite WAL mode lets them write
concurrently with serialized commits.

Output
------
    data/encoded_corpus.db
        encoded_sentences        (sentence-level, schema in preencoder.py)
        encoded_multilevel       (word/phrase/sentence/paragraph, schema in
                                  backend/multilevel_preprocessor.py)
                                 — only populated when --multilevel is passed.

Idempotent: skips sources already marked complete in the relevant
`encoding_progress` / `encoding_progress_multilevel` table.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sqlite3
import sys
import time

import numpy as np

# Allow running from repo root.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.multilevel_preprocessor import (    # noqa: E402
    MULTILEVEL_DB_PATH,
    init_multilevel_db,
    is_source_encoded_multilevel,
    multilevel_stream,
)
from backend.preencoder import (    # noqa: E402
    DB_PATH,
    encode_source,
    init_db,
    is_source_encoded,
    progress_for,
)


# ============================================================
# Sentence-level encoding (existing baseline path)
# ============================================================

def _encode_one(payload: tuple[str, str, str]) -> tuple[str, str, int, float]:
    """Worker entrypoint. payload = (source_file, domain, db_path).
    Returns (source, domain, n_written, seconds)."""
    source, domain, db_path = payload
    n, secs = encode_source(source, domain, db_path)
    return source, domain, n, secs


# ============================================================
# Multi-resolution encoding (--multilevel)
# ============================================================

def _encode_one_multilevel(
    payload: tuple[str, str, str],
) -> tuple[str, str, int, float]:
    """Worker entrypoint for the multilevel path. Reads the source file,
    streams multilevel items, encodes each, writes to encoded_multilevel.
    Returns (source, domain, n_written, seconds)."""
    source, domain, db_path = payload
    return source, domain, *_encode_source_multilevel(source, domain, db_path)


def _encode_source_multilevel(
    source_file: str, domain: str, db_path: str,
) -> tuple[int, float]:
    # Lazy import — keeps the GloVe binary out of the parent process.
    from backend.input import ENCODER_TEXT_ID, encode_text

    t0 = time.perf_counter()

    with open(source_file, "r", encoding="utf-8", errors="replace") as f:
        full_text = f.read()
    items = list(multilevel_stream(full_text))
    n_total = len(items)

    conn = sqlite3.connect(db_path, timeout=60.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    cur = conn.cursor()

    cur.execute(
        "INSERT OR REPLACE INTO encoding_progress_multilevel "
        "(source_file, domain, total_items, encoded_items, completed) "
        "VALUES (?, ?, ?, 0, 0)",
        (source_file, domain, n_total),
    )
    conn.commit()

    encoder_id = ENCODER_TEXT_ID
    encoded_at = time.time()

    batch: list[tuple] = []
    written = 0
    BATCH_SIZE = 500
    for position, item in enumerate(items):
        rep = encode_text(item.text)
        batch.append((
            source_file, domain, item.text, position,
            rep.astype(np.float32, copy=False).tobytes(),
            encoder_id, item.level, item.surprise_multiplier, encoded_at,
        ))
        if len(batch) >= BATCH_SIZE:
            cur.executemany(
                "INSERT INTO encoded_multilevel "
                "(source_file, domain, sentence, position, representation, "
                " encoder_id, level, surprise_multiplier, encoded_at) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                batch,
            )
            written += len(batch)
            batch.clear()
            cur.execute(
                "UPDATE encoding_progress_multilevel "
                "SET encoded_items = ? WHERE source_file = ?",
                (written, source_file),
            )
            conn.commit()

    if batch:
        cur.executemany(
            "INSERT INTO encoded_multilevel "
            "(source_file, domain, sentence, position, representation, "
            " encoder_id, level, surprise_multiplier, encoded_at) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            batch,
        )
        written += len(batch)

    cur.execute(
        "UPDATE encoding_progress_multilevel "
        "SET encoded_items = ?, completed = 1 WHERE source_file = ?",
        (written, source_file),
    )
    conn.commit()
    conn.close()

    return written, time.perf_counter() - t0


def parse_source_arg(arg: str) -> tuple[str, str]:
    if ":" in arg:
        path, domain = arg.rsplit(":", 1)
    else:
        path, domain = arg, "unknown"
    return path, domain


def _resolve_sources(args: argparse.Namespace) -> list[tuple[str, str]]:
    """Either parse the explicit `path:domain` args or read curriculum.json.
    For curriculum mode the JSON may use either the sequential `sequence`
    schema (book-step entries) or the interleaved `sources` schema; we
    accept both."""
    if args.sources:
        return [parse_source_arg(s) for s in args.sources]
    import json as _json
    with open(args.curriculum, "r", encoding="utf-8") as f:
        curr = _json.load(f)
    if "sources" in curr:
        out = [
            (s["source"], s.get("domain", "unknown"))
            for s in curr["sources"]
        ]
    else:
        out = [
            (s["source"], s.get("domain", "unknown"))
            for s in curr.get("sequence", [])
            if s.get("type") == "book"
        ]
    print(f"  curriculum {args.curriculum}: {len(out)} sources")
    return out


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
    ap.add_argument("--multilevel", action="store_true",
                    help="Encode word/phrase/sentence/paragraph items into the "
                         "encoded_multilevel table instead of the sentence-only "
                         "encoded_sentences table. Surprise multipliers per level "
                         "are stored alongside each row.")
    args = ap.parse_args()

    requested = _resolve_sources(args)

    if args.multilevel:
        init_multilevel_db(args.db)
        # The multilevel path uses its own progress table so a sentence-only
        # run and a multilevel run can coexist in the same DB without one
        # masking the other. Sources live in MULTILEVEL_DB_PATH by default;
        # honor the --db override so tests can isolate.
        db_path = args.db if args.db != DB_PATH else MULTILEVEL_DB_PATH
        if db_path != args.db:
            init_multilevel_db(db_path)
        pending: list[tuple[str, str, str]] = []
        for src, domain in requested:
            if not os.path.exists(src):
                print(f"  [skip] missing file: {src}")
                continue
            if is_source_encoded_multilevel(src, db_path):
                print(f"  [skip] already multilevel-encoded: {src}")
                continue
            pending.append((src, domain, db_path))

        if not pending:
            print("\nnothing to do.")
            return 0

        workers = args.workers or min(len(pending), mp.cpu_count())
        print(f"\nmultilevel-encoding {len(pending)} source(s) with {workers} worker(s)…")
        print(f"  db: {db_path}\n")

        t0 = time.perf_counter()
        if workers <= 1 or len(pending) == 1:
            results = [_encode_one_multilevel(p) for p in pending]
        else:
            with mp.Pool(workers) as pool:
                results = pool.map(_encode_one_multilevel, pending)
        duration = time.perf_counter() - t0

        total = sum(r[2] for r in results)
        print(f"\nencoded {total:,} multilevel items across "
              f"{len(results)} source(s) in {duration:.1f}s")
        for source, domain, n, secs in results:
            rate = n / max(secs, 1e-6)
            print(f"  {source} [{domain}]: {n:,} items in {secs:.1f}s ({rate:,.0f}/s)")
        return 0

    # ---- sentence-only path (default, unchanged) ----
    init_db(args.db)

    pending = []
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
