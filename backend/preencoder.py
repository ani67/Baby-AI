"""Pre-encode corpus sentences into a SQLite database.

This is the encoding-once-then-reuse path. ingest_book.py / run_curriculum.py
can read pre-encoded float32[D_REP] vectors directly from `encoded_corpus.db`
instead of running encode_text per sentence on every ingestion run.

Schema (locked v1):
    encoded_sentences(id, source_file, domain, sentence, position,
                      representation, encoder_id, encoded_at)
    encoding_progress(source_file PK, domain, total_sentences,
                      encoded_sentences, completed)

WAL mode is enabled so multiple worker processes can write concurrently
with serialized transactions. Each worker handles one source file end
to end, batching 500-row inserts. Workers don't share state.
"""
from __future__ import annotations

import os
import sqlite3
import time
from typing import Iterable

import numpy as np


DB_PATH = "data/encoded_corpus.db"
BATCH_SIZE = 500


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS encoded_sentences (
    id          INTEGER PRIMARY KEY,
    source_file TEXT NOT NULL,
    domain      TEXT NOT NULL,
    sentence    TEXT NOT NULL,
    position    INTEGER NOT NULL,
    representation BLOB NOT NULL,
    encoder_id  TEXT NOT NULL,
    encoded_at  REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_es_source ON encoded_sentences(source_file);
CREATE INDEX IF NOT EXISTS idx_es_domain ON encoded_sentences(domain);

CREATE TABLE IF NOT EXISTS encoding_progress (
    source_file TEXT PRIMARY KEY,
    domain      TEXT NOT NULL,
    total_sentences INTEGER NOT NULL,
    encoded_sentences INTEGER NOT NULL,
    completed   INTEGER DEFAULT 0
);
"""


def init_db(db_path: str = DB_PATH) -> None:
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=30.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    conn.close()


def is_source_encoded(source_file: str, db_path: str = DB_PATH) -> bool:
    if not os.path.exists(db_path):
        return False
    conn = sqlite3.connect(db_path, timeout=30.0)
    try:
        row = conn.execute(
            "SELECT completed FROM encoding_progress WHERE source_file = ?",
            (source_file,),
        ).fetchone()
        return bool(row and row[0])
    finally:
        conn.close()


def preprocess_text_file(path: str) -> list[str]:
    """Same preprocessing as ingest_book — kept-sentence list, 5–40 words."""
    # Inline import to break a circular dependency cycle (ingest_book imports
    # backend.* but we don't want backend.* importing ingest_book at module
    # load time).
    from ingest_book import keep_sentence, split_sentences, strip_book

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        raw = f.read()
    text = strip_book(raw)
    sentences = split_sentences(text)
    return [s for s in sentences if keep_sentence(s)]


def encode_source(
    source_file: str,
    domain: str,
    db_path: str = DB_PATH,
    batch_size: int = BATCH_SIZE,
) -> tuple[int, float]:
    """Encode every kept sentence in `source_file` and write to `db_path`.

    Returns (count_encoded, wall_seconds). Idempotent against a prior
    completed run (caller should check `is_source_encoded` first; this
    function will happily re-insert if called regardless).
    """
    # Lazy import so the heavy GloVe binary load happens once per worker
    # process, not at module import time.
    from backend.input import ENCODER_TEXT_ID, encode_text

    t0 = time.perf_counter()
    sentences = preprocess_text_file(source_file)
    n_total = len(sentences)

    conn = sqlite3.connect(db_path, timeout=60.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")    # acceptable durability for a derived corpus
    cur = conn.cursor()

    # Mark started.
    cur.execute(
        "INSERT OR REPLACE INTO encoding_progress "
        "(source_file, domain, total_sentences, encoded_sentences, completed) "
        "VALUES (?, ?, ?, 0, 0)",
        (source_file, domain, n_total),
    )
    conn.commit()

    encoder_id = ENCODER_TEXT_ID
    encoded_at = time.time()

    batch: list[tuple] = []
    written = 0
    for i, sentence in enumerate(sentences):
        rep = encode_text(sentence)
        # tobytes serializes float32 in C order.
        batch.append((source_file, domain, sentence, i,
                      rep.astype(np.float32, copy=False).tobytes(),
                      encoder_id, encoded_at))
        if len(batch) >= batch_size:
            cur.executemany(
                "INSERT INTO encoded_sentences "
                "(source_file, domain, sentence, position, representation, encoder_id, encoded_at) "
                "VALUES (?,?,?,?,?,?,?)",
                batch,
            )
            written += len(batch)
            batch.clear()
            cur.execute(
                "UPDATE encoding_progress SET encoded_sentences = ? WHERE source_file = ?",
                (written, source_file),
            )
            conn.commit()

    if batch:
        cur.executemany(
            "INSERT INTO encoded_sentences "
            "(source_file, domain, sentence, position, representation, encoder_id, encoded_at) "
            "VALUES (?,?,?,?,?,?,?)",
            batch,
        )
        written += len(batch)

    cur.execute(
        "UPDATE encoding_progress SET encoded_sentences = ?, completed = 1 WHERE source_file = ?",
        (written, source_file),
    )
    conn.commit()
    conn.close()

    return written, time.perf_counter() - t0


def fetch_encoded(
    source_file: str,
    db_path: str = DB_PATH,
) -> Iterable[tuple[str, np.ndarray, int]]:
    """Yield (sentence, representation, position) for one previously-encoded source."""
    conn = sqlite3.connect(db_path, timeout=30.0)
    try:
        cur = conn.execute(
            "SELECT sentence, representation, position FROM encoded_sentences "
            "WHERE source_file = ? ORDER BY position",
            (source_file,),
        )
        for sentence, rep_blob, position in cur:
            rep = np.frombuffer(rep_blob, dtype=np.float32).copy()
            yield sentence, rep, position
    finally:
        conn.close()


def progress_for(source_file: str, db_path: str = DB_PATH) -> dict | None:
    if not os.path.exists(db_path):
        return None
    conn = sqlite3.connect(db_path, timeout=30.0)
    try:
        row = conn.execute(
            "SELECT domain, total_sentences, encoded_sentences, completed "
            "FROM encoding_progress WHERE source_file = ?",
            (source_file,),
        ).fetchone()
        if not row:
            return None
        return {
            "source_file": source_file,
            "domain": row[0],
            "total_sentences": row[1],
            "encoded_sentences": row[2],
            "completed": bool(row[3]),
        }
    finally:
        conn.close()
