"""Multi-resolution text preprocessor.

Same source text → words + phrases + sentences + paragraphs.
Each granularity reveals different structure in the same material.

Lower-level items (words, phrases) carry a higher surprise multiplier so they
cross threshold more easily; paragraphs carry a lower multiplier so they're
rarer. The multiplier is consumed by `PredictionEngine.set_surprise_multiplier`.

    word       → 1.5  (lower effective threshold → more writes)
    phrase     → 1.3
    sentence   → 1.0  (current baseline unchanged)
    paragraph  → 0.8  (higher threshold → fewer writes)

The bottom of this module also owns the `encoded_multilevel` SQLite table
(read + write helpers). It lives here rather than in `preencoder.py`
because the table's schema (level + surprise_multiplier columns) and the
producer (extract_multilevel) are inseparable concerns.
"""
from __future__ import annotations

import os
import re
import sqlite3
from dataclasses import dataclass
from typing import Iterable, Iterator

import numpy as np
import spacy


_nlp = None


def get_nlp():
    """Lazy-load the spaCy English pipeline. One process loads it once."""
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


@dataclass
class PreprocessedItem:
    text: str
    level: str              # 'word' | 'phrase' | 'sentence' | 'paragraph'
    surprise_multiplier: float


def extract_multilevel(text: str) -> list[PreprocessedItem]:
    """Yield word, phrase, sentence, and paragraph items from one paragraph
    of text. Deduplicates words and phrases within the call so a paragraph
    that mentions "justice" five times produces one word item, not five.
    """
    items: list[PreprocessedItem] = []
    doc = get_nlp()(text)

    # LEVEL 0 — significant words (lemmatized, deduplicated).
    seen_words: set[str] = set()
    for token in doc:
        if (
            token.pos_ in ("NOUN", "VERB", "ADJ", "PROPN")
            and not token.is_stop
            and not token.is_punct
            and len(token.lemma_) > 3
            and token.lemma_.isalpha()
            and token.lemma_ not in seen_words
        ):
            seen_words.add(token.lemma_)
            items.append(PreprocessedItem(
                text=token.lemma_, level="word", surprise_multiplier=1.5,
            ))

    # LEVEL 1 — noun phrases (2-5 words).
    seen_phrases: set[str] = set()
    for chunk in doc.noun_chunks:
        phrase = chunk.text.lower().strip()
        wc = len(phrase.split())
        if 2 <= wc <= 5 and phrase not in seen_phrases and len(phrase) > 6:
            seen_phrases.add(phrase)
            items.append(PreprocessedItem(
                text=phrase, level="phrase", surprise_multiplier=1.3,
            ))

    # LEVEL 2 — sentences (existing baseline, unchanged).
    for sent in doc.sents:
        wc = len(sent.text.split())
        if 5 <= wc <= 40:
            items.append(PreprocessedItem(
                text=sent.text.strip(), level="sentence", surprise_multiplier=1.0,
            ))

    # LEVEL 3 — paragraph itself.
    wc = len(text.split())
    if 20 <= wc <= 200:
        items.append(PreprocessedItem(
            text=text.strip(), level="paragraph", surprise_multiplier=0.8,
        ))

    return items


def extract_paragraphs(full_text: str) -> list[str]:
    """Split a full document into paragraphs by blank-line separators.
    Paragraphs >300 words are chunked into 200-word slices so a single
    very-long paragraph still produces useful paragraph-level items.
    """
    raw = re.split(r"\n\s*\n", full_text)
    paragraphs: list[str] = []
    for p in raw:
        p = p.strip()
        wc = len(p.split())
        if 20 <= wc <= 300:
            paragraphs.append(p)
        elif wc > 300:
            words = p.split()
            for i in range(0, len(words), 200):
                chunk = " ".join(words[i:i + 200])
                if len(chunk.split()) >= 20:
                    paragraphs.append(chunk)
    return paragraphs


def multilevel_stream(full_text: str) -> Iterator[PreprocessedItem]:
    """Stream all multilevel items across every paragraph of a document.
    Order within a paragraph is words → phrases → sentences → paragraph.
    """
    for paragraph in extract_paragraphs(full_text):
        for item in extract_multilevel(paragraph):
            yield item


# ============================================================
# encoded_multilevel SQLite table
# ============================================================

MULTILEVEL_DB_PATH = "data/encoded_corpus.db"

MULTILEVEL_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS encoded_multilevel (
    id          INTEGER PRIMARY KEY,
    source_file TEXT NOT NULL,
    domain      TEXT NOT NULL,
    sentence    TEXT NOT NULL,
    position    INTEGER NOT NULL,
    representation BLOB NOT NULL,
    encoder_id  TEXT NOT NULL,
    level       TEXT NOT NULL,
    surprise_multiplier REAL NOT NULL,
    encoded_at  REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_em_source ON encoded_multilevel(source_file);
CREATE INDEX IF NOT EXISTS idx_em_domain ON encoded_multilevel(domain);
CREATE INDEX IF NOT EXISTS idx_em_level  ON encoded_multilevel(level);

CREATE TABLE IF NOT EXISTS encoding_progress_multilevel (
    source_file TEXT PRIMARY KEY,
    domain      TEXT NOT NULL,
    total_items INTEGER NOT NULL,
    encoded_items INTEGER NOT NULL,
    completed   INTEGER DEFAULT 0
);
"""


def init_multilevel_db(db_path: str = MULTILEVEL_DB_PATH) -> None:
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=30.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(MULTILEVEL_SCHEMA_SQL)
    conn.commit()
    conn.close()


def is_source_encoded_multilevel(
    source_file: str, db_path: str = MULTILEVEL_DB_PATH,
) -> bool:
    if not os.path.exists(db_path):
        return False
    conn = sqlite3.connect(db_path, timeout=30.0)
    try:
        row = conn.execute(
            "SELECT completed FROM encoding_progress_multilevel WHERE source_file = ?",
            (source_file,),
        ).fetchone()
        return bool(row and row[0])
    finally:
        conn.close()


def fetch_encoded_multilevel(
    source_file: str,
    db_path: str = MULTILEVEL_DB_PATH,
) -> Iterable[tuple[str, np.ndarray, int, str, float]]:
    """Yield (sentence, representation, position, level, surprise_multiplier)
    for one previously-encoded source, ordered by position.
    """
    conn = sqlite3.connect(db_path, timeout=30.0)
    try:
        cur = conn.execute(
            "SELECT sentence, representation, position, level, surprise_multiplier "
            "FROM encoded_multilevel "
            "WHERE source_file = ? ORDER BY position",
            (source_file,),
        )
        for sentence, rep_blob, position, level, mult in cur:
            rep = np.frombuffer(rep_blob, dtype=np.float32).copy()
            yield sentence, rep, int(position), level, float(mult)
    finally:
        conn.close()
