"""Component 9: Knowledge Store

External facts — retrieved, never hallucinated. Separate from model weights.
Grows from user corrections and teacher responses. Facts with confidence > 0.7
are prepended to prompts before inference.

Storage:
  data/facts.json     — full fact data (human-readable)
  data/chroma_db/     — ChromaDB vector index (collection: "knowledge_store")
"""

import json
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import chromadb

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import component3

# ---------- Fact dataclass ----------


@dataclass
class Fact:
    id: str
    fact: str
    source: str           # "user_correction" | "teacher" | "search"
    confidence: float     # 0-1
    times_retrieved: int
    timestamp: float


# ---------- module state (set by init) ----------

_data_dir: Path | None = None
_facts_file: Path | None = None
_facts: dict[str, Fact] = {}
_collection: chromadb.Collection | None = None
_initialised = False

# ---------- init / reset ----------


def init(data_dir: str | Path | None = None):
    """Initialise the knowledge store.

    Args:
        data_dir: Directory for facts.json and chroma_db/.
                  Defaults to <project_root>/data.
    """
    global _data_dir, _facts_file, _facts, _collection, _initialised

    if data_dir is None:
        _data_dir = Path(__file__).resolve().parent.parent / "data"
    else:
        _data_dir = Path(data_dir)

    _data_dir.mkdir(parents=True, exist_ok=True)
    _facts_file = _data_dir / "facts.json"

    # Load facts from disk
    _facts = _load_facts_from_disk()

    # ChromaDB persistent client — same directory as component4, different collection
    chroma_dir = _data_dir / "chroma_db"
    client = chromadb.PersistentClient(path=str(chroma_dir))
    _collection = client.get_or_create_collection("knowledge_store")

    # Sync: re-index any facts missing from ChromaDB
    existing_ids = set(_collection.get()["ids"]) if _collection.count() > 0 else set()
    missing = [f for f in _facts.values() if f.id not in existing_ids]
    if missing:
        _collection.add(
            ids=[f.id for f in missing],
            documents=[f.fact for f in missing],
        )

    _initialised = True


def _ensure_init():
    if not _initialised:
        init()


# ---------- persistence helpers ----------


def _load_facts_from_disk() -> dict[str, Fact]:
    if _facts_file is None or not _facts_file.exists():
        return {}
    with open(_facts_file, "r") as f:
        raw = json.load(f)
    return {fact["id"]: Fact(**fact) for fact in raw}


def _save_facts_to_disk():
    if _facts_file is None:
        return
    _facts_file.parent.mkdir(parents=True, exist_ok=True)
    with open(_facts_file, "w") as f:
        json.dump([asdict(fact) for fact in _facts.values()], f, indent=2)


# ---------- public API ----------


def store_fact(
    fact: str,
    source: str,
    confidence: float,
) -> str:
    """Store a new fact. Writes to disk immediately. Returns fact_id."""
    _ensure_init()
    import time

    fact_id = uuid.uuid4().hex[:16]
    fact_obj = Fact(
        id=fact_id,
        fact=fact,
        source=source,
        confidence=round(confidence, 4),
        times_retrieved=0,
        timestamp=time.time(),
    )

    _facts[fact_id] = fact_obj
    _save_facts_to_disk()
    _collection.add(ids=[fact_id], documents=[fact])

    return fact_id


def retrieve_facts(query: str, n: int = 5) -> list[Fact]:
    """Return up to N facts whose text is most similar to the query.

    Increments times_retrieved on each returned fact and persists.
    """
    _ensure_init()

    if _collection.count() == 0:
        return []

    actual_n = min(n, _collection.count())
    results = _collection.query(query_texts=[query], n_results=actual_n)

    matched_ids = results["ids"][0]
    facts = []
    for fid in matched_ids:
        fact = _facts.get(fid)
        if fact is not None:
            fact.times_retrieved += 1
            facts.append(fact)

    if facts:
        _save_facts_to_disk()

    return facts


def get_fact_count() -> int:
    """Total number of stored facts."""
    _ensure_init()
    return len(_facts)


# ---------- prompt augmentation ----------


def _augment_prompt(prompt: str) -> str:
    """Prepend high-confidence facts to a prompt.

    Retrieves top-5 facts similar to the prompt. Any with confidence > 0.7
    are prepended as context. No-op if knowledge store not yet initialised.
    """
    if not _initialised:
        return prompt

    if _collection.count() == 0:
        return prompt

    facts = retrieve_facts(prompt, n=5)
    high_conf = [f for f in facts if f.confidence > 0.7]

    if not high_conf:
        return prompt

    fact_texts = ". ".join(f.fact for f in high_conf)
    return f"Relevant facts: {fact_texts}. Now answer: {prompt}"


# ---------- hook into component3 ----------

# Wrap component3.query to prepend facts before inference
_original_query = component3.query


def _augmented_query(prompt: str, max_tokens: int = 256) -> str:
    """Query with fact augmentation: retrieve relevant facts, prepend to prompt."""
    augmented = _augment_prompt(prompt)
    return _original_query(augmented, max_tokens)


component3.query = _augmented_query

# Wrap component3.submit_correction to auto-store corrections as facts
_original_submit_correction = component3.submit_correction


def _augmented_submit_correction(prompt: str, correct_completion: str) -> None:
    """Submit correction and auto-store it as a fact. No-op if not initialised."""
    _original_submit_correction(prompt, correct_completion)
    if _initialised:
        store_fact(correct_completion, "user_correction", 0.95)


component3.submit_correction = _augmented_submit_correction
