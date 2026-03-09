"""Component 4: Episodic Store

Persistent memory of interactions. Survives restarts. Each entry scored.
The "hippocampus."

Storage:
  data/episodes.json   — full episode data (human-readable)
  data/chroma_db/      — ChromaDB vector index for similarity search

ChromaDB uses its built-in all-MiniLM-L6-v2 ONNX model for embeddings
(~80 MB, no separate sentence-transformers dependency needed).
"""

import json
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import chromadb

# ---------- Episode dataclass ----------


@dataclass
class Episode:
    id: str
    prompt: str
    response: str
    correction: Optional[str]
    timestamp: float
    importance_score: float = 1.0
    times_referenced: int = 0


# ---------- module state (set by init) ----------

_data_dir: Path | None = None
_episodes_file: Path | None = None
_episodes: dict[str, Episode] = {}
_collection: chromadb.Collection | None = None
_initialised = False

# Scoring hook — set by Component 5 to auto-score new episodes
_score_fn = None  # Optional[Callable[[Episode], float]], set by Component 5


# ---------- init / reset ----------


def init(data_dir: str | Path | None = None):
    """Initialise the episodic store.

    Args:
        data_dir: Directory for episodes.json and chroma_db/.
                  Defaults to <project_root>/data.
    """
    global _data_dir, _episodes_file, _episodes, _collection, _initialised

    if data_dir is None:
        _data_dir = Path(__file__).resolve().parent.parent / "data"
    else:
        _data_dir = Path(data_dir)

    _data_dir.mkdir(parents=True, exist_ok=True)
    _episodes_file = _data_dir / "episodes.json"

    # Load episodes from disk
    _episodes = _load_episodes_from_disk()

    # ChromaDB persistent client
    chroma_dir = _data_dir / "chroma_db"
    client = chromadb.PersistentClient(path=str(chroma_dir))
    _collection = client.get_or_create_collection("episodes")

    # Sync: re-index any episodes missing from ChromaDB
    existing_ids = set(_collection.get()["ids"]) if _collection.count() > 0 else set()
    missing = [ep for ep in _episodes.values() if ep.id not in existing_ids]
    if missing:
        _collection.add(
            ids=[ep.id for ep in missing],
            documents=[ep.prompt for ep in missing],
        )

    _initialised = True


def _ensure_init():
    if not _initialised:
        init()


# ---------- persistence helpers ----------


def _load_episodes_from_disk() -> dict[str, Episode]:
    if _episodes_file is None or not _episodes_file.exists():
        return {}
    with open(_episodes_file, "r") as f:
        raw = json.load(f)
    return {ep["id"]: Episode(**ep) for ep in raw}


def _save_episodes_to_disk():
    if _episodes_file is None:
        return
    with open(_episodes_file, "w") as f:
        json.dump([asdict(ep) for ep in _episodes.values()], f, indent=2)


# ---------- public API ----------


def store_episode(
    prompt: str,
    response: str,
    correction: Optional[str],
    timestamp: float,
) -> str:
    """Store a new episode. Writes to disk immediately. Returns episode_id."""
    _ensure_init()

    episode_id = uuid.uuid4().hex[:16]
    episode = Episode(
        id=episode_id,
        prompt=prompt,
        response=response,
        correction=correction,
        timestamp=timestamp,
    )

    # Auto-score if Component 5 has registered a scoring function
    if _score_fn is not None:
        episode.importance_score = _score_fn(episode)

    _episodes[episode_id] = episode
    _save_episodes_to_disk()
    _collection.add(ids=[episode_id], documents=[prompt])

    return episode_id


def get_recent_episodes(n: int = 50) -> list[Episode]:
    """Return the N most recent episodes, sorted newest-first."""
    _ensure_init()
    sorted_eps = sorted(_episodes.values(), key=lambda e: e.timestamp, reverse=True)
    return sorted_eps[:n]


def get_similar_episodes(prompt: str, n: int = 10) -> list[Episode]:
    """Return up to N episodes whose prompts are most similar to the query.

    Increments times_referenced on each returned episode and persists.
    """
    _ensure_init()

    if _collection.count() == 0:
        return []

    actual_n = min(n, _collection.count())
    results = _collection.query(query_texts=[prompt], n_results=actual_n)

    matched_ids = results["ids"][0]
    episodes = []
    for eid in matched_ids:
        ep = _episodes.get(eid)
        if ep is not None:
            ep.times_referenced += 1
            episodes.append(ep)

    if episodes:
        _save_episodes_to_disk()

    return episodes


def get_episode_by_id(episode_id: str) -> Optional[Episode]:
    """Retrieve a single episode by ID."""
    _ensure_init()
    return _episodes.get(episode_id)


def update_episode(episode: Episode):
    """Update an existing episode in-memory and on disk."""
    _ensure_init()
    _episodes[episode.id] = episode
    _save_episodes_to_disk()


def get_episode_count() -> int:
    """Total number of stored episodes."""
    _ensure_init()
    return len(_episodes)
