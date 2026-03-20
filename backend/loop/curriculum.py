import os
import random
import sqlite3
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Toggle: "live" (Ollama teacher) or "precomputed" (embedding_cache)
# ---------------------------------------------------------------------------

CURRICULUM_SOURCE = os.getenv("CURRICULUM_SOURCE", "live")


@dataclass
class CurriculumItem:
    id: str
    stage: int
    item_type: str                              # "image"|"image_pair"|"video"|"concept"
    input_vector: torch.Tensor | None           # pre-encoded input
    expected_vector: torch.Tensor | None        # pre-encoded expected answer
    label: str | None
    description: str | None
    context: str | None
    template_slots: dict = field(default_factory=dict)
    stage_relevance: float = 1.0
    image_path: str | None = None               # path to image on disk (for vision models)
    image_url: str | None = None                # remote URL for frontend thumbnail
    precomputed: bool = False                    # True when item came from embedding_cache


class EmptyPoolError(Exception):
    pass


# ---------------------------------------------------------------------------
# Embedding cache reader
# ---------------------------------------------------------------------------

def _bytes_to_tensor(blob: bytes) -> torch.Tensor:
    """Unpack a BLOB of float32 values into a normalised torch tensor."""
    n = len(blob) // 4
    vals = struct.unpack(f"{n}f", blob)
    return F.normalize(torch.tensor(vals, dtype=torch.float32), dim=0)


class _EmbeddingCache:
    """Lazy reader over the embedding_cache table in the existing SQLite db."""

    def __init__(self, db_path: str):
        self._db_path = db_path
        self._conn: sqlite3.Connection | None = None
        self._ids: list[int] = []
        self._count: int = 0
        self._cursor_idx: int = 0  # sequential position for round-robin
        self._step_times: list[float] = []
        self._has_image_url: bool = False

    def open(self) -> int:
        """Open the database and index image_ids.  Returns count."""
        if not os.path.exists(self._db_path):
            return 0
        try:
            self._conn = sqlite3.connect(self._db_path)
            self._conn.row_factory = sqlite3.Row
            # Check table exists
            row = self._conn.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='embedding_cache'"
            ).fetchone()
            if row[0] == 0:
                self._conn.close()
                self._conn = None
                return 0
            self._ids = [
                r[0] for r in self._conn.execute(
                    "SELECT image_id FROM embedding_cache ORDER BY image_id"
                ).fetchall()
            ]
            self._count = len(self._ids)
            random.shuffle(self._ids)
            # Detect image_url column for backwards compat
            cols = {r[1] for r in self._conn.execute("PRAGMA table_info(embedding_cache)").fetchall()}
            self._has_image_url = "image_url" in cols
        except Exception as e:
            print(f"[curriculum] embedding_cache open error: {e}", flush=True)
            self._conn = None
            self._count = 0
        return self._count

    def sample(self) -> CurriculumItem | None:
        """Return one random item from the cache."""
        if not self._conn or self._count == 0:
            return None
        # Round-robin through shuffled IDs; re-shuffle on wrap
        if self._cursor_idx >= self._count:
            random.shuffle(self._ids)
            self._cursor_idx = 0
        iid = self._ids[self._cursor_idx]
        self._cursor_idx += 1

        if self._has_image_url:
            row = self._conn.execute(
                "SELECT image_emb, caption_emb, caption_text, image_url FROM embedding_cache WHERE image_id=?",
                (iid,),
            ).fetchone()
        else:
            row = self._conn.execute(
                "SELECT image_emb, caption_emb, caption_text FROM embedding_cache WHERE image_id=?",
                (iid,),
            ).fetchone()
        if row is None:
            return None

        image_emb = _bytes_to_tensor(row["image_emb"])
        caption_emb = _bytes_to_tensor(row["caption_emb"])
        caption_text = row["caption_text"]
        image_url = row["image_url"] if self._has_image_url else None

        return CurriculumItem(
            id=f"coco_{iid}",
            stage=0,
            item_type="image",
            input_vector=image_emb,
            expected_vector=caption_emb,
            label=None,
            description=caption_text,
            context=caption_text,
            template_slots={"description": caption_text},
            stage_relevance=1.0,
            image_path=None,
            image_url=image_url,
            precomputed=True,
        )

    def record_step_time(self, ms: float) -> None:
        self._step_times.append(ms)

    def avg_step_ms(self) -> float:
        if not self._step_times:
            return 0.0
        return sum(self._step_times) / len(self._step_times)

    @property
    def count(self) -> int:
        return self._count


# ---------------------------------------------------------------------------
# Curriculum
# ---------------------------------------------------------------------------

class Curriculum:
    """
    The pool of experiences available at each stage.

    When CURRICULUM_SOURCE=precomputed, items come from embedding_cache
    (pre-computed CLIP embeddings from MS-COCO).  Otherwise the original
    live image/concept pools are used.
    """

    def __init__(self, data_dir: str = "backend/data", db_path: str | None = None):
        self._pools: dict[int, list[CurriculumItem]] = {
            0: [], 1: [], 2: [], 3: [], 4: [],
        }
        self._data_dir = data_dir
        self._source = CURRICULUM_SOURCE
        self._step_count = 0

        # Precomputed cache (lazy)
        self._cache: _EmbeddingCache | None = None
        if self._source == "precomputed":
            _default_db = str(Path(__file__).resolve().parent.parent / "state" / "dev.db")
            resolved_db = db_path or os.getenv("DB_PATH", _default_db)
            self._cache = _EmbeddingCache(resolved_db)
            count = self._cache.open()
            if count > 0:
                print(f"[curriculum] source=precomputed items={count}", flush=True)
            else:
                print("[curriculum] source=precomputed but embedding_cache is empty — falling back to live", flush=True)
                self._source = "live"
                self._cache = None

        if self._source == "live":
            print("[curriculum] source=live", flush=True)

        # Always load the live pools (needed for add_teacher_vocabulary, etc.)
        self._load_stage_0()
        self._load_concepts()

    def next_item(self, stage: int, model_state: dict) -> CurriculumItem:
        self._step_count += 1
        t0 = time.time()

        if self._source == "precomputed" and self._cache is not None:
            item = self._cache.sample()
            if item is not None:
                ms = (time.time() - t0) * 1000
                self._cache.record_step_time(ms)
                if self._step_count % 100 == 0:
                    print(
                        f"[curriculum] step={self._step_count} avg_step_ms={self._cache.avg_step_ms():.2f}",
                        flush=True,
                    )
                return item
            # Cache exhausted or error — fall through to live pools

        # Live mode
        pool = self._pools.get(stage, []) + self._pools.get(stage - 1, [])
        if not pool:
            pool = [item for items in self._pools.values() for item in items]
        if not pool:
            raise EmptyPoolError(f"No curriculum items available for stage {stage}")

        item = random.choice(pool)

        if self._step_count % 100 == 0 and self._source == "live":
            ms = (time.time() - t0) * 1000
            print(f"[curriculum] step={self._step_count} avg_step_ms={ms:.2f}", flush=True)

        return item

    def add_item(self, item: CurriculumItem) -> None:
        self._pools[item.stage].append(item)

    def add_image(self, image, label: str | None = None, image_path: str | None = None) -> CurriculumItem:
        item = CurriculumItem(
            id=f"img_{uuid4().hex[:8]}",
            stage=0,
            item_type="image",
            input_vector=None,
            expected_vector=None,
            label=label,
            description=f"image{' of ' + label if label else ''}",
            context=None,
            template_slots={"description": label or "this image"},
            stage_relevance=1.0,
            image_path=image_path,
        )
        self._pools[0].append(item)
        return item

    def add_teacher_vocabulary(self, word: str) -> None:
        item = CurriculumItem(
            id=f"word_{word}",
            stage=4,
            item_type="concept",
            input_vector=None,
            expected_vector=None,
            label=word,
            description=word,
            context=None,
            template_slots={"concept": word},
            stage_relevance=1.0,
        )
        if not any(i.label == word for i in self._pools[4]):
            self._pools[4].append(item)

    def _load_stage_0(self) -> None:
        stage0_dir = Path(self._data_dir) / "stage0"
        if stage0_dir.exists():
            for category_dir in stage0_dir.iterdir():
                if not category_dir.is_dir():
                    continue
                label = category_dir.name
                for img_path in list(category_dir.glob("*.jpg")) + list(category_dir.glob("*.png")):
                    item = CurriculumItem(
                        id=f"img_{img_path.stem}",
                        stage=0,
                        item_type="image",
                        input_vector=None,
                        expected_vector=None,
                        label=label,
                        description=f"a {label}",
                        context=None,
                        template_slots={"description": f"a {label}"},
                        stage_relevance=1.0,
                        image_path=str(img_path),
                    )
                    self._pools[0].append(item)

    def _load_concepts(self) -> None:
        """Load text-only concept items from concepts.txt."""
        concepts_path = Path(self._data_dir) / "stage0" / "concepts.txt"
        if not concepts_path.exists():
            return
        count = 0
        for line in concepts_path.read_text().splitlines():
            word = line.strip()
            if not word:
                continue
            item = CurriculumItem(
                id=f"concept_{word}",
                stage=0,
                item_type="concept",
                input_vector=None,
                expected_vector=None,
                label=word,
                description=f"a {word}",
                context=None,
                template_slots={"description": f"a {word}", "concept": word},
                stage_relevance=0.7,
            )
            # Avoid duplicates
            if not any(i.id == item.id for i in self._pools[0]):
                self._pools[0].append(item)
                count += 1
        if count:
            print(f"Loaded {count} text concepts from concepts.txt")

        # Fallback if nothing loaded at all
        if not any(items for items in self._pools.values()):
            fallback_concepts = [
                "dog", "cat", "tree", "car", "bird",
                "fish", "house", "ball", "sun", "flower",
            ]
            for concept in fallback_concepts:
                item = CurriculumItem(
                    id=f"concept_{concept}",
                    stage=0,
                    item_type="concept",
                    input_vector=None,
                    expected_vector=None,
                    label=concept,
                    description=f"a {concept}",
                    context=None,
                    template_slots={"description": f"a {concept}"},
                    stage_relevance=1.0,
                )
                self._pools[0].append(item)
            print("WARNING: No images in data/stage0/ — using fallback concept items.")
