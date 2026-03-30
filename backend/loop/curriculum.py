import os
import random
import sqlite3
import struct
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Toggle: "live" (Ollama teacher) or "precomputed" (embedding_cache)
# ---------------------------------------------------------------------------

CURRICULUM_SOURCE = os.getenv("CURRICULUM_SOURCE", "precomputed")


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
    patches: torch.Tensor | None = None         # C.3: (49, 512) patch-level features
    sequence: list[torch.Tensor] | None = None  # sequential vectors (per-word or per-patch)


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
        self._step_times: deque[float] = deque(maxlen=1000)
        self._has_image_url: bool = False
        self._patches: dict[int, torch.Tensor] | None = None  # C.3: lazy-loaded
        self._patches_path: str | None = None
        # Sequential episode state
        self._episode_ids: list[int] = []       # pre-fetched IDs for current episode
        self._episode_cursor: int = 0
        self._episode_length: int = 16
        self._category_list: list[str] = []     # all categories, shuffled
        self._category_cursor: int = 0

    def open(self) -> int:
        """Open the database and index image_ids.  Returns count."""
        if not os.path.exists(self._db_path):
            return 0
        try:
            self._conn = sqlite3.connect(self._db_path, check_same_thread=False, timeout=10)
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
            # patch_features.pt is 11GB — skip loading to avoid OOM.
            # Patches are optional enrichment, not needed for core training.
            self._patches_path = None
            # Sequential episodes: discover categories
            self._init_categories()
        except Exception as e:
            print(f"[curriculum] embedding_cache open error: {e}", flush=True)
            self._conn = None
            self._count = 0
        return self._count

    def _init_categories(self) -> None:
        """Discover available categories for sequential episode sampling."""
        if not self._conn:
            return
        try:
            rows = self._conn.execute(
                "SELECT DISTINCT category FROM embedding_cache "
                "WHERE category IS NOT NULL AND category != ''"
            ).fetchall()
            self._category_list = [r[0] for r in rows]
            random.shuffle(self._category_list)
            if self._category_list:
                print(f"[curriculum] {len(self._category_list)} categories for sequential episodes", flush=True)
        except Exception:
            self._category_list = []

    def _start_episode(self) -> bool:
        """Pick next category and pre-fetch episode IDs. Returns True if episode started."""
        if not self._conn or not self._category_list:
            return False
        # Round-robin through categories, shuffle on wrap
        if self._category_cursor >= len(self._category_list):
            random.shuffle(self._category_list)
            self._category_cursor = 0
        category = self._category_list[self._category_cursor]
        self._category_cursor += 1
        rows = self._conn.execute(
            "SELECT image_id FROM embedding_cache WHERE category=? ORDER BY RANDOM() LIMIT ?",
            (category, self._episode_length),
        ).fetchall()
        if not rows:
            return False
        self._episode_ids = [r[0] for r in rows]
        self._episode_cursor = 0
        self._episode_category = category
        print(f"[curriculum] episode: {category} ({len(self._episode_ids)} items)", flush=True)
        return True

    def sample_sequential(self) -> CurriculumItem | None:
        """Return next item from current episode, starting a new one if needed."""
        if self._episode_cursor >= len(self._episode_ids):
            if not self._start_episode():
                return self.sample()
        if self._episode_cursor >= len(self._episode_ids):
            return self.sample()
        iid = self._episode_ids[self._episode_cursor]
        self._episode_cursor += 1
        return self._fetch_item(iid)

    def _fetch_item(self, iid: int) -> CurriculumItem | None:
        """Fetch a single CurriculumItem by image_id."""
        if not self._conn:
            return None
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
        # Lazy-load patch features on first access
        if self._patches is None and self._patches_path is not None:
            self._patches = torch.load(self._patches_path, weights_only=False)
            print(f"[curriculum] loaded {len(self._patches)} patch features", flush=True)
        patches = self._patches.get(iid) if self._patches else None
        return CurriculumItem(
            id=f"coco_{iid}",
            stage=0,
            item_type="image",
            input_vector=image_emb,
            expected_vector=caption_emb,
            label=getattr(self, '_episode_category', None),
            description=caption_text,
            context=caption_text,
            template_slots={"description": caption_text},
            stage_relevance=1.0,
            image_path=None,
            image_url=image_url,
            precomputed=True,
            patches=patches,
        )

    def sample(self) -> CurriculumItem | None:
        """Return one random item from the cache (fallback for non-sequential)."""
        if not self._conn or self._count == 0:
            return None
        if self._cursor_idx >= self._count:
            random.shuffle(self._ids)
            self._cursor_idx = 0
        iid = self._ids[self._cursor_idx]
        self._cursor_idx += 1
        return self._fetch_item(iid)

    def sample_batch(self, n: int) -> list[CurriculumItem]:
        """Return up to n items from the cache (sequential episodes)."""
        items: list[CurriculumItem] = []
        for _ in range(n):
            item = self.sample_sequential()
            if item is None:
                break
            items.append(item)
        return items

    def sample_adversarial(self, n: int, category_weights: dict[str, float]) -> list[CurriculumItem]:
        """Sample n items, weighted toward categories the model is worst at."""
        items: list[CurriculumItem] = []
        if not self._conn:
            return items
        # Try to get items matching worst categories first (sorted ascending by weight = worst first)
        for category, weight in sorted(category_weights.items(), key=lambda x: -x[1]):
            if len(items) >= n:
                break
            limit = max(1, int(n * weight))
            # Use indexed category column (exact match, ~12ms) instead of
            # LIKE '%word%' (~142ms per query). Falls back to LIKE if no category column.
            try:
                if self._has_image_url:
                    rows = self._conn.execute(
                        "SELECT image_id, image_emb, caption_emb, caption_text, image_url "
                        "FROM embedding_cache WHERE category=? ORDER BY RANDOM() LIMIT ?",
                        (category, limit),
                    ).fetchall()
                else:
                    rows = self._conn.execute(
                        "SELECT image_id, image_emb, caption_emb, caption_text "
                        "FROM embedding_cache WHERE category=? ORDER BY RANDOM() LIMIT ?",
                        (category, limit),
                    ).fetchall()
            except Exception:
                # Fallback if category column doesn't exist
                rows = self._conn.execute(
                    "SELECT image_id, image_emb, caption_emb, caption_text "
                    "FROM embedding_cache WHERE caption_text LIKE ? ORDER BY RANDOM() LIMIT ?",
                    (f'%{category}%', limit),
                ).fetchall()
            for row in rows:
                if len(items) >= n:
                    break
                image_emb = _bytes_to_tensor(row["image_emb"])
                caption_emb = _bytes_to_tensor(row["caption_emb"])
                caption_text = row["caption_text"]
                image_url = row["image_url"] if self._has_image_url else None
                items.append(CurriculumItem(
                    id=f"coco_{row['image_id']}", stage=0, item_type="image",
                    input_vector=image_emb, expected_vector=caption_emb,
                    label=None, description=caption_text, context=caption_text,
                    template_slots={"description": caption_text},
                    stage_relevance=1.0, image_path=None, image_url=image_url,
                    precomputed=True,
                ))
        # Fill remaining with random samples
        while len(items) < n:
            item = self.sample()
            if item:
                items.append(item)
            else:
                break
        return items

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
            _default_db = os.getenv("DB_PATH", "data/dev.db")
            resolved_db = db_path or _default_db
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
            item = self._cache.sample_sequential()
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

    def next_batch(self, n: int, stage: int, model_state: dict, category_weights: dict[str, float] | None = None) -> list[CurriculumItem]:
        """Return up to n curriculum items at once. Precomputed uses bulk cache read.

        If category_weights is provided, uses adversarial sampling — categories
        the model is worst at get proportionally more exposure.
        """
        if self._source == "precomputed" and self._cache is not None:
            # Sequential episodes (default path)
            items = self._cache.sample_batch(n)
            if items:
                self._step_count += len(items)
                return items

        # Live fallback: sample n items individually
        return [self.next_item(stage, model_state) for _ in range(n)]

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
