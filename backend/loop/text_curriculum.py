"""
Text curriculum: serves text training data as CurriculumItems.

Loads structured text data from text_curriculum.json and reasoning_tasks.json,
encodes on demand via any text encoder with an .encode() method, and gates
difficulty by model step so the brain progresses from simple sentences to
multi-sentence reasoning.

Progressive difficulty schedule:
    step 0-5K:    level 1 only (simple noun/verb sentences)
    step 5K-15K:  levels 1-2 (+ relationships, prepositions)
    step 15K-30K: levels 1-3 (+ QA, cause/effect)
    step 30K+:    all levels (+ multi-sentence, reasoning)
"""

import json
import logging
import random
from pathlib import Path

import torch

from .curriculum import CurriculumItem

logger = logging.getLogger(__name__)

# Difficulty gates: (step_threshold, max_level)
_LEVEL_GATES = [
    (0,     1),
    (5_000, 2),
    (15_000, 3),
    (30_000, 999),  # all levels
]


def _max_level_for_step(step: int) -> int:
    """Return the highest curriculum level unlocked at this training step."""
    level = 1
    for threshold, max_lv in _LEVEL_GATES:
        if step >= threshold:
            level = max_lv
    return level


class TextCurriculum:
    """
    Serves text training data alongside the image curriculum.

    Each item is encoded on demand through the provided text_encoder,
    keeping the class encoder-agnostic (works with NativeTextEncoder,
    CLIP, or any encoder exposing .encode(str) -> Tensor).
    """

    def __init__(self, data_dir: str, text_encoder):
        self._data_dir = Path(data_dir)
        self._encoder = text_encoder
        self._items: list[dict] = []
        self._cursor: int = 0

        self._load_file("text_curriculum.json")
        self._load_file("reasoning_tasks.json")

        if self._items:
            random.shuffle(self._items)
            logger.info("[text_curriculum] loaded %d items from %s", len(self._items), data_dir)
            print(f"[text_curriculum] loaded {len(self._items)} items", flush=True)
        else:
            logger.warning("[text_curriculum] no text data found in %s", data_dir)
            print(f"[text_curriculum] WARNING: no text data found in {data_dir}", flush=True)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_file(self, filename: str) -> None:
        """Load a JSON file of text items. Silently skip if missing."""
        path = self._data_dir / filename
        if not path.exists():
            logger.info("[text_curriculum] %s not found, skipping", path)
            return
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("[text_curriculum] failed to load %s: %s", path, e)
            print(f"[text_curriculum] WARNING: failed to load {path}: {e}", flush=True)
            return

        # Support both top-level list and {"items": [...]} wrapper
        if isinstance(data, dict):
            data = data.get("items", [])
        if not isinstance(data, list):
            logger.warning("[text_curriculum] unexpected format in %s", path)
            return

        for i, entry in enumerate(data):
            if not isinstance(entry, dict):
                continue
            item = {
                "index": len(self._items),
                "text": entry.get("text", ""),
                "question": entry.get("question", ""),
                "answer": entry.get("answer", ""),
                "category": entry.get("category", "text"),
                "level": int(entry.get("level", 1)),
                "source": filename,
            }
            # Must have either text or question+answer
            if item["text"] or (item["question"] and item["answer"]):
                self._items.append(item)

    # ------------------------------------------------------------------
    # Encoding (on demand)
    # ------------------------------------------------------------------

    def _encode(self, text: str) -> torch.Tensor:
        """Encode text through the provided encoder. Returns (512,) tensor."""
        return self._encoder.encode(text)

    # ------------------------------------------------------------------
    # Serving
    # ------------------------------------------------------------------

    def _eligible_items(self, model_step: int) -> list[dict]:
        """Return items whose level is unlocked at the current step."""
        max_lv = _max_level_for_step(model_step)
        return [it for it in self._items if it["level"] <= max_lv]

    def next_item(self, model_step: int = 0) -> CurriculumItem | None:
        """
        Return the next text training item as a CurriculumItem.

        QA items: input = encoded question, expected = encoded answer.
        Plain text: input = expected = encoded text (self-supervised).

        Returns None if no items are available.
        """
        eligible = self._eligible_items(model_step)
        if not eligible:
            return None

        # Round-robin with shuffle on wrap
        if self._cursor >= len(eligible):
            random.shuffle(self._items)  # reshuffle full pool for variety
            self._cursor = 0

        # Pick from eligible pool at cursor (mod to stay in bounds)
        item = eligible[self._cursor % len(eligible)]
        self._cursor += 1

        return self._make_curriculum_item(item)

    def next_batch(self, n: int, model_step: int = 0) -> list[CurriculumItem]:
        """Return up to n text items at the current difficulty level."""
        eligible = self._eligible_items(model_step)
        if not eligible:
            return []

        # Sample with replacement if n > eligible count
        if n >= len(eligible):
            selected = eligible[:]
            random.shuffle(selected)
        else:
            selected = random.sample(eligible, n)

        items = []
        for raw in selected:
            ci = self._make_curriculum_item(raw)
            if ci is not None:
                items.append(ci)
        return items

    def _make_curriculum_item(self, raw: dict) -> CurriculumItem | None:
        """Convert a raw dict into a CurriculumItem with encoded vectors."""
        try:
            is_qa = bool(raw["question"] and raw["answer"])

            if is_qa:
                input_vec = self._encode(raw["question"])
                expected_vec = self._encode(raw["answer"])
                description = f"Q: {raw['question']} A: {raw['answer']}"
            else:
                vec = self._encode(raw["text"])
                input_vec = vec
                expected_vec = vec
                description = raw["text"]

            return CurriculumItem(
                id=f"text_{raw['index']}",
                stage=0,
                item_type="text",
                input_vector=input_vec,
                expected_vector=expected_vec,
                label=raw["category"],
                description=description,
                context=description,
                template_slots={"description": description},
                stage_relevance=1.0,
                precomputed=True,
            )
        except Exception as e:
            logger.warning("[text_curriculum] failed to encode item %d: %s", raw["index"], e)
            return None

    @property
    def size(self) -> int:
        """Total number of text items loaded."""
        return len(self._items)

    def level_counts(self) -> dict[int, int]:
        """Return {level: count} for diagnostics."""
        counts: dict[int, int] = {}
        for item in self._items:
            lv = item["level"]
            counts[lv] = counts.get(lv, 0) + 1
        return counts
