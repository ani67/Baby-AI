"""
Text curriculum: serves text training data as CurriculumItems.

Loads structured text data from text_curriculum.json, reasoning_tasks.json,
and all datasets in data/datasets/*.json. Encodes on demand via any text
encoder with an .encode() method, and gates difficulty by model step so
the brain progresses from simple sentences to multi-sentence reasoning.

Progressive difficulty schedule:
    step 0-5K:    level 1 only (simple noun/verb sentences)
    step 5K-15K:  levels 1-2 (+ relationships, prepositions)
    step 15K-30K: levels 1-3 (+ QA, cause/effect)
    step 30K+:    all levels (+ multi-sentence, reasoning)

Dataset sampling weights (80/20 reasoning focus):
    2x: math, coding, science_reasoning, reasoning, commonsense_reasoning,
        pronoun_reasoning
    1x: facts, factual_qa, commonsense_completion
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

# Categories that get 2x sampling weight (reasoning-heavy).
# Everything else gets 1x.
_HIGH_WEIGHT_CATEGORIES = frozenset({
    "math", "coding", "science_reasoning", "reasoning",
    "commonsense_reasoning", "pronoun_reasoning",
})


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
        self._load_file("text_diverse.json")
        self._load_file("text_conversations.json")
        self._load_file("text_commonsense.json")
        self._load_file("reasoning_tasks.json")
        self._load_datasets_dir()

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

    def _load_datasets_dir(self) -> None:
        """Load all JSON files from data/datasets/.

        Each file has {"items": [...], "count": N, "source": "name"}.
        Large factual datasets (triviaqa, simple_wikipedia) are capped to
        prevent drowning out reasoning items. Reasoning/math/coding datasets
        are loaded in full.
        """
        datasets_dir = self._data_dir / "datasets"
        if not datasets_dir.exists():
            return

        # Cap for low-weight (factual) datasets to prevent domination
        FACTUAL_CAP = 20_000

        loaded_counts: dict[str, int] = {}
        for path in sorted(datasets_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("[text_curriculum] failed to load %s: %s", path, e)
                continue

            if isinstance(data, dict):
                entries = data.get("items", [])
            elif isinstance(data, list):
                entries = data
            else:
                continue

            if not entries:
                continue

            # Determine category from first entry to decide capping
            sample_cat = entries[0].get("category", "text") if isinstance(entries[0], dict) else "text"
            is_high_weight = sample_cat in _HIGH_WEIGHT_CATEGORIES

            # Cap factual datasets to prevent them drowning reasoning items
            if not is_high_weight and len(entries) > FACTUAL_CAP:
                random.shuffle(entries)
                entries = entries[:FACTUAL_CAP]

            count = 0
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                item = {
                    "index": len(self._items),
                    "text": entry.get("text", ""),
                    "question": entry.get("question", ""),
                    "answer": entry.get("answer", ""),
                    "category": entry.get("category", "text"),
                    "level": int(entry.get("level", 1)),
                    "source": path.name,
                }
                if item["text"] or (item["question"] and item["answer"]):
                    self._items.append(item)
                    count += 1

            loaded_counts[path.name] = count

        if loaded_counts:
            total = sum(loaded_counts.values())
            breakdown = ", ".join(f"{k}={v}" for k, v in sorted(loaded_counts.items()))
            print(f"[text_curriculum] datasets: {total} items ({breakdown})", flush=True)

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

    def _weighted_sample(self, eligible: list[dict], n: int) -> list[dict]:
        """Sample n items from eligible pool with 2x weight for reasoning categories."""
        if not eligible:
            return []
        eligible_set = {id(it) for it in eligible}
        # Build weighted pool from eligible items only
        weighted = []
        for it in eligible:
            weight = 2 if it["category"] in _HIGH_WEIGHT_CATEGORIES else 1
            weighted.extend([it] * weight)
        if not weighted:
            return eligible[:n]
        if n >= len(weighted):
            random.shuffle(weighted)
            # Deduplicate (items appear multiple times due to weighting)
            seen = set()
            result = []
            for it in weighted:
                idx = it["index"]
                if idx not in seen:
                    seen.add(idx)
                    result.append(it)
            return result
        # Sample with replacement then deduplicate
        selected = []
        seen = set()
        attempts = 0
        while len(selected) < n and attempts < n * 4:
            it = random.choice(weighted)
            idx = it["index"]
            if idx not in seen:
                seen.add(idx)
                selected.append(it)
            attempts += 1
        return selected

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

        # Weighted random pick (reasoning/math/coding favored 2x)
        item = self._weighted_sample(eligible, 1)
        if not item:
            return None

        return self._make_curriculum_item(item[0])

    def next_batch(self, n: int, model_step: int = 0) -> list[CurriculumItem]:
        """Return up to n text items at the current difficulty level, weighted by category."""
        eligible = self._eligible_items(model_step)
        if not eligible:
            return []

        selected = self._weighted_sample(eligible, n)

        items = []
        for raw in selected:
            ci = self._make_curriculum_item(raw)
            if ci is not None:
                items.append(ci)
        return items

    def _make_curriculum_item(self, raw: dict) -> CurriculumItem | None:
        """Convert a raw dict into a CurriculumItem with encoded vectors."""
        try:
            has_question = bool(raw["question"] and raw["answer"])
            has_text_answer = bool(raw["text"] and raw["answer"])
            is_qa = has_question or has_text_answer

            if has_question:
                input_vec = self._encode(raw["question"])
                expected_vec = self._encode(raw["answer"])
                description = f"Q: {raw['question']} A: {raw['answer']}"
            elif has_text_answer:
                input_vec = self._encode(raw["text"])
                expected_vec = self._encode(raw["answer"])
                description = f"Q: {raw['text']} A: {raw['answer']}"
            else:
                vec = self._encode(raw["text"])
                input_vec = vec
                expected_vec = vec
                description = raw["text"]

            # Sequential encoding: per-word vectors with positional encoding
            # Only when encoder supports it (NativeTextEncoder after switch)
            sequence = None
            if hasattr(self._encoder, 'encode_sequential'):
                try:
                    # Use answer text for sequence (that's what brain learns to predict)
                    seq_text = raw["answer"] if (raw["answer"] and (raw["question"] or raw["text"])) else raw["text"]
                    if seq_text and len(seq_text.split()) > 2:
                        sequence = self._encoder.encode_sequential(seq_text)
                except Exception:
                    pass  # fall back to non-sequential

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
                sequence=sequence,
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


class PreEncodedTextCurriculum(TextCurriculum):
    """TextCurriculum variant that serves pre-encoded CLIP vectors.

    Used by parallel training workers to avoid loading CLIP (~400MB) per process.
    Parent pre-encodes all items once, workers use cached vectors.

    After native encoder switch (cos_sim > 0.65), re-encodes via the lightweight
    native encoder instead of using stale CLIP vectors.
    """

    def __init__(self, pre_encoded_items: list[dict], native_text_encoder=None):
        """Initialize from pre-encoded item dicts.

        Each dict has: index, text, question, answer, category, level, source,
        input_vec (Tensor), expected_vec (Tensor).
        """
        self._items = []
        self._cursor = 0
        self._encoder = native_text_encoder
        self._pre_encoded: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._clip_targets: dict[str, torch.Tensor] = {}
        self._use_pre_encoded = True

        for item in pre_encoded_items:
            idx = item["index"]
            raw = {
                "index": idx,
                "text": item.get("text", ""),
                "question": item.get("question", ""),
                "answer": item.get("answer", ""),
                "category": item.get("category", "text"),
                "level": item.get("level", 1),
                "source": item.get("source", "pre_encoded"),
            }
            self._items.append(raw)
            self._pre_encoded[idx] = (item["input_vec"], item["expected_vec"])

            # Cache CLIP expected_vector keyed by description (survives native switch)
            has_question = bool(raw["question"] and raw["answer"])
            has_text_answer = bool(raw["text"] and raw["answer"])
            if has_question:
                desc = f"Q: {raw['question']} A: {raw['answer']}"
            elif has_text_answer:
                desc = f"Q: {raw['text']} A: {raw['answer']}"
            else:
                desc = raw["text"]
            self._clip_targets[desc] = item["expected_vec"].detach()

        if self._items:
            random.shuffle(self._items)
            print(f"[text_curriculum] pre-encoded: {len(self._items)} items", flush=True)

    def _encode(self, text: str) -> torch.Tensor:
        """Encode via native encoder (only called after CLIP→native switch)."""
        if self._encoder is None:
            raise RuntimeError("No encoder available — pre-encoded vectors should be used")
        return self._encoder.encode(text)

    def _make_curriculum_item(self, raw: dict) -> CurriculumItem | None:
        """Build item from pre-encoded vectors, or re-encode via native encoder."""
        idx = raw["index"]

        # After native encoder switch, encode fresh (native encoder is tiny)
        if not self._use_pre_encoded or idx not in self._pre_encoded:
            return super()._make_curriculum_item(raw)

        try:
            input_vec, expected_vec = self._pre_encoded[idx]

            has_question = bool(raw["question"] and raw["answer"])
            has_text_answer = bool(raw["text"] and raw["answer"])

            if has_question:
                description = f"Q: {raw['question']} A: {raw['answer']}"
            elif has_text_answer:
                description = f"Q: {raw['text']} A: {raw['answer']}"
            else:
                description = raw["text"]

            return CurriculumItem(
                id=f"text_{idx}",
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
            logger.warning("[pre_encoded] failed for item %d: %s", idx, e)
            return None

    def switch_to_native(self):
        """Drop pre-encoded CLIP vectors, start using native encoder.

        Preserves _clip_targets so distill_step() can still train the native
        encoder against CLIP outputs (not its own).
        """
        self._use_pre_encoded = False
        self._pre_encoded.clear()
        print(
            f"[text_curriculum] switched to native encoder, freed pre-encoded cache "
            f"(kept {len(self._clip_targets)} CLIP targets for distillation)",
            flush=True,
        )

    def get_clip_target(self, description: str) -> torch.Tensor | None:
        """Return the cached CLIP vector for a description, or None."""
        return self._clip_targets.get(description)
