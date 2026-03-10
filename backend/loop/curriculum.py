import random
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

import torch


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


class EmptyPoolError(Exception):
    pass


class Curriculum:
    """
    The pool of experiences available at each stage.
    """

    def __init__(self, data_dir: str = "backend/data"):
        self._pools: dict[int, list[CurriculumItem]] = {
            0: [], 1: [], 2: [], 3: [], 4: [],
        }
        self._data_dir = data_dir
        self._load_stage_0()
        self._load_concepts()

    def next_item(self, stage: int, model_state: dict) -> CurriculumItem:
        pool = self._pools.get(stage, []) + self._pools.get(stage - 1, [])
        if not pool:
            pool = [item for items in self._pools.values() for item in items]
        if not pool:
            raise EmptyPoolError(f"No curriculum items available for stage {stage}")

        return random.choice(pool)

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
