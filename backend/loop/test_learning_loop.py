import asyncio
import sys
import os
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn.functional as F
import pytest

from model.baby_model import BabyModel
from state.store import StateStore
from loop.orchestrator import LearningLoop, LoopState, StepResult
from loop.curriculum import Curriculum, CurriculumItem


# ── Mock components ──


@dataclass
class MockTeacherResponse:
    answer: str = "It is a dog. Dogs are animals."
    model: str = "mock"
    duration_ms: int = 50
    tokens_used: int = 10
    truncated: bool = False


class MockTeacher:
    async def ask(self, question: str, stage: int, context=None, image_bytes=None):
        return MockTeacherResponse()

    async def health_check(self):
        return True


class TimeoutTeacher:
    async def ask(self, question: str, stage: int, context=None, image_bytes=None):
        raise TeacherTimeoutError("timeout")

    async def health_check(self):
        return True


class UnavailableTeacher:
    async def ask(self, question: str, stage: int, context=None, image_bytes=None):
        raise TeacherUnavailableError("unavailable")

    async def health_check(self):
        return False


class TeacherTimeoutError(Exception):
    pass


class TeacherUnavailableError(Exception):
    pass


class MockTextEncoder:
    def encode(self, text: str) -> torch.Tensor:
        # Deterministic but different per text
        torch.manual_seed(hash(text) % 2**31)
        return F.normalize(torch.randn(512), dim=0)


class MockTextDecoder:
    def decode(self, vector, max_words=30, temperature=0.7):
        return "hello world dog"


class MockVizEmitter:
    async def emit_step(self, **kwargs):
        pass


class MockCurriculum:
    """Curriculum with pre-built items that don't need real encoders."""

    def __init__(self):
        self._items = []
        labels = ["dog", "cat", "tree", "car", "bird",
                  "fish", "house", "ball", "sun", "flower"]
        for i, label in enumerate(labels):
            torch.manual_seed(i)
            self._items.append(CurriculumItem(
                id=f"test_{label}",
                stage=0,
                item_type="image",
                input_vector=F.normalize(torch.randn(512), dim=0),
                expected_vector=None,
                label=label,
                description=f"a {label}",
                context=None,
                template_slots={"description": f"a {label}"},
                stage_relevance=1.0,
            ))

    def next_item(self, stage: int, model_state: dict) -> CurriculumItem:
        import random
        return random.choice(self._items)

    def add_teacher_vocabulary(self, word: str):
        pass


def make_test_loop(teacher=None):
    model = BabyModel(initial_clusters=4, nodes_per_cluster=4)
    if teacher is None:
        teacher = MockTeacher()
    store = StateStore(path=":memory:")
    encoder = (None, MockTextEncoder(), None)
    decoder = MockTextDecoder()
    viz = MockVizEmitter()
    curriculum = MockCurriculum()

    loop = LearningLoop(
        model=model,
        teacher=teacher,
        encoder=encoder,
        decoder=decoder,
        store=store,
        viz_emitter=viz,
        curriculum=curriculum,
    )
    return loop


# ── Tests ──


@pytest.mark.asyncio
async def test_single_step_produces_step_result():
    loop = make_test_loop()
    result = await loop.step_once()
    assert not result.skipped
    assert result.question != ""
    assert result.answer != ""
    assert result.curiosity_score > 0
    print("PASS: test_single_step_produces_step_result")


@pytest.mark.asyncio
async def test_state_transitions():
    loop = make_test_loop()
    assert loop.get_status().state == "idle"
    task = asyncio.create_task(loop.start())
    await asyncio.sleep(0.2)
    assert loop.get_status().state == "running"
    await loop.pause()
    assert loop.get_status().state == "paused"
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    print("PASS: test_state_transitions")


@pytest.mark.asyncio
async def test_step_once_pauses_after():
    loop = make_test_loop()
    await loop.pause()
    result = await loop.step_once()
    assert loop.get_status().state == "paused"
    print("PASS: test_step_once_pauses_after")


@pytest.mark.asyncio
async def test_teacher_timeout_skips_step():
    loop = make_test_loop(teacher=TimeoutTeacher())
    result = await loop.step_once()
    assert result.skipped
    assert result.reason == "teacher_timeout"
    print("PASS: test_teacher_timeout_skips_step")


@pytest.mark.asyncio
async def test_teacher_unavailable_sets_error():
    loop = make_test_loop(teacher=UnavailableTeacher())
    with pytest.raises(TeacherUnavailableError):
        await loop.step_once()
    assert loop.get_status().state == "error"
    print("PASS: test_teacher_unavailable_sets_error")


@pytest.mark.asyncio
async def test_model_updates_after_step():
    loop = make_test_loop()
    for _ in range(5):
        await loop.step_once()
    assert loop.model.step == 5
    print("PASS: test_model_updates_after_step")


@pytest.mark.asyncio
async def test_dialogue_logged_after_step():
    loop = make_test_loop()
    await loop.step_once()
    dialogues = loop.store.get_dialogues()
    assert len(dialogues) == 1
    print("PASS: test_dialogue_logged_after_step")


@pytest.mark.asyncio
async def test_curiosity_drives_variety():
    """Model should not ask the same question repeatedly."""
    loop = make_test_loop()
    questions = []
    for _ in range(10):
        result = await loop.step_once()
        questions.append(result.question)
    unique = len(set(questions))
    assert unique >= 6, f"Only {unique} unique questions out of 10"
    print(f"PASS: test_curiosity_drives_variety ({unique}/10 unique)")


@pytest.mark.asyncio
async def test_human_message_returns_string():
    loop = make_test_loop()
    response = await loop.human_message("What is a dog?")
    assert isinstance(response, str)
    assert len(response) > 0
    print("PASS: test_human_message_returns_string")


@pytest.mark.asyncio
async def test_reset_clears_model():
    loop = make_test_loop()
    for _ in range(20):
        await loop.step_once()
    await loop.reset()
    assert loop.model.step == 0
    assert loop.get_status().state == "idle"
    print("PASS: test_reset_clears_model")


if __name__ == "__main__":
    async def run_all():
        await test_single_step_produces_step_result()
        await test_state_transitions()
        await test_step_once_pauses_after()
        await test_teacher_timeout_skips_step()
        await test_teacher_unavailable_sets_error()
        await test_model_updates_after_step()
        await test_dialogue_logged_after_step()
        await test_curiosity_drives_variety()
        await test_human_message_returns_string()
        await test_reset_clears_model()
        print("\nAll 10 tests passed.")

    asyncio.run(run_all())
