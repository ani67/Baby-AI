import asyncio
import sys
import os

import pytest

# Allow imports from parent package
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from teacher.bridge import TeacherBridge, TeacherUnavailableError


# ── Integration tests — require Ollama running with phi4-mini ──


@pytest.mark.integration
@pytest.mark.asyncio
async def test_health_check_when_running():
    bridge = TeacherBridge()
    assert await bridge.health_check() is True
    await bridge.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_basic_ask():
    bridge = TeacherBridge()
    response = await bridge.ask(
        question="What is a dog?",
        stage=0,
    )
    assert len(response.answer) > 0
    assert response.duration_ms > 0
    assert response.tokens_used > 0
    await bridge.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_stage_0_answer_is_short():
    bridge = TeacherBridge()
    response = await bridge.ask(
        question="What is a tree?",
        stage=0,
    )
    word_count = len(response.answer.split())
    assert word_count < 25, f"Stage 0 answer too long: {word_count} words"
    await bridge.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ask_with_context():
    bridge = TeacherBridge()
    response = await bridge.ask(
        question="What is the difference between these two things?",
        stage=2,
        context="Thing 1: a dog. Thing 2: a cat.",
    )
    assert len(response.answer) > 0
    await bridge.close()


# ── Unit tests — no Ollama required ──


@pytest.mark.asyncio
async def test_unavailable_raises():
    bridge = TeacherBridge(host="http://localhost:9999", timeout=2.0)
    with pytest.raises(TeacherUnavailableError):
        await bridge.ask("test", stage=0)
    await bridge.close()


@pytest.mark.asyncio
async def test_health_check_returns_false_when_down():
    bridge = TeacherBridge(host="http://localhost:9999", timeout=2.0)
    result = await bridge.health_check()
    assert result is False
    await bridge.close()


@pytest.mark.asyncio
async def test_list_models_returns_empty_when_down():
    bridge = TeacherBridge(host="http://localhost:9999", timeout=2.0)
    result = await bridge.list_available_models()
    assert result == []
    await bridge.close()
