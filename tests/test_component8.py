"""Tests for Component 8: Teacher Ensemble

All tests use mock teachers — no live Ollama or API key required.
"""

import os
import shutil
import sys
import tempfile
import time
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import component4
import component8
from component8 import (
    TeacherResponse,
    _build_consensus,
    _responses_agree,
    request_teacher_guidance,
    enqueue_query,
    reset,
)

_tmpdir = None


def setup_module():
    global _tmpdir
    _tmpdir = tempfile.mkdtemp(prefix="teacher_test_")
    component4.init(data_dir=_tmpdir)


def teardown_module():
    if _tmpdir and os.path.exists(_tmpdir):
        shutil.rmtree(_tmpdir)


def setup_function():
    """Reset rate limiter between tests."""
    reset()


# ---------- mock helpers ----------

_MOCK_REASONING_AGREE = (
    "REASONING: Step 1: Light from the sun enters the atmosphere. "
    "Step 2: Shorter wavelengths (blue) scatter more than longer ones (red). "
    "Step 3: This is called Rayleigh scattering. "
    "Step 4: We see blue light scattered in all directions.\n"
    "ANSWER: The sky is blue because of Rayleigh scattering of sunlight."
)

_MOCK_REASONING_AGREE_2 = (
    "REASONING: First, sunlight contains all colors. "
    "Second, when light hits air molecules, blue wavelengths scatter more. "
    "Third, this phenomenon is Rayleigh scattering. "
    "Fourth, scattered blue light reaches our eyes from all directions.\n"
    "ANSWER: The sky is blue because of Rayleigh scattering of sunlight."
)

_MOCK_REASONING_DISAGREE = (
    "REASONING: I cannot determine a clear explanation for this phenomenon.\n"
    "ANSWER: 42 is the ultimate numerical result here."
)


# ---------- tests ----------


def test_teacher_response_dataclass():
    """TeacherResponse should have all required fields."""
    tr = TeacherResponse(
        prompt="test",
        reasoning_trace="trace",
        confidence=0.9,
        sources=["ollama_phi4mini"],
        training_worthy=True,
    )
    assert tr.prompt == "test"
    assert tr.confidence == 0.9
    assert tr.training_worthy is True


def test_consensus_single_source():
    """Single source should give confidence 0.5."""
    result = _build_consensus({"ollama_phi4mini": _MOCK_REASONING_AGREE})
    assert result.confidence == 0.5
    assert result.sources == ["ollama_phi4mini"]
    assert result.training_worthy is False  # 0.5 <= 0.6


def test_consensus_two_agree():
    """Two agreeing sources should give confidence 0.9."""
    result = _build_consensus({
        "ollama_phi4mini": _MOCK_REASONING_AGREE,
        "gemini_flash": _MOCK_REASONING_AGREE_2,
    })
    assert result.confidence == 0.9
    assert result.training_worthy is True
    assert len(result.sources) == 2


def test_consensus_two_disagree():
    """Two disagreeing sources should give confidence 0.2."""
    result = _build_consensus({
        "ollama_phi4mini": _MOCK_REASONING_AGREE,
        "gemini_flash": _MOCK_REASONING_DISAGREE,
    })
    assert result.confidence == 0.2
    assert result.training_worthy is False


def test_responses_agree_same_answer():
    """Similar ANSWER sections should agree."""
    assert _responses_agree(_MOCK_REASONING_AGREE, _MOCK_REASONING_AGREE_2) is True


def test_responses_disagree_different_answer():
    """Different ANSWER sections should disagree."""
    assert _responses_agree(_MOCK_REASONING_AGREE, _MOCK_REASONING_DISAGREE) is False


def test_training_worthy_threshold():
    """training_worthy should be True only when confidence > 0.6."""
    high = _build_consensus({
        "ollama_phi4mini": _MOCK_REASONING_AGREE,
        "gemini_flash": _MOCK_REASONING_AGREE_2,
    })
    assert high.training_worthy is True  # 0.9 > 0.6

    single = _build_consensus({"ollama_phi4mini": _MOCK_REASONING_AGREE})
    assert single.training_worthy is False  # 0.5 <= 0.6

    low = _build_consensus({
        "ollama_phi4mini": _MOCK_REASONING_AGREE,
        "gemini_flash": _MOCK_REASONING_DISAGREE,
    })
    assert low.training_worthy is False  # 0.2 <= 0.6


@mock.patch("component8._ollama_available", return_value=True)
@mock.patch("component8._query_ollama", return_value=_MOCK_REASONING_AGREE)
@mock.patch("component8._gemini_available", return_value=True)
@mock.patch("component8._query_gemini", return_value=_MOCK_REASONING_AGREE_2)
def test_request_teacher_guidance_both_agree(mock_gem_q, mock_gem_a, mock_oll_q, mock_oll_a):
    """With both mocked teachers agreeing, should get high confidence."""
    result = request_teacher_guidance("Why is the sky blue?")
    assert result is not None
    assert result.confidence == 0.9
    assert result.training_worthy is True
    assert "REASONING:" in result.reasoning_trace
    assert len(result.reasoning_trace) > 100
    assert len(result.sources) == 2


@mock.patch("component8._ollama_available", return_value=True)
@mock.patch("component8._query_ollama", return_value=_MOCK_REASONING_AGREE)
@mock.patch("component8._gemini_available", return_value=False)
def test_request_teacher_guidance_single_source(mock_gem_a, mock_oll_q, mock_oll_a):
    """With only one teacher, confidence should be 0.5."""
    result = request_teacher_guidance("What is 2+2?")
    assert result is not None
    assert result.confidence == 0.5
    assert result.sources == ["ollama_phi4mini"]


@mock.patch("component8._ollama_available", return_value=False)
@mock.patch("component8._gemini_available", return_value=False)
def test_request_teacher_guidance_no_sources(mock_gem_a, mock_oll_a):
    """With no teachers available, should return None."""
    result = request_teacher_guidance("test prompt")
    assert result is None


@mock.patch("component8._ollama_available", return_value=True)
@mock.patch("component8._query_ollama", return_value=_MOCK_REASONING_AGREE)
@mock.patch("component8._gemini_available", return_value=False)
def test_rate_limiting(mock_gem_a, mock_oll_q, mock_oll_a):
    """Should return None after exceeding rate limit."""
    # Make 10 queries (max per hour)
    for i in range(10):
        result = request_teacher_guidance(f"prompt {i}")
        assert result is not None

    # 11th should be rate-limited
    result = request_teacher_guidance("one more")
    assert result is None


@mock.patch("component8._ollama_available", return_value=True)
@mock.patch("component8._query_ollama", return_value=_MOCK_REASONING_AGREE)
@mock.patch("component8._gemini_available", return_value=False)
def test_context_included_in_prompt(mock_gem_a, mock_oll_q, mock_oll_a):
    """Context should be prepended to the prompt when provided."""
    result = request_teacher_guidance("What is X?", context="X = 42")
    assert result is not None
    # Check that ollama was called with the context included
    call_args = mock_oll_q.call_args[0][0]
    assert "X = 42" in call_args


def test_log_file_created():
    """Teacher queries should be logged to disk."""
    with mock.patch("component8._ollama_available", return_value=True), \
         mock.patch("component8._query_ollama", return_value=_MOCK_REASONING_AGREE), \
         mock.patch("component8._gemini_available", return_value=False):
        request_teacher_guidance("log test")

    assert component8._LOG_FILE.exists()
    with open(component8._LOG_FILE) as f:
        lines = f.readlines()
    assert len(lines) >= 1
    entry = json.loads(lines[-1])
    assert entry["prompt"] == "log test"


def test_enqueue_does_not_block():
    """enqueue_query should return immediately."""
    t0 = time.time()
    enqueue_query("test prompt", context=None)
    elapsed = time.time() - t0
    assert elapsed < 0.1, f"enqueue_query took {elapsed:.3f}s, should be instant"


import json
