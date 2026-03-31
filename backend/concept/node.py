"""ConceptNode and TypedEdge — the atomic units of Concept Brain v3."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

import torch
from torch import Tensor


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


def _now_step() -> int:
    return int(time.time())


@dataclass
class ConceptNode:
    id: str = field(default_factory=_new_id)
    vector: Tensor = field(default_factory=lambda: torch.zeros(512))
    name: str | None = None
    modalities: set[str] = field(default_factory=set)
    modality_vectors: dict[str, Tensor] = field(default_factory=dict)
    modality_weights: dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    observation_count: int = 0
    activation: float = 0.0
    cluster_id: str | None = None
    created_at: int = field(default_factory=_now_step)
    last_accessed: int = field(default_factory=_now_step)


@dataclass
class TypedEdge:
    source_id: str = ""
    target_id: str = ""
    relation: str = ""
    strength: float = 0.5
    evidence: int = 1
    created_at: int = field(default_factory=_now_step)
    last_used: int = field(default_factory=_now_step)
