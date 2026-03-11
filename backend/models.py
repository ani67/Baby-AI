from pydantic import BaseModel
from typing import Any


class StageRequest(BaseModel):
    stage: int


class SpeedRequest(BaseModel):
    delay_ms: int


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    message: str
    step: int
    stage: int


class ImageUrlRequest(BaseModel):
    url: str
    label: str | None = None


class BulkImageUrlRequest(BaseModel):
    urls: list[str]


class BulkImageResult(BaseModel):
    url: str
    label: str | None = None
    ok: bool
    error: str | None = None


class BulkImageUploadResponse(BaseModel):
    total: int
    added: int
    results: list[BulkImageResult]


class ImageUploadResponse(BaseModel):
    item_id: str
    label: str | None
    message: str


class ResetRequest(BaseModel):
    architecture_state: str
    signal_quality: str
    why_reset: str
    what_was_learned: str


class StatusResponse(BaseModel):
    state: str
    step: int
    stage: int
    delay_ms: int
    error_message: str | None
    graph_summary: dict
    teacher_healthy: bool


class StepResponse(BaseModel):
    step: int = 0
    question: str = ""
    answer: str = ""
    curiosity_score: float = 0.0
    is_positive: bool = True
    delta_summary: dict = {}
    growth_events: list = []
    duration_ms: int = 0
    skipped: bool = False
    reason: str = ""


class SnapshotResponse(BaseModel):
    type: str
    step: int
    stage: int
    nodes: list[dict]
    clusters: list[dict]
    edges: list[dict]
    model_stats: dict
