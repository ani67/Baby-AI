"""
Backend API — FastAPI server that wires all components together.
"""

import asyncio
import io
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from models import (
    StatusResponse, StepResponse, StageRequest, SpeedRequest,
    ChatRequest, ChatResponse, ImageUrlRequest, ImageUploadResponse,
    BulkImageUrlRequest, BulkImageResult, BulkImageUploadResponse,
    SnapshotResponse,
)

logger = logging.getLogger(__name__)


def _build_components_mock():
    """Build lightweight mock components for testing (no CLIP, no Ollama)."""
    from dataclasses import dataclass
    import torch
    import torch.nn.functional as F

    from state.store import StateStore
    from model.baby_model import BabyModel
    from viz.emitter import VizEmitter
    from loop.orchestrator import LearningLoop
    from loop.curriculum import Curriculum, CurriculumItem

    @dataclass
    class MockTeacherResponse:
        answer: str = "It is a thing."
        model: str = "mock"
        duration_ms: int = 10
        tokens_used: int = 5
        truncated: bool = False

    class MockTeacher:
        async def ask(self, question, stage, context=None, image_bytes=None):
            return MockTeacherResponse()
        async def health_check(self):
            return True

    class MockTextEncoder:
        def encode(self, text):
            torch.manual_seed(hash(text) % 2**31)
            return F.normalize(torch.randn(512), dim=0)

    class MockTextDecoder:
        def decode(self, vector, max_words=30, temperature=0.7):
            return "hello world"

    class MockCurriculum:
        def __init__(self):
            self._items = []
            for i, label in enumerate(["dog", "cat", "tree", "car", "bird",
                                        "fish", "house", "ball", "sun", "flower"]):
                torch.manual_seed(i)
                self._items.append(CurriculumItem(
                    id=f"test_{label}", stage=0, item_type="image",
                    input_vector=F.normalize(torch.randn(512), dim=0),
                    expected_vector=None, label=label,
                    description=f"a {label}", context=None,
                    template_slots={"description": f"a {label}"},
                    stage_relevance=1.0,
                ))
        def next_item(self, stage, model_state):
            import random
            return random.choice(self._items)
        def add_teacher_vocabulary(self, word):
            pass
        def add_image(self, image, label=None, image_path=None):
            import torch
            item = CurriculumItem(
                id=f"img_uploaded", stage=0, item_type="image",
                input_vector=F.normalize(torch.randn(512), dim=0),
                expected_vector=None, label=label,
                description=label or "uploaded",
                context=None,
                template_slots={"description": label or "uploaded"},
                stage_relevance=1.0,
            )
            self._items.append(item)
            return item

    store = StateStore(path=":memory:")
    model = BabyModel(initial_clusters=4, nodes_per_cluster=4)
    teacher = MockTeacher()
    encoder = (None, MockTextEncoder(), None)
    decoder = MockTextDecoder()
    emitter = VizEmitter(snapshot_interval=50, projection_interval=200)
    curriculum = MockCurriculum()

    loop = LearningLoop(
        model=model, teacher=teacher, encoder=encoder,
        decoder=decoder, store=store, viz_emitter=emitter,
        curriculum=curriculum,
    )

    return loop, emitter, store, curriculum


def _build_components_real(config):
    """Build real components with CLIP, Ollama, etc."""
    from state.store import StateStore
    from model.baby_model import BabyModel
    from viz.emitter import VizEmitter
    from loop.orchestrator import LearningLoop
    from loop.curriculum import Curriculum
    from encoder.clip_mlx import CLIPWrapper
    from encoder.encoder import ImageEncoder, TextEncoder, VideoEncoder
    from encoder.decoder import TextDecoder
    from teacher.bridge import TeacherBridge

    store = StateStore(config.db_path)

    print("Loading encoders (CLIP ~5s)...")
    clip = CLIPWrapper()
    image_enc = ImageEncoder(clip)
    text_enc = TextEncoder(clip)
    video_enc = VideoEncoder(image_enc)
    decoder = TextDecoder(vocab_size=2048)
    print("Encoders ready.")

    teacher = TeacherBridge(
        host=config.ollama_url,
        model=config.teacher_model,
    )

    model = BabyModel(
        initial_clusters=config.initial_clusters,
        nodes_per_cluster=config.nodes_per_cluster,
    )

    # Restore from latest checkpoint if available
    latest_ckpt = store.get_latest_checkpoint()
    if latest_ckpt is not None:
        try:
            checkpoint = store.load_checkpoint(latest_ckpt["id"])
            model.restore_from_checkpoint(checkpoint)
            print(f"Restored from checkpoint at step {checkpoint['step']}.")
            model.cleanup_excess_clusters()
            model.reconnect_orphaned_clusters()
        except Exception as e:
            print(f"Warning: Could not restore checkpoint: {e}")
            print("Starting fresh.")
    else:
        print("No checkpoint found, starting fresh.")

    curriculum = Curriculum(data_dir=config.data_dir)

    emitter = VizEmitter(
        snapshot_interval=config.snapshot_interval,
        projection_interval=config.projection_interval,
    )

    loop = LearningLoop(
        model=model, teacher=teacher,
        encoder=(image_enc, text_enc, video_enc),
        decoder=decoder, store=store,
        viz_emitter=emitter, curriculum=curriculum,
    )

    # Sync orchestrator stage from restored model
    if model.step > 0:
        loop._stage = model.stage

    return loop, emitter, store, curriculum


@asynccontextmanager
async def lifespan(app: FastAPI):
    import os
    testing = os.environ.get("TESTING", "0") == "1"

    if testing:
        loop, emitter, store, curriculum = _build_components_mock()
    else:
        from config import Config
        config = Config()
        loop, emitter, store, curriculum = _build_components_real(config)

        # Warm up teacher
        try:
            await loop.teacher.ask("ping", stage=0)
            print(f"Teacher reachable.")
        except Exception as e:
            print(f"WARNING: Teacher not reachable at startup: {e}")
            print("Start Ollama and press Start to begin.")

    app.state.loop = loop
    app.state.emitter = emitter
    app.state.store = store
    app.state.curriculum = curriculum

    print("Ready.")
    yield

    await loop.pause()
    print("Shutdown complete.")


app = FastAPI(title="Developmental AI", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve uploaded images as static files
import os
os.makedirs("data/stage0", exist_ok=True)
app.mount("/images/data", StaticFiles(directory="data"), name="images")


# ── WebSocket ──


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    emitter = app.state.emitter
    await emitter.connect(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        await emitter.disconnect(ws)
    except Exception:
        await emitter.disconnect(ws)


# ── Loop control ──


@app.post("/start", response_model=StatusResponse)
async def start():
    loop = app.state.loop
    status = loop.get_status()
    if status.state == "running":
        return status
    asyncio.create_task(loop.start())
    await asyncio.sleep(0.05)  # let the task start
    return loop.get_status()


@app.post("/pause", response_model=StatusResponse)
async def pause():
    loop = app.state.loop
    await loop.pause()
    return loop.get_status()


@app.post("/resume", response_model=StatusResponse)
async def resume():
    loop = app.state.loop
    status = loop.get_status()
    if status.state == "running":
        return status
    asyncio.create_task(loop.resume())
    await asyncio.sleep(0.05)
    return loop.get_status()


@app.post("/step", response_model=StepResponse)
async def step_once():
    loop = app.state.loop
    if loop.get_status().state == "running":
        return StepResponse(skipped=True, reason="loop_running")
    result = await loop.step_once()
    return StepResponse(
        step=result.step,
        question=result.question,
        answer=result.answer,
        curiosity_score=result.curiosity_score,
        is_positive=result.is_positive,
        delta_summary=result.delta_summary,
        growth_events=result.growth_events,
        duration_ms=result.duration_ms,
        skipped=result.skipped,
        reason=result.reason,
    )


@app.post("/reset", response_model=StatusResponse)
async def reset():
    loop = app.state.loop
    await loop.reset()
    return loop.get_status()


@app.get("/status", response_model=StatusResponse)
async def status():
    loop = app.state.loop
    return loop.get_status()


# ── Stage and speed ──


@app.post("/stage", response_model=StatusResponse)
async def set_stage(body: StageRequest):
    loop = app.state.loop
    if not 0 <= body.stage <= 4:
        raise HTTPException(status_code=400, detail="Stage must be 0-4")
    loop.set_stage(body.stage)
    return loop.get_status()


@app.post("/speed", response_model=StatusResponse)
async def set_speed(body: SpeedRequest):
    loop = app.state.loop
    loop.set_speed(body.delay_ms)
    return loop.get_status()


# ── Human chat ──


@app.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest):
    loop = app.state.loop
    response = await loop.human_message(body.message)
    return ChatResponse(
        message=response,
        step=loop.model.step,
        stage=loop.get_status().stage,
    )


# ── Image upload ──


@app.post("/image", response_model=ImageUploadResponse)
async def upload_image(
    file: UploadFile = File(...),
    label: str | None = Form(None),
):
    if file.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(
            status_code=400,
            detail="Only JPEG and PNG images are accepted",
        )

    data = await file.read()
    if len(data) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image too large (max 10MB)")

    try:
        from PIL import Image
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image data")

    # Save to disk so it persists across restarts
    from pathlib import Path
    from uuid import uuid4
    file_label = label or file.filename or "upload"
    save_dir = Path("data/stage0") / file_label.replace(" ", "_")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{uuid4().hex[:8]}.jpg"
    image.save(str(save_path), "JPEG")

    curriculum = app.state.curriculum
    item = curriculum.add_image(image, label=label, image_path=str(save_path))

    return ImageUploadResponse(
        item_id=item.id,
        label=label,
        message="Image saved and added to curriculum.",
    )


@app.post("/image-url", response_model=ImageUploadResponse)
async def upload_image_url(body: ImageUrlRequest):
    import httpx
    fetch_headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Referer": "",
    }
    try:
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True, headers=fetch_headers) as client:
            resp = await client.get(body.url)
            resp.raise_for_status()
    except Exception as e:
        print(f"[image-url] fetch failed: {e}")
        raise HTTPException(
            status_code=400,
            detail="Could not fetch image — try downloading it and uploading directly instead.",
        )

    content_type = resp.headers.get("content-type", "")
    print(f"[image-url] fetched {body.url} content-type={content_type} size={len(resp.content)}")

    data = resp.content
    if len(data) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image too large (max 10MB)")

    try:
        from PIL import Image
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Could not fetch image — try downloading it and uploading directly instead.",
        )

    # Auto-derive label from URL filename if not provided
    label = body.label
    if not label:
        from urllib.parse import urlparse
        from pathlib import PurePosixPath
        import re as _re
        path = urlparse(body.url).path
        stem = PurePosixPath(path).stem
        # Sanitize: keep only alphanumeric, spaces, hyphens, underscores
        label = stem.replace("-", " ").replace("_", " ").strip()
        label = _re.sub(r'[^a-zA-Z0-9 ]', '', label).strip()
        if not label or len(label) > 60:
            label = "image"

    # Save to disk so it persists across restarts
    from pathlib import Path
    from uuid import uuid4
    safe_dir_name = label.replace(" ", "_")[:60]
    save_dir = Path("data/stage0") / safe_dir_name
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{uuid4().hex[:8]}.jpg"
    image.save(str(save_path), "JPEG")
    print(f"[image-url] saved: {save_path} (label={label})")

    curriculum = app.state.curriculum
    item = curriculum.add_image(image, label=label, image_path=str(save_path))

    return ImageUploadResponse(
        item_id=item.id,
        label=label,
        message=f"Image saved and added to curriculum.",
    )


@app.post("/images-bulk", response_model=BulkImageUploadResponse)
async def upload_images_bulk(body: BulkImageUrlRequest):
    import httpx
    from PIL import Image
    from pathlib import Path
    from uuid import uuid4
    from urllib.parse import urlparse
    from pathlib import PurePosixPath

    results: list[BulkImageResult] = []
    added = 0

    fetch_headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Referer": "",
    }
    async with httpx.AsyncClient(timeout=15.0, follow_redirects=True, headers=fetch_headers) as client:
        for url in body.urls:
            url = url.strip()
            if not url:
                continue
            try:
                resp = await client.get(url)
                resp.raise_for_status()

                data = resp.content
                if len(data) > 10 * 1024 * 1024:
                    results.append(BulkImageResult(url=url, ok=False, error="Too large"))
                    continue

                image = Image.open(io.BytesIO(data)).convert("RGB")

                # Derive label from filename
                import re as _re
                path = urlparse(url).path
                stem = PurePosixPath(path).stem
                label = stem.replace("-", " ").replace("_", " ").strip()
                label = _re.sub(r'[^a-zA-Z0-9 ]', '', label).strip()
                if not label or len(label) > 60:
                    label = "image"

                safe_dir_name = label.replace(" ", "_")[:60]
                save_dir = Path("data/stage0") / safe_dir_name
                save_dir.mkdir(parents=True, exist_ok=True)
                save_path = save_dir / f"{uuid4().hex[:8]}.jpg"
                image.save(str(save_path), "JPEG")
                print(f"[images-bulk] saved: {save_path} (label={label})")

                curriculum = app.state.curriculum
                curriculum.add_image(image, label=label, image_path=str(save_path))

                results.append(BulkImageResult(url=url, label=label, ok=True))
                added += 1
            except Exception as e:
                results.append(BulkImageResult(url=url, ok=False, error="Could not fetch image — try downloading it and uploading directly instead."))

    return BulkImageUploadResponse(total=len(results), added=added, results=results)


# ── Snapshot ──


@app.get("/snapshot", response_model=SnapshotResponse)
async def snapshot():
    emitter = app.state.emitter
    return emitter.get_current_snapshot()


# ── Health ──


@app.get("/health")
async def health():
    return {"status": "ok"}
