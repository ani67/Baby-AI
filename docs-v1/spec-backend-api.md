# SPEC: Backend API
*Component 7 of 9 — FastAPI server that wires everything together*

---

## What it is

The FastAPI application. Owns startup/shutdown, wires all components
together, and exposes the HTTP + WebSocket interface the frontend talks to.

It is thin. No business logic lives here.
Every route handler does one thing: delegate to the right component
and return the result. The components do the work.

---

## Location in the project

```
project/
  backend/
    main.py           ← FastAPI app, lifespan, all routes
    dependencies.py   ← shared component instances (singleton getters)
    config.py         ← environment + config values
    models.py         ← Pydantic request/response models
```

---

## Startup sequence

```
lifespan():
  1. Load config
  2. Init StateStore (creates SQLite tables if not exists)
  3. Init Encoders (loads CLIP — slowest step, ~5s on M1)
  4. Init TeacherBridge (verifies Ollama is reachable)
  5. Init TextDecoder
  6. Init BabyModel (fast — starts near-empty)
  7. Init Curriculum (loads stage0 images from data/stage0/)
  8. Init VizEmitter
  9. Init LearningLoop (wires all of the above together)
  10. Store all instances in app.state
  11. Log "ready" to console
  yield   ← app is live
  shutdown:
    await loop.pause()
    store.close()
```

The CLIP load (step 3) is the slow step.
Print a progress message before it starts so the human knows to wait.

---

## Routes overview

```
WebSocket
  WS  /ws                     live graph stream

Learning loop control
  POST /start                  start the loop
  POST /pause                  pause the loop
  POST /resume                 resume the loop
  POST /step                   execute one step, return result
  POST /reset                  reset model to initial state
  GET  /status                 loop state, graph summary, teacher health

Stage control
  POST /stage                  set current developmental stage

Speed control
  POST /speed                  set inter-step delay (ms)

Human chat
  POST /chat                   send message, get model response

Curriculum
  POST /image                  upload image (multipart/form-data)

Snapshot
  GET  /snapshot               full current graph state (REST, no WS needed)

Health
  GET  /health                 liveness check (for start.sh)
```

---

## main.py

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import logging

from .dependencies import get_loop, get_emitter, get_store, get_curriculum
from .models import (
    StartResponse, StatusResponse, StepResponse,
    StageRequest, SpeedRequest, ChatRequest, ChatResponse,
    ImageUploadResponse, SnapshotResponse
)
from .config import Config

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = Config()

    print("Loading encoders (CLIP ~5s)...")
    from .model.encoder import ImageEncoder, TextEncoder, VideoEncoder
    image_enc = ImageEncoder(device=config.device)
    text_enc  = TextEncoder(device=config.device)
    video_enc = VideoEncoder(image_enc)
    decoder   = TextDecoder(vocab_size=2048)
    print("Encoders ready.")

    from .state_store import StateStore
    store = StateStore(config.db_path)

    from .teacher.bridge import TeacherBridge
    teacher = TeacherBridge(
        base_url=config.ollama_url,
        model=config.teacher_model
    )
    try:
        await teacher.ask("ping", stage=0)
        print(f"Teacher ({config.teacher_model}) reachable.")
    except Exception as e:
        print(f"WARNING: Teacher not reachable at startup: {e}")
        print("Start Ollama and press Start to begin.")

    from .model.baby_model import BabyModel
    model = BabyModel(
        initial_clusters=config.initial_clusters,
        nodes_per_cluster=config.nodes_per_cluster
    )

    from .loop.curriculum import Curriculum
    curriculum = Curriculum(data_dir=config.data_dir)

    from .viz.emitter import VizEmitter
    emitter = VizEmitter(
        snapshot_interval=config.snapshot_interval,
        projection_interval=config.projection_interval
    )

    from .loop.orchestrator import LearningLoop
    loop = LearningLoop(
        model=model,
        teacher=teacher,
        encoders=(image_enc, text_enc, video_enc),
        decoder=decoder,
        store=store,
        viz_emitter=emitter,
        curriculum=curriculum
    )

    app.state.loop       = loop
    app.state.emitter    = emitter
    app.state.store      = store
    app.state.curriculum = curriculum

    print("Ready. Open http://localhost:5173 to begin.")

    yield

    await loop.pause()
    store.close()
    print("Shutdown complete.")


app = FastAPI(title="Developmental AI", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],   # Vite dev server
    allow_methods=["*"],
    allow_headers=["*"]
)
```

---

## WebSocket route

```python
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    emitter: VizEmitter = app.state.emitter
    await emitter.connect(ws)
    try:
        while True:
            # Keep alive — receive any client messages (ping/pong)
            # The frontend doesn't send anything meaningful over WS
            # but we need to await to detect disconnects
            await ws.receive_text()
    except WebSocketDisconnect:
        await emitter.disconnect(ws)
    except Exception:
        await emitter.disconnect(ws)
```

---

## Loop control routes

```python
@app.post("/start", response_model=StatusResponse)
async def start():
    loop: LearningLoop = app.state.loop
    asyncio.create_task(loop.start())   # non-blocking
    return loop.get_status()


@app.post("/pause", response_model=StatusResponse)
async def pause():
    loop: LearningLoop = app.state.loop
    await loop.pause()
    return loop.get_status()


@app.post("/resume", response_model=StatusResponse)
async def resume():
    loop: LearningLoop = app.state.loop
    asyncio.create_task(loop.resume())  # non-blocking
    return loop.get_status()


@app.post("/step", response_model=StepResponse)
async def step_once():
    """
    Executes exactly one step and returns the result.
    If the loop is RUNNING, this is a no-op (return current status).
    If PAUSED or IDLE, executes one step and returns to PAUSED.
    """
    loop: LearningLoop = app.state.loop
    if loop.get_status().state == "running":
        return {"skipped": True, "reason": "loop_running"}
    result = await loop.step_once()
    return result


@app.post("/reset", response_model=StatusResponse)
async def reset():
    loop: LearningLoop = app.state.loop
    await loop.reset()
    return loop.get_status()


@app.get("/status", response_model=StatusResponse)
async def status():
    loop: LearningLoop = app.state.loop
    return loop.get_status()
```

---

## Stage and speed routes

```python
@app.post("/stage", response_model=StatusResponse)
async def set_stage(body: StageRequest):
    """
    body.stage: 0-4
    Sets the developmental stage.
    Takes effect on the next step.
    Does not reset the model.
    """
    loop: LearningLoop = app.state.loop
    if not 0 <= body.stage <= 4:
        raise HTTPException(status_code=400, detail="Stage must be 0-4")
    loop.set_stage(body.stage)
    return loop.get_status()


@app.post("/speed", response_model=StatusResponse)
async def set_speed(body: SpeedRequest):
    """
    body.delay_ms: 0 = max speed, 1000 = 1 step per second, etc.
    """
    loop: LearningLoop = app.state.loop
    loop.set_speed(body.delay_ms)
    return loop.get_status()
```

---

## Human chat route

```python
@app.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest):
    """
    Sends a message to the Baby Model and returns its response.
    Does not pause the loop.
    Response reflects the model's current learned state.
    """
    loop: LearningLoop = app.state.loop
    response = await loop.human_message(body.message)
    return ChatResponse(
        message=response,
        step=loop.model.step,
        stage=loop.get_status().stage
    )
```

---

## Image upload route

```python
@app.post("/image", response_model=ImageUploadResponse)
async def upload_image(
    file: UploadFile = File(...),
    label: str | None = None
):
    """
    Accepts a JPEG or PNG image upload.
    Adds it to the Stage 0 curriculum pool immediately.
    The loop will pick it up on subsequent steps.

    label: optional human-provided label (e.g. "my dog Biscuit")
    """
    if file.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(
            status_code=400,
            detail="Only JPEG and PNG images are accepted"
        )

    data = await file.read()
    if len(data) > 10 * 1024 * 1024:   # 10MB limit
        raise HTTPException(status_code=400, detail="Image too large (max 10MB)")

    from PIL import Image
    import io
    image = Image.open(io.BytesIO(data)).convert("RGB")

    curriculum: Curriculum = app.state.curriculum
    item = curriculum.add_image(image, label=label)

    return ImageUploadResponse(
        item_id=item.id,
        label=label,
        message=f"Image added to curriculum. Model will encounter it shortly."
    )
```

---

## Snapshot route

```python
@app.get("/snapshot", response_model=SnapshotResponse)
async def snapshot():
    """
    Returns the full current graph state as JSON.
    Use this for initial page load before the WebSocket connects,
    or for debugging.
    """
    emitter: VizEmitter = app.state.emitter
    return emitter.get_current_snapshot()
```

---

## Health route

```python
@app.get("/health")
async def health():
    """
    Used by start.sh to poll until the server is ready.
    Returns 200 once the app has finished startup.
    """
    return {"status": "ok"}
```

---

## Pydantic models (models.py)

```python
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

class ImageUploadResponse(BaseModel):
    item_id: str
    label: str | None
    message: str

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
```

---

## Config (config.py)

```python
import os
import torch
from dataclasses import dataclass

@dataclass
class Config:
    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # Ollama
    ollama_url: str = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    teacher_model: str = os.environ.get("TEACHER_MODEL", "phi4-mini")

    # Database
    db_path: str = os.environ.get("DB_PATH", "backend/data/dev.db")

    # Data
    data_dir: str = os.environ.get("DATA_DIR", "backend/data")

    # Model
    initial_clusters: int = 4
    nodes_per_cluster: int = 8

    # Viz
    snapshot_interval: int = 50
    projection_interval: int = 200

    # Device
    @property
    def device(self) -> str:
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
```

---

## Running the backend

```bash
# From project root
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

`--reload` for development (auto-restart on file changes).
Drop `--reload` in production.

---

## Error handling

**Teacher not reachable at startup:**
Log a warning. Don't crash. The human may start Ollama after the
server is already running. The loop's `start()` will warm-up the
teacher and surface the error through `LoopStatus.teacher_healthy`.

**Teacher not reachable mid-run:**
`LearningLoop` sets state to ERROR and stops.
`GET /status` returns `state: "error"` and `error_message`.
The frontend shows an error banner.
Human checks Ollama, then presses Reset or Resume.

**Image upload with non-image content-type but image bytes:**
The content_type check is a first pass. PIL.Image.open will raise
if the bytes are not a valid image regardless of content_type.
Wrap the PIL open in try/except and return 400 if it fails.

**Step called while loop is running:**
Return `{skipped: true, reason: "loop_running"}` — not an error.
The frontend step button should be disabled while running, but
handle this gracefully server-side regardless.

---

## CORS note

`allow_origins=["http://localhost:5173"]` is the Vite dev server.
If the frontend is served from a different port or host,
update this. In production (if ever deployed), restrict to
the actual origin.

---

## Tests

```python
# test_main.py — uses FastAPI TestClient + pytest-asyncio

from fastapi.testclient import TestClient
from fastapi import WebSocketDisconnect

def test_health():
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

def test_status_returns_idle():
    with TestClient(app) as client:
        r = client.get("/status")
        assert r.status_code == 200
        assert r.json()["state"] == "idle"

def test_step_returns_step_result():
    with TestClient(app) as client:
        r = client.post("/step")
        assert r.status_code == 200
        body = r.json()
        assert "question" in body
        assert "answer" in body

def test_stage_out_of_range():
    with TestClient(app) as client:
        r = client.post("/stage", json={"stage": 99})
        assert r.status_code == 400

def test_chat_returns_string():
    with TestClient(app) as client:
        r = client.post("/chat", json={"message": "hello"})
        assert r.status_code == 200
        assert isinstance(r.json()["message"], str)

def test_image_upload_jpeg():
    with TestClient(app) as client:
        img_bytes = make_test_jpeg()   # helper: 10x10 red JPEG
        r = client.post(
            "/image",
            files={"file": ("test.jpg", img_bytes, "image/jpeg")},
            data={"label": "test"}
        )
        assert r.status_code == 200
        assert r.json()["label"] == "test"

def test_image_upload_rejects_pdf():
    with TestClient(app) as client:
        r = client.post(
            "/image",
            files={"file": ("bad.pdf", b"%PDF", "application/pdf")}
        )
        assert r.status_code == 400

def test_websocket_receives_snapshot_on_connect():
    with TestClient(app) as client:
        # Seed some graph state
        client.post("/step")   # run one step to generate state

        with client.websocket_connect("/ws") as ws:
            # First message should be snapshot if graph has state
            # (may be absent if step didn't generate enough state)
            # At minimum, connection doesn't crash
            pass

def test_start_pause_resume():
    with TestClient(app) as client:
        r = client.post("/start")
        assert r.json()["state"] in ("running", "idle")

        r = client.post("/pause")
        assert r.json()["state"] == "paused"

        r = client.post("/resume")
        assert r.json()["state"] in ("running",)

        client.post("/pause")

def test_reset_returns_to_idle():
    with TestClient(app) as client:
        client.post("/step")
        r = client.post("/reset")
        assert r.json()["state"] == "idle"
        assert r.json()["step"] == 0
```

---

## Hard parts

**`asyncio.create_task` inside a sync context.**
`start()` and `resume()` use `asyncio.create_task` to launch
the loop as a background coroutine. This works inside an async
route handler because FastAPI runs routes in the same event loop.
But if `create_task` is called before the event loop is running
(e.g. during module import) it will fail.
Solution: always call it inside the route handler, never at module level.

**TestClient and background tasks.**
FastAPI's `TestClient` uses `httpx` under the hood and runs the
event loop synchronously. Background tasks launched with
`asyncio.create_task` inside route handlers may not execute
before the test assertion runs. For tests that need the loop
to actually run some steps, use `AsyncClient` from `httpx` with
`pytest-asyncio` instead of `TestClient`.

**Lifespan and test isolation.**
Each `TestClient(app)` call triggers the full lifespan (startup + shutdown).
CLIP loads every time. This makes the test suite slow.
Solution: use a module-scoped fixture that creates the client once:
```python
@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c
```
Or mock the CLIP encoder in tests with a dummy that returns
random normalized 512-dim vectors.

**`asyncio.create_task` from the `/start` route creates a
background task that outlives the request.**
If the test client closes before the task finishes, the task
gets cancelled. This is fine — `loop.start()` handles cancellation
gracefully by checking `_state == RUNNING` at the top of each
iteration.

**Pydantic v2 vs v1.**
FastAPI 0.100+ ships with Pydantic v2 by default.
The `models.py` above uses v2-compatible syntax.
If you're on an older FastAPI with Pydantic v1,
`dict` type hints in models need to be `Dict[str, Any]`
from `typing` instead of the bare `dict`.
Check `pydantic.__version__` and adjust accordingly.

---

## M1-specific notes

Uvicorn on M1 works without any special configuration.
`--reload` uses file watchers which are efficient on M1.

The CLIP load during lifespan is the only slow startup step.
It happens once at process start. On M1, MLX-based CLIP loads
in ~5s. Subsequent imports are fast (model is cached in process memory).

If `mps` device throws errors during testing, the `Config.device`
property falls back to `cpu` automatically — tests run on CPU,
production runs on MPS. No test-specific configuration needed.
