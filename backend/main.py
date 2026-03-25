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
    SnapshotResponse, ResetRequest,
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

    curriculum = Curriculum(data_dir=config.data_dir, db_path=config.db_path)

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
async def reset(body: ResetRequest):
    # Save experiment notes before resetting
    from pathlib import Path
    from datetime import datetime

    loop = app.state.loop
    store = app.state.store
    step = loop.model.step
    stage = loop.get_status().stage

    notes_dir = Path("data/experiment_notes")
    notes_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    notes_path = notes_dir / f"reset_step{step}_{ts}.md"

    notes_path.write_text(
        f"# Experiment Reset — Step {step}, Stage {stage}\n"
        f"**Date:** {datetime.now().isoformat()}\n\n"
        f"## Architecture State\n{body.architecture_state}\n\n"
        f"## Signal Quality\n{body.signal_quality}\n\n"
        f"## Why Reset\n{body.why_reset}\n\n"
        f"## What Was Learned\n{body.what_was_learned}\n"
    )
    print(f"[reset] notes saved to {notes_path}", flush=True)

    # Delete all checkpoint .pt files on disk
    ckpt_dir = Path("data/checkpoints")
    deleted = 0
    if ckpt_dir.exists():
        for pt_file in ckpt_dir.glob("*.pt"):
            pt_file.unlink()
            deleted += 1

    # Clear SQLite tables so next run starts fresh
    store.clear_for_reset()

    print(f"[reset] deleted {deleted} checkpoint files, cleared dialogue history", flush=True)

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


# ── Cluster labels ──


STOPWORDS = {
    "a", "an", "the", "is", "it", "in", "on", "of", "to", "and", "or",
    "for", "with", "that", "this", "was", "are", "be", "has", "had",
    "not", "but", "from", "they", "we", "you", "he", "she", "its",
    "by", "at", "as", "do", "if", "so", "no", "up", "out", "about",
    "their", "there", "these", "those", "would", "could", "should",
    "which", "where", "after", "being", "other", "every", "still",
    "while", "since", "until", "before", "between", "through", "during",
    "what", "when", "how", "who", "will", "can", "may", "very", "just",
    "than", "then", "also", "into", "over", "such", "some", "any",
    "each", "more", "most", "been", "have", "were", "our", "your",
    "them", "him", "her", "his", "my", "me", "us", "i",
}


def _compute_cluster_labels(store, graph) -> dict[str, list[str]]:
    """Shared helper: compute top-5 emergent label words per cluster."""
    import json
    from collections import Counter

    active_ids = [c.id for c in graph.clusters if not c.dormant]
    rows = store.get_recent_dialogues_for_clusters(limit=2000)

    cluster_texts: dict[str, list[str]] = {cid: [] for cid in active_ids}
    for clusters_json, answer in rows:
        try:
            cluster_list = json.loads(clusters_json)
        except (json.JSONDecodeError, TypeError):
            continue
        top3 = cluster_list[:3]
        for cid in top3:
            if cid in cluster_texts and len(cluster_texts[cid]) < 50:
                cluster_texts[cid].append(answer)

    labels: dict[str, list[str]] = {}
    for cid in active_ids:
        texts = cluster_texts[cid]
        if not texts:
            labels[cid] = []
            continue
        counter: Counter = Counter()
        for text in texts:
            for word in text.lower().split():
                clean = word.strip(".,!?;:\"'()-[]{}").lower()
                if len(clean) > 2 and clean.isalpha() and clean not in STOPWORDS:
                    counter[clean] += 1
        labels[cid] = [word for word, _ in counter.most_common(5)]
    return labels


import re as _re
_CLUSTER_ID_RE = _re.compile(r'^(c_\d+)([a-z]*)$')


def _cluster_parent_id(cid: str) -> str | None:
    """Derive parent cluster ID from BUD naming convention.

    c_00a  → c_00   (depth-1 parent)
    c_00ab → c_00a  (strip last suffix letter)
    c_00   → None   (root — no alphabetic suffix)
    """
    m = _CLUSTER_ID_RE.match(cid)
    if not m:
        return None
    base, suffix = m.group(1), m.group(2)
    if not suffix:
        return None  # root-level cluster, no parent
    return base + suffix[:-1]  # strip last letter


def _cluster_depth(cid: str) -> int:
    """Depth = number of BUD suffix letters after the numeric part."""
    m = _CLUSTER_ID_RE.match(cid)
    if not m:
        return 0
    return len(m.group(2))


@app.get("/clusters/labels")
async def cluster_labels():
    """
    For each active cluster, find the 50 most recent dialogue entries where
    that cluster was in the top 3 activated. Extract text from those entries.
    Return the 5 most frequent non-stopwords as the cluster's emergent label.
    """
    store = app.state.store
    graph = app.state.loop.model.graph
    return {"labels": _compute_cluster_labels(store, graph)}


@app.get("/clusters/tree")
async def cluster_tree():
    """Return cluster parent-child tree derived from BUD naming convention.

    c_00 → children c_00a, c_00b
    c_00a → children c_00aa, c_00ab

    Since bud() removes parent clusters from the graph, we insert phantom
    parent nodes when siblings exist but their parent doesn't.  This
    reconstructs the full BUD lineage tree for visualization.
    """
    store = app.state.store
    loop = app.state.loop
    graph = loop.model.graph

    try:
        labels = _compute_cluster_labels(store, graph)
    except Exception as e:
        print(f"[tree] labels computation failed: {e}", flush=True)
        labels = {}

    # Build node map for real clusters
    node_map: dict[str, dict] = {}
    for cluster in graph.clusters:
        cid = cluster.id
        parent = _cluster_parent_id(cid)
        node_map[cid] = {
            "id": cid,
            "depth": _cluster_depth(cid),
            "parent": parent,
            "dormant": cluster.dormant,
            "cluster_type": cluster.cluster_type,
            "labels": labels.get(cid, []),
            "phantom": False,
        }

    # Insert phantom parents where needed: if c_00a exists but c_00 doesn't,
    # create a phantom c_00 node so the tree connects.  Walk up recursively.
    phantoms_needed: set[str] = set()
    for node in list(node_map.values()):
        parent = node["parent"]
        while parent and parent not in node_map and parent not in phantoms_needed:
            phantoms_needed.add(parent)
            parent = _cluster_parent_id(parent)

    for pid in phantoms_needed:
        node_map[pid] = {
            "id": pid,
            "depth": _cluster_depth(pid),
            "parent": _cluster_parent_id(pid),
            "dormant": True,
            "cluster_type": None,
            "labels": [],
            "phantom": True,
        }

    # Build edges: link each node to its parent if parent is in node_map
    edges = []
    for cid, node in node_map.items():
        parent = node["parent"]
        if parent and parent in node_map:
            edges.append({"source": parent, "target": cid})
        elif parent:
            node["parent"] = None

    nodes = list(node_map.values())
    max_depth = max((n["depth"] for n in nodes), default=0)

    # Debug log — first 20 clusters and their resolved parents
    sample = sorted(node_map.keys())[:20]
    print(
        f"[tree] clusters={len(graph.clusters)} nodes={len(nodes)} "
        f"edges={len(edges)} phantoms={len(phantoms_needed)} max_depth={max_depth}",
        flush=True,
    )
    return {"nodes": nodes, "edges": edges, "max_depth": max_depth}


@app.get("/dashboard")
async def dashboard():
    import json
    import math
    from collections import Counter, defaultdict

    store = app.state.store
    loop = app.state.loop
    graph = loop.model.graph

    gs = graph.summary()
    categories = store.get_category_performance()

    # ── Structure metrics ──

    # 1. Spatial clustering (silhouette-like score)
    # For each category, compute mean intra-category distance vs inter-category distance.
    # Uses cluster identity vectors as positions (512-d, more reliable than 3D projections).
    labels_data = _compute_cluster_labels(store, graph)
    category_clusters: dict[str, list] = defaultdict(list)
    cluster_identities: dict[str, list[float]] = {}
    for cluster in graph.clusters:
        if cluster.dormant:
            continue
        identity = cluster.identity
        cluster_identities[cluster.id] = identity
        words = labels_data.get(cluster.id, [])
        for word in words[:1]:  # primary label only
            category_clusters[word].append(cluster.id)

    spatial_score = None
    if len(category_clusters) >= 3:
        import torch
        intra_dists = []
        inter_dists = []
        cats_with_multiple = {cat: cids for cat, cids in category_clusters.items() if len(cids) >= 2}
        for cat, cids in list(cats_with_multiple.items())[:10]:
            ids_list = list(cluster_identities.keys())
            for i in range(len(cids)):
                ci = cluster_identities.get(cids[i])
                if ci is None:
                    continue
                for j in range(i + 1, len(cids)):
                    cj = cluster_identities.get(cids[j])
                    if cj is None:
                        continue
                    intra_dists.append(torch.dot(ci, cj).item())
                # Compare to random other-category cluster
                for other_cid in ids_list[:5]:
                    if other_cid not in cids:
                        co = cluster_identities.get(other_cid)
                        if co is not None:
                            inter_dists.append(torch.dot(ci, co).item())
        if intra_dists and inter_dists:
            avg_intra = sum(intra_dists) / len(intra_dists)
            avg_inter = sum(inter_dists) / len(inter_dists)
            # Higher = better separation (intra similar, inter different)
            spatial_score = round(avg_intra - avg_inter, 4)

    # 2. Sibling label coherence
    # For BUD siblings (c_05a, c_05b), how many label words do they share?
    all_ids = {c.id for c in graph.clusters}
    sibling_pairs = 0
    shared_words_total = 0
    for cluster in graph.clusters:
        parent = _cluster_parent_id(cluster.id)
        if parent is None:
            continue
        # Find sibling
        sibling_suffix = cluster.id[-1]
        other_suffix = 'b' if sibling_suffix == 'a' else 'a'
        sibling_id = cluster.id[:-1] + other_suffix
        if sibling_id in all_ids and sibling_id > cluster.id:  # avoid double-counting
            words_a = set(labels_data.get(cluster.id, []))
            words_b = set(labels_data.get(sibling_id, []))
            if words_a and words_b:
                shared = len(words_a & words_b)
                shared_words_total += shared
                sibling_pairs += 1
    sibling_coherence = round(shared_words_total / max(sibling_pairs, 1), 2)

    # 3. Layer abstraction gradient
    # Do higher layers have more general labels? Measure label word frequency by layer.
    layer_labels: dict[float, list[str]] = defaultdict(list)
    for cluster in graph.clusters:
        if not cluster.dormant:
            words = labels_data.get(cluster.id, [])
            layer_labels[cluster.layer_index].extend(words)
    layer_diversity = {}
    for layer_idx in sorted(layer_labels.keys()):
        words = layer_labels[layer_idx]
        if words:
            unique_ratio = len(set(words)) / len(words)
            layer_diversity[int(layer_idx)] = round(unique_ratio, 3)

    # 4. Co-firing communities
    # Cluster the co-firing pairs into groups using connected components.
    cofiring_pairs = store.get_cofiring_pairs()
    strong_pairs = [p for p in cofiring_pairs if p["count"] > 10]
    if strong_pairs:
        # Simple union-find for connected components
        parent_map: dict[str, str] = {}
        def find(x):
            while parent_map.get(x, x) != x:
                parent_map[x] = parent_map.get(parent_map[x], parent_map[x])
                x = parent_map[x]
            return x
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent_map[ra] = rb
        for p in strong_pairs[:2000]:  # cap for speed
            union(p["a"], p["b"])
        components: dict[str, list] = defaultdict(list)
        all_cofired = set()
        for p in strong_pairs[:2000]:
            all_cofired.add(p["a"])
            all_cofired.add(p["b"])
        for cid in all_cofired:
            components[find(cid)].append(cid)
        community_sizes = sorted([len(v) for v in components.values()], reverse=True)
        num_communities = len([s for s in community_sizes if s >= 3])
    else:
        community_sizes = []
        num_communities = 0

    return {
        "step": loop.model.step,
        "clusters": gs["cluster_count"],
        "layers": gs["layer_count"],
        "edges": gs["edge_count"],
        "nodes": gs["node_count"],
        "growth_rate": gs["cluster_count"] / max(loop.model.step / 1000, 1),
        "edge_ratio": gs["edge_count"] / max(gs["cluster_count"], 1),
        "categories": {
            "best": categories[-5:] if len(categories) >= 5 else categories,
            "worst": categories[:5],
            "total_tracked": len(categories),
        },
        "structure": {
            "spatial_score": spatial_score,  # >0 = clusters of same category are closer than different
            "sibling_coherence": sibling_coherence,  # avg shared label words between BUD siblings
            "layer_diversity": layer_diversity,  # unique/total label ratio per layer (higher layers should be lower = more general)
            "cofiring_communities": num_communities,  # number of distinct co-firing groups (>= 3 members)
            "community_sizes": community_sizes[:10],  # top 10 community sizes
        },
        "memory_buffer": {
            "norm": round(loop.model._activation_buffer.norm().item(), 4),
            "decay": loop.model.buffer_decay,
            "weight": loop.model.buffer_weight,
            "top_k": loop.model.buffer_top_k,
        },
        "homeostasis": {
            "edge_ratio_target": loop.model._target_edge_ratio,
            "edge_ratio_actual": round(gs["edge_count"] / max(gs["cluster_count"], 1), 1),
            "activation_coverage": round(loop.model._activation_coverage, 3),
            "curiosity_buffer": len(loop.model._low_resonance_inputs),
        },
    }


@app.get("/clusters/cofiring")
async def cluster_cofiring():
    """Return co-firing pairs with normalized strength."""
    store = app.state.store
    pairs = store.get_cofiring_pairs()
    if not pairs:
        return {"pairs": []}
    max_count = max(p["count"] for p in pairs)
    return {
        "pairs": [
            {
                "a": p["a"],
                "b": p["b"],
                "count": p["count"],
                "strength": p["count"] / max_count if max_count > 0 else 0,
            }
            for p in pairs
        ]
    }


# ── Debug ──


@app.get("/debug/cluster/{cluster_id}")
async def debug_cluster(cluster_id: str):
    """Deep inspection of a single cluster's internal state."""
    import json

    loop = app.state.loop
    graph = loop.model.graph
    store = app.state.store

    cluster = graph.get_cluster(cluster_id)
    if cluster is None:
        raise HTTPException(status_code=404, detail=f"Cluster {cluster_id} not found")

    # Identity texture variance
    tile = graph._tile_index.get(cluster_id)
    texture_variance = tile.compute_variance() if tile and tile._texture is not None else None

    # Mean activation over last 100 steps (from node activation histories)
    node_activations = []
    for node in cluster.nodes:
        if node.alive and node.activation_history:
            node_activations.extend(list(node.activation_history)[-100:])
    mean_activation_100 = sum(node_activations) / len(node_activations) if node_activations else 0.0

    # Per-node weight norms (proxy for FF update magnitude)
    node_details = []
    for node in cluster.nodes:
        node_details.append({
            "id": node.id,
            "alive": node.alive,
            "weight_norm": node.weights.norm().item(),
            "bias": node.bias.item() if node.bias.numel() == 1 else 0.0,
            "mean_activation": node.mean_activation,
            "activation_variance": node.activation_variance,
            "age": node.age,
            "plasticity": node.plasticity,
        })

    # Positive/negative step counts from recent dialogues
    rows = store.get_recent_dialogues_for_clusters(limit=500)
    positive_count = 0
    negative_count = 0
    total_appearances = 0
    for clusters_json, answer in rows:
        try:
            cluster_list = json.loads(clusters_json)
        except (json.JSONDecodeError, TypeError):
            continue
        if cluster_id in cluster_list:
            total_appearances += 1

    # Also check delta_summary for is_positive
    raw_rows = store._conn.execute(
        "SELECT clusters_active, delta_summary FROM dialogues ORDER BY id DESC LIMIT 500"
    ).fetchall()
    for row in raw_rows:
        try:
            clist = json.loads(row[0])
            if cluster_id not in clist:
                continue
            ds = json.loads(row[1])
            if ds.get("is_positive"):
                positive_count += 1
            else:
                negative_count += 1
        except (json.JSONDecodeError, TypeError):
            continue

    # BUD tree: parent and children
    all_ids = {c.id for c in graph.clusters}
    parent = _cluster_parent_id(cluster_id)
    if parent and parent not in all_ids:
        parent = f"{parent} (phantom)"
    children = []
    for c in graph.clusters:
        cp = _cluster_parent_id(c.id)
        if cp == cluster_id:
            children.append(c.id)

    return {
        "id": cluster_id,
        "cluster_type": cluster.cluster_type,
        "layer_index": cluster.layer_index,
        "dormant": cluster.dormant,
        "age": cluster.age,
        "plasticity": cluster.plasticity,
        "node_count": len(cluster.nodes),
        "texture_variance": texture_variance,
        "mean_activation_100": round(mean_activation_100, 4),
        "activation_bimodality": round(cluster.activation_bimodality, 4),
        "output_coherence": round(cluster.output_coherence, 4),
        "positive_steps": positive_count,
        "negative_steps": negative_count,
        "total_appearances_last500": total_appearances,
        "bud_parent": parent,
        "bud_children": children,
        "bud_depth": _cluster_depth(cluster_id),
        "nodes": node_details,
    }


# ── Snapshot ──


@app.get("/snapshot", response_model=SnapshotResponse)
async def snapshot():
    emitter = app.state.emitter
    return emitter.get_current_snapshot()


# ── Health ──


@app.get("/health")
async def health():
    return {"status": "ok"}
