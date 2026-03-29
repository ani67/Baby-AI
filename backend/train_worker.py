"""
Training worker — runs in a separate process from the HTTP server.

Owns the brain, model, MPS GPU, and SQLite. Runs a clean pipeline
where each stage is independent, timed, and fail-safe.

Pipeline per batch:
    1. prepare_batch()    → items + replay        (data loading)
    2. compute_batch()    → changes, activations  (learning)
    3. distill_step()     → cos_sim               (native encoder training)
    4. track_categories() → category stats        (evaluation)
    5. grow_and_prune()   → growth events         (architecture)
    6. record_cofiring()  → pairs flushed         (community tracking)
    7. checkpoint()       → saved                 (persistence)
    8. publish_state()    → shared state file      (communication)

Usage:
    python train_worker.py
"""

import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn.functional as F

from loop.shared_state import write_state


# ── Command interface ──

COMMAND_FILE = os.path.join(os.path.dirname(__file__), "data", "train_command.json")


def read_command() -> str | None:
    try:
        with open(COMMAND_FILE) as f:
            cmd = json.load(f)
        os.unlink(COMMAND_FILE)
        return cmd.get("command")
    except (FileNotFoundError, json.JSONDecodeError):
        return None


CHAT_REQUEST_FILE = os.path.join(os.path.dirname(__file__), "data", "chat_request.json")
CHAT_RESPONSE_FILE = os.path.join(os.path.dirname(__file__), "data", "chat_response.json")


def handle_chat(loop):
    """Check for chat request from server, process it, write response."""
    try:
        with open(CHAT_REQUEST_FILE) as f:
            req = json.load(f)
        os.unlink(CHAT_REQUEST_FILE)
    except (FileNotFoundError, json.JSONDecodeError):
        return

    message = req.get("message", "")
    req_type = req.get("type", "chat")

    try:
        if req_type == "correct":
            # Correction: encode and update weights
            input_vec = loop.text_encoder.encode(message)
            loop.model.brain.forward(input_vec)
            loop.model.brain.update(input_vec, input_vec)
            pred, _ = loop.model.brain.forward(input_vec)
            response = loop.decoder.decode(pred, max_words=8, model_step=loop.model.step)
        else:
            # Chat: encode input, forward, decode response
            input_vec = loop.text_encoder.encode(message)
            output_vec, _ = loop.model.brain.forward(input_vec)
            response = loop.decoder.generate(
                output_vec, brain=loop.model.brain,
                max_tokens=6, model_step=loop.model.step,
            )

        os.makedirs(os.path.dirname(CHAT_RESPONSE_FILE), exist_ok=True)
        with open(CHAT_RESPONSE_FILE, "w") as f:
            json.dump({"message": response}, f)
        print(f"[chat] '{message}' → '{response}'", flush=True)
    except Exception as e:
        with open(CHAT_RESPONSE_FILE, "w") as f:
            json.dump({"message": f"(error: {e})"}, f)
        print(f"[chat] error: {e}", flush=True)


def write_command(command: str):
    os.makedirs(os.path.dirname(COMMAND_FILE), exist_ok=True)
    with open(COMMAND_FILE, "w") as f:
        json.dump({"command": command}, f)


# ── Build ──

def build_components():
    from config import Config
    from state.store import StateStore
    from model.baby_model_v2 import BabyModelV2 as BabyModel
    from loop.orchestrator import LearningLoop
    from loop.curriculum import Curriculum
    from encoder.clip_mlx import CLIPWrapper
    from encoder.encoder import ImageEncoder, TextEncoder, VideoEncoder
    from encoder.decoder import GroundedDecoder
    from encoder.native_text import NativeTextEncoder
    from encoder.native_vision import NativeVisionEncoder
    from teacher.bridge import TeacherBridge
    from viz.emitter import VizEmitter

    config = Config()
    store = StateStore(config.db_path)

    print("Loading encoders (CLIP ~5s)...")
    clip = CLIPWrapper()
    image_enc = ImageEncoder(clip)
    text_enc = TextEncoder(clip)
    video_enc = VideoEncoder(image_enc)
    decoder = GroundedDecoder(text_encoder=text_enc, db_path=config.db_path)
    print("Encoders ready.")

    native_text = NativeTextEncoder(decoder.vocab, decoder.word_embeddings)
    native_vision = NativeVisionEncoder()
    nt_path = os.path.join(os.path.dirname(config.db_path), "checkpoints", "native_text.pt")
    nv_path = os.path.join(os.path.dirname(config.db_path), "checkpoints", "native_vision.pt")
    if os.path.exists(nt_path):
        try:
            native_text.load_state_dict(torch.load(nt_path, weights_only=True))
            print("Restored native text encoder.")
        except Exception as e:
            print(f"Warning: native text checkpoint incompatible ({e}), starting fresh.")
    if os.path.exists(nv_path):
        try:
            native_vision.load(nv_path)
            print("Restored native vision encoder.")
        except Exception as e:
            print(f"Warning: native vision checkpoint incompatible ({e}), starting fresh.")
            os.unlink(nv_path)  # delete incompatible checkpoint

    teacher = TeacherBridge(host=config.ollama_url, model=config.teacher_model)
    model = BabyModel(initial_clusters=config.initial_clusters, nodes_per_cluster=config.nodes_per_cluster)

    latest = store.get_latest_checkpoint()
    if latest:
        try:
            ckpt = store.load_checkpoint(latest["id"])
            model.restore_from_checkpoint(ckpt)
            print(f"Restored from checkpoint at step {ckpt['step']}.")
            model.cleanup_excess_clusters()
            model.reconnect_orphaned_clusters()
        except Exception as e:
            print(f"Warning: checkpoint restore failed: {e}")
    else:
        print("No checkpoint, starting fresh.")

    curriculum = Curriculum(data_dir=config.data_dir, db_path=config.db_path)

    # Pre-load patches
    if hasattr(curriculum, '_cache') and curriculum._cache:
        cache = curriculum._cache
        if cache._patches is None and cache._patches_path:
            print("Loading patch features...")
            cache._patches = torch.load(cache._patches_path, weights_only=False)
            print(f"Loaded {len(cache._patches)} patch features.")

    emitter = VizEmitter(snapshot_interval=config.snapshot_interval, projection_interval=config.projection_interval)
    emitter._last_graph_json = model.graph.to_json()
    emitter._step = model.step

    loop = LearningLoop(
        model=model, teacher=teacher,
        encoder=(image_enc, text_enc, video_enc),
        decoder=decoder, store=store,
        viz_emitter=emitter, curriculum=curriculum,
        native_text_encoder=native_text,
        native_vision_encoder=native_vision,
    )
    if model.step > 0:
        loop._stage = model.stage

    return loop


# ── Pipeline stages ──

def prepare_batch(loop, batch_size=32, batch_count=0):
    """Stage 1: Load data. Returns (items, replay) or None."""
    graph_summary = loop._cached_graph_summary

    # Cache category weights, refresh every 50 micro-batches
    if not hasattr(loop, '_cat_weights_cache') or batch_count % 50 == 0:
        try:
            cats = loop.store.get_category_performance()
            if len(cats) >= 10:
                max_sim = max(c["avg_sim"] for c in cats) or 1.0
                loop._cat_weights_cache = {c["category"]: max(0.1, 1.0 - c["avg_sim"] / max_sim) for c in cats}
            else:
                loop._cat_weights_cache = None
        except Exception:
            loop._cat_weights_cache = None
    cat_weights = loop._cat_weights_cache

    items = loop.curriculum.next_batch(batch_size, stage=loop._stage, model_state=graph_summary, category_weights=cat_weights)
    if not items:
        return None

    # Mix text (20%)
    if loop._text_curriculum:
        text_items = loop._text_curriculum.next_batch(max(1, len(items) // 5), model_step=loop.model.step)
        if text_items:
            items.extend(text_items)

    # Saturation cap
    active = [c for c in loop.model.graph.clusters if not c.dormant]
    saturated = sum(1 for c in active if c.mean_activation > 0.85)
    if saturated / max(len(active), 1) > 0.2:
        items = items[:min(8, len(items))]

    replay = loop.memory.sample_replay(n=8, category_weights=cat_weights)
    return items, replay


def compute_batch(loop, items, replay):
    """Stage 2: Forward + update. The actual learning. Returns (changes, prediction, activations, elapsed_ms)."""
    t0 = time.perf_counter()
    changes, prediction, activations, anchor, elapsed, all_acts = loop._batch_compute(items, replay)
    return changes, prediction, activations, elapsed


def distill_step(loop, items, items_processed):
    """Stage 3: Native text encoder distillation — EVERY BATCH.
    The distill step is cheap (~5ms) vs batch compute (~2s).
    Running every batch gives 13x more learning signal."""
    if loop.native_text_encoder is None:
        return
    import random
    candidates = [i for i in items if i.description and i.expected_vector is not None]
    if not candidates:
        return
    # Distill on up to 3 items per batch (more signal, still cheap)
    for item in random.sample(candidates, min(3, len(candidates))):
        loss = loop.native_text_encoder.distill_step(item.description, item.expected_vector)
        loop.metrics.record_text_distill(1.0 - loss)
    if items_processed % 7500 < 50:
        loop._save_native_checkpoints()

    # Auto-switch: when distillation is good enough, use native encoder for curriculum
    cos_sim = loop.metrics.snapshot()["distillation"].get("text_cosine_sim")
    if cos_sim is not None and cos_sim > 0.65:
        _switch_to_native_text(loop, cos_sim)


def _switch_to_native_text(loop, cos_sim=0.0):
    """Switch text curriculum to native encoder."""
    if loop._text_curriculum and loop._text_curriculum._encoder is not loop.native_text_encoder:
        loop._text_curriculum._encoder = loop.native_text_encoder
        print(f"[distill] SWITCHED text curriculum to native encoder (cos_sim={cos_sim:.3f})", flush=True)


_vision_distill_images = None  # cached list of (image_path, clip_embedding) pairs


def distill_vision_step(loop, items_processed):
    """Stage 3b: Native vision encoder distillation — every batch.
    Uses local stage0 images + their CLIP embeddings as targets."""
    global _vision_distill_images
    if loop.native_vision_encoder is None:
        return

    # Lazy-load local images + compute CLIP embeddings once
    if _vision_distill_images is None:
        import glob
        image_paths = glob.glob("data/stage0/**/*.jpg", recursive=True)
        image_paths += glob.glob("data/stage0/**/*.png", recursive=True)
        if not image_paths:
            return
        # Compute CLIP embeddings for all local images (one-time, cached)
        import PIL.Image
        _vision_distill_images = []
        for path in image_paths[:200]:  # cap at 200 for memory
            try:
                img = PIL.Image.open(path).convert("RGB")
                clip_vec = loop.image_encoder.encode(img)
                _vision_distill_images.append((path, img, clip_vec))
            except Exception:
                continue
        if _vision_distill_images:
            print(f"[vision_distill] cached {len(_vision_distill_images)} local images", flush=True)

    if not _vision_distill_images:
        return

    # Distill on 2 random images per batch
    import random
    samples = random.sample(_vision_distill_images, min(2, len(_vision_distill_images)))
    for path, img, clip_vec in samples:
        loss = loop.native_vision_encoder.distill_step([img], clip_vec.unsqueeze(0))
        loop.metrics.record_vision_distill(1.0 - loss)

    # Save checkpoint alongside text encoder
    if items_processed % 7500 < 50:
        try:
            loop.native_vision_encoder.save("data/checkpoints/native_vision.pt")
        except Exception:
            pass


def track_categories(loop, items, items_processed):
    """Stage 4: Category performance tracking (every 1500 items)."""
    if items_processed % 1500 > 50:
        return
    import random
    sample = random.sample(items, min(4, len(items)))
    for item in sample:
        if item.expected_vector is None:
            continue
        cat = item.label
        if not cat and item.description:
            import re
            m = re.search(r'\b(dog|cat|bird|car|bus|person|horse|elephant)\b', item.description, re.I)
            cat = m.group(1).lower() if m else None
        if cat:
            pred, _ = loop.model.forward(item.expected_vector, return_activations=False)
            sim = torch.dot(pred, F.normalize(item.expected_vector, dim=0)).item()
            loop.store.update_category_performance(cat, sim, sim > 0.2, loop.model.step)


def grow_and_prune(loop):
    """Stage 5: Growth check (bud, dormancy, connect)."""
    return loop.model.growth_check(loop.store)


def record_cofiring(loop, activations, items_processed):
    """Stage 6: Co-firing pairs (z-score filtered, flush every 7000 items)."""
    if not activations:
        return
    scores = list(activations.values())
    mean_s = sum(scores) / len(scores)
    std_s = (sum((v - mean_s) ** 2 for v in scores) / max(len(scores), 1)) ** 0.5
    sig = [cid for cid, v in activations.items() if v > mean_s + std_s]
    for i in range(len(sig)):
        for j in range(i + 1, len(sig)):
            loop._cofiring_buffer.append((sig[i], sig[j]))

    if len(loop._cofiring_buffer) > 5000:
        loop._cofiring_buffer = loop._cofiring_buffer[-5000:]
    if items_processed % 7000 < 50 and loop._cofiring_buffer:
        loop.store.batch_update_cofiring(loop._cofiring_buffer, loop.model.step)
        loop._cofiring_buffer = []


def run_reasoning(loop, items_processed):
    """Stage 7: Reasoning tasks (every 1500 items, with state isolation)."""
    if loop._reasoning_trainer is None or items_processed % 1500 > 50:
        return
    saved = loop.model.brain.activation_buffer.clone()
    try:
        result = loop._reasoning_trainer.train_step()
        loop.metrics.record_reasoning(result['type'], result['correct'], result.get('similarity', 0.0))
    except Exception:
        pass
    loop.model.brain.activation_buffer = saved


def save_checkpoint(loop, items_processed=0):
    """Stage 8: Save model checkpoint (every 15000 items)."""
    if loop.model.step <= 0 or (items_processed > 0 and items_processed % 15000 > 50):
        return
    try:
        state_dict = {}
        if hasattr(loop.model, 'brain'):
            state_dict["brain_state"] = loop.model.brain.state_dict()
        state_dict["_activation_buffer"] = loop.model._activation_buffer
        if hasattr(loop.model, '_working_memory'):
            state_dict["working_memory"] = loop.model._working_memory.state_dict()
        graph_json = loop.model.graph.to_json()
        loop.store.save_checkpoint(
            step=loop.model.step, stage=loop._stage,
            model_state_dict=state_dict, graph_json=graph_json,
        )
        loop.store.prune_old_snapshots()
        # Also save native encoder weights alongside brain checkpoint
        loop._save_native_checkpoints()
        if loop.native_vision_encoder is not None:
            try:
                loop.native_vision_encoder.save("data/checkpoints/native_vision.pt")
            except Exception:
                pass
    except Exception as e:
        print(f"[checkpoint] error: {e}", flush=True)


_viz_cache = {"clusters": [], "edges": [], "nodes": [], "step": 0}
_activity = {}  # latest fired neurons + input info
_pca_components = None  # PCA projection axes (2, dim)
_pca_step = 0
_community_map = {}  # cluster_id → community_id
_community_step = 0
_dialogue_buffer = []  # last 50 Q&A entries
_dashboard_cache = {}


@torch.no_grad()
def _compute_pca_positions(loop):
    """Compute 3D positions via PCA on active neuron weights.
    PCA finds the 2 axes of MOST variance — similar neurons cluster naturally.
    Y = layer_index (vertical structure), X/Z = first 2 principal components.
    Recompute PCA axes every 5000 steps (stable between recomputes)."""
    global _pca_components, _pca_step
    brain = loop.model.brain
    n = brain.n
    if n == 0:
        return {}

    active_mask = ~brain.dormant[:n]
    active_idx = active_mask.nonzero().squeeze(1)
    n_active = len(active_idx)
    if n_active < 3:
        return {}

    weights = brain.weights[active_idx]  # (n_active, dim) on device

    # Recompute PCA axes periodically (eigendecompose is cheap for 512×512)
    if _pca_components is None or loop.model.step - _pca_step > 5000:
        centered = weights - weights.mean(dim=0, keepdim=True)
        # Covariance: (dim, n_active) @ (n_active, dim) = (dim, dim)
        cov = (centered.T @ centered) / max(n_active - 1, 1)
        # Top 2 eigenvectors (on CPU — eigendecompose not on MPS)
        cov_cpu = cov.cpu().float()
        eigenvalues, eigenvectors = torch.linalg.eigh(cov_cpu)
        # Last 2 eigenvectors = largest eigenvalues
        _pca_components = eigenvectors[:, -2:].T.to(brain.device)  # (2, dim)
        _pca_step = loop.model.step

    # Project: (n_active, dim) @ (dim, 2) → (n_active, 2)
    projected = (weights @ _pca_components.T).cpu()  # (n_active, 2)

    # Scale to reasonable visual range (±5 units)
    scale = projected.abs().quantile(0.95).clamp(min=0.1).item()
    projected = projected / scale * 5.0

    layers = brain.layer_indices[active_idx].cpu()

    positions = {}
    for i, gidx in enumerate(active_idx.cpu().tolist()):
        cid = brain.cluster_ids[gidx]
        positions[cid] = [
            float(projected[i, 0]),      # X: PC1
            float(layers[i]) * 1.5,      # Y: layer (compressed)
            float(projected[i, 1]),       # Z: PC2
        ]
    return positions


def _compute_communities(loop):
    """Assign neurons to communities from cofiring data. Cache result."""
    global _community_map, _community_step
    if loop.model.step - _community_step < 3000 and _community_map:
        return _community_map

    try:
        pairs = loop.store.get_cofiring_pairs()
        strong = [p for p in pairs if p["count"] > 10]
        if not strong:
            return _community_map

        parent = {}
        def find(x):
            while parent.get(x, x) != x:
                parent[x] = parent.get(parent[x], parent[x])
                x = parent[x]
            return x
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb: parent[ra] = rb

        for p in strong[:2000]:
            union(p["a"], p["b"])

        # Assign community IDs (0-based)
        from collections import defaultdict
        components = defaultdict(list)
        all_ids = set()
        for p in strong[:2000]:
            all_ids.add(p["a"]); all_ids.add(p["b"])
        for cid in all_ids:
            components[find(cid)].append(cid)

        # Sort by size, assign stable IDs
        sorted_comms = sorted(components.values(), key=len, reverse=True)
        new_map = {}
        for comm_id, members in enumerate(sorted_comms):
            for cid in members:
                new_map[cid] = comm_id

        _community_map = new_map
        _community_step = loop.model.step
    except Exception:
        pass
    return _community_map


def _compute_viz(loop):
    """Build viz data: ONLY active neurons, PCA positions, community colors, clean edges."""
    brain = loop.model.brain
    n = brain.n
    positions = _compute_pca_positions(loop)
    communities = _compute_communities(loop)

    # ONLY active neurons (not dormant — those are dead weight visually)
    clusters = []
    nodes = []
    active_mask = ~brain.dormant[:n]
    fire_rates = brain.fire_rates[:n].cpu()

    for i in range(n):
        if brain.dormant[i].item():
            continue  # skip dormant — don't send to frontend
        cid = brain.cluster_ids[i]
        pos = positions.get(cid)
        if pos is None:
            continue
        comm = communities.get(cid, -1)
        # Encode community as cluster_type for frontend coloring
        # Frontend maps cluster_type → color. We'll use "comm_N" format.
        ctype = f"comm_{comm}" if comm >= 0 else "unknown"

        age = int(brain.ages[i].item())
        clusters.append({
            "id": cid,
            "cluster_type": ctype,
            "dormant": False,
            "layer_index": float(brain.layer_indices[i].item()),
            "age": age,
            "pos": pos,
        })
        nid = f"n_{i:04d}"
        nodes.append({
            "id": nid,
            "cluster": cid,
            "pos": pos,
            "activation_mean": float(fire_rates[i].item()),
            "alive": True,
            "age": age,
        })

    # Inter-community edges only (top 100 by strength)
    # Skip edges within the same community (reduces spaghetti)
    edges = []
    for (i, j), s in sorted(brain._edge_strengths.items(), key=lambda x: x[1], reverse=True):
        if i >= n or j >= n:
            continue
        if brain.dormant[i].item() or brain.dormant[j].item():
            continue
        cid_i = brain.cluster_ids[i]
        cid_j = brain.cluster_ids[j]
        comm_i = communities.get(cid_i, -1)
        comm_j = communities.get(cid_j, -2)
        if comm_i == comm_j and comm_i >= 0:
            continue  # skip intra-community edges
        edges.append({
            "from": cid_i,
            "to": cid_j,
            "strength": round(s, 4),
        })
        if len(edges) >= 100:
            break

    return {"clusters": clusters, "nodes": nodes, "edges": edges}


def _compute_dashboard(loop):
    """Compute dashboard metrics (spatial score, communities, categories)."""
    try:
        cats = loop.store.get_category_performance()
        best = cats[-5:] if len(cats) >= 5 else cats
        worst = cats[:5]

        # Community count from cofiring
        cofiring_pairs = loop.store.get_cofiring_pairs()
        strong = [p for p in cofiring_pairs if p["count"] > 10]
        if strong:
            parent_map = {}
            def find(x):
                while parent_map.get(x, x) != x:
                    parent_map[x] = parent_map.get(parent_map[x], parent_map[x])
                    x = parent_map[x]
                return x
            def union(a, b):
                ra, rb = find(a), find(b)
                if ra != rb: parent_map[ra] = rb
            for p in strong[:2000]:
                union(p["a"], p["b"])
            from collections import defaultdict
            components = defaultdict(list)
            all_ids = set()
            for p in strong[:2000]:
                all_ids.add(p["a"]); all_ids.add(p["b"])
            for cid in all_ids:
                components[find(cid)].append(cid)
            sizes = sorted([len(v) for v in components.values()], reverse=True)
            communities = len([s for s in sizes if s >= 3])
        else:
            communities = 0
            sizes = []

        gs = loop._cached_graph_summary
        return {
            "spatial_score": None,  # expensive, skip for now
            "communities": communities,
            "community_sizes": sizes[:10],
            "categories": {
                "best": [{"category": c["category"], "avg_sim": c["avg_sim"]} for c in best],
                "worst": [{"category": c["category"], "avg_sim": c["avg_sim"]} for c in worst],
                "total_tracked": len(cats),
            },
            "growth_rate": round(gs.get("cluster_count", 0) / max(loop.model.step / 1000, 1), 1),
            "edge_ratio": round(gs.get("edge_count", 0) / max(gs.get("cluster_count", 1), 1), 1),
        }
    except Exception:
        return {}


def add_dialogue(items, prediction, loop):
    """Add training Q&A to dialogue buffer — pick the most interesting item."""
    global _dialogue_buffer
    if not items:
        return
    import random as _rng

    # Pick a random item (not always the last — show variety)
    item = _rng.choice(items)
    desc = item.description or ""
    is_text = item.item_type == "text" if hasattr(item, "item_type") else False
    category = item.label or ""

    # Get model's response for this specific item
    model_response = ""
    similarity = 0.0
    try:
        if item.expected_vector is not None:
            item_pred, _ = loop.model.forward(item.expected_vector, return_activations=False)
            model_response = loop.decoder.decode(item_pred, max_words=8, model_step=loop.model.step)
            similarity = float(torch.dot(item_pred, F.normalize(item.expected_vector, dim=0)).item())
    except Exception:
        pass

    # Format the question to show what TYPE of learning this is
    if is_text:
        question = f"[text/{category}] {desc[:80]}"
    else:
        question = f"[image/{category}] {desc[:80]}"

    _dialogue_buffer.append({
        "step": loop.model.step,
        "question": question,
        "answer": f"brain says: {model_response}" if model_response else "(no response)",
        "model_answer": model_response,
        "curiosity_score": round(max(0, 1.0 - similarity), 2),  # higher = more confused
        "is_positive": similarity > 0.2,
        "stage": loop._stage,
        "timestamp": time.time(),
        "image_url": getattr(item, "image_url", None),
    })
    if len(_dialogue_buffer) > 50:
        _dialogue_buffer = _dialogue_buffer[-50:]


def publish(loop, full=False):
    """Publish rich state for HTTP server + frontend."""
    global _viz_cache, _dashboard_cache
    try:
        loop._update_cached_status()

        # Viz data: recompute every 500 steps or on full refresh
        if full or loop.model.step - _viz_cache.get("step", 0) > 500:
            _viz_cache = _compute_viz(loop)
            _viz_cache["step"] = loop.model.step

        # Dashboard: recompute every 3000 steps
        if full or loop.model.step - _dashboard_cache.get("_step", 0) > 3000:
            _dashboard_cache = _compute_dashboard(loop)
            _dashboard_cache["_step"] = loop.model.step

        write_state({
            "step": loop.model.step,
            "stage": loop._stage,
            "state": "running",
            "graph_summary": loop._cached_graph_summary,
            "metrics": loop.metrics.snapshot(),
            "viz": {
                "clusters": _viz_cache.get("clusters", []),
                "nodes": _viz_cache.get("nodes", []),
                "edges": _viz_cache.get("edges", []),
            },
            "dashboard": _dashboard_cache,
            "activity": _activity,
            "dialogue": _dialogue_buffer[-50:],
        })
    except Exception as e:
        print(f"[publish] error: {e}", flush=True)


# ── Main loop ──

def run(loop):
    """Clean training loop. Each stage independent, timed, fail-safe."""
    import asyncio

    # Teacher warmup (needs event loop for the async HTTP call)
    ev = asyncio.new_event_loop()
    asyncio.set_event_loop(ev)
    try:
        ev.run_until_complete(loop.teacher.ask("Hello", stage=0))
        print("Teacher reachable.", flush=True)
    except Exception:
        print("Teacher not reachable.", flush=True)

    print("Training started.", flush=True)

    # If native text checkpoint exists, switch immediately (don't wait for cos_sim)
    if os.path.exists(os.path.join("data", "checkpoints", "native_text.pt")):
        _switch_to_native_text(loop, cos_sim=0.0)
        print("[startup] native text encoder checkpoint found — using native from start", flush=True)

    publish(loop)  # initial state

    batch_count = 0
    items_processed = 0
    state = "running"

    while True:
        # Check commands
        cmd = read_command()
        if cmd == "pause":
            state = "paused"
            write_state({"step": loop.model.step, "stage": loop._stage, "state": "paused",
                         "graph_summary": loop._cached_graph_summary, "metrics": loop.metrics.snapshot()})
            print("[worker] paused", flush=True)
            while True:
                time.sleep(0.5)
                cmd = read_command()
                if cmd in ("resume", "start"):
                    state = "running"
                    print("[worker] resumed", flush=True)
                    break
                if cmd == "stop":
                    return
        elif cmd == "stop":
            return

        # ── Chat check (between batches, instant response) ──
        handle_chat(loop)

        # ── PIPELINE ──
        try:
            # 1. Prepare
            batch_data = prepare_batch(loop, batch_count=batch_count)
            if batch_data is None:
                time.sleep(0.1)
                continue
            items, replay = batch_data

            # 2. Compute (the real work)
            changes, prediction, activations, elapsed_ms = compute_batch(loop, items, replay)
            batch_count += 1
            items_processed += len(items)

            # Capture activity for frontend pulse view
            global _activity
            last_item = items[-1]
            top_fired = sorted(activations.items(), key=lambda x: x[1], reverse=True)[:20]
            _activity = {
                "fired": [cid for cid, _ in top_fired],
                "scores": [round(s, 3) for _, s in top_fired],
                "input": (last_item.description or "")[:100],
                "input_type": "text" if getattr(last_item, "item_type", "") == "text" else "image",
            }

            # 3. Distill text + vision (every batch)
            try:
                distill_step(loop, items, items_processed)
            except Exception as e:
                print(f"[distill_text] error: {e}", flush=True)
            try:
                distill_vision_step(loop, items_processed)
            except Exception as e:
                print(f"[distill_vision] error: {e}", flush=True)

            # 4. Categories (every 1500 items)
            try:
                track_categories(loop, items, items_processed)
            except Exception as e:
                print(f"[categories] error: {e}", flush=True)

            # 5. Growth
            try:
                grow_and_prune(loop)
            except Exception as e:
                print(f"[growth] error: {e}", flush=True)

            # 6. Cofiring (flush every 7000 items)
            try:
                record_cofiring(loop, activations, items_processed)
            except Exception as e:
                print(f"[cofiring] error: {e}", flush=True)

            # 7. Reasoning (every 1500 items)
            try:
                run_reasoning(loop, items_processed)
            except Exception as e:
                print(f"[reasoning] error: {e}", flush=True)

            # 8. Decoder training + dialogue capture
            try:
                teacher_answer = items[-1].description or ""
                teacher_clip = items[-1].expected_vector if items[-1].expected_vector is not None else prediction
                loop.decoder.train_step(teacher_clip, teacher_answer)
                add_dialogue(items, prediction, loop)
            except Exception as e:
                print(f"[decoder] error: {e}", flush=True)

            # 9. Checkpoint (every 15000 items)
            try:
                save_checkpoint(loop, items_processed)
            except Exception as e:
                print(f"[checkpoint] error: {e}", flush=True)

            # 10. Publish state (every batch for responsive UI)
            publish(loop)

            # Log (every 1500 items)
            if items_processed % 1500 < 50:
                gs = loop._cached_graph_summary
                print(
                    f"[worker] step={loop.model.step} active={gs.get('node_count', '?')} "
                    f"items={items_processed} batch={batch_count} elapsed={elapsed_ms:.0f}ms",
                    flush=True,
                )

        except Exception as e:
            print(f"[worker] batch error: {e}", flush=True)
            traceback.print_exc()
            time.sleep(1)


if __name__ == "__main__":
    print("=" * 60)
    print("BABY AI TRAINING WORKER")
    print("=" * 60)

    loop = build_components()
    print("Ready.")

    try:
        run(loop)
    except KeyboardInterrupt:
        print("\n[worker] interrupted, saving checkpoint...")
        save_checkpoint(loop)
        publish(loop)
        print("[worker] done.")
