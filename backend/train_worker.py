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
        native_text.load_state_dict(torch.load(nt_path, weights_only=True))
        print("Restored native text encoder.")
    if os.path.exists(nv_path):
        native_vision.load(nv_path)
        print("Restored native vision encoder.")

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
    """Stage 3: Native text encoder distillation (every 500 items)."""
    if loop.native_text_encoder is None or items_processed % 500 != 0:
        return
    import random
    candidates = [i for i in items if i.description and i.expected_vector is not None]
    if not candidates:
        return
    item = random.choice(candidates)
    loss = loop.native_text_encoder.distill_step(item.description, item.expected_vector)
    loop.metrics.record_text_distill(1.0 - loss)
    if items_processed % 7500 == 0:
        loop._save_native_checkpoints()


def track_categories(loop, items, items_processed):
    """Stage 4: Category performance tracking (every 1500 items)."""
    if items_processed % 1500 != 0:
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
    if items_processed % 7000 == 0 and loop._cofiring_buffer:
        loop.store.batch_update_cofiring(loop._cofiring_buffer, loop.model.step)
        loop._cofiring_buffer = []


def run_reasoning(loop, items_processed):
    """Stage 7: Reasoning tasks (every 1500 items, with state isolation)."""
    if loop._reasoning_trainer is None or items_processed % 1500 != 0:
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
    if loop.model.step <= 0 or (items_processed > 0 and items_processed % 15000 != 0):
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
    except Exception as e:
        print(f"[checkpoint] error: {e}", flush=True)


_cached_graph_json = None
_graph_json_step = 0

def publish(loop, full=False):
    """Publish current state to shared file for HTTP server.
    full=True forces graph_json refresh (expensive). Otherwise uses cached."""
    global _cached_graph_json, _graph_json_step
    try:
        loop._update_cached_status()
        # Only recompute graph_json every 500 items or on full refresh
        if full or _cached_graph_json is None or loop.model.step - _graph_json_step > 500:
            _cached_graph_json = loop.model.graph.to_json()
            _graph_json_step = loop.model.step
        write_state({
            "step": loop.model.step,
            "stage": loop._stage,
            "state": "running",
            "graph_summary": loop._cached_graph_summary,
            "metrics": loop.metrics.snapshot(),
            "graph_json": _cached_graph_json,
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

            # 3. Distill (every 500 items)
            try:
                distill_step(loop, items, items_processed)
            except Exception as e:
                print(f"[distill] error: {e}", flush=True)

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

            # 8. Decoder training
            try:
                teacher_answer = items[-1].description or ""
                teacher_clip = items[-1].expected_vector if items[-1].expected_vector is not None else prediction
                loop.decoder.train_step(teacher_clip, teacher_answer)
            except Exception as e:
                print(f"[decoder] error: {e}", flush=True)

            # 9. Checkpoint (every 15000 items)
            try:
                save_checkpoint(loop, items_processed)
            except Exception as e:
                print(f"[checkpoint] error: {e}", flush=True)

            # 10. Publish state (every 50 items for responsive UI)
            if items_processed % 50 == 0:
                publish(loop)

            # Log (every 1500 items)
            if items_processed % 1500 == 0:
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
