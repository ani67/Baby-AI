"""
Parallel trainer — memory-efficient multi-worker training.

Pre-encodes all training data in the parent process (CLIP loaded once),
then spawns lightweight workers that never touch CLIP.

Architecture:
    Parent (one-time, ~5min):
      1. Load CLIP + encoders
      2. Pre-encode 46K text items + 786 vision items
      3. Save decoder embeddings
      4. Free CLIP (~400MB returned)
      5. Launch workers

    Worker 0 (~350MB):           Worker 1 (~350MB):
      Brain + native encoders      Brain + native encoders
      Pre-encoded text/vision      Pre-encoded text/vision
      Decoder (for publish)        No decoder
      Periodic merge ──────────────Periodic merge

Memory: ~800MB parent (peak during encode, drops after) + ~350MB per worker
vs old: ~1.5GB per worker × 3 + parent = 40GB OOM

Usage:
    python parallel_trainer.py              # 2 workers (default)
    python parallel_trainer.py --workers 3  # 3 workers (needs 32GB+)
"""

import gc
import glob
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

MERGE_INTERVAL = 100  # steps between weight reconciliation
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CACHE_PATH = os.path.join(DATA_DIR, "checkpoints", "pre_encoded_cache.pt")


# ── Pre-encoding (parent process) ──

def _load_raw_text_items() -> list[dict]:
    """Load all text curriculum JSON files, return raw item dicts."""
    items = []
    filenames = [
        "text_curriculum.json", "text_diverse.json",
        "text_conversations.json", "text_commonsense.json",
        "reasoning_tasks.json",
    ]
    for filename in filenames:
        path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(path):
            continue
        try:
            data = json.loads(open(path).read())
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(data, dict):
            data = data.get("items", [])
        if not isinstance(data, list):
            continue
        for entry in data:
            if not isinstance(entry, dict):
                continue
            item = {
                "index": len(items),
                "text": entry.get("text", ""),
                "question": entry.get("question", ""),
                "answer": entry.get("answer", ""),
                "category": entry.get("category", "text"),
                "level": int(entry.get("level", 1)),
                "source": filename,
            }
            if item["text"] or (item["question"] and item["answer"]):
                items.append(item)
    return items


def _pre_encode_text(raw_items: list[dict], text_enc) -> list[dict]:
    """Encode all text items through CLIP. Returns items with input_vec/expected_vec."""
    import torch

    # Collect unique texts
    texts_needed = set()
    for item in raw_items:
        if item["question"] and item["answer"]:
            texts_needed.add(item["question"])
            texts_needed.add(item["answer"])
        elif item["text"] and item["answer"]:
            texts_needed.add(item["text"])
            texts_needed.add(item["answer"])
        elif item["text"]:
            texts_needed.add(item["text"])

    # Batch encode through CLIP
    text_list = list(texts_needed)
    encoded = {}
    batch_size = 64
    print(f"[pre-encode] encoding {len(text_list)} unique texts...", flush=True)
    for i in range(0, len(text_list), batch_size):
        batch = text_list[i:i + batch_size]
        vecs = text_enc.encode_batch(batch)
        for j, t in enumerate(batch):
            encoded[t] = vecs[j].cpu()
        if (i // batch_size) % 50 == 0 and i > 0:
            print(f"[pre-encode] {i}/{len(text_list)} texts encoded", flush=True)
    print(f"[pre-encode] done: {len(encoded)} unique texts encoded", flush=True)

    # Attach vectors to items
    for item in raw_items:
        has_question = bool(item["question"] and item["answer"])
        has_text_answer = bool(item["text"] and item["answer"])

        if has_question:
            item["input_vec"] = encoded[item["question"]]
            item["expected_vec"] = encoded[item["answer"]]
        elif has_text_answer:
            item["input_vec"] = encoded[item["text"]]
            item["expected_vec"] = encoded[item["answer"]]
        elif item["text"]:
            vec = encoded[item["text"]]
            item["input_vec"] = vec
            item["expected_vec"] = vec

    return raw_items


def _pre_encode_vision(image_enc, native_vision) -> list[dict]:
    """Encode stage0 images through both CLIP and native ConvNet.
    Includes 16-patch sequential vectors with positional encoding."""
    import PIL.Image
    import torch

    image_paths = glob.glob(os.path.join(DATA_DIR, "stage0", "**", "*.jpg"), recursive=True)
    image_paths += glob.glob(os.path.join(DATA_DIR, "stage0", "**", "*.png"), recursive=True)
    if not image_paths:
        print("[pre-encode] no stage0 images found", flush=True)
        return []

    items = []
    for path in image_paths:
        try:
            img = PIL.Image.open(path).convert("RGB")
            native_vec = native_vision.encode(img).cpu()
            clip_vec = image_enc.encode(img).cpu()
            label = os.path.basename(os.path.dirname(path))
            # 16-patch sequential vectors (4×4 grid with positional encoding)
            patch_seq = None
            try:
                patch_seq = native_vision.encode_patches_grid(img)
            except Exception:
                pass
            items.append({
                "input_vector": native_vec,
                "expected_vector": clip_vec,
                "label": label,
                "image_path": path,
                "sequence": patch_seq,
            })
        except Exception:
            continue

    print(f"[pre-encode] encoded {len(items)} vision items from stage0", flush=True)
    return items


def pre_encode_all():
    """Load CLIP once, pre-encode everything, return data + brain dims. Frees CLIP after."""
    import torch
    from config import Config
    from state.store import StateStore
    from model.baby_model_v2 import BabyModelV2 as BabyModel
    from encoder.clip_mlx import CLIPWrapper
    from encoder.encoder import ImageEncoder, TextEncoder
    from encoder.native_vision import NativeVisionEncoder
    from encoder.decoder import GroundedDecoder

    config = Config()
    store = StateStore(config.db_path)

    # Get brain dimensions from checkpoint (no CLIP needed for this)
    model = BabyModel(initial_clusters=config.initial_clusters,
                      nodes_per_cluster=config.nodes_per_cluster)
    latest = store.get_latest_checkpoint()
    if latest:
        try:
            ckpt = store.load_checkpoint(latest["id"])
            model.restore_from_checkpoint(ckpt)
            print(f"Brain from checkpoint: step={ckpt['step']}", flush=True)
        except Exception as e:
            print(f"Checkpoint restore failed: {e}", flush=True)

    brain_n = model.brain.n
    brain_dim = model.brain.dim
    brain_max = model.brain.max_size
    initial_weights = model.brain.weights[:brain_n].cpu().numpy()
    del model, store
    gc.collect()

    # Check how many items we'd encode (for cache staleness check)
    raw_text = _load_raw_text_items()
    n_text = len(raw_text)
    image_paths = glob.glob(os.path.join(DATA_DIR, "stage0", "**", "*.jpg"), recursive=True)
    image_paths += glob.glob(os.path.join(DATA_DIR, "stage0", "**", "*.png"), recursive=True)
    n_vision = len(image_paths)

    # Try loading from cache
    pre_text = None
    pre_vision = None
    if os.path.exists(CACHE_PATH):
        try:
            cache = torch.load(CACHE_PATH, weights_only=False)
            if cache.get("n_text") == n_text and cache.get("n_vision") == n_vision:
                pre_text = cache["pre_text"]
                pre_vision = cache["pre_vision"]
                print(f"[pre-encode] loaded from cache ({n_text} text, {n_vision} vision)", flush=True)
            else:
                print(f"[pre-encode] cache stale (cached {cache.get('n_text')}/{cache.get('n_vision')}, "
                      f"current {n_text}/{n_vision}), re-encoding...", flush=True)
            del cache
        except Exception as e:
            print(f"[pre-encode] cache load failed ({e}), re-encoding...", flush=True)

    if pre_text is None:
        print("[pre-encode] cache miss, encoding from scratch...", flush=True)

        # Load CLIP (the only time in the entire parallel run)
        print("Loading CLIP (one-time)...", flush=True)
        clip = CLIPWrapper()
        image_enc = ImageEncoder(clip)
        text_enc = TextEncoder(clip)

        # Load native vision encoder for stage0 encoding
        native_vision = NativeVisionEncoder()
        nv_path = os.path.join(os.path.dirname(config.db_path), "checkpoints", "native_vision.pt")
        if os.path.exists(nv_path):
            try:
                native_vision.load(nv_path)
                print("Restored native vision encoder for pre-encoding.", flush=True)
            except Exception:
                pass

        # Save decoder embeddings for workers
        decoder = GroundedDecoder(text_encoder=text_enc, db_path=config.db_path)
        emb_path = os.path.join(os.path.dirname(config.db_path), "checkpoints", "decoder_embeddings.pt")
        decoder.save_embeddings(emb_path)
        print(f"Saved decoder embeddings to {emb_path}", flush=True)
        del decoder

        # Pre-encode text
        pre_text = _pre_encode_text(raw_text, text_enc)

        # Pre-encode vision
        pre_vision = _pre_encode_vision(image_enc, native_vision)

        # Save cache
        torch.save({"n_text": n_text, "n_vision": n_vision,
                     "pre_text": pre_text, "pre_vision": pre_vision}, CACHE_PATH)
        print(f"[pre-encode] saved cache ({n_text} text, {n_vision} vision)", flush=True)

        # Free CLIP and all encoding infrastructure
        del clip, image_enc, text_enc, native_vision
        gc.collect()
        try:
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass
        print("CLIP freed. Pre-encoding complete.", flush=True)
    else:
        # Cache hit — decoder embeddings should already exist from a prior encode
        emb_path = os.path.join(os.path.dirname(config.db_path), "checkpoints", "decoder_embeddings.pt")
        if not os.path.exists(emb_path):
            print("Loading CLIP for decoder embeddings only...", flush=True)
            clip = CLIPWrapper()
            text_enc = TextEncoder(clip)
            decoder = GroundedDecoder(text_encoder=text_enc, db_path=config.db_path)
            decoder.save_embeddings(emb_path)
            print(f"Saved decoder embeddings to {emb_path}", flush=True)
            del decoder, clip, text_enc
            gc.collect()

    return {
        "brain_n": brain_n,
        "brain_dim": brain_dim,
        "brain_max": brain_max,
        "initial_weights": initial_weights,
        "pre_text": pre_text,
        "pre_vision": pre_vision,
    }


# ── Worker process ──

def worker_process(worker_id, n_workers, shared_weights, shared_shape,
                   merge_lock, merge_counter, step_counter, stop_flag,
                   pre_encoded_text, pre_encoded_vision):
    """Lightweight training worker — no CLIP, uses pre-encoded data."""
    import torch
    import torch.nn.functional as F

    from train_worker import build_worker_components, prepare_batch, compute_batch
    from train_worker import distill_step, distill_vision_step_parallel
    from train_worker import grow_and_prune, publish
    from train_worker import _switch_to_native_text
    from loop.shared_state import write_state

    loop = build_worker_components(worker_id, pre_encoded_text, pre_encoded_vision)

    # Switch to native text if checkpoint exists
    if os.path.exists(os.path.join("data", "checkpoints", "native_text.pt")):
        _switch_to_native_text(loop, cos_sim=0.0)

    brain = loop.model.brain
    local_steps = 0
    batch_count = 0
    items_processed = 0

    while not stop_flag.value:
        # ── TRAIN ──
        try:
            batch_data = prepare_batch(loop, batch_count=batch_count)
            if batch_data is None:
                time.sleep(0.1)
                continue

            items, replay = batch_data
            changes, prediction, activations, elapsed_ms = compute_batch(loop, items, replay)
            batch_count += 1
            items_processed += len(items)
            local_steps += 1

            try:
                distill_step(loop, items, items_processed)
            except Exception:
                pass

            try:
                distill_vision_step_parallel(loop, items_processed)
            except Exception:
                pass

            try:
                grow_and_prune(loop)
            except Exception:
                pass

            if worker_id == 0:
                with step_counter.get_lock():
                    step_counter.value = loop.model.step
                if local_steps % 10 == 0:
                    publish(loop)

            # Flush MPS memory cache periodically
            if batch_count % 50 == 0:
                try:
                    if torch.backends.mps.is_available():
                        torch.mps.empty_cache()
                except Exception:
                    pass

        except Exception as e:
            print(f"[worker-{worker_id}] error: {e}", flush=True)
            time.sleep(0.5)
            continue

        # ── MERGE (lock-based, no barrier) ──
        # Each worker independently merges every MERGE_INTERVAL local steps.
        # Uses shared memory as the "reference" weights. Each worker:
        #   1. Reads shared weights
        #   2. Averages with its own: merged = (shared + local) / 2
        #   3. Writes merged back to shared and applies to local brain
        if local_steps % MERGE_INTERVAL == 0:
            n = min(brain.n, shared_shape[0])

            weights_np = brain.weights[:n].cpu().numpy()
            shared_arr = np.frombuffer(shared_weights.get_obj(), dtype=np.float32).reshape(shared_shape)

            with merge_lock:
                merged = (shared_arr[:n, :] + weights_np) * 0.5
                shared_arr[:n, :] = merged
                with merge_counter.get_lock():
                    merge_counter.value += 1
                    mc = merge_counter.value

            merged_t = torch.from_numpy(merged.copy()).to(brain.device)
            brain.weights[:n] = F.normalize(merged_t, dim=1)

            print(
                f"[merge] worker={worker_id} step={loop.model.step} "
                f"n={n} total_merges={mc}",
                flush=True,
            )

    print(f"[worker-{worker_id}] stopped", flush=True)


# ── Coordinator ──

def _get_memory_mb() -> float | None:
    """Get current process tree memory usage in MB."""
    try:
        import resource
        # rusage is per-process, but gives a rough idea
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return usage.ru_maxrss / (1024 * 1024)  # macOS reports in bytes
    except Exception:
        return None


def run_parallel(n_workers=2):
    """Launch memory-efficient parallel training."""
    import torch

    print("=" * 60)
    print(f"PARALLEL TRAINER — {n_workers} workers (memory-efficient)")
    print("=" * 60)

    # Phase 1: Pre-encode everything (CLIP loaded once, then freed)
    print("\n--- Phase 1: Pre-encoding ---", flush=True)
    data = pre_encode_all()
    brain_n = data["brain_n"]
    brain_dim = data["brain_dim"]
    brain_max = data["brain_max"]
    print(f"Brain: {brain_n} neurons, {brain_dim} dims, max {brain_max}", flush=True)
    print(f"Pre-encoded: {len(data['pre_text'])} text, {len(data['pre_vision'])} vision", flush=True)

    # Phase 2: Shared memory for weight merging
    print("\n--- Phase 2: Shared memory ---", flush=True)
    shared_shape = (brain_max, brain_dim)
    total_floats = shared_shape[0] * shared_shape[1]
    shared_weights = mp.Array('f', total_floats, lock=True)

    shared_arr = np.frombuffer(shared_weights.get_obj(), dtype=np.float32).reshape(shared_shape)
    shared_arr[:brain_n, :] = data["initial_weights"]
    del data["initial_weights"]  # free the copy

    merge_lock = mp.Lock()
    merge_counter = mp.Value('i', 0)
    step_counter = mp.Value('i', 0)
    stop_flag = mp.Value('b', False)

    # Phase 3: Launch lightweight workers
    print(f"\n--- Phase 3: Launching {n_workers} workers ---", flush=True)
    workers = []
    for i in range(n_workers):
        p = mp.Process(
            target=worker_process,
            args=(i, n_workers, shared_weights, shared_shape,
                  merge_lock, merge_counter, step_counter, stop_flag,
                  data["pre_text"], data["pre_vision"]),
        )
        p.start()
        workers.append(p)
        print(f"Launched worker {i} (PID {p.pid})", flush=True)

    # Free parent's copy of pre-encoded data (workers have their own via pickle)
    del data
    gc.collect()

    # Phase 4: Monitor
    print("\n--- Training ---", flush=True)
    try:
        while True:
            time.sleep(10)
            alive = sum(1 for p in workers if p.is_alive())
            if alive == 0:
                print("All workers stopped.", flush=True)
                break

            mem = _get_memory_mb()
            mem_str = f" mem={mem:.0f}MB" if mem else ""
            print(
                f"[coordinator] step={step_counter.value} "
                f"workers={alive}/{n_workers} "
                f"merges={merge_counter.value}{mem_str}",
                flush=True,
            )
    except KeyboardInterrupt:
        print("\nStopping workers...", flush=True)
        stop_flag.value = True
        for p in workers:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()

    print("Parallel training stopped.", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2,
                        help="Number of parallel workers (default: 2, safe for 16GB)")
    args = parser.parse_args()

    run_parallel(n_workers=args.workers)
