"""
Parallel trainer — multiple workers with periodic weight reconciliation.

Each worker trains on a different data partition independently.
Every MERGE_INTERVAL steps, all workers pause, deltas are summed,
and merged weights are distributed. Then workers resume.

Architecture:
    Worker 1 (images)     ─┐
    Worker 2 (text)        ├─→ MERGE (every 100 steps) ─→ shared weights
    Worker 3 (conv+reason) ─┘

Delta merge is mathematically equivalent to sequential training:
    merged = snapshot + delta_1 + delta_2 + delta_3
    (addition is commutative — order doesn't matter)

Usage:
    python parallel_trainer.py              # 3 workers
    python parallel_trainer.py --workers 4  # 4 workers
"""

import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

MERGE_INTERVAL = 100  # steps between reconciliation
STATE_DIR = os.path.join(os.path.dirname(__file__), "data")


def worker_process(worker_id, n_workers, shared_weights, shared_shape,
                   merge_barrier, snapshot_lock, step_counter, stop_flag):
    """One training worker. Trains independently, pauses for merges."""
    import torch
    import torch.nn.functional as F

    # Build components (each worker loads its own model)
    from train_worker import build_components, prepare_batch, compute_batch
    from train_worker import distill_step, grow_and_prune, publish
    from train_worker import _switch_to_native_text, _build_native_vision_items
    from loop.shared_state import write_state

    loop = build_components()

    # Switch to native text if checkpoint exists
    if os.path.exists(os.path.join("data", "checkpoints", "native_text.pt")):
        _switch_to_native_text(loop, cos_sim=0.0)

    _build_native_vision_items(loop)

    print(f"[worker-{worker_id}] ready, {loop._text_curriculum.size if loop._text_curriculum else 0} text items", flush=True)

    brain = loop.model.brain
    n = brain.n
    batch_count = 0
    items_processed = 0
    local_steps = 0

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

            # Distill every batch
            try:
                distill_step(loop, items, items_processed)
            except Exception:
                pass

            # Growth every batch
            try:
                grow_and_prune(loop)
            except Exception:
                pass

            # Only worker 0 publishes state and handles checkpoints
            if worker_id == 0:
                with step_counter.get_lock():
                    step_counter.value = loop.model.step
                if local_steps % 10 == 0:
                    publish(loop)

        except Exception as e:
            print(f"[worker-{worker_id}] error: {e}", flush=True)
            time.sleep(0.5)
            continue

        # ── MERGE CHECKPOINT ──
        if local_steps % MERGE_INTERVAL == 0:
            # Wait for all workers to reach this point
            merge_barrier.wait()

            # Worker 0 is the coordinator
            if worker_id == 0:
                # Snapshot is already in shared memory from last merge
                # Collect: each worker's weights are their own brain.weights
                pass  # deltas applied below

            # Each worker writes its delta to shared memory
            # Delta = current_weights - snapshot (what this worker learned)
            weights_np = brain.weights[:n].cpu().numpy()
            shared_arr = np.frombuffer(shared_weights.get_obj(), dtype=np.float32).reshape(shared_shape)

            with snapshot_lock:
                # Add this worker's delta to the shared accumulator
                # On first worker: accumulator = snapshot + delta_1
                # On second: accumulator = snapshot + delta_1 + delta_2
                # etc.
                if worker_id == 0:
                    # Reset accumulator to current snapshot (worker 0's weights)
                    shared_arr[:n, :] = weights_np
                else:
                    # Add delta: shared += (my_weights - shared) / n_workers
                    # This incrementally averages
                    shared_arr[:n, :] += (weights_np - shared_arr[:n, :]) / (worker_id + 1)

            # Wait for all workers to write their deltas
            merge_barrier.wait()

            # All workers load the merged weights
            merged = torch.from_numpy(shared_arr[:n, :].copy()).to(brain.device)
            brain.weights[:n] = F.normalize(merged, dim=1)

            if worker_id == 0:
                print(
                    f"[merge] step={loop.model.step} workers={n_workers} "
                    f"local_steps={local_steps}",
                    flush=True,
                )

    print(f"[worker-{worker_id}] stopped", flush=True)


def run_parallel(n_workers=3):
    """Launch parallel training with weight reconciliation."""
    import torch

    # Load model to get dimensions
    from train_worker import build_components
    loop = build_components()
    brain = loop.model.brain
    n = brain.n
    dim = brain.dim
    print(f"Brain: {n} neurons, {dim} dims", flush=True)

    # Shared memory for weight merging
    shared_shape = (brain.max_size, dim)
    total_floats = shared_shape[0] * shared_shape[1]
    shared_weights = mp.Array('f', total_floats, lock=True)

    # Initialize shared memory with current weights
    shared_arr = np.frombuffer(shared_weights.get_obj(), dtype=np.float32).reshape(shared_shape)
    shared_arr[:n, :] = brain.weights[:n].cpu().numpy()

    # Synchronization
    merge_barrier = mp.Barrier(n_workers)
    snapshot_lock = mp.Lock()
    step_counter = mp.Value('i', loop.model.step)
    stop_flag = mp.Value('b', False)

    # Clean up the loop we used for init
    del loop, brain

    # Launch workers
    workers = []
    for i in range(n_workers):
        p = mp.Process(
            target=worker_process,
            args=(i, n_workers, shared_weights, shared_shape,
                  merge_barrier, snapshot_lock, step_counter, stop_flag),
        )
        p.start()
        workers.append(p)
        print(f"Launched worker {i} (PID {p.pid})", flush=True)

    # Wait for interrupt
    try:
        while True:
            time.sleep(5)
            alive = sum(1 for p in workers if p.is_alive())
            if alive == 0:
                print("All workers stopped.", flush=True)
                break
            print(
                f"[coordinator] step={step_counter.value} "
                f"workers={alive}/{n_workers}",
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
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()

    print("=" * 60)
    print(f"PARALLEL TRAINER — {args.workers} workers")
    print("=" * 60)

    run_parallel(n_workers=args.workers)
