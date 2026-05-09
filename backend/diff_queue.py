"""Diff queue for parallel concept-graph ingestion (Phase 7 perf).

Architecture
------------

    +-------------+     +-------------+         +-------------+
    | reader 0    |     | reader 1    |   ...   | reader N-1  |
    | (process)   |     | (process)   |         | (process)   |
    |             |     |             |         |             |
    | encode +    |     | encode +    |         | encode +    |
    | local dedup |     | local dedup |         | local dedup |
    +------+------+     +------+------+         +------+------+
           |                   |                       |
           v                   v                       v
       +--------------------------------------------------+
       |   multiprocessing.Queue   (ConceptDiff records)  |
       +--------------------------+-----------------------+
                                  |
                                  v
                        +--------------------+
                        |  writer thread     |
                        |  (main process)    |
                        |                    |
                        |  loop.cycle(...)   |
                        |  → find_or_match   |
                        |  → graph mutate    |
                        +--------------------+

Why this shape:
  - readers are the CPU bottleneck (encoding, GloVe lookup, multilevel
    preprocessing). Run them in parallel processes.
  - the graph (FAISS index, node table, edges) is single-writer by design;
    find_or_match is the dedup primitive and must be authoritative —
    so one writer thread holds the graph and applies every diff.
  - readers do best-effort local dedup against a sliding window so they
    don't flood the queue with identical concepts; the writer's
    find_or_match is the final authority.

Two queues:
  - _queue          : readers → writer; ConceptDiff records.
  - _snapshot_queue : writer → readers; size 1 GraphSnapshot, latest wins.
                      Readers can use the snapshot for prediction priors
                      (a future hook — Phase 7 keeps it simple and just
                      ships the latest FAISS index bytes).
"""
from __future__ import annotations

import multiprocessing as mp
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ConceptDiff:
    """A proposed graph mutation from a reader process.

    Vectors are passed as bytes (np.ndarray.tobytes) so the dataclass is
    pickle-friendly for multiprocessing.Queue without dragging numpy
    serialization overhead into the hot path.
    """
    representation: bytes           # float32[D_REP] serialized via tobytes()
    surprise: float
    affect_snapshot: bytes          # float32[N_AFF] serialized
    name_hint: str
    source_domain: str
    source_name: str
    level: str                      # 'word' | 'phrase' | 'sentence' | 'paragraph'
    surprise_multiplier: float
    edges_to_add: list = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


@dataclass
class GraphSnapshot:
    """Read-only snapshot of graph state for reader prediction.

    Lightweight — just the FAISS index (serialized) and the parallel
    id-map. Updated by the writer every PARALLEL_INGESTION_SNAPSHOT_INTERVAL
    writes; readers pull the latest from a size-1 queue (overwrites are
    fine; only the freshest matters).
    """
    faiss_index_bytes: bytes
    id_map: list
    node_count: int
    updated_at: float


class DiffQueue:
    """Cross-process queue between N readers and 1 writer.

    Uses multiprocessing.Queue (which pickles + ships across the fork
    boundary). Bounded so a runaway reader can't grow memory unbounded;
    when full, put_diff returns False and the reader retries.
    """

    # macOS BoundedSemaphore caps at SEM_VALUE_MAX = 32767, so the queue
    # size must stay below that. The queue is backpressure, not buffer —
    # the writer keeps up at ~10K items/s, so a 30K cap gives ~3 s of
    # headroom at peak parallel reader throughput.
    QUEUE_MAXSIZE = 30_000

    def __init__(self) -> None:
        self._queue: mp.Queue = mp.Queue(maxsize=self.QUEUE_MAXSIZE)
        self._snapshot_queue: mp.Queue = mp.Queue(maxsize=1)
        self._done_event = mp.Event()

    # ---- diff path (readers → writer) ----

    def put_diff(self, diff: ConceptDiff, timeout: float = 1.0) -> bool:
        """Push a diff. Returns True on success, False on queue-full /
        timeout. Readers must handle the False case — busy-wait or drop
        based on calling policy."""
        try:
            self._queue.put(diff, timeout=timeout)
            return True
        except Exception:
            return False

    def get_diff(self, timeout: float = 0.1) -> Optional[ConceptDiff]:
        """Pop a diff. Returns None on timeout (writer can poll done_event
        between empty pulls)."""
        try:
            return self._queue.get(timeout=timeout)
        except Exception:
            return None

    # ---- snapshot path (writer → readers) ----

    def push_snapshot(self, snapshot: GraphSnapshot) -> None:
        """Replace the snapshot in the size-1 queue. Drains any stale
        snapshot first so readers always see the freshest state."""
        try:
            self._snapshot_queue.get_nowait()
        except Exception:
            pass
        try:
            self._snapshot_queue.put_nowait(snapshot)
        except Exception:
            pass

    def get_snapshot(self) -> Optional[GraphSnapshot]:
        """Non-blocking read of the latest snapshot. Returns None if no
        snapshot has been pushed yet, or the queue was just drained."""
        try:
            return self._snapshot_queue.get_nowait()
        except Exception:
            return None

    # ---- coordination ----

    def signal_done(self) -> None:
        self._done_event.set()

    def is_done(self) -> bool:
        return self._done_event.is_set()

    def qsize(self) -> int:
        try:
            return self._queue.qsize()
        except (NotImplementedError, OSError):
            # qsize is best-effort on macOS — fall back to 0 if it raises.
            return 0

    def close(self) -> None:
        """Release the underlying mp.Queue resources. macOS caps total
        semaphores per session; without explicit close() back-to-back
        manager.run() calls can exhaust them. Idempotent.

        We cancel_join_thread on the snapshot queue because the writer
        process pushed snapshots to it but no reader currently consumes
        them — a feeder thread is sitting on a full pipe. join_thread()
        would deadlock waiting for an unblocked write. cancel detaches
        the feeder so the queue can release its semaphore.
        """
        for q in (self._queue, self._snapshot_queue):
            try:
                q.cancel_join_thread()
            except Exception:
                pass
            try:
                q.close()
            except Exception:
                pass
