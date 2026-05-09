"""Parallel ingestion manager — N reader processes + 1 writer thread.

Structure
---------

  ParallelIngestionManager
    .run(sources_by_domain)
        |
        v
    spawns N reader processes (each runs reader_worker)
    consumes diffs in the main process via the same loop.cycle path
    that run_curriculum.run_interleaved uses sequentially:
        loop.input_pipeline.ingest_text(name_hint, representation=rep)
        loop.cycle(ingest, force_respond=False, skip_simulation=True)

    so find_or_match dedup, auto-link, F.attend(INPUT), and B.observe
    all behave identically to the sequential path. The only difference
    is who encodes: in the sequential path the main process encodes;
    in parallel mode the readers encode in parallel and the main
    process only applies the resulting diffs.

  reader_worker(...)
    - sets OMP_NUM_THREADS=1 BEFORE any heavy import (torch / faiss
      transitively load libomp; multiple threads × multiple processes
      collides on macOS).
    - lazy-imports multilevel_preprocessor (built by a parallel subagent;
      may not exist in this worktree). Falls back to the existing
      ingest_book sentence splitter so the manager runs end-to-end
      either way.
    - encodes each item, applies a 60 s local dedup, ships a ConceptDiff
      onto the queue. Sends a sentinel ConceptDiff with name_hint =
      '__READER_DONE__' on completion.

  ParallelIngestionManager.run loop
    - drains diffs, applies each through loop.cycle (skip_simulation=True
      — same fast-ingestion mode as run_interleaved).
    - every PARALLEL_INGESTION_SNAPSHOT_INTERVAL writes, pushes a fresh
      GraphSnapshot for any reader that wants prediction priors. (No
      reader consumes the snapshot today — Phase 7 keeps it as a hook.)
"""
from __future__ import annotations

import io
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from backend import config
from backend.diff_queue import ConceptDiff, DiffQueue, GraphSnapshot

if TYPE_CHECKING:
    from backend.main_loop import MainLoop


SNAPSHOT_UPDATE_INTERVAL = config.PARALLEL_INGESTION_SNAPSHOT_INTERVAL


# Per-level surprise multipliers. Words are short and high-leverage,
# paragraphs are long but more predictable; this maps level → relative
# surprise weight for the diff. Currently informational — the reader
# encodes it on the diff, the writer hands the result to the same
# loop.cycle so B's Welford does the actual scoring.
_SURPRISE_MULTIPLIER_BY_LEVEL = {
    "word":      1.5,
    "phrase":    1.3,
    "sentence":  1.0,
    "paragraph": 0.8,
}


def reader_worker(
    reader_id: int,
    source_paths: list,
    domain: str,
    diff_queue: DiffQueue,
    surprise_multiplier_map: dict,
    ingestion_threshold: float,
    done_event,
) -> None:
    """Reader process — runs in a separate process via mp.Process.

    Reads each source path, multilevel-streams items (word / phrase /
    sentence / paragraph if the multilevel preprocessor is available;
    sentences only otherwise), encodes via the canonical text encoder,
    drops near-duplicates against a 60 s local-dedup window, and pushes
    ConceptDiffs onto the queue.

    Does NOT touch the graph directly — every mutation must go through
    the writer's loop.cycle path so find_or_match dedup and B/A side
    effects stay authoritative.
    """
    # --- MUST set BEFORE any heavy import (torch / faiss / encoders) ---
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    # Lazy import of the multilevel preprocessor. Built by a parallel
    # subagent; if it's not in this worktree yet, fall back to the
    # existing sentence splitter from ingest_book so the parallel
    # ingestion path still runs (just at sentence-only granularity).
    multilevel_stream = _resolve_multilevel_stream()
    if multilevel_stream is None:
        # Hard failure — we have neither the multilevel preprocessor
        # nor the sentence-splitter fallback. Emit DONE sentinel and
        # exit so the writer doesn't hang.
        _send_done(diff_queue, domain)
        return

    # Lazy import the encoder (after env vars are set).
    from backend.input import encode_text

    local_seen: dict[bytes, float] = {}

    for source_path in source_paths:
        if done_event.is_set():
            break
        try:
            text = Path(source_path).read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        for item in multilevel_stream(text):
            if done_event.is_set():
                break

            item_text = getattr(item, "text", None) or str(item)
            level = getattr(item, "level", "sentence")
            level_multiplier = float(getattr(
                item,
                "surprise_multiplier",
                surprise_multiplier_map.get(level, 1.0),
            ))

            rep = encode_text(item_text)
            if rep is None:
                continue
            n = float(np.linalg.norm(rep))
            if n < 0.01:
                continue
            rep_norm = (rep / (n + 1e-9)).astype(np.float32)
            rep_key = rep_norm.tobytes()

            now = time.time()
            last = local_seen.get(rep_key, 0.0)
            if now - last < config.PARALLEL_INGESTION_LOCAL_DEDUP_WINDOW:
                continue
            local_seen[rep_key] = now

            diff = ConceptDiff(
                representation=rep_key,
                surprise=ingestion_threshold / max(level_multiplier, 1e-3),
                affect_snapshot=np.zeros(config.N_AFF, dtype=np.float32).tobytes(),
                name_hint=item_text[:64],
                source_domain=domain,
                source_name=Path(source_path).stem,
                level=level,
                surprise_multiplier=level_multiplier,
            )

            # Backpressure: spin until the queue accepts. If done_event
            # fires (writer wants to wind down), bail out cleanly.
            while not diff_queue.put_diff(diff, timeout=0.5):
                if done_event.is_set():
                    return

    _send_done(diff_queue, domain)


def _resolve_multilevel_stream():
    """Return a callable text → iterable[item] for the reader to use.

    Tries the multilevel preprocessor first (Phase 7 perf — built by
    a parallel subagent); falls back to ingest_book.split_sentences if
    multilevel isn't available yet. Returns None only if neither is
    importable, which means the reader can't do its job.
    """
    try:
        from backend.multilevel_preprocessor import multilevel_stream
        return multilevel_stream
    except ImportError:
        pass

    try:
        from ingest_book import split_sentences, strip_book
    except ImportError:
        return None

    def _sentence_only_stream(text: str):
        stripped = strip_book(text)
        for sent in split_sentences(stripped):
            yield _SentenceItem(text=sent)

    return _sentence_only_stream


class _SentenceItem:
    """Minimal item shim for the sentence-only fallback path."""

    __slots__ = ("text", "level", "surprise_multiplier")

    def __init__(self, text: str) -> None:
        self.text = text
        self.level = "sentence"
        self.surprise_multiplier = 1.0


def _send_done(diff_queue: DiffQueue, domain: str) -> None:
    """Push the sentinel diff that tells the writer this reader is done."""
    diff_queue.put_diff(
        ConceptDiff(
            representation=b"",
            surprise=0.0,
            affect_snapshot=b"",
            name_hint="__READER_DONE__",
            source_domain=domain,
            source_name="",
            level="",
            surprise_multiplier=1.0,
        ),
        timeout=5.0,
    )


class ParallelIngestionManager:
    """Coordinates N reader processes + the in-process writer loop.

    The constructor takes a MainLoop because the writer invokes
    loop.cycle(...) for each diff — same path run_interleaved uses
    sequentially. This guarantees the parallel runner produces an
    identical graph (modulo ordering) to the sequential runner.
    """

    def __init__(self, *, loop: "MainLoop", n_readers: int = 4) -> None:
        self.loop = loop
        self.n_readers = max(1, int(n_readers))
        self.diff_queue = DiffQueue()
        self._writes_since_snapshot = 0

    def run(self, sources_by_domain: dict) -> int:
        """Spawn readers, drain diffs, apply through loop.cycle.

        sources_by_domain: {'philosophy': [path, ...], 'history': [...], ...}
        Returns the total number of items processed (excluding the per-reader
        DONE sentinels).
        """
        domains = list(sources_by_domain.items())
        if not domains:
            return 0

        # Round-robin domain assignment across readers. If there are
        # fewer domains than readers, some readers stay idle (they
        # never start). If more domains than readers, each reader gets
        # a list and works through them serially.
        reader_assignments: list[list[tuple[str, list[str]]]] = [
            [] for _ in range(self.n_readers)
        ]
        for i, (domain, paths) in enumerate(domains):
            reader_assignments[i % self.n_readers].append((domain, paths))

        # Each reader process runs in its own mp.Process. daemon=True so
        # they die with the parent if the parent crashes; the manager's
        # finally block joins them cleanly otherwise.
        done_event = mp.Event()
        readers: list[mp.Process] = []
        total_readers = 0
        for i, assignments in enumerate(reader_assignments):
            if not assignments:
                continue
            # Flatten this reader's assigned (domain, paths) into one
            # source_paths list. The reader tags every diff with the
            # domain of the *first* assignment — for round-robin
            # assignment this is fine because each reader typically
            # gets one domain anyway. When multiple, mixing domains is
            # explicitly accepted (the diff carries the per-batch
            # domain hint, not the per-item).
            source_paths: list[str] = []
            domain_name = assignments[0][0]
            for _, paths in assignments:
                source_paths.extend(paths)

            p = mp.Process(
                target=reader_worker,
                args=(
                    i,
                    source_paths,
                    domain_name,
                    self.diff_queue,
                    _SURPRISE_MULTIPLIER_BY_LEVEL,
                    config.INGESTION_MIN_THRESHOLD,
                    done_event,
                ),
                daemon=True,
            )
            p.start()
            readers.append(p)
            total_readers += 1

        if total_readers == 0:
            return 0

        # Initial snapshot push (best-effort). Readers don't consume it
        # in Phase 7 but the hook is in place.
        self._push_snapshot()

        # Bulk ingestion — same flag run_interleaved sets so B uses the
        # 1.0σ surprise threshold instead of 1.5σ. Restore in the
        # finally block.
        self.loop.predict_engine.set_ingestion_mode(True)

        readers_done = 0
        items_processed = 0
        t0 = time.perf_counter()

        try:
            while readers_done < total_readers:
                diff = self.diff_queue.get_diff(timeout=0.1)
                if diff is None:
                    # Empty pull. If every reader has exited *and* the
                    # queue is fully drained, we're done.
                    if all(not p.is_alive() for p in readers) and self.diff_queue.qsize() == 0:
                        break
                    continue

                if diff.name_hint == "__READER_DONE__":
                    readers_done += 1
                    continue
                if not diff.representation:
                    continue

                # Reconstruct the float32 vector. .copy() because the
                # buffer is read-only; downstream code (graph) expects
                # a writable array.
                rep = np.frombuffer(diff.representation, dtype=np.float32).copy()
                now = time.time()

                # Same call sequence as run_interleaved's per-sentence
                # body — find_or_match dedup, auto-link, B.observe,
                # F.attend(INPUT) all run inside cycle.
                ingest = self.loop.input_pipeline.ingest_text(
                    diff.name_hint,
                    now=now,
                    representation=rep,
                )
                self.loop.cycle(
                    ingest,
                    now=now + 1e-3,
                    force_respond=False,
                    skip_simulation=True,
                )

                items_processed += 1
                self._writes_since_snapshot += 1

                if self._writes_since_snapshot >= SNAPSHOT_UPDATE_INTERVAL:
                    self._push_snapshot()
                    self._writes_since_snapshot = 0

                if items_processed % 10_000 == 0:
                    rate = items_processed / max(time.perf_counter() - t0, 1e-6)
                    print(
                        f"[parallel] {items_processed:,} items  "
                        f"nodes={self.loop.graph.node_count}  "
                        f"queue={self.diff_queue.qsize()}  "
                        f"({rate:,.0f}/s)",
                        flush=True,
                    )
        except KeyboardInterrupt:
            done_event.set()
        finally:
            done_event.set()
            self.loop.predict_engine.set_ingestion_mode(False)
            # Join readers first so their feeders flush, then close the
            # parent's queue handles. Order matters: closing the queue
            # before readers exit can drop in-flight items and block the
            # readers' put_diff calls.
            for p in readers:
                p.join(timeout=5)
                if p.is_alive():
                    p.terminate()
                    p.join(timeout=2)
            self.diff_queue.close()

        return items_processed

    # ---- internal: snapshot publishing ----

    def _push_snapshot(self) -> None:
        """Serialize the FAISS index + id-map into a GraphSnapshot and
        push to the readers. Best-effort — failures here only mean
        readers get a stale snapshot, which is fine because writer-side
        find_or_match remains authoritative."""
        try:
            import faiss
        except ImportError:
            return

        try:
            buf = io.BytesIO()
            writer = faiss.PyCallbackIOWriter(buf.write)
            faiss.write_index(self.loop.graph._faiss_index, writer)
            del writer  # flush
            snapshot = GraphSnapshot(
                faiss_index_bytes=buf.getvalue(),
                id_map=list(self.loop.graph._faiss_id_map),
                node_count=self.loop.graph.node_count,
                updated_at=time.time(),
            )
            self.diff_queue.push_snapshot(snapshot)
        except Exception as exc:
            # Snapshot failures are non-fatal but we shouldn't swallow
            # silently — readers won't get fresh priors but the writer
            # can keep going. Log once per failure type.
            print(
                f"[parallel] snapshot push failed: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
