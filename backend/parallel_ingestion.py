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
import json
import multiprocessing as mp
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np

from backend import config
from backend.diff_queue import ConceptDiff, DiffQueue, GraphSnapshot

if TYPE_CHECKING:
    from backend.main_loop import MainLoop


SNAPSHOT_UPDATE_INTERVAL = config.PARALLEL_INGESTION_SNAPSHOT_INTERVAL
SAVE_INTERVAL            = config.PARALLEL_INGESTION_SAVE_INTERVAL
SLEEP_CHECK_EVERY        = config.PARALLEL_INGESTION_SLEEP_CHECK_EVERY
SLEEP_DURATION           = config.PARALLEL_INGESTION_SLEEP_DURATION
WRITER_BATCH_SIZE        = config.WRITER_BATCH_SIZE


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
        try:
            _send_done(diff_queue, domain)
        finally:
            diff_queue.close()
        return

    # Lazy import the encoder (after env vars are set).
    from backend.input import encode_text

    local_seen: dict[bytes, float] = {}

    try:
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
    finally:
        # On *abnormal* shutdown (writer set done_event), the queue may
        # be full and the feeder thread blocked on the pipe — joining
        # it would hang the child indefinitely, which is the v0.6
        # "curriculum stop hung in mp.Queue cleanup" symptom. Detach
        # the feeder so the child can exit immediately. We accept that
        # any items still buffered get dropped — they're already lost
        # because the writer isn't consuming.
        #
        # On *normal* shutdown the writer is still draining; we must
        # NOT cancel the feeder here or the trailing items (including
        # the DONE sentinel) get dropped, which is what reduced the
        # parallel test's items_processed.
        if done_event.is_set():
            try:
                diff_queue.close()
            except Exception:
                pass


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

    def __init__(
        self,
        *,
        loop: "MainLoop",
        n_readers: int = 4,
        save_callback: Optional[Callable[[], None]] = None,
        save_interval: int = SAVE_INTERVAL,
        journal_path: Optional[str] = None,
        mind_name: Optional[str] = None,
    ) -> None:
        self.loop = loop
        self.n_readers = max(1, int(n_readers))
        self.diff_queue = DiffQueue()
        self._writes_since_snapshot = 0
        self._writes_since_save = 0
        self._writes_since_sleep_check = 0
        # save_callback is invoked from the writer thread (main process)
        # every save_interval items, on signals (SIGTERM/SIGINT), and
        # once more in the finally. None disables periodic save and
        # leaves the caller responsible for the final persist.save.
        self._save_callback = save_callback
        self._save_interval = max(1, int(save_interval))
        self._stop_requested = False
        self._n_saves = 0
        # Surprise-journal writes. The sequential `run_interleaved`
        # path appends to surprised_sentences.jsonl directly; the
        # parallel writer needs to do the same so future training has
        # an up-to-date corpus. Buffered so we don't fsync per item.
        self._journal_path: Optional[str] = journal_path
        self._journal_buffer: list[str] = []
        self._journal_flush_every = 100
        self._journal_records_written = 0

        # Native-head training trigger. Counts surprises since the last
        # train and the journal-line count at last train. Both guards
        # together prevent a wasted train on a mostly-static corpus.
        # Initialised at first save (lazy — corpus may not exist yet).
        self._surprises_at_last_train: Optional[int] = None
        self._corpus_lines_at_last_train: Optional[int] = None
        self._mind_name_for_train: Optional[str] = mind_name

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

        # Install SIGTERM (and re-install SIGINT) so that an external
        # `kill` of the curriculum runner sets done_event + flushes a
        # save before we exit. SIGINT is also caught here so we don't
        # rely on KeyboardInterrupt firing from inside the cycle —
        # MPS / numpy can sometimes swallow it. Restored in finally.
        prev_sigterm = signal.signal(signal.SIGTERM, self._on_signal)
        try:
            prev_sigint = signal.signal(signal.SIGINT, self._on_signal)
        except ValueError:
            # Not main thread — happens in tests; SIGINT handler is
            # already installed by the runtime.
            prev_sigint = None

        try:
            batch: list[ConceptDiff] = []
            batch_target = max(1, int(WRITER_BATCH_SIZE))
            while readers_done < total_readers:
                if self._stop_requested:
                    # Flush partial batch before bailing — these diffs
                    # were already pulled off the queue, so dropping
                    # them silently would lose work.
                    if batch:
                        items_processed = self._apply_batch_and_bookkeep(
                            batch, items_processed, t0,
                        )
                        batch = []
                    done_event.set()
                    break

                diff = self.diff_queue.get_diff(timeout=0.1)
                if diff is None:
                    # Empty pull. Flush partial batch first so we don't
                    # sit on already-dequeued diffs while the queue is
                    # quiet — pacing matters for the snapshot/save cadence.
                    if batch:
                        items_processed = self._apply_batch_and_bookkeep(
                            batch, items_processed, t0,
                        )
                        batch = []
                    # If every reader has exited *and* the queue is
                    # fully drained, we're done.
                    if all(not p.is_alive() for p in readers) and self.diff_queue.qsize() == 0:
                        break
                    continue

                if diff.name_hint == "__READER_DONE__":
                    # Flush partial batch before recording the DONE so
                    # bookkeeping (snapshot, save) fires for trailing
                    # items belonging to this reader.
                    if batch:
                        items_processed = self._apply_batch_and_bookkeep(
                            batch, items_processed, t0,
                        )
                        batch = []
                    readers_done += 1
                    continue
                if not diff.representation:
                    continue

                batch.append(diff)
                if len(batch) >= batch_target:
                    items_processed = self._apply_batch_and_bookkeep(
                        batch, items_processed, t0,
                    )
                    batch = []
        except KeyboardInterrupt:
            done_event.set()
            self._stop_requested = True
        finally:
            done_event.set()
            self.loop.predict_engine.set_ingestion_mode(False)
            # Restore signal handlers BEFORE we start the join loop —
            # otherwise a second Ctrl+C during cleanup re-enters our
            # handler instead of producing the usual KeyboardInterrupt,
            # which makes the runner unkillable.
            try:
                signal.signal(signal.SIGTERM, prev_sigterm)
                if prev_sigint is not None:
                    signal.signal(signal.SIGINT, prev_sigint)
            except (ValueError, TypeError):
                pass
            # Join readers first so their feeders flush, then close the
            # parent's queue handles. Order matters: closing the queue
            # before readers exit can drop in-flight items and block the
            # readers' put_diff calls. Each reader installs its own
            # cancel_join_thread in its finally so the join times out
            # quickly even if the feeder was stuck.
            for p in readers:
                p.join(timeout=5)
                if p.is_alive():
                    p.terminate()
                    p.join(timeout=2)
            self.diff_queue.close()
            # Flush any remaining surprise-journal records so a SIGTERM
            # mid-run doesn't drop the last <100 entries.
            if self._journal_buffer:
                self._flush_journal_buffer()
            # Final save after readers are gone so the on-disk state
            # matches the final in-memory graph. Idempotent if we just
            # saved a few items ago — but cheap insurance.
            if self._save_callback is not None:
                self._do_save(reason="final")

        return items_processed

    def _apply_batch_and_bookkeep(
        self,
        batch: list[ConceptDiff],
        items_processed: int,
        t0: float,
    ) -> int:
        """Apply a batch of diffs through the per-item cycle path, then
        run the same per-item bookkeeping (snapshot push, save, sleep,
        rate logging) the unbatched loop did. Returns the updated
        `items_processed` counter.

        The batch itself doesn't change the per-item correctness path
        (cycle still runs once per diff). What batching buys is one
        upstream MPS query — `predict_batch` — that future callers can
        consume to short-circuit cycle's per-item nearest() lookup.
        Today's writer just exercises predict_batch as a warm pass; the
        speedup is bounded by however much of the cycle's work is in
        that one matmul.
        """
        if not batch:
            return items_processed

        now = time.time()
        self._process_batch(batch, now)

        for diff in batch:
            items_processed += 1
            self._writes_since_snapshot += 1
            self._writes_since_save += 1
            self._writes_since_sleep_check += 1

            if self._writes_since_snapshot >= SNAPSHOT_UPDATE_INTERVAL:
                self._push_snapshot()
                self._writes_since_snapshot = 0

            if (self._save_callback is not None
                    and self._writes_since_save >= self._save_interval):
                self._do_save(reason="periodic")
                self._writes_since_save = 0
                # After each periodic save the mind on disk is
                # consistent with the live loop — safe moment to fork
                # a training subprocess that reads the disk state.
                self._maybe_train_native_head()

            if self._writes_since_sleep_check >= SLEEP_CHECK_EVERY:
                self._writes_since_sleep_check = 0
                self._maybe_sleep_for_ceiling()

            if items_processed % 10_000 == 0:
                rate = items_processed / max(time.perf_counter() - t0, 1e-6)
                print(
                    f"[parallel] {items_processed:,} items  "
                    f"nodes={self.loop.graph.node_count}  "
                    f"queue={self.diff_queue.qsize()}  "
                    f"saves={self._n_saves}  "
                    f"({rate:,.0f}/s)",
                    flush=True,
                )
        return items_processed

    def _process_batch(self, batch: list[ConceptDiff], now: float) -> None:
        """Apply a batch of diffs to the loop. Per-item state mutation
        goes through loop.cycle (find_or_match dedup, auto-link,
        F.attend, B.observe).

        Note on predict_batch: the batched MPS query was tried as a
        single up-front matmul plumbed through ingest_text(prediction=),
        but it produced a *correctness* divergence: within-batch writes
        invalidate the predictions other items in the same batch were
        scored against. A 10K-item bench showed batch=64 yielding 4×
        fewer nodes (46 vs 199) than batch=1 on identical input,
        because surprise scoring diverged when later items in a batch
        couldn't see earlier items' writes. The right place for
        predict_batch is read-only paths (active inference, replay
        scoring) — not the write path.

        We keep the batch *collection* boundary (drains the queue in
        chunks of WRITER_BATCH_SIZE) for cleaner snapshot/save
        cadence, but each item still runs its own predict() inside
        cycle. The actual hot-path speedup needs a different
        architecture (e.g. a write-aware batching that re-queries
        after each new write, or persistent index residency on MPS).
        """
        if not batch:
            return

        # Per-item: cycle runs the authoritative observation path. We
        # advance `now` by a microsecond per item so each cycle's
        # last_observation_t / tick lands distinct (matches the
        # unbatched loop's now+1e-3 stamping).
        for i, diff in enumerate(batch):
            row = np.frombuffer(diff.representation, dtype=np.float32).copy()
            t_item = now + i * 1e-6

            ingest = self.loop.input_pipeline.ingest_text(
                diff.name_hint,
                now=t_item,
                representation=row,
            )
            self.loop.predict_engine.set_surprise_multiplier(
                diff.surprise_multiplier
            )
            try:
                self.loop.cycle(
                    ingest,
                    now=t_item + 1e-3,
                    force_respond=False,
                    skip_simulation=True,
                )
            finally:
                self.loop.predict_engine.clear_surprise_multiplier()

            # Surprise-journal write. Same gating as the unbatched path:
            # journal records track 1:1 with new graph structure.
            gap = ingest.gap
            if (
                self._journal_path is not None
                and gap.is_surprise
                and gap.was_new_write
            ):
                self._journal_buffer.append(
                    json.dumps({
                        "sentence":           diff.name_hint,
                        "affect":             self.loop.affect.composite(t_item).tolist(),
                        "concept_embeddings": [],
                        "surprise_score":     float(gap.surprise_score),
                        "concept_id":         int(gap.concept_id) if gap.concept_id is not None else None,
                        "level":              diff.level,
                        "t":                  float(t_item),
                    }, ensure_ascii=False)
                )
                if len(self._journal_buffer) >= self._journal_flush_every:
                    self._flush_journal_buffer()

    def _flush_journal_buffer(self) -> None:
        """Append the buffered surprise-journal lines to disk and clear
        the buffer. Best-effort — IO failure here only loses log
        records, not graph state."""
        if not self._journal_buffer or self._journal_path is None:
            return
        try:
            os.makedirs(
                os.path.dirname(self._journal_path) or ".", exist_ok=True
            )
            with open(self._journal_path, "a", encoding="utf-8") as f:
                f.write("\n".join(self._journal_buffer))
                f.write("\n")
            self._journal_records_written += len(self._journal_buffer)
        except OSError as exc:
            print(
                f"[parallel] journal flush failed: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
        finally:
            self._journal_buffer.clear()

    # ---- internal: signal + save ----

    def _on_signal(self, signum, _frame) -> None:
        """Signal handler for SIGTERM / SIGINT during run(). Sets the
        stop flag so the run loop exits cleanly at the next iteration;
        the finally block then runs the final save and joins readers.

        Kept extremely small — signal context limits what we can do
        safely. The actual save happens inside the run loop, not here.
        """
        # Stamp the reason so the next save log line says why we're
        # stopping. print() is technically not signal-safe but we're
        # already in a controlled shutdown; the line lands or it
        # doesn't.
        self._stop_requested = True
        try:
            print(
                f"\n[parallel] received signal {signum} — stopping after current item",
                flush=True,
            )
        except Exception:
            pass

    def _maybe_sleep_for_ceiling(self) -> None:
        """Fire a short sleep when node_count has crossed the concept
        ceiling. Sleep does (in this order) replay → abstraction
        formation → ceiling-prune; the prune step is the only thing
        that ever bounds node_count during a parallel run, since the
        parallel path has no curriculum step boundaries to hang sleep
        off of. Cheap when graph is below ceiling — just one int compare.
        """
        if self.loop.graph.node_count <= config.CONCEPT_CEILING:
            return
        pre_nodes = self.loop.graph.node_count
        pre_edges = self.loop.graph.edge_count
        # Pause ingestion mode for the duration of sleep so B's surprise
        # threshold is the standard 1.5σ during replays — same gating
        # the sequential interleaved path uses around its sleep().
        self.loop.predict_engine.set_ingestion_mode(False)
        try:
            t0 = time.perf_counter()
            slept = self.loop.sleep(
                now=time.time(), duration_seconds=SLEEP_DURATION,
            )
            dur = time.perf_counter() - t0
            print(
                f"[parallel] sleep @ ceiling: nodes "
                f"{pre_nodes:,}->{self.loop.graph.node_count:,}  "
                f"edges {pre_edges:,}->{self.loop.graph.edge_count:,}  "
                f"replays={slept.replays_fired} "
                f"abstractions={slept.abstractions_formed} "
                f"({dur:.1f}s)",
                flush=True,
            )
            # If a save_callback is wired, persist immediately after a
            # ceiling sleep — abstractions and prune both happen here
            # and we want them on disk before resuming ingestion.
            if self._save_callback is not None:
                self._do_save(reason="post-sleep")
                self._writes_since_save = 0
        except Exception as exc:
            print(
                f"[parallel] sleep FAILED: {type(exc).__name__}: {exc}",
                flush=True,
            )
        finally:
            self.loop.predict_engine.set_ingestion_mode(True)

    def _maybe_train_native_head(self) -> None:
        """Fire native_head_v2 training when surprise count + corpus
        growth + memory all clear. Cheap when not due — three int
        compares. The interleaved path has the same logic in
        run_curriculum.run_interleaved; this is the parallel-mode
        port so an unattended parallel run also benefits from
        periodic LM updates.

        Trigger: surprises_since_last_train >= TRAIN_LM_EVERY_N_SURPRISES.
        Guards: corpus floor (LM_TRAIN_MIN_CORPUS), corpus growth
        (MIN_CORPUS_GROWTH_TO_RETRAIN). Memory check is best-effort.
        """
        if self._mind_name_for_train is None:
            return  # caller didn't wire mind_name → cannot subprocess
        if self._journal_path is None:
            return  # no corpus to train on

        surprise_count = self.loop.predict_engine.surprise_count
        if self._surprises_at_last_train is None:
            # First call after startup — initialise high-water marks
            # so the first train requires fresh growth, not historical.
            self._surprises_at_last_train = surprise_count
            self._corpus_lines_at_last_train = self._count_corpus_lines()
            return
        if (surprise_count - self._surprises_at_last_train
                < config.TRAIN_LM_EVERY_N_SURPRISES):
            return

        # Trigger threshold crossed — apply guards.
        corpus_lines = self._count_corpus_lines()
        if corpus_lines < config.LM_TRAIN_MIN_CORPUS:
            print(
                f"[parallel] train_lm SKIPPED — corpus has only "
                f"{corpus_lines} lines (< {config.LM_TRAIN_MIN_CORPUS} "
                f"floor); marker reset",
                flush=True,
            )
            self._surprises_at_last_train = surprise_count
            return

        prev_lines = self._corpus_lines_at_last_train or 0
        growth = corpus_lines - prev_lines
        if growth < config.MIN_CORPUS_GROWTH_TO_RETRAIN:
            print(
                f"[parallel] train_lm SKIPPED — corpus grew only "
                f"{growth} lines since last train "
                f"(< {config.MIN_CORPUS_GROWTH_TO_RETRAIN} floor); "
                f"marker reset, will retry next window",
                flush=True,
            )
            self._surprises_at_last_train = surprise_count
            return

        print(
            f"[parallel] train_lm @ surprises={surprise_count:,}  "
            f"corpus_lines={corpus_lines:,}  growth={growth:,}",
            flush=True,
        )

        # Pause ingestion mode for the subprocess so any concurrent
        # save sees a consistent state. The subprocess loads from
        # disk so the running loop is undisturbed; we just need a
        # fresh checkpoint first.
        self.loop.predict_engine.set_ingestion_mode(False)
        try:
            if self._save_callback is not None:
                self._do_save(reason="pre-train")
            t0 = time.perf_counter()
            cmd = [
                sys.executable,
                os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "scripts", "train_native_head.py",
                ),
                "--mind", self._mind_name_for_train,
                "--epochs", "10",
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            dur = time.perf_counter() - t0
            if proc.returncode != 0:
                print(
                    f"[parallel] train_lm FAILED in {dur:.1f}s",
                    flush=True,
                )
                # Don't reset markers on failure — try again next
                # window (memory might have freed up, etc.)
                return
            # Print final loss line if present.
            for line in proc.stdout.strip().splitlines()[-3:]:
                print(f"  [train_lm] {line}", flush=True)
            print(
                f"[parallel] train_lm OK in {dur:.1f}s",
                flush=True,
            )
            # Reload the live expression head so subsequent generation
            # in this process uses the freshly-trained weights.
            try:
                self.loop.expression.reload_language_head()
            except Exception as exc:
                print(
                    f"[parallel] reload_language_head failed: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
            self._surprises_at_last_train = surprise_count
            self._corpus_lines_at_last_train = corpus_lines
        finally:
            self.loop.predict_engine.set_ingestion_mode(True)

    def _count_corpus_lines(self) -> int:
        if self._journal_path is None or not os.path.exists(self._journal_path):
            return 0
        n = 0
        with open(self._journal_path, "r", encoding="utf-8") as f:
            for _ in f:
                n += 1
        return n

    def _do_save(self, *, reason: str) -> None:
        if self._save_callback is None:
            return
        try:
            t0 = time.perf_counter()
            self._save_callback()
            self._n_saves += 1
            dur = time.perf_counter() - t0
            print(
                f"[parallel] {reason} save complete in {dur*1000:.0f}ms "
                f"(saves={self._n_saves}, nodes={self.loop.graph.node_count})",
                flush=True,
            )
        except Exception as exc:
            print(
                f"[parallel] {reason} save FAILED: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )

    # ---- internal: snapshot publishing ----

    def _push_snapshot(self) -> None:
        """Serialize the MPSConceptIndex matrix + id-map into a
        GraphSnapshot and push to the readers. Best-effort — failures
        here only mean readers get a stale snapshot, which is fine
        because writer-side find_or_match remains authoritative."""
        try:
            mat = self.loop.graph._index.to_numpy_matrix()
            if mat is None:
                return
            buf = io.BytesIO()
            np.save(buf, mat, allow_pickle=False)
            snapshot = GraphSnapshot(
                faiss_index_bytes=buf.getvalue(),  # field name kept for compat
                id_map=list(self.loop.graph._index._id_map),
                node_count=self.loop.graph.node_count,
                updated_at=time.time(),
            )
            self.diff_queue.push_snapshot(snapshot)
        except Exception as exc:
            print(
                f"[parallel] snapshot push failed: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
