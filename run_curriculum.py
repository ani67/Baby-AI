"""Run a curriculum end-to-end against one mind.

Usage
-----
    python3 run_curriculum.py --mind first --curriculum curriculum.json
    python3 run_curriculum.py --mind first --until alice_in_wonderland
    python3 run_curriculum.py --mind first --reset

Step types
----------
    book        Ingest a text source. If data/encoded_corpus.db has the
                source pre-encoded, stream representations from there
                (skip encoding entirely; ~5K sentences/sec). Otherwise
                fall back to live encoding.

    sleep       MainLoop.sleep(now, duration_seconds=step.duration_seconds).

    train_lm    Subprocess: python3 scripts/train_language_head.py
                --mind {paths.mind_name} --epochs {step.epochs}.
                Runs in a fresh process so the GloVe + torch model
                loads in isolation; weights land in paths.language_head.

    dialogue    Each prompt run through MainLoop.cycle with
                force_respond=True. All outputs logged to paths.dialogue_log
                (JSONL).

Resumable
---------
    paths.curriculum_progress holds {completed: [step_name, ...]}.
    Steps already in `completed` are skipped on rerun.
"""
from __future__ import annotations

# OMP threading conflict fix — same rationale as backend/graph.py. Set
# AT THE TOP before any heavy import (especially anything that pulls
# torch transitively, which loads libomp). The trainer subprocess we
# spawn inherits these env vars, so its torch import is also safe.
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import subprocess
import sys
import time
import traceback
from collections import Counter, deque

try:
    import psutil
    _PSUTIL_OK = True
except ImportError:
    _PSUTIL_OK = False


# Phase 7 stability fix: skip LM training if free RAM falls below this
# floor. Loading GPT-2 (~500 MB) + the trainer's optimizer state on a
# memory-pressured laptop has crashed before. Skipping a single training
# round and letting the system recover is cheaper than losing a long
# ingestion run.
LM_TRAIN_MIN_FREE_GB = 2.0


def _have_enough_memory_for_training() -> tuple[bool, float]:
    """Returns (ok, available_gb). When psutil isn't installed we
    assume ok and report 0.0 — caller logs and proceeds."""
    if not _PSUTIL_OK:
        return True, 0.0
    avail = psutil.virtual_memory().available / (1024 ** 3)
    return avail >= LM_TRAIN_MIN_FREE_GB, avail

# Allow running from repo root.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.affect import AffectStack                  # noqa: E402
from backend.attention import Attention                  # noqa: E402
from backend.expression import Expression                # noqa: E402
from backend.graph import ConceptGraph                   # noqa: E402
from backend.identity import Identity                    # noqa: E402
from backend.input import InputPipeline                  # noqa: E402
from backend.main_loop import MainLoop                   # noqa: E402
from backend.mind_paths import MindPaths                 # noqa: E402
from backend.persistence import MindPersistence          # noqa: E402
from backend.predict import PredictionEngine             # noqa: E402
from backend.multilevel_preprocessor import (             # noqa: E402
    MULTILEVEL_DB_PATH,
    fetch_encoded_multilevel,
    is_source_encoded_multilevel,
    multilevel_stream,
)
from backend.preencoder import (                         # noqa: E402
    DB_PATH as ENCODED_DB_PATH,
    fetch_encoded,
    is_source_encoded,
)
from backend.interleaved_reader import (                  # noqa: E402
    InterleavedReader,
    SourceSpec,
)
from backend.simulation import SimulationReplay          # noqa: E402


# ============================================================
# Mind construction (mirrors ingest_book and api)
# ============================================================

def construct_mind(birth_seed: int = 42, paths: MindPaths | None = None) -> MainLoop:
    now = time.time()
    a = AffectStack(birth_seed=birth_seed, t_birth=now)
    g = ConceptGraph()
    p = PredictionEngine(affect=a, graph=g)
    sim = SimulationReplay(affect=a, graph=g, predict_engine=p)
    ident = Identity(
        affect=a, graph=g, predict_engine=p, simulation=sim,
        birth_seed=birth_seed, birth_time=now,
    )
    h = InputPipeline(affect=a, graph=g, predict_engine=p, identity=ident)
    f = Attention(affect=a, graph=g)
    gx = Expression(
        affect=a, graph=g, predict_engine=p, identity=ident, input_pipeline=h,
        lm_weights_path=paths.language_head if paths else None,
        lm_vocab_path=paths.vocab if paths else None,
    )
    return MainLoop(
        affect=a, graph=g, predict_engine=p, simulation=sim,
        identity=ident, attention=f, expression=gx, input_pipeline=h,
    )


def _seed_for_mind_name(mind_name: str) -> int:
    """Deterministic per-name birth seed so two minds with different names
    are born different by default. `--birth-seed` overrides this."""
    import hashlib
    h = hashlib.sha1(mind_name.encode("utf-8")).digest()
    return int.from_bytes(h[:4], "little")


def load_or_construct(paths: MindPaths, birth_seed: int) -> MainLoop:
    if os.path.exists(paths.db):
        loop = MindPersistence.load(paths.db)
        # Rewire LM paths in case the loaded Expression was constructed with
        # default global paths (older saves predate the lm_weights_path arg).
        loop.expression._lm_weights_path = paths.language_head
        loop.expression._lm_vocab_path = paths.vocab
        # Phase 7 fix: persistence.MindPersistence.load builds Expression
        # without lm_weights_path, so _cd_root stays None and the GPT-2
        # conditioned-decoder loader silently skips. Re-point it here so
        # reload_language_head can find both v2 and v1 deltas. Without
        # this, in-curriculum probes fall back to the LSTM head even
        # when a fresh GPT-2 v2 is on disk.
        loop.expression._cd_root = (
            os.path.dirname(paths.language_head) or "."
        )
        from backend.language_head import CONDITIONED_DECODER_FILENAME
        loop.expression._cd_weights_path = os.path.join(
            loop.expression._cd_root, CONDITIONED_DECODER_FILENAME,
        )
        loop.expression._language_head = None
        loop.expression._conditioned_decoder = None
        loop.expression._lm_vocab = None
        loop.expression._lm_load_attempted = False
        loop.expression._cd_load_attempted = False
        # Phase 7: persistence no longer auto-builds the FAISS index.
        # Curriculum needs it for the interleaved write+nearest hot
        # loop (33× speedup over brute-force-with-rebuild). Build now.
        loop.graph._rebuild_faiss_index()
        return loop
    fresh = construct_mind(birth_seed=birth_seed, paths=paths)
    fresh.graph._rebuild_faiss_index()
    return fresh


# ============================================================
# Curriculum progress
# ============================================================

def load_curriculum_progress(paths: MindPaths) -> dict:
    if not os.path.exists(paths.curriculum_progress):
        return {"completed": []}
    try:
        with open(paths.curriculum_progress, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"completed": []}


def save_curriculum_progress(paths: MindPaths, progress: dict) -> None:
    tmp = paths.curriculum_progress + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2)
    os.replace(tmp, paths.curriculum_progress)


# ============================================================
# Step handlers
# ============================================================

def run_book_step(loop: MainLoop, paths: MindPaths, step: dict) -> dict:
    """Ingest a book source into the mind. Uses pre-encoded representations
    from data/encoded_corpus.db when available; falls back to on-the-fly
    encoding via ingest_book's preprocessing path otherwise.
    """
    source = step["source"]
    domain = step.get("domain", "unknown")
    name = step["name"]

    if not os.path.exists(source):
        return {"skipped": True, "reason": f"source missing: {source}"}

    h = loop.input_pipeline
    g = loop.graph
    persist = MindPersistence(paths.db)

    # Register the source as an agent for refers_to anchoring.
    handle = name[:32] if name else os.path.basename(source)[:32]
    agent_id = h.register_agent(handle, now=time.time())

    # Surprised-sentence training log (LM corpus).
    os.makedirs(os.path.dirname(paths.surprised_log) or ".", exist_ok=True)
    train_log = open(paths.surprised_log, "a", encoding="utf-8")

    nodes_before    = g.node_count
    surprises_before = loop.predict_engine.surprise_count

    pre_encoded = is_source_encoded(source, ENCODED_DB_PATH)
    print(f"  [book] {name} [{domain}]  pre-encoded={pre_encoded}  ingestion_mode=ON")

    # Phase 7: bulk book ingestion uses the relaxed surprise threshold
    # (1.0σ instead of 1.5σ) so the language-head training corpus grows
    # fast enough to override GPT-2's web-text priors. Restored to False
    # in the finally block so subsequent live conversation uses the
    # calibrated default.
    loop.predict_engine.set_ingestion_mode(True)

    # Capture the kept-sentence list to paths.book_text_log so train_lm
    # has the full corpus for vocab building.
    sentences_for_vocab: list[str] = []

    n_done = 0
    t0 = time.perf_counter()

    try:
        if pre_encoded:
            # Pre-encoded path — skip encode_text per sentence.
            for sentence, rep, _position in fetch_encoded(source, ENCODED_DB_PATH):
                sentences_for_vocab.append(sentence)
                now = time.time()
                ingest = h.ingest_text(
                    sentence, now=now, agent_id=agent_id,
                    representation=rep,
                )
                loop.cycle(ingest, now=now + 1e-3, force_respond=False, skip_simulation=True)

                if ingest.gap.is_surprise:
                    composite = loop.affect.composite(now)
                    train_log.write(json.dumps({
                        "sentence":       sentence,
                        "affect":         [float(x) for x in composite],
                        "surprise_score": float(ingest.gap.surprise_score),
                        "concept_id":     ingest.gap.concept_id,
                        "t":              float(now),
                    }) + "\n")

                n_done += 1
                if n_done % 500 == 0:
                    train_log.flush()
                    arousal = loop.affect.current_arousal(time.time())
                    rate = n_done / max(time.perf_counter() - t0, 1e-6)
                    print(f"      [{n_done:>6,d}]  nodes={g.node_count:>5d}  "
                          f"edges={g.edge_count:>5d}  arousal={arousal:.3f}  "
                          f"({rate:.0f}/s)")
                if n_done % 2000 == 0:
                    persist.save(loop, now=time.time())
        else:
            # On-the-fly encoding path — same logic as ingest_book.py.
            from ingest_book import load_sentences
            sentences = load_sentences(source)
            sentences_for_vocab = sentences
            for sentence in sentences:
                now = time.time()
                ingest = h.ingest_text(sentence, now=now, agent_id=agent_id)
                loop.cycle(ingest, now=now + 1e-3, force_respond=False, skip_simulation=True)
                if ingest.gap.is_surprise:
                    composite = loop.affect.composite(now)
                    train_log.write(json.dumps({
                        "sentence":       sentence,
                        "affect":         [float(x) for x in composite],
                        "surprise_score": float(ingest.gap.surprise_score),
                        "concept_id":     ingest.gap.concept_id,
                        "t":              float(now),
                    }) + "\n")
                n_done += 1
                if n_done % 500 == 0:
                    train_log.flush()
                    arousal = loop.affect.current_arousal(time.time())
                    rate = n_done / max(time.perf_counter() - t0, 1e-6)
                    print(f"      [{n_done:>6,d}]  nodes={g.node_count:>5d}  "
                          f"edges={g.edge_count:>5d}  arousal={arousal:.3f}  "
                          f"({rate:.0f}/s)")
                if n_done % 2000 == 0:
                    persist.save(loop, now=time.time())
    finally:
        train_log.close()
        loop.predict_engine.set_ingestion_mode(False)

    # Append to the book-text-log (used by train_lm vocab build); keep
    # accumulating across book steps so the vocab covers the full curriculum.
    if sentences_for_vocab:
        os.makedirs(os.path.dirname(paths.book_text_log) or ".", exist_ok=True)
        with open(paths.book_text_log, "a", encoding="utf-8") as f:
            for s in sentences_for_vocab:
                f.write(s + "\n")

    duration = time.perf_counter() - t0
    persist.save(loop, now=time.time())

    return {
        "name":            name,
        "source":          source,
        "domain":          domain,
        "sentences":       n_done,
        "nodes_added":     g.node_count - nodes_before,
        "surprises_added": loop.predict_engine.surprise_count - surprises_before,
        "throughput":      n_done / max(duration, 1e-6),
        "duration_s":      duration,
    }


def run_sleep_step(loop: MainLoop, paths: MindPaths, step: dict) -> dict:
    duration = float(step.get("duration_seconds", 60.0))
    print(f"  [sleep] {step['name']}  duration_budget={duration}s")
    result = loop.sleep(now=time.time(), duration_seconds=duration)
    print(f"      replays={result.replays_fired}  "
          f"abstractions={result.abstractions_formed}  "
          f"actual={result.duration_actual:.2f}s")
    MindPersistence(paths.db).save(loop, now=time.time())
    return {
        "name":               step["name"],
        "replays_fired":      result.replays_fired,
        "abstractions_formed": result.abstractions_formed,
        "duration_actual":    result.duration_actual,
    }


def run_train_lm_step(paths: MindPaths, step: dict) -> dict:
    epochs = int(step.get("epochs", 50))
    print(f"  [train_lm] {step['name']}  epochs={epochs}  (subprocess)")
    cmd = [
        sys.executable, "scripts/train_native_head.py",
        "--mind", paths.mind_name,
        "--epochs", str(epochs),
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    duration = time.perf_counter() - t0
    if proc.returncode != 0:
        print("    [train_lm] FAILED:")
        print(proc.stdout[-2000:])
        print(proc.stderr[-2000:])
        return {"name": step["name"], "ok": False, "duration_s": duration}
    # Print the last few lines of training output for visibility.
    tail = proc.stdout.strip().splitlines()[-6:]
    for line in tail:
        print(f"    {line}")
    return {"name": step["name"], "ok": True, "epochs": epochs, "duration_s": duration}


def _top_k_concepts_by_activation(loop: MainLoop, k: int = 5) -> list[tuple[int, str, int]]:
    """Sorted (cid, name, activation_count) for the top-k most activated
    concepts. Names are cleaned of trailing newlines and truncated to 60ch."""
    nodes = sorted(
        loop.graph.nodes.values(),
        key=lambda n: -int(getattr(n, "activation_count", 0)),
    )[:k]
    out = []
    for n in nodes:
        name = (getattr(n, "name", "") or "").replace("\n", " ").strip()
        out.append((int(n.concept_id), name[:60], int(n.activation_count)))
    return out


def _milestone_report(
    loop: MainLoop,
    n_pulled: int,
    domain_window: deque,
    surprises_at_milestone: int,
    t0: float,
) -> None:
    """Print the every-10K-sentence milestone snapshot."""
    now = time.time()
    arousal  = float(loop.affect.current_arousal(now))
    nodes    = loop.graph.node_count
    edges    = loop.graph.edge_count
    surprise = loop.predict_engine.surprise_count
    rate     = n_pulled / max(time.perf_counter() - t0, 1e-6)

    print()
    print(f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"   [milestone @ {n_pulled:,}]  rate={rate:,.0f}/s")
    print(f"     nodes={nodes:,}  edges={edges:,}  surprises={surprise:,}  "
          f"arousal={arousal:.3f}")

    # Domain distribution of last 1000 sentences (whatever is in the deque).
    if domain_window:
        c = Counter(domain_window)
        n = len(domain_window)
        dist_strs = [f"{d}:{100*v/n:.0f}%" for d, v in c.most_common()]
        print(f"     last_{n}_domains: {' '.join(dist_strs)}")

    # Top-5 most activated concepts.
    top5 = _top_k_concepts_by_activation(loop, k=5)
    if top5:
        print(f"     top_5_activated:")
        for cid, name, count in top5:
            label = name if name else f"<unnamed:{cid}>"
            print(f"       cid={cid:<6d} act={count:>4d}  {label!r}")
    print(f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print()


def _train_lm_with_curve(paths: MindPaths, epochs: int) -> tuple[bool, list[float]]:
    """Train the native head as a subprocess; parse the per-epoch loss line.
    Returns (ok, [losses]).

    Phase 8+: the active mouth is native_head_v2 (commit 6b6ed38),
    not the GPT-2 LM head. Train script is scripts/train_native_head.py
    which writes data/{mind}/native_head_v2.pt; expression.py prefers
    that file when present. Trainer defaults to the live unfiltered
    surprise journal — no --corpus override here.
    """
    import re
    cmd = [
        sys.executable, "scripts/train_native_head.py",
        "--mind", paths.mind_name,
        "--epochs", str(epochs),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    losses: list[float] = []
    for line in proc.stdout.splitlines():
        m = re.search(r"epoch \d+/\d+\s+loss=([\d.]+)", line)
        if m:
            losses.append(float(m.group(1)))
    if proc.returncode != 0:
        print("    [train_lm] FAILED:")
        print(proc.stdout[-1500:])
        print(proc.stderr[-1500:])
        return False, losses
    # Print a compact tail of the trainer's own output so we still see what
    # it reported.
    for line in proc.stdout.strip().splitlines()[-4:]:
        print(f"      {line}")
    return True, losses


def _sample_response(loop: MainLoop, prompt: str = "what is beautiful") -> str:
    """Run a one-shot force_respond cycle and return the surface text.
    Used by the post-training milestone report — non-destructive enough at
    one cycle (the cycle does fire B.observe + write_on_surprise if applicable).
    """
    h = loop.input_pipeline
    examiner = h.register_agent("post_train_probe", now=time.time())
    now = time.time()
    ing = h.ingest_text(prompt, agent_id=examiner, now=now)
    cyc = loop.cycle(ing, now=now + 1e-3, force_respond=True)
    return cyc.emitted_surface or "(no surface)"


# ============================================================
# Multi-resolution interleaved reader (used when --multilevel)
# ============================================================

class _MultilevelInterleavedReader:
    """Mirror of InterleavedReader that yields multilevel items.

    Each yielded record carries (text, representation, level, multiplier,
    domain, source_name, position). Sources draw probability proportional
    to weight × size_factor where size_factor = min(1.0, 5000 / N_items).

    Pre-encoded items must be in `encoded_multilevel` (run encode_corpus.py
    --multilevel first). Sources missing from that table are skipped with
    a warning.
    """

    def __init__(
        self,
        sources: list[SourceSpec],
        db_path: str = MULTILEVEL_DB_PATH,
        seed: int | None = None,
    ) -> None:
        import random as _random
        self._rng = _random.Random(seed)
        self._sources: list[dict] = []
        for spec in sources:
            if not is_source_encoded_multilevel(spec.source_file, db_path):
                print(f"[multilevel] [skip] not pre-encoded: {spec.source_file}")
                continue
            rows = list(fetch_encoded_multilevel(spec.source_file, db_path))
            if not rows:
                print(f"[multilevel] [skip] empty source: {spec.source_file}")
                continue
            size_factor = min(1.0, 5000.0 / len(rows))
            self._sources.append({
                "spec": spec,
                "rows": rows,
                "cursor": 0,
                "size_factor": size_factor,
                "active": True,
                "n_pulled": 0,
            })

    @property
    def total_items(self) -> int:
        return sum(len(s["rows"]) for s in self._sources)

    @property
    def active_sources(self) -> int:
        return sum(1 for s in self._sources if s["active"])

    def per_source_pulls(self) -> dict[str, int]:
        return {s["spec"].name: s["n_pulled"] for s in self._sources}

    def __iter__(self):
        return self

    def __next__(self):
        active = [s for s in self._sources if s["active"]]
        if not active:
            raise StopIteration
        weights = [s["spec"].weight * s["size_factor"] for s in active]
        chosen = self._rng.choices(active, weights=weights, k=1)[0]
        idx = chosen["cursor"]
        row = chosen["rows"][idx]
        chosen["cursor"] = idx + 1
        chosen["n_pulled"] += 1
        if chosen["cursor"] >= len(chosen["rows"]):
            chosen["active"] = False
        sentence, rep, position, level, multiplier = row
        return {
            "text":           sentence,
            "representation": rep,
            "level":          level,
            "multiplier":     multiplier,
            "domain":         chosen["spec"].domain,
            "source_name":    chosen["spec"].name,
            "position":       position,
        }


def run_interleaved(
    paths: MindPaths,
    curriculum_path: str,
    max_sentences: int | None = None,
    birth_seed: int | None = None,
    multilevel: bool = False,
) -> dict:
    """Phase 7 — interleaved curriculum mode.

    Reads `sources` from curriculum JSON, builds an InterleavedReader,
    pulls sentences in weighted random rotation, fires loop.cycle with
    skip_simulation=True for each. Sleep / idle / train_lm are triggered
    by total-volume counters, not per-source completion.

    On completion: runs the self_examination dialogue with the LM that
    was trained during the run (if any).
    """
    with open(curriculum_path, "r", encoding="utf-8") as f:
        curr = json.load(f)

    from backend.config import (                                    # noqa: E402
        LM_TRAIN_MIN_CORPUS,
        MIN_CORPUS_GROWTH_TO_RETRAIN,
        TRAIN_LM_EVERY_N_SURPRISES,
    )
    sleep_every  = int(curr.get("sleep_every_n_sentences", 8000))
    idle_every   = int(curr.get("idle_every_n_sentences",  2000))
    # Phase 7 perf fix: default cadence comes from config (2000); a
    # curriculum file can override only by *raising* the threshold —
    # we cap at the config minimum so a stale JSON doesn't pin us back
    # to the v0.4 frequent-train regime.
    train_after  = max(
        int(curr.get("train_lm_every_n_surprises", TRAIN_LM_EVERY_N_SURPRISES)),
        TRAIN_LM_EVERY_N_SURPRISES,
    )
    print(f"[interleaved] train_lm_every_n_surprises={train_after} "
          f"(min corpus floor: {LM_TRAIN_MIN_CORPUS} surprised sentences)")

    sources = [
        SourceSpec(
            name=s["name"], source_file=s["source"],
            domain=s.get("domain", "unknown"),
            weight=float(s.get("weight", 1.0)),
        )
        for s in curr["sources"]
    ]
    print(f"[interleaved] {len(sources)} source(s) declared in {curriculum_path}")

    if multilevel:
        ml_reader = _MultilevelInterleavedReader(sources)
        print(f"[interleaved] [multilevel] {ml_reader.active_sources} pre-encoded sources, "
              f"{ml_reader.total_items:,} multilevel items available")
        if ml_reader.active_sources == 0:
            raise RuntimeError(
                "no multilevel-encoded sources — run "
                "`python3 encode_corpus.py --multilevel --curriculum <path>` first"
            )
        reader = ml_reader  # type: ignore[assignment]
    else:
        reader = InterleavedReader(sources)
        print(f"[interleaved] {reader.active_sources} encoded sources, "
              f"{reader.total_sentences:,} total sentences available")
        if reader.active_sources == 0:
            raise RuntimeError("no encoded sources — run encode_corpus.py first")

    seed = birth_seed if birth_seed is not None else _seed_for_mind_name(paths.mind_name)
    loop = load_or_construct(paths, birth_seed=seed)
    print(f"[interleaved] mind: {paths.mind_name} (birth_seed={seed})")
    print(f"[interleaved]   loaded: {loop.graph.node_count} nodes,"
          f" {loop.graph.edge_count} edges, cycle_count={loop.cycle_count}")

    h = loop.input_pipeline
    persist = MindPersistence(paths.db)

    # Surprised-sentence training corpus — same JSONL the LSTM/GPT2 trainers
    # consume. Append; this run extends whatever was logged before.
    os.makedirs(os.path.dirname(paths.surprised_log) or ".", exist_ok=True)
    train_log = open(paths.surprised_log, "a", encoding="utf-8")

    # Source-name agents so refers_to chains anchor cleanly. One per source.
    name_to_agent_id: dict[str, int] = {}
    for spec in sources:
        name_to_agent_id[spec.name] = h.register_agent(spec.name, now=time.time())

    nodes_before     = loop.graph.node_count
    edges_before     = loop.graph.edge_count
    surprises_before = loop.predict_engine.surprise_count

    surprises_since_train = 0
    n_pulled = 0
    n_done = 0
    t0 = time.perf_counter()
    pending_train = False

    # Corpus-growth gate: skip training if the surprise journal hasn't
    # grown by at least MIN_CORPUS_GROWTH_TO_RETRAIN since the last
    # training run. Initialized to current corpus size so the first
    # train fires only after MIN_CORPUS_GROWTH_TO_RETRAIN new lines.
    def _count_corpus_lines() -> int:
        # Counter for the corpus-growth guard. Phase 8+ trains on the
        # unfiltered live journal (paths.surprised_log) — same source
        # train_native_head.py reads by default.
        if not os.path.exists(paths.surprised_log):
            return 0
        n = 0
        with open(paths.surprised_log, "r", encoding="utf-8") as f:
            for _ in f:
                n += 1
        return n
    corpus_size_at_last_train = _count_corpus_lines()

    domain_distribution: dict[str, int] = {}
    domain_window: deque = deque(maxlen=1000)
    milestone_every = 10_000
    next_milestone = milestone_every

    # Stream the kept-sentence list to paths.book_text_log for vocab-builds.
    bt_log = open(paths.book_text_log, "a", encoding="utf-8")

    # Phase 7 ingestion mode (1.0σ threshold) for the entire interleaved run.
    # The post-run dialogue is invoked by a separate process, so this flag
    # only affects bulk book ingestion.
    loop.predict_engine.set_ingestion_mode(True)
    print("[interleaved] ingestion_mode=ON (1.0σ threshold)")
    if multilevel:
        print("[interleaved] multilevel=ON — per-item surprise multipliers active")

    # Per-level pull counters (multilevel only) for the post-run summary.
    level_pulls: dict[str, int] = {}

    try:
        for sent in reader:
            if max_sentences is not None and n_pulled >= max_sentences:
                break

            # Normalize: legacy InterleavedSentence vs multilevel dict.
            if multilevel:
                text         = sent["text"]
                rep          = sent["representation"]
                domain       = sent["domain"]
                source_name  = sent["source_name"]
                level        = sent["level"]
                multiplier   = sent["multiplier"]
            else:
                text         = sent.sentence
                rep          = sent.representation
                domain       = sent.domain
                source_name  = sent.source_name
                level        = "sentence"
                multiplier   = 1.0

            now = time.time()
            agent_id = name_to_agent_id[source_name]

            # Apply the level-specific surprise multiplier just for this
            # item; the finally-clear keeps it from leaking into anything
            # downstream (e.g. the post-run dialogue).
            loop.predict_engine.set_surprise_multiplier(multiplier)
            try:
                ingest = h.ingest_text(
                    text, now=now, agent_id=agent_id,
                    representation=rep,
                )
                loop.cycle(ingest, now=now + 1e-3, force_respond=False, skip_simulation=True)
            finally:
                loop.predict_engine.clear_surprise_multiplier()

            domain_distribution[domain] = domain_distribution.get(domain, 0) + 1
            domain_window.append(domain)
            bt_log.write(text + "\n")
            if multilevel:
                level_pulls[level] = level_pulls.get(level, 0) + 1

            if ingest.gap.is_surprise:
                composite = loop.affect.composite(now)
                train_log.write(json.dumps({
                    "sentence":       text,
                    "affect":         [float(x) for x in composite],
                    "surprise_score": float(ingest.gap.surprise_score),
                    "concept_id":     ingest.gap.concept_id,
                    "t":              float(now),
                    "domain":         domain,
                    "source":         source_name,
                    "level":          level,
                }) + "\n")
                surprises_since_train += 1
                if surprises_since_train >= train_after:
                    pending_train = True

            n_pulled += 1
            n_done += 1

            if n_done % 500 == 0:
                train_log.flush(); bt_log.flush()
                rate = n_pulled / max(time.perf_counter() - t0, 1e-6)
                arousal = loop.affect.current_arousal(time.time())
                print(f"  [{n_pulled:>6,d}]  nodes={loop.graph.node_count:>5d}  "
                      f"edges={loop.graph.edge_count:>5d}  arousal={arousal:.3f}  "
                      f"({rate:,.0f}/s)", flush=True)

            # Every 10K sentences: rich milestone report (nodes, edges,
            # surprises, arousal, last-1000 domain distribution, top-5
            # most activated concepts).
            if n_pulled >= next_milestone:
                _milestone_report(
                    loop=loop,
                    n_pulled=n_pulled,
                    domain_window=domain_window,
                    surprises_at_milestone=loop.predict_engine.surprise_count,
                    t0=t0,
                )
                next_milestone += milestone_every

            if n_done % idle_every == 0:
                replayed = loop.idle(now=time.time(), max_replays=3)
                print(f"    [idle @ {n_pulled:,}]  replayed={replayed.replayed_count}",
                      flush=True)

            if n_done % sleep_every == 0 or pending_train:
                # Pre-sleep snapshot for the rich after-sleep report.
                nodes_before  = loop.graph.node_count
                edges_before  = loop.graph.edge_count
                slept = loop.sleep(now=time.time(), duration_seconds=20.0)
                nodes_after  = loop.graph.node_count
                edges_after  = loop.graph.edge_count
                print(f"    [sleep @ {n_pulled:,}]  replays={slept.replays_fired}  "
                      f"abstractions={slept.abstractions_formed}  "
                      f"nodes:{nodes_before:,}->{nodes_after:,}  "
                      f"edges:{edges_before:,}->{edges_after:,}  "
                      f"actual={slept.duration_actual:.2f}s",
                      flush=True)
                # Save before triggering the LM training subprocess —
                # train_language_head loads the mind from disk.
                train_log.close(); train_log = None
                bt_log.close(); bt_log = None
                persist.save(loop, now=time.time())
                if pending_train:
                    # Min-corpus guard — don't train on a sliver. The
                    # surprised_sentences.jsonl is the trainer's input; if
                    # it has < LM_TRAIN_MIN_CORPUS lines, the resulting
                    # model is just noise. Counter reset so the next
                    # window has a clean shot.
                    surprise_lines = 0
                    if os.path.exists(paths.surprised_log):
                        with open(paths.surprised_log, "r", encoding="utf-8") as f:
                            for _ in f:
                                surprise_lines += 1
                    if surprise_lines < LM_TRAIN_MIN_CORPUS:
                        print(f"    [train_lm @ {n_pulled:,}]  SKIPPED — "
                              f"corpus has only {surprise_lines} surprised "
                              f"sentences (< {LM_TRAIN_MIN_CORPUS} floor); "
                              f"counter reset", flush=True)
                        surprises_since_train = 0
                        pending_train = False
                        train_log = open(paths.surprised_log, "a", encoding="utf-8")
                        bt_log = open(paths.book_text_log, "a", encoding="utf-8")
                        continue
                    # Corpus-growth guard — if the journal hasn't grown
                    # by MIN_CORPUS_GROWTH_TO_RETRAIN since the last
                    # training run, skip. The native head can't move
                    # meaningfully on <2K new examples; better to wait
                    # and let surprise accumulate than burn a 10-min
                    # subprocess on noise-level delta.
                    current_corpus_size = _count_corpus_lines()
                    growth = current_corpus_size - corpus_size_at_last_train
                    if growth < MIN_CORPUS_GROWTH_TO_RETRAIN:
                        print(f"    [train_lm @ {n_pulled:,}]  SKIPPED — "
                              f"corpus grew only {growth} lines since last "
                              f"train (< {MIN_CORPUS_GROWTH_TO_RETRAIN} floor); "
                              f"counter reset, will retry next window",
                              flush=True)
                        surprises_since_train = 0
                        pending_train = False
                        train_log = open(paths.surprised_log, "a", encoding="utf-8")
                        bt_log = open(paths.book_text_log, "a", encoding="utf-8")
                        continue
                    # Memory guard — don't fire the trainer when the
                    # machine is already pressured. Loading GPT-2 (~500 MB)
                    # + optimizer state on top of an already-tight system
                    # is the documented crash mode on 16 GB M1 / Phase 7.
                    have_mem, avail_gb = _have_enough_memory_for_training()
                    if not have_mem:
                        print(f"    [train_lm @ {n_pulled:,}]  SKIPPED — "
                              f"only {avail_gb:.1f} GB free "
                              f"(< {LM_TRAIN_MIN_FREE_GB:.1f} GB floor); "
                              f"counter reset, will retry next window",
                              flush=True)
                        surprises_since_train = 0
                        pending_train = False
                        # Re-open the logs that were closed for the
                        # subprocess hand-off, then continue.
                        train_log = open(paths.surprised_log, "a", encoding="utf-8")
                        bt_log = open(paths.book_text_log, "a", encoding="utf-8")
                        continue
                    print(f"    [train_lm @ {n_pulled:,}]  "
                          f"surprises_since_last={surprises_since_train}  "
                          f"free_ram={avail_gb:.1f}GB",
                          flush=True)
                    ok, losses = _train_lm_with_curve(paths, epochs=10)
                    if losses:
                        # Compact loss-curve line + final loss.
                        curve = " ".join(f"{x:.3f}" for x in losses)
                        print(f"      loss_curve: {curve}")
                        print(f"      final_loss: {losses[-1]:.4f}", flush=True)
                    if ok:
                        loop.expression.reload_language_head()
                        # Sample response after the training stage —
                        # uses the freshly-reloaded LM.
                        try:
                            resp = _sample_response(loop, "what is beautiful")
                            print(f"      probe 'what is beautiful' -> {resp!r}",
                                  flush=True)
                        except Exception as exc:
                            print(f"      [probe failed: {exc}]", flush=True)
                    surprises_since_train = 0
                    pending_train = False
                    corpus_size_at_last_train = current_corpus_size
                # Re-open the logs after the subprocess completes.
                train_log = open(paths.surprised_log, "a", encoding="utf-8")
                bt_log = open(paths.book_text_log, "a", encoding="utf-8")
    except KeyboardInterrupt:
        print("\n[interleaved] interrupted; saving …")
    except Exception as exc:
        # Phase 7 crash recovery — checkpoint the mind state before the
        # process unwinds so a long ingestion run survives an unexpected
        # crash (MPS OOM, torch internals, anything that bubbles up
        # through cycle()). After saving we re-raise so the caller still
        # sees the original traceback / non-zero exit.
        print(f"\n[interleaved] CRASH at sentence {n_pulled:,}: "
              f"{type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        try:
            if train_log is not None:
                train_log.close()
            if bt_log is not None:
                bt_log.close()
            loop.predict_engine.set_ingestion_mode(False)
            persist.save(loop, now=time.time())
            print(f"[interleaved] crash-checkpoint saved to {paths.db}", flush=True)
        except Exception as save_exc:
            print(f"[interleaved] crash-checkpoint ALSO failed: {save_exc}",
                  flush=True)
        raise

    if train_log is not None:
        train_log.close()
    if bt_log is not None:
        bt_log.close()

    loop.predict_engine.set_ingestion_mode(False)
    persist.save(loop, now=time.time())
    duration = time.perf_counter() - t0

    summary = {
        "n_pulled":            n_pulled,
        "duration_s":          duration,
        "throughput":          n_pulled / max(duration, 1e-6),
        "domain_distribution": domain_distribution,
        "per_source_pulls":    reader.per_source_pulls(),
        "nodes_added":         loop.graph.node_count - nodes_before,
        "edges_added":         loop.graph.edge_count - edges_before,
        "surprises_added":     loop.predict_engine.surprise_count - surprises_before,
        "final_node_count":    loop.graph.node_count,
        "final_edge_count":    loop.graph.edge_count,
    }
    if multilevel:
        summary["level_pulls"] = level_pulls
    print()
    unit = "items" if multilevel else "sentences"
    print(f"[interleaved] pulled {n_pulled:,} {unit} in {duration:.1f}s "
          f"({summary['throughput']:,.0f}/s)")
    print(f"[interleaved] graph: +{summary['nodes_added']} nodes,  "
          f"+{summary['edges_added']} edges,  "
          f"+{summary['surprises_added']} surprises")
    print(f"[interleaved] domain distribution:")
    for d, n in sorted(domain_distribution.items(), key=lambda kv: -kv[1]):
        print(f"    {d:14s}  {n:>5,d}  ({100*n/max(n_pulled,1):5.1f}%)")
    if multilevel and level_pulls:
        print(f"[interleaved] level distribution:")
        for lvl, n in sorted(level_pulls.items(), key=lambda kv: -kv[1]):
            print(f"    {lvl:14s}  {n:>5,d}  ({100*n/max(n_pulled,1):5.1f}%)")

    return summary


def _train_lm_subprocess(paths: MindPaths, epochs: int) -> dict:
    """Helper: invoke native-head training in a subprocess so the
    torch/GloVe binaries can load fresh and don't leak memory across
    many train cycles inside the long-running interleaved runner.
    Phase 8+: native_head_v2 is the active mouth, not GPT-2.
    Trainer defaults to the unfiltered surprise journal — no
    --corpus override needed."""
    cmd = [
        sys.executable, "scripts/train_native_head.py",
        "--mind", paths.mind_name,
        "--epochs", str(epochs),
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    duration = time.perf_counter() - t0
    if proc.returncode != 0:
        print("    [train_lm] FAILED:")
        print(proc.stdout[-1500:])
        print(proc.stderr[-1500:])
        return {"ok": False, "duration_s": duration}
    for line in proc.stdout.strip().splitlines()[-4:]:
        print(f"    {line}")
    return {"ok": True, "duration_s": duration}


def run_dialogue_step(loop: MainLoop, paths: MindPaths, step: dict) -> dict:
    prompts = step.get("prompts", [])
    print(f"  [dialogue] {step['name']}  prompts={len(prompts)}")
    h = loop.input_pipeline
    persist = MindPersistence(paths.db)
    agent_id = h.register_agent("examiner", now=time.time())

    log_path = paths.dialogue_log
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    answers = []
    with open(log_path, "a", encoding="utf-8") as logf:
        for prompt in prompts:
            now = time.time()
            res = h.ingest_text(prompt, agent_id=agent_id, now=now)
            cyc = loop.cycle(res, now=now + 1e-3, force_respond=True)
            answer = cyc.emitted_surface
            decision = type(cyc.expression_decision).__name__ if cyc.expression_decision else None
            entry = {
                "step": step["name"],
                "t":    now,
                "prompt": prompt,
                "decision": decision,
                "answer": answer,
            }
            answers.append(entry)
            logf.write(json.dumps(entry) + "\n")
            print(f"    Q: {prompt!r}")
            print(f"       → ({decision}) {answer!r}")

    persist.save(loop, now=time.time())
    return {"name": step["name"], "answers": answers}


# ============================================================
# Parallel ingestion (Phase 7 — N readers + 1 writer)
# ============================================================

def run_parallel_ingestion(
    paths: MindPaths,
    curriculum_path: str,
    n_readers: int,
    birth_seed: int,
) -> dict:
    """Phase 7 — parallel ingestion mode.

    Reads the curriculum's `sources` list, groups source paths by domain,
    spawns n_readers reader processes that encode in parallel, and the
    main process drives loop.cycle for each diff via ParallelIngestionManager.

    No max_sentences cap — readers stream through the assigned sources end
    to end. Use --interleave (without --parallel) for capped runs.
    """
    from backend.parallel_ingestion import ParallelIngestionManager  # noqa: E402

    with open(curriculum_path, "r", encoding="utf-8") as f:
        curr = json.load(f)

    sources_by_domain: dict[str, list[str]] = {}
    for s in curr.get("sources", []):
        path = s["source"]
        if not os.path.exists(path):
            print(f"[parallel] skipping missing source: {path}")
            continue
        domain = s.get("domain", "unknown")
        sources_by_domain.setdefault(domain, []).append(path)

    if not sources_by_domain:
        raise RuntimeError(
            f"no usable sources in {curriculum_path} "
            f"(check `sources[].source` paths)"
        )

    seed = birth_seed
    loop = load_or_construct(paths, birth_seed=seed)
    print(f"[parallel] mind: {paths.mind_name} (birth_seed={seed})")
    print(f"[parallel]   loaded: {loop.graph.node_count} nodes, "
          f"{loop.graph.edge_count} edges, cycle_count={loop.cycle_count}")
    print(f"[parallel] domains: "
          f"{', '.join(f'{d}={len(p)}' for d, p in sources_by_domain.items())}")
    print(f"[parallel] n_readers={n_readers}")

    persist = MindPersistence(paths.db)

    nodes_before     = loop.graph.node_count
    edges_before     = loop.graph.edge_count
    surprises_before = loop.predict_engine.surprise_count

    # Periodic save while the parallel run is in flight — protects
    # against an unkillable mp.Queue cleanup chain dropping a multi-hour
    # run on the floor (the v0.6 "curriculum stop hung" symptom).
    def _save_now() -> None:
        persist.save(loop, now=time.time())

    t0 = time.perf_counter()
    mgr = ParallelIngestionManager(
        loop=loop,
        n_readers=n_readers,
        save_callback=_save_now,
        journal_path=paths.surprised_log,
    )
    items_processed = mgr.run(sources_by_domain)
    duration = time.perf_counter() - t0

    # Manager already issues a final save in its finally block when a
    # save_callback is provided, so we don't need a redundant one here.

    summary = {
        "items_processed":   items_processed,
        "duration_s":        duration,
        "throughput":        items_processed / max(duration, 1e-6),
        "nodes_added":       loop.graph.node_count - nodes_before,
        "edges_added":       loop.graph.edge_count - edges_before,
        "surprises_added":   loop.predict_engine.surprise_count - surprises_before,
        "final_node_count":  loop.graph.node_count,
        "final_edge_count":  loop.graph.edge_count,
        "n_readers":         n_readers,
    }
    print()
    print(f"[parallel] processed {items_processed:,} items in {duration:.1f}s "
          f"({summary['throughput']:,.0f}/s)")
    print(f"[parallel] graph: +{summary['nodes_added']} nodes,  "
          f"+{summary['edges_added']} edges,  "
          f"+{summary['surprises_added']} surprises")
    return summary


# ============================================================
# Driver
# ============================================================

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mind", required=True, help="mind name (paths under data/{mind}/)")
    ap.add_argument("--curriculum", default="curriculum.json")
    ap.add_argument("--reset", action="store_true",
                    help="discard data/{mind}/ before starting (start from scratch)")
    ap.add_argument("--until", default=None,
                    help="run up to and including the named step, then stop")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="run at most N steps from where we are")
    ap.add_argument("--birth-seed", type=int, default=None,
                    help="explicit birth seed; default = sha1(mind_name)[:4] so "
                         "different mind_names produce structurally different minds")
    ap.add_argument("--interleave", action="store_true",
                    help="Phase 7 interleaved mode — read curriculum's `sources` array "
                         "and pull sentences in weighted random rotation across all "
                         "sources simultaneously, instead of exhausting one before the next.")
    ap.add_argument("--max-sentences", type=int, default=None,
                    help="cap total sentences pulled (interleave mode only)")
    ap.add_argument("--multilevel", action="store_true",
                    help="(interleave mode) pull from the multilevel encoded "
                         "table — words / phrases / sentences / paragraphs — "
                         "with per-level surprise multipliers. Requires running "
                         "encode_corpus.py --multilevel first.")
    ap.add_argument("--parallel", action="store_true",
                    help="Phase 7 perf — N reader processes + 1 writer thread. "
                         "Requires --interleave (uses the curriculum's `sources` "
                         "list grouped by domain).")
    ap.add_argument("--n-readers", type=int, default=4,
                    help="number of parallel reader processes (default: 4); "
                         "only used when --parallel is set")
    args = ap.parse_args()

    birth_seed = args.birth_seed if args.birth_seed is not None else _seed_for_mind_name(args.mind)

    paths = MindPaths(args.mind)
    paths.ensure_dirs()

    if args.reset:
        import shutil
        if os.path.exists(paths.root):
            shutil.rmtree(paths.root)
            print(f"removed {paths.root}")
        paths.ensure_dirs()

    # ---- Phase 7: interleaved mode short-circuits the sequential path ----
    if args.interleave:
        if args.parallel:
            run_parallel_ingestion(
                paths=paths,
                curriculum_path=args.curriculum,
                n_readers=args.n_readers,
                birth_seed=birth_seed,
            )
            return 0
        run_interleaved(
            paths=paths,
            curriculum_path=args.curriculum,
            max_sentences=args.max_sentences,
            birth_seed=birth_seed,
            multilevel=args.multilevel,
        )
        return 0
    if args.multilevel:
        raise SystemExit("--multilevel currently only supported with --interleave")

    with open(args.curriculum, "r", encoding="utf-8") as f:
        curriculum = json.load(f)

    progress = load_curriculum_progress(paths)
    completed: set[str] = set(progress.get("completed", []))

    print(f"\nmind: {args.mind}")
    print(f"curriculum: {args.curriculum}  ({len(curriculum['sequence'])} steps total)")
    print(f"already completed: {len(completed)}\n")

    loop = load_or_construct(paths, birth_seed=birth_seed)
    print(f"loaded mind: {loop.graph.node_count} nodes, {loop.graph.edge_count} edges,"
          f" cycle_count={loop.cycle_count}, birth_seed={birth_seed}\n")

    n_run = 0
    try:
        for step in curriculum["sequence"]:
            name = step.get("name", "<unnamed>")
            if name in completed:
                print(f"[skip already done] {name}")
                continue

            stype = step.get("type")
            print(f"\n=== STEP: {name} (type={stype}) ===")

            if stype == "book":
                summary = run_book_step(loop, paths, step)
            elif stype == "sleep":
                summary = run_sleep_step(loop, paths, step)
            elif stype == "train_lm":
                summary = run_train_lm_step(paths, step)
                # After training, refresh in-process LM in case there's
                # a dialogue step coming up.
                loop.expression.reload_language_head()
            elif stype == "dialogue":
                summary = run_dialogue_step(loop, paths, step)
            else:
                print(f"   [unknown type] {stype}; skipping")
                summary = {"unknown": stype}

            completed.add(name)
            progress["completed"] = sorted(completed)
            save_curriculum_progress(paths, progress)
            n_run += 1

            if args.until and name == args.until:
                print(f"\nreached --until {args.until}; stopping")
                break
            if args.max_steps is not None and n_run >= args.max_steps:
                print(f"\nreached --max-steps {args.max_steps}; stopping")
                break
    except KeyboardInterrupt:
        print("\ninterrupted; saving …")

    MindPersistence(paths.db).save(loop, now=time.time())
    save_curriculum_progress(paths, progress)

    print()
    print(f"final state: {loop.graph.node_count} nodes,"
          f" {loop.graph.edge_count} edges,"
          f" {loop.graph.pin_count} pins,"
          f" cycle_count={loop.cycle_count}")
    print(f"completed steps: {len(completed)}/{len(curriculum['sequence'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
