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

import argparse
import json
import os
import subprocess
import sys
import time

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
from backend.preencoder import (                         # noqa: E402
    DB_PATH as ENCODED_DB_PATH,
    fetch_encoded,
    is_source_encoded,
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
        loop.expression._language_head = None
        loop.expression._lm_vocab = None
        loop.expression._lm_load_attempted = False
        return loop
    return construct_mind(birth_seed=birth_seed, paths=paths)


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
    print(f"  [book] {name} [{domain}]  pre-encoded={pre_encoded}")

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
        sys.executable, "scripts/train_language_head.py",
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
