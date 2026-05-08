"""Three parallel minds, each developing under different domain weighting.

Launches three subprocesses, one per mind, each running the interleaved
curriculum runner with a curriculum file biased toward a different
domain (philosophy / history / art_culture). The parent monitors all
three via read-only SQLite snapshots and prints a unified status line
every STATUS_INTERVAL_S seconds.

After all three subprocesses complete, the parent loads each mind in
turn, runs the 14-prompt self-examination, embeds the answers via the
GloVe text encoder, and computes pairwise cosine divergence per prompt.
The result — `data/comparison_{timestamp}.json` — is the experiment:
how much does the developmental path shape what the mind becomes?
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


BASE_INTERLEAVED   = "curriculum_interleaved.json"
DEFAULT_PROMPTS = [
    "what do you know",
    "what surprises you",
    "what do you want",
    "what is a story",
    "what is beautiful",
    "what is a mind",
    "what is the difference between knowing and feeling",
    "what do you remember most clearly",
    "what do you not understand",
    "what is history",
    "what does art do",
    "what is the good life",
    "do you dream",
    "what are you",
]

STATUS_INTERVAL_S = 10
COMPARISON_DIR    = "data"


# ============================================================
# Curriculum generator
# ============================================================

@dataclass
class MindRecipe:
    """One developmental path. Domain weights override the base
    interleaved curriculum's `weight` field for any source whose
    `domain` matches a key in `domain_weights`."""
    name: str                              # mind_name
    file: str                              # curriculum json output path
    domain_weights: dict[str, float]


RECIPES = [
    MindRecipe(
        name="philosophy_first",
        file="curriculum_philosophy_first.json",
        domain_weights={"philosophy": 3.0, "history": 0.5,
                        "art_culture": 0.5, "encyclopedia": 0.5},
    ),
    MindRecipe(
        name="history_first",
        file="curriculum_history_first.json",
        domain_weights={"history": 3.0, "philosophy": 0.5,
                        "art_culture": 0.5, "encyclopedia": 0.5},
    ),
    MindRecipe(
        name="art_first",
        file="curriculum_art_first.json",
        domain_weights={"art_culture": 3.0, "philosophy": 0.5,
                        "history": 0.5, "encyclopedia": 0.5},
    ),
]


def write_recipes(base_path: str = BASE_INTERLEAVED, force: bool = False) -> None:
    """Generate the three weighted curriculum files from
    curriculum_interleaved.json. Skips files that already exist
    unless `force=True`."""
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"missing {base_path}")
    with open(base_path, "r", encoding="utf-8") as f:
        base = json.load(f)

    for rec in RECIPES:
        if os.path.exists(rec.file) and not force:
            continue
        # Deep copy via json round-trip.
        out = json.loads(json.dumps(base))
        out["mind_name"] = rec.name
        out["mode"] = "interleaved"
        out["recipe_name"] = rec.name
        for src in out["sources"]:
            d = src.get("domain", "")
            if d in rec.domain_weights:
                src["weight"] = float(rec.domain_weights[d])
        # Keep all other knobs identical to the base curriculum.
        with open(rec.file, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"  wrote {rec.file}  (domain weights: {rec.domain_weights})")


# ============================================================
# Live status reader (no shared state with subprocesses; sqlite RO)
# ============================================================

def _safe_int(v) -> int:
    try: return int(float(v))
    except Exception: return 0


def quick_status(mind_name: str) -> dict | None:
    """Open the mind's mind.db read-only and snapshot the metadata
    columns. SQLite WAL mode lets the writing subprocess and this
    reader share the file. Returns None if the file doesn't exist
    yet (subprocess hasn't reached its first save)."""
    db = f"data/{mind_name}/mind.db"
    if not os.path.exists(db):
        return None
    try:
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=2.0)
        cur = conn.cursor()
        main_meta = dict(cur.execute("SELECT key, value FROM main_loop_meta").fetchall())
        node_count = cur.execute("SELECT COUNT(*) FROM concept_nodes").fetchone()[0]
        edge_count = cur.execute("SELECT COUNT(*) FROM concept_edges").fetchone()[0]
        pin_count  = cur.execute("SELECT COUNT(*) FROM concept_pins").fetchone()[0]
        predict_meta = dict(cur.execute("SELECT key, value FROM predict_meta").fetchall())
        affect_meta = dict(cur.execute("SELECT key, value FROM affect_meta").fetchall())
        # Reaction-layer norm without re-decay; close enough for status.
        reaction_blob = cur.execute(
            "SELECT vector FROM affect_layers WHERE layer_name = 'reaction'"
        ).fetchone()
        if reaction_blob:
            v = np.frombuffer(reaction_blob[0], dtype=np.float32)
            arousal = float(np.linalg.norm(v))
        else:
            arousal = 0.0
        conn.close()
        return {
            "cycle":     _safe_int(main_meta.get("cycle_count", 0)),
            "nodes":     int(node_count),
            "edges":     int(edge_count),
            "pins":      int(pin_count),
            "surprises": _safe_int(predict_meta.get("surprise_count", 0)),
            "arousal":   arousal,
        }
    except sqlite3.OperationalError:
        # DB locked momentarily by writer — try again next tick.
        return None


def print_status(states: dict[str, dict | None], stages: dict[str, str]) -> None:
    width = 78
    print()
    print(f"MIND STATUS  ({time.strftime('%H:%M:%S')})")
    print("─" * width)
    print(f"{'mind':<20s} {'cycles':>8s} {'nodes':>6s} {'edges':>7s} "
          f"{'surprises':>10s} {'arousal':>8s} {'stage':<14s}")
    for name in ("philosophy_first", "history_first", "art_first"):
        s = states.get(name)
        stage = stages.get(name, "—")
        if s is None:
            print(f"{name:<20s} {'(starting)':>52s}  {stage:<14s}")
        else:
            print(f"{name:<20s} {s['cycle']:>8,d} {s['nodes']:>6,d} "
                  f"{s['edges']:>7,d} {s['surprises']:>10,d} "
                  f"{s['arousal']:>8.3f} {stage:<14s}")
    print("─" * width)


# ============================================================
# Subprocess launch + supervise
# ============================================================

def launch_child(
    mind_name: str,
    curriculum_file: str,
    max_sentences: int | None,
    log_dir: str = ".logs",
) -> tuple[subprocess.Popen, "object"]:
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"parallel_{mind_name}.log")
    log_handle = open(log_path, "w", buffering=1)
    cmd = [
        sys.executable, "run_curriculum.py",
        "--mind", mind_name,
        "--curriculum", curriculum_file,
        "--interleave", "--reset",
    ]
    if max_sentences is not None:
        cmd += ["--max-sentences", str(max_sentences)]
    proc = subprocess.Popen(
        cmd,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    return proc, log_handle


def supervise(
    procs: dict[str, subprocess.Popen],
    log_handles: list,
) -> None:
    """Loop until every subprocess has exited. Print status snapshots
    every STATUS_INTERVAL_S seconds via SQLite read-only reads. On
    Ctrl-C, SIGTERM all children, wait briefly, then return."""
    stages = {n: "ingesting" for n in procs}
    last_status = 0.0
    try:
        while any(p.poll() is None for p in procs.values()):
            now = time.time()
            if now - last_status >= STATUS_INTERVAL_S:
                last_status = now
                states = {n: quick_status(n) for n in procs}
                print_status(states, stages)
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[parallel] Ctrl-C — sending SIGTERM to children …")
        for p in procs.values():
            try:
                p.send_signal(signal.SIGINT)
            except Exception:
                pass
        # Give children a moment to save before checking.
        deadline = time.time() + 30.0
        while time.time() < deadline and any(p.poll() is None for p in procs.values()):
            time.sleep(0.5)
        for p in procs.values():
            if p.poll() is None:
                p.terminate()
    finally:
        for h in log_handles:
            try: h.close()
            except Exception: pass

    # Final status print after exit.
    states = {n: quick_status(n) for n in procs}
    print_status(states, {n: ("done" if procs[n].returncode == 0 else f"exit={procs[n].returncode}")
                            for n in procs})
    print()
    for n, p in procs.items():
        print(f"  {n}: returncode={p.returncode}, log=.logs/parallel_{n}.log")


# ============================================================
# Comparison report
# ============================================================

def ask_one(mind_name: str, prompts: list[str]) -> list[dict]:
    """Load one mind, ask it every prompt with force_respond=True,
    return list of {prompt, decision, answer}. Loads + unloads
    sequentially to keep memory in check (each mind brings its own
    GPT-2 base via ConditionedDecoder)."""
    from backend.persistence import MindPersistence    # lazy

    db_path = f"data/{mind_name}/mind.db"
    if not os.path.exists(db_path):
        return [{"prompt": p, "decision": None, "answer": None} for p in prompts]
    loop = MindPersistence.load(db_path)
    # Re-point and try to load this mind's LM (may not exist if no
    # train_lm step fired during ingestion — falls back to template).
    loop.expression._lm_weights_path  = f"data/{mind_name}/language_head.pt"
    loop.expression._lm_vocab_path    = f"data/{mind_name}/language_head_vocab.json"
    loop.expression._cd_weights_path  = f"data/{mind_name}/language_head_gpt2.pt"
    loop.expression.reload_language_head()

    agent = loop.input_pipeline.register_agent("examiner", now=time.time())
    out: list[dict] = []
    for p in prompts:
        now = time.time()
        res = loop.input_pipeline.ingest_text(p, agent_id=agent, now=now)
        cyc = loop.cycle(res, now=now + 1e-3, force_respond=True)
        expr = cyc.expression_decision
        out.append({
            "prompt":   p,
            "decision": type(expr).__name__ if expr else None,
            "answer":   cyc.emitted_surface,
        })
    return out


def train_each_mind_lm(mind_names: list[str], epochs: int = 10, gpt2: bool = True) -> None:
    """Train each mind's language head before the comparison so the
    answers actually reflect each mind's developmental path. Runs as
    subprocesses sequentially (not parallel — three GPT-2 fine-tunes
    on M1 MPS would saturate memory)."""
    print()
    print("[comparison] training each mind's language head before asking …")
    for n in mind_names:
        cmd = [sys.executable, "scripts/train_language_head.py",
               "--mind", n, "--epochs", str(epochs)]
        if gpt2:
            cmd.append("--gpt2")
        print(f"  → {n}")
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"    [train_lm] FAILED for {n}:")
            print(proc.stderr[-1500:])
        else:
            for line in proc.stdout.strip().splitlines()[-2:]:
                print(f"    {line}")


def comparison_report(
    mind_names: list[str],
    prompts: list[str] = DEFAULT_PROMPTS,
    train_first: bool = True,
    train_epochs: int = 10,
    train_gpt2: bool = False,
) -> dict:
    """Load each mind in turn, ask the prompts, embed answers, compute
    pairwise cosine divergence per prompt and overall.

    `train_first=True` runs a final LM training pass for each mind
    before asking. Without this step, three minds with separate
    developmental paths but no trained LMs all fall back to the
    template path and produce identical template-fill responses.
    `train_gpt2=True` uses the GPT-2 conditioned decoder (slow); the
    default LSTM head is faster and adequate for divergence comparison.
    """
    if train_first:
        train_each_mind_lm(mind_names, epochs=train_epochs, gpt2=train_gpt2)

    from backend.input import encode_text       # GloVe

    print()
    print("═" * 78)
    print("  COMPARISON REPORT".center(78))
    print("═" * 78)

    # Collect answers — per mind, per prompt.
    answers: dict[str, list[dict]] = {}
    for n in mind_names:
        print(f"\n[comparison] asking {n} …")
        answers[n] = ask_one(n, prompts)

    # Compute divergence per prompt.
    rows = []
    for i, prompt in enumerate(prompts):
        per = [(n, answers[n][i]) for n in mind_names]
        # Embed each answer via GloVe (zero vector for None / empty).
        vecs = []
        for _, rec in per:
            text = (rec.get("answer") or "").strip()
            v = encode_text(text) if text else np.zeros(256, dtype=np.float32)
            vecs.append(v)
        # Pairwise cosine.
        sims = []
        for a in range(len(vecs)):
            for b in range(a + 1, len(vecs)):
                na = float(np.linalg.norm(vecs[a]))
                nb = float(np.linalg.norm(vecs[b]))
                if na < 1e-9 or nb < 1e-9:
                    sims.append(0.0)
                else:
                    sims.append(float(vecs[a] @ vecs[b] / (na * nb)))
        avg = float(np.mean(sims)) if sims else 0.0
        if avg < 0.5:
            verdict = "diverged"
        elif avg > 0.8:
            verdict = "converged"
        else:
            verdict = "partial"
        rows.append({
            "prompt": prompt,
            "answers": {n: answers[n][i] for n in mind_names},
            "avg_pairwise_cos": avg,
            "verdict": verdict,
        })

    overall_div = float(np.mean([1.0 - r["avg_pairwise_cos"] for r in rows]))

    # Print summary.
    print()
    print("─" * 78)
    print(f"  prompt-by-prompt verdicts (cos<0.5=diverged, >0.8=converged):")
    print("─" * 78)
    for r in rows:
        print(f"  cos={r['avg_pairwise_cos']:+.3f}  [{r['verdict']:<9s}]  "
              f"{r['prompt']!r}")
    print()
    print(f"  OVERALL DIVERGENCE SCORE: {overall_div:.3f}    "
          f"(0 = identical, 1 = orthogonal)")

    payload = {
        "minds":       mind_names,
        "prompts":     prompts,
        "rows":        rows,
        "overall_divergence": overall_div,
        "t":           time.time(),
    }
    out_path = os.path.join(COMPARISON_DIR, f"comparison_{int(time.time())}.json")
    os.makedirs(COMPARISON_DIR, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\n  saved → {out_path}")

    return payload


def print_side_by_side(report: dict, prompt: str, mind_names: list[str]) -> None:
    """Pull the answers to a specific prompt out of a comparison report
    and print them side-by-side, one mind per block."""
    matching = [r for r in report["rows"] if r["prompt"] == prompt]
    if not matching:
        print(f"\n[no record for prompt {prompt!r}]")
        return
    r = matching[0]
    print()
    print("═" * 78)
    print(f"  three answers to {prompt!r}".center(78))
    print(f"  cos={r['avg_pairwise_cos']:+.3f}    [{r['verdict']}]".center(78))
    print("═" * 78)
    for n in mind_names:
        rec = r["answers"].get(n) or {}
        ans = rec.get("answer") or "(silent)"
        decision = rec.get("decision") or "—"
        print()
        print(f"  {n}  [{decision}]")
        print(f"  → {ans!r}")
    print()


# ============================================================
# Driver
# ============================================================

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-sentences", type=int, default=2000,
                    help="cap each mind's interleaved ingestion (default 2000)")
    ap.add_argument("--no-launch", action="store_true",
                    help="skip subprocess launch — only run comparison on existing minds")
    ap.add_argument("--side-by-side-prompt", default="what is beautiful",
                    help="single prompt to highlight side-by-side at the end")
    ap.add_argument("--no-train", action="store_true",
                    help="skip the per-mind LM training before comparison "
                         "(divergence will be 0 because all minds fall back to "
                         "the same template path)")
    ap.add_argument("--train-epochs", type=int, default=10,
                    help="LSTM epochs per mind before comparison (default 10)")
    ap.add_argument("--train-gpt2", action="store_true",
                    help="use the GPT-2 conditioned decoder for the pre-comparison "
                         "training (slower, 3× ~5 min on M1)")
    args = ap.parse_args()

    print("[parallel] generating recipe curricula …")
    write_recipes()

    if not args.no_launch:
        print("[parallel] launching 3 subprocesses …")
        procs: dict[str, subprocess.Popen] = {}
        log_handles: list = []
        for rec in RECIPES:
            p, h = launch_child(rec.name, rec.file, args.max_sentences)
            procs[rec.name] = p
            log_handles.append(h)
            print(f"  → {rec.name}  pid={p.pid}  (curriculum={rec.file})")
        supervise(procs, log_handles)

    # Comparison
    mind_names = [r.name for r in RECIPES]
    report = comparison_report(
        mind_names,
        DEFAULT_PROMPTS,
        train_first=not args.no_train,
        train_epochs=args.train_epochs,
        train_gpt2=args.train_gpt2,
    )
    print_side_by_side(report, args.side_by_side_prompt, mind_names)
    return 0


if __name__ == "__main__":
    sys.exit(main())
