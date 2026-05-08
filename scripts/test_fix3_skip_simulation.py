"""Fix 3 — verify skip_simulation=True in book ingestion does NOT touch
the OUTPUT-layer Welford stats. With simulation enabled, G's
request_expression path fires B.observe at OUTPUT for each candidate; with
it skipped, OUTPUT count must stay at zero.

Also reports the wall-clock difference per 500-sentence ingest.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.affect import AffectStack            # noqa: E402
from backend.attention import Attention            # noqa: E402
from backend.expression import Expression          # noqa: E402
from backend.graph import ConceptGraph             # noqa: E402
from backend.identity import Identity              # noqa: E402
from backend.input import InputPipeline            # noqa: E402
from backend.main_loop import MainLoop             # noqa: E402
from backend.predict import PredictionEngine       # noqa: E402
from backend.preencoder import DB_PATH as ENCODED_DB_PATH  # noqa: E402
from backend.simulation import SimulationReplay   # noqa: E402

import sqlite3


def fetch_n(n: int) -> list[tuple[str, np.ndarray]]:
    conn = sqlite3.connect(f"file:{ENCODED_DB_PATH}?mode=ro", uri=True)
    rows = conn.execute(
        "SELECT sentence, representation FROM encoded_sentences "
        "ORDER BY id LIMIT ?", (n,),
    ).fetchall()
    conn.close()
    out = []
    for s, blob in rows:
        rep = np.frombuffer(blob, dtype=np.float32).astype(np.float32, copy=True)
        out.append((s, rep))
    return out


def build_loop() -> MainLoop:
    now = time.time()
    a = AffectStack(birth_seed=33, t_birth=now)
    g = ConceptGraph()
    p = PredictionEngine(affect=a, graph=g)
    sim = SimulationReplay(affect=a, graph=g, predict_engine=p)
    ident = Identity(
        affect=a, graph=g, predict_engine=p, simulation=sim,
        birth_seed=33, birth_time=now,
    )
    h = InputPipeline(affect=a, graph=g, predict_engine=p, identity=ident)
    f = Attention(affect=a, graph=g)
    gx = Expression(
        affect=a, graph=g, predict_engine=p, identity=ident, input_pipeline=h,
    )
    return MainLoop(
        affect=a, graph=g, predict_engine=p, simulation=sim,
        identity=ident, attention=f, expression=gx, input_pipeline=h,
    )


def run(label: str, sentences, skip_sim: bool) -> dict:
    loop = build_loop()
    h = loop.input_pipeline
    agent = h.register_agent("fix3", now=time.time())

    t0 = time.perf_counter()
    for s, rep in sentences:
        now = time.time()
        ing = h.ingest_text(s, now=now, agent_id=agent, representation=rep)
        loop.cycle(ing, now=now + 1e-3, force_respond=False,
                   skip_simulation=skip_sim)
    dt = time.perf_counter() - t0

    return {
        "label":     label,
        "skip_sim":  skip_sim,
        "dur_s":     dt,
        "rate":      len(sentences) / max(dt, 1e-9),
        "nodes":     loop.graph.node_count,
        "input_count":  loop.predict_engine.layer_stats("INPUT")["count"],
        "proc_count":   loop.predict_engine.layer_stats("PROCESSING")["count"],
        "output_count": loop.predict_engine.layer_stats("OUTPUT")["count"],
    }


def main() -> int:
    sentences = fetch_n(500)
    print(f"[fix3] pulled {len(sentences)} pre-encoded sentences")
    print()

    a = run("skip_simulation=True ", sentences, skip_sim=True)
    b = run("skip_simulation=False", sentences, skip_sim=False)

    print(f"  {'mode':<22s}  {'sec':>6s}  {'sent/s':>8s}  "
          f"{'INPUT':>6s}  {'PROC':>5s}  {'OUTPUT':>6s}  {'nodes':>6s}")
    for r in (a, b):
        print(f"  {r['label']:<22s}  {r['dur_s']:>6.2f}  "
              f"{r['rate']:>8.0f}  {r['input_count']:>6d}  "
              f"{r['proc_count']:>5d}  {r['output_count']:>6d}  "
              f"{r['nodes']:>6d}")

    print()
    speedup = b["dur_s"] / max(a["dur_s"], 1e-9)
    print(f"[fix3] speedup with skip_simulation=True: {speedup:.2f}x")

    assert a["output_count"] == 0, \
        f"skip_simulation=True still incremented OUTPUT count: {a['output_count']}"
    print(f"[fix3] OUTPUT count with skip_simulation=True: "
          f"{a['output_count']} (expect 0) — PASS")

    return 0


if __name__ == "__main__":
    sys.exit(main())
