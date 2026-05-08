"""A1 test — ingestion mode vs normal mode, same 500 sentences, fresh minds.

Pulls the first 500 pre-encoded sentences from data/encoded_corpus.db,
constructs a fresh mind, ingests them with ingestion_mode=False; then a
second fresh mind with ingestion_mode=True. Reports surprise counts and
ratio. Target per the spec: 2-3× more surprises in ingestion mode.
"""
from __future__ import annotations

import os
import sqlite3
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


N_SENTENCES = 500
BIRTH_SEED  = 4242


def fetch_first_n(n: int) -> list[tuple[str, np.ndarray]]:
    """Pull the first n encoded sentences across all sources, in DB rowid order."""
    conn = sqlite3.connect(f"file:{ENCODED_DB_PATH}?mode=ro", uri=True)
    rows = conn.execute(
        "SELECT sentence, representation FROM encoded_sentences ORDER BY id LIMIT ?",
        (n,),
    ).fetchall()
    conn.close()
    out: list[tuple[str, np.ndarray]] = []
    for sentence, blob in rows:
        rep = np.frombuffer(blob, dtype=np.float32)
        if rep.shape != (256,):
            raise ValueError(f"bad rep shape: {rep.shape}")
        out.append((sentence, rep.astype(np.float32, copy=True)))
    return out


def build_fresh_mind(birth_seed: int) -> MainLoop:
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
    )
    return MainLoop(
        affect=a, graph=g, predict_engine=p, simulation=sim,
        identity=ident, attention=f, expression=gx, input_pipeline=h,
    )


def run_one(label: str, sentences: list[tuple[str, np.ndarray]], ingestion_mode: bool) -> dict:
    loop = build_fresh_mind(BIRTH_SEED)
    loop.predict_engine.set_ingestion_mode(ingestion_mode)
    h = loop.input_pipeline
    agent = h.register_agent("a1_test", now=time.time())

    surprises = 0
    nodes_before = loop.graph.node_count
    for sentence, rep in sentences:
        now = time.time()
        ing = h.ingest_text(sentence, now=now, agent_id=agent, representation=rep)
        loop.cycle(ing, now=now + 1e-3, force_respond=False, skip_simulation=True)
        if ing.gap.is_surprise:
            surprises += 1

    layer_stats = loop.predict_engine.layer_stats("INPUT")
    return {
        "label":           label,
        "ingestion_mode":  ingestion_mode,
        "n":               len(sentences),
        "surprises":       surprises,
        "nodes":           loop.graph.node_count - nodes_before,
        "edges":           loop.graph.edge_count,
        "stats_count":     layer_stats["count"],
        "stats_mean":      layer_stats["mean"],
        "stats_stddev":    layer_stats["stddev"],
        "threshold_mag":   layer_stats["threshold"],
    }


def main() -> int:
    print(f"[A1 test] pulling first {N_SENTENCES} sentences from {ENCODED_DB_PATH} …")
    sentences = fetch_first_n(N_SENTENCES)
    print(f"  got {len(sentences)} sentences")
    print()

    print("[A1 test] run 1 — normal mode (1.5σ)")
    r1 = run_one("normal",     sentences, ingestion_mode=False)
    print(f"  surprises = {r1['surprises']}/{r1['n']}  ({100*r1['surprises']/r1['n']:.1f}%)")
    print(f"  nodes_added = {r1['nodes']}  edges = {r1['edges']}")
    print(f"  Welford INPUT: count={r1['stats_count']} mean={r1['stats_mean']:.4f} "
          f"stddev={r1['stats_stddev']:.4f} threshold_mag={r1['threshold_mag']!s}")
    print()

    print("[A1 test] run 2 — ingestion mode (1.0σ + cold floor 0.06)")
    r2 = run_one("ingestion", sentences, ingestion_mode=True)
    print(f"  surprises = {r2['surprises']}/{r2['n']}  ({100*r2['surprises']/r2['n']:.1f}%)")
    print(f"  nodes_added = {r2['nodes']}  edges = {r2['edges']}")
    print(f"  Welford INPUT: count={r2['stats_count']} mean={r2['stats_mean']:.4f} "
          f"stddev={r2['stats_stddev']:.4f} threshold_mag={r2['threshold_mag']!s}")
    print()

    ratio = r2["surprises"] / max(r1["surprises"], 1)
    print(f"[A1 test] ratio  ingestion / normal  =  {ratio:.2f}×")
    print(f"          target                       =  2.0–3.0×")
    if 1.5 <= ratio <= 4.0:
        print("[A1 test] PASS — ratio inside acceptable band")
        return 0
    else:
        print("[A1 test] OUT OF BAND — investigate")
        return 1


if __name__ == "__main__":
    sys.exit(main())
