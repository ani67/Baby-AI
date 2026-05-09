"""FastAPI + WebSocket front door.

One mind per process. Construction happens in the lifespan handler so the
mind is ready before the first request lands. All mutating endpoints
serialize through a single asyncio Lock — the mind is single-writer by
design and the API guards that contract.

Endpoints:
  POST /ingest                — text in, runs loop.cycle, returns CycleResult
  POST /idle                  — runs loop.idle, returns IdleResult
  POST /sleep                 — runs loop.sleep, returns SleepResult
  GET  /state                 — current snapshot for the stats / affect panels
  GET  /graph                 — full concept graph for the 3D visualization
  WS   /ws                    — pushes every cycle / idle / sleep event live
  POST /save  (bonus)         — persist to data/mind.db
  POST /load  (bonus)         — restore from data/mind.db

State serialization keeps the payloads small enough for live streaming —
embeddings (256 floats per node) are NOT shipped over the wire; only the
scalar/derived fields the frontend renders.
"""
from __future__ import annotations

import asyncio
import os
import time
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.affect import AffectStack
from backend.attention import Attention
from backend.expression import Expression
from backend.graph import ConceptGraph, EdgeType
from backend.identity import (
    ChosenCandidate,
    Identity,
    RevisionRequest,
    SuppressionRequest,
)
from backend.input import InputPipeline
from backend.main_loop import CycleResult, IdleResult, MainLoop, SleepResult
from backend.mind_paths import MindPaths
from backend.persistence import MindPersistence
from backend.predict import PredictionEngine
from backend.simulation import SimulationReplay


# Multi-mind: pick which mind this server backs via MIND_NAME (default 'first').
# Falls back to legacy MIND_DB env var if set, for backward compat with older
# single-mind setups.
MIND_NAME = os.environ.get("MIND_NAME", "first")
PATHS = MindPaths(MIND_NAME)
PATHS.ensure_dirs()
DB_PATH = os.environ.get("MIND_DB", PATHS.db)


def _read_db_node_count(db_path: str) -> int:
    """Read the current concept_nodes count straight from SQLite.

    Used by the lifespan shutdown handler to detect stale-snapshot
    saves: if our in-memory graph holds materially fewer nodes than
    what's on disk *right now*, another process rewrote the DB while
    we were running and saving here would silently roll back the
    richer state. Returns 0 on any error so the guard fails open
    (we still save when we can't read disk).
    """
    import sqlite3
    try:
        conn = sqlite3.connect(db_path, timeout=5.0)
        try:
            count = conn.execute(
                "SELECT COUNT(*) FROM concept_nodes"
            ).fetchone()[0]
        finally:
            conn.close()
        return int(count)
    except Exception:
        return 0


def construct_mind(birth_seed: int = 42) -> MainLoop:
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
        lm_weights_path=PATHS.language_head,
        lm_vocab_path=PATHS.vocab,
    )
    return MainLoop(
        affect=a, graph=g, predict_engine=p, simulation=sim,
        identity=ident, attention=f, expression=gx, input_pipeline=h,
    )


# Process-global mind state. Wrapped in a small mutable holder so lifespan can
# swap it on /load without losing the lock identity.
state: dict[str, Any] = {
    "loop": None,
    "lock": None,
    "ws_clients": set(),
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["lock"] = asyncio.Lock()
    if os.path.exists(DB_PATH):
        try:
            loop = MindPersistence.load(DB_PATH)
            # Re-point the loaded Expression at this mind's per-mind LM
            # files (in case the persisted state was written before the
            # multi-mind split). reload_language_head() picks them up.
            #
            # Phase 7: persistence.MindPersistence.load constructs Expression
            # without lm_weights_path, leaving _cd_root=None — so CD
            # discovery silently skipped before this fix. Re-point _cd_root
            # and _cd_weights_path explicitly so the GPT-2 v2 / v1 deltas
            # can be picked up by reload_language_head.
            loop.expression._lm_weights_path = PATHS.language_head
            loop.expression._lm_vocab_path = PATHS.vocab
            loop.expression._cd_root = (
                os.path.dirname(PATHS.language_head) or "."
            )
            from backend.language_head import CONDITIONED_DECODER_FILENAME
            loop.expression._cd_weights_path = os.path.join(
                loop.expression._cd_root, CONDITIONED_DECODER_FILENAME,
            )
            loop.expression.reload_language_head()
            # Rebuild the cosine index over loaded nodes. Persistence
            # leaves it empty (the FAISS-era stability fix); without
            # this rebuild, graph.nearest() returns None for every
            # query, so find_or_match misses, the input pipeline
            # writes a fresh concept with zero edges, and spread has
            # nothing to propagate through. Active sets stay at 1.
            t_idx = time.perf_counter()
            loop.graph._rebuild_index()
            print(
                f"[mind:{MIND_NAME}] cosine index rebuilt: "
                f"ntotal={loop.graph._index.ntotal:,} "
                f"({(time.perf_counter() - t_idx) * 1000:.0f}ms)"
            )
            state["loop"] = loop
            print(f"[mind:{MIND_NAME}] restored from {DB_PATH} "
                  f"({loop.graph.node_count} nodes, {loop.graph.edge_count} edges)")
        except Exception as exc:  # corrupted DB → start fresh
            print(f"[mind:{MIND_NAME}] failed to restore from {DB_PATH}: {exc}; starting fresh")
            state["loop"] = construct_mind()
    else:
        state["loop"] = construct_mind()
        print(f"[mind:{MIND_NAME}] new mind constructed (no save at {DB_PATH})")
    yield
    # Best-effort save on shutdown.
    try:
        os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
        # Stale-snapshot guard: if disk has materially more nodes than
        # we hold in RAM, we loaded an old snapshot (or another process
        # rewrote the DB while we were running). Saving here would
        # silently roll back the richer state. Refuse instead.
        #
        # Real-world incident this prevents (v0.7 ingestion run): an old
        # uvicorn process loaded mind 'first' at v0.5 baseline (6,474
        # nodes) hours before the curriculum, held that snapshot in RAM
        # the entire run, and on SIGTERM its shutdown save clobbered
        # the freshly-trained 77K mind back to 6,475. Disk had 12×
        # more nodes than RAM at that moment — this guard would have
        # caught it.
        in_mem_nodes = state["loop"].graph.node_count
        on_disk_nodes = _read_db_node_count(DB_PATH)
        if on_disk_nodes > 0 and in_mem_nodes < on_disk_nodes * 0.5:
            print(
                f"[mind] REFUSING SAVE — in-memory has {in_mem_nodes:,} "
                f"nodes, disk has {on_disk_nodes:,}. Stale-snapshot "
                f"protection: would shrink the on-disk graph by "
                f">{100 - 100 * in_mem_nodes / max(on_disk_nodes, 1):.0f}%."
            )
        else:
            MindPersistence(DB_PATH).save(state["loop"], now=time.time())
            print(f"[mind] saved {in_mem_nodes:,} nodes to {DB_PATH} "
                  f"on shutdown")
    except Exception as exc:
        print(f"[mind] save on shutdown failed: {exc}")


app = FastAPI(lifespan=lifespan, title="The Mind")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------------------------------------------------
# Request / response shapes
# ----------------------------------------------------------------------

class IngestRequest(BaseModel):
    text: str
    agent_handle: str | None = None
    # When True, the cycle's WAIT default flips to EXPRESS — the social
    # constraint that humans get responses. Default True when an agent_handle
    # is provided (a human spoke), default False otherwise (internal/test paths).
    force_respond: bool | None = None


class IdleRequest(BaseModel):
    max_replays: int = 1


class SleepRequest(BaseModel):
    duration_seconds: float = 2.0


class SeedRequest(BaseModel):
    """Pre-train the graph with structured input before the user starts
    talking. Texts are ingested in sequence, each followed by a cycle so
    the spread/replay/abstraction machinery still runs. Optional default
    agent_handle attributes everything to one source.
    """
    texts: list[str]
    agent_handle: str | None = "world"
    inter_step_delay_s: float = 0.0   # purely for diagnostic pacing


# ----------------------------------------------------------------------
# Serialization helpers
# ----------------------------------------------------------------------

def _v(arr: np.ndarray) -> list[float]:
    return [float(x) for x in arr]


def _node_alignment(node, composite: np.ndarray) -> float:
    """Cosine of a node's running affect with the system's current composite.
    In [-1, 1]; the frontend uses this to color nodes on a warm/cool axis.
    """
    rs = node.affect_trace.running_state
    rn = float(np.linalg.norm(rs)); cn = float(np.linalg.norm(composite))
    if rn < 1e-9 or cn < 1e-9:
        return 0.0
    return float(rs @ composite / (rn * cn))


def _node_arousal_proxy(node) -> float:
    """L2 norm of the node's running affect — how much the node has been
    'felt' on average. Distinct from the system's current arousal."""
    return float(np.linalg.norm(node.affect_trace.running_state))


def serialize_cycle(c: CycleResult, loop: MainLoop, now: float) -> dict:
    expr: dict | None = None
    d = c.expression_decision
    if isinstance(d, ChosenCandidate):
        expr = {
            "type": "chosen",
            "surface": d.candidate.surface_text,
            "expression_gap": d.expression_gap,
            "score": d.score,
        }
    elif isinstance(d, RevisionRequest):
        expr = {"type": "revision", "reason": d.reason, "best_gap": d.best_gap}
    elif isinstance(d, SuppressionRequest):
        expr = {"type": "suppression", "reason": d.reason, "best_gap": d.best_gap}

    # Phase 7 workstream B: expose the budget the cycle saw (computed
    # against the post-processing active set, same now). Useful for the
    # frontend ConversationLog and for end-to-end test harnesses verifying
    # that the length signal landed.
    proc_set = c.processed_active_set or c.input_active_set or {}
    budget = loop.compute_expression_budget(proc_set, now=now)
    n_sentences = 0
    if c.emitted_surface:
        n_sentences = sum(1 for s in c.emitted_surface.split(". ") if s.strip())

    return {
        "type":         "cycle",
        "stimulus_id":  c.stimulus_id,
        "now":          now,
        "active_set":   {str(cid): float(v) for cid, v in c.spread.active_set.items()},
        "processed_active_set_size": len(proc_set),
        "traversed_edges": [
            # ((src, dst, edge_type), propagated_activation)
            {
                "source":     int(src),
                "target":     int(dst),
                "type":       et.value,
                "activation": float(act),
            }
            for ((src, dst, et), act) in c.spread.traversed_edges
        ],
        "phase":        c.attention_frame.phase.value,
        "mode":         c.attention_frame.mode.value,
        "arousal":      float(c.attention_frame.arousal),
        "action": {
            "kind":              c.action.action.value,
            "target_concept_id": c.action.target_concept_id,
            "score":             c.action.score,
        },
        "expression":      expr,
        "emitted_surface": c.emitted_surface,
        "budget":          int(budget),
        "n_sentences":     int(n_sentences),
        "stats":           serialize_state_lite(loop, now),
    }


def serialize_state(loop: MainLoop, now: float) -> dict:
    a = loop.affect; g = loop.graph
    composite = a.composite(now)
    arousal = a.current_arousal(now)

    # Top 10 concepts by activation_count
    nodes_ranked = sorted(g.nodes.values(), key=lambda n: -n.activation_count)[:10]

    return {
        "now":                  now,
        "cycle_count":          loop.cycle_count,
        "last_observation_t":   loop.last_observation_t,
        "node_count":           g.node_count,
        "edge_count":           g.edge_count,
        "pin_count":            g.pin_count,
        "replay_buffer_size":   loop.simulation.buffer_size,
        "agent_count":          loop.input_pipeline.agent_count,
        "observation_count":    loop.predict_engine.observation_count,
        "surprise_count":       loop.predict_engine.surprise_count,
        "consolidation_active": bool(a._consolidation_active),
        "composite":            _v(composite),
        "character":            _v(a.character.vector),
        "arousal":              float(arousal),
        "self_concept_id":      loop.identity.spine.self_concept_id,
        "mind_uuid":            loop.identity.spine.mind_uuid,
        "top_active_concepts":  [
            {
                "id":               n.concept_id,
                "name":             n.name[:64],
                "activation_count": n.activation_count,
                "alignment":        _node_alignment(n, composite),
                "arousal":          _node_arousal_proxy(n),
            }
            for n in nodes_ranked
        ],
        "recent_emissions": loop.input_pipeline.emitted_log[-10:],
    }


def serialize_state_lite(loop: MainLoop, now: float) -> dict:
    """Subset of /state that goes inside cycle payloads, so the frontend
    can update stats live without a separate /state poll per cycle."""
    a = loop.affect; g = loop.graph
    composite = a.composite(now)
    return {
        "cycle_count":          loop.cycle_count,
        "node_count":           g.node_count,
        "edge_count":           g.edge_count,
        "pin_count":            g.pin_count,
        "replay_buffer_size":   loop.simulation.buffer_size,
        "consolidation_active": bool(a._consolidation_active),
        "composite":            _v(composite),
        "arousal":              float(a.current_arousal(now)),
    }


def serialize_graph(loop: MainLoop, now: float) -> dict:
    g = loop.graph; a = loop.affect
    composite = a.composite(now)
    nodes = []
    for cid, node in g.nodes.items():
        nodes.append({
            "id":               cid,
            "name":             node.name[:64],
            "activation_count": node.activation_count,
            "surprise_at_birth": float(node.surprise_at_birth),
            "alignment":        _node_alignment(node, composite),   # [-1, 1]
            "arousal":          _node_arousal_proxy(node),
            "is_pinned":        g.is_pinned(cid),
            "last_activated":   float(node.last_activated),
            "created_at":       float(node.created_at),
        })
    edges = [
        {
            "source":           int(e.source_id),
            "target":           int(e.target_id),
            "type":             e.type.value,
            "weight":           float(e.weight),
            "activation_count": int(e.activation_count),
        }
        for e in g._edges.values()
    ]
    return {
        "nodes":            nodes,
        "edges":            edges,
        "self_concept_id":  loop.identity.spine.self_concept_id,
        "now":              now,
    }


# ----------------------------------------------------------------------
# WebSocket broadcast
# ----------------------------------------------------------------------

async def broadcast(payload: dict) -> None:
    dead: list[WebSocket] = []
    for ws in list(state["ws_clients"]):
        try:
            await ws.send_json(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        state["ws_clients"].discard(ws)


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    state["ws_clients"].add(ws)
    # Send initial state so a fresh client has something to render immediately.
    try:
        async with state["lock"]:
            initial = serialize_state(state["loop"], now=time.time())
        await ws.send_json({"type": "state", **initial})
    except Exception:
        pass
    try:
        while True:
            # Drop pings/whatever the client sends; we only push.
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        state["ws_clients"].discard(ws)


# ----------------------------------------------------------------------
# Endpoints
# ----------------------------------------------------------------------

@app.post("/ingest")
async def ingest(req: IngestRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text must be non-empty")

    # Default policy: a human (any agent_handle) gets a response; internal /
    # automated paths don't trigger force_respond unless explicit.
    force = req.force_respond if req.force_respond is not None else bool(req.agent_handle)

    async with state["lock"]:
        loop: MainLoop = state["loop"]
        now = time.time()
        agent_id: int | None = None
        if req.agent_handle:
            agent_id = loop.input_pipeline.register_agent(req.agent_handle, now=now)
        ingest_result = loop.input_pipeline.ingest_text(text, agent_id=agent_id, now=now)
        cycle_result = loop.cycle(ingest_result, now=now + 1e-3, force_respond=force)
        payload = serialize_cycle(cycle_result, loop, now=now + 2e-3)

    await broadcast(payload)
    return payload


@app.post("/seed")
async def seed(req: SeedRequest):
    """Bulk-ingest a list of texts to build graph density before the user
    starts talking. Each text runs through the full predict→observe→cycle
    pipeline, but with force_respond=False so the seeding doesn't fill the
    emission log. Returns a per-text summary plus the final stats.
    """
    if not req.texts:
        raise HTTPException(status_code=400, detail="texts must be non-empty")

    summaries = []
    async with state["lock"]:
        loop: MainLoop = state["loop"]
        agent_id: int | None = None
        if req.agent_handle:
            agent_id = loop.input_pipeline.register_agent(req.agent_handle, now=time.time())
        for i, raw in enumerate(req.texts):
            text = raw.strip()
            if not text:
                continue
            now = time.time()
            ingest_result = loop.input_pipeline.ingest_text(text, agent_id=agent_id, now=now)
            cycle_result = loop.cycle(ingest_result, now=now + 1e-3, force_respond=False)
            summaries.append({
                "text":         text,
                "stimulus_id":  cycle_result.stimulus_id,
                "concept_id":   ingest_result.gap.concept_id,
                "is_surprise":  bool(ingest_result.gap.is_surprise),
                "was_new":      bool(ingest_result.gap.was_new_write),
                "action":       cycle_result.action.action.value,
            })
            if req.inter_step_delay_s > 0:
                await asyncio.sleep(req.inter_step_delay_s)

        final = serialize_state_lite(loop, now=time.time())

    payload = {"type": "seed_complete", "count": len(summaries), "stats": final, "summaries": summaries}
    await broadcast(payload)
    return payload


@app.post("/idle")
async def idle(req: IdleRequest = IdleRequest()):
    async with state["lock"]:
        loop: MainLoop = state["loop"]
        now = time.time()
        result = loop.idle(now=now, max_replays=req.max_replays)
        payload = {
            "type": "idle",
            "now":  result.now,
            "replayed_count": result.replayed_count,
            "stats": serialize_state_lite(loop, now=result.now),
        }
    await broadcast(payload)
    return payload


@app.post("/sleep")
async def sleep(req: SleepRequest = SleepRequest()):
    async with state["lock"]:
        loop: MainLoop = state["loop"]
        now = time.time()
        result = loop.sleep(now=now, duration_seconds=req.duration_seconds)
        payload = {
            "type": "sleep",
            "now":  now,
            "replays_fired":       result.replays_fired,
            "abstractions_formed": result.abstractions_formed,
            "duration_actual":     result.duration_actual,
            "stats": serialize_state_lite(loop, now=time.time()),
        }
    await broadcast(payload)
    return payload


@app.get("/state")
async def get_state():
    async with state["lock"]:
        return serialize_state(state["loop"], now=time.time())


@app.get("/graph")
async def get_graph():
    async with state["lock"]:
        return serialize_graph(state["loop"], now=time.time())


# Stable per-process ordering of EdgeType for the binary encoder. We
# pack the enum index / 10 into a float channel so the shader can
# branch on it without a string lookup.
_EDGE_TYPE_ORDER = list(EdgeType)
_EDGE_TYPE_INDEX = {t: i for i, t in enumerate(_EDGE_TYPE_ORDER)}


@app.get("/graph/binary")
async def get_graph_binary():
    """Compact binary representation of the graph for the GPU-instanced
    frontend renderer. ~7× smaller than the JSON /graph payload at
    73K nodes / 416K edges (8.6 MB vs 57 MB), and the frontend can
    parse it into typed arrays without a JSON.parse pass.

    Layout (little-endian throughout):

        header                 : i32 node_count, i32 edge_count
        nodes  (28 bytes each) : i32 id,
                                  f32 ex, f32 ey, f32 ez,    (embedding[0:3])
                                  f32 activation_norm,
                                  f32 surprise_at_birth,
                                  f32 affect_arousal_proxy
        edges  (16 bytes each) : i32 source_idx, i32 target_idx,
                                  f32 weight,
                                  f32 type_idx_norm           (enum_index / 10)

    `source_idx` / `target_idx` are positions in the node array (NOT
    concept_ids). The frontend uses them directly as indices into the
    position texture in the shader.
    """
    import struct
    from fastapi.responses import Response
    import numpy as np

    async with state["lock"]:
        loop: MainLoop = state["loop"]
        g = loop.graph
        nodes = list(g.nodes.values())
        edges = list(g._edges.values())
        node_count = len(nodes)
        edge_count = len(edges)

        # Build id → index map and the node payload in one pass.
        node_buf = bytearray(node_count * 28)
        node_index: dict[int, int] = {}
        for i, n in enumerate(nodes):
            node_index[n.concept_id] = i
            emb = n.embedding
            ex = float(emb[0]) if emb.shape[0] > 0 else 0.0
            ey = float(emb[1]) if emb.shape[0] > 1 else 0.0
            ez = float(emb[2]) if emb.shape[0] > 2 else 0.0
            arousal_proxy = float(np.linalg.norm(n.affect_trace.running_state))
            struct.pack_into(
                "<i6f",
                node_buf,
                i * 28,
                int(n.concept_id),
                ex, ey, ez,
                min(float(n.activation_count) / 100.0, 1.0),
                float(n.surprise_at_birth),
                arousal_proxy,
            )

        edge_buf = bytearray(edge_count * 16)
        for i, e in enumerate(edges):
            src_idx = node_index.get(int(e.source_id), 0)
            tgt_idx = node_index.get(int(e.target_id), 0)
            type_norm = _EDGE_TYPE_INDEX.get(e.type, 0) / 10.0
            struct.pack_into(
                "<iiff",
                edge_buf,
                i * 16,
                src_idx,
                tgt_idx,
                float(e.weight),
                float(type_norm),
            )

        header = struct.pack("<ii", node_count, edge_count)
        payload = header + bytes(node_buf) + bytes(edge_buf)

    return Response(content=payload, media_type="application/octet-stream")


@app.post("/save")
async def save():
    async with state["lock"]:
        os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
        MindPersistence(DB_PATH).save(state["loop"], now=time.time())
        size = os.path.getsize(DB_PATH)
    return {"path": DB_PATH, "size_bytes": size}


@app.post("/load")
async def load():
    if not os.path.exists(DB_PATH):
        raise HTTPException(status_code=404, detail=f"no save at {DB_PATH}")
    async with state["lock"]:
        state["loop"] = MindPersistence.load(DB_PATH)
    return {"loaded_from": DB_PATH}
