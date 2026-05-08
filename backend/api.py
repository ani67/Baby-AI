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
from backend.graph import ConceptGraph
from backend.identity import (
    ChosenCandidate,
    Identity,
    RevisionRequest,
    SuppressionRequest,
)
from backend.input import InputPipeline
from backend.main_loop import CycleResult, IdleResult, MainLoop, SleepResult
from backend.persistence import MindPersistence
from backend.predict import PredictionEngine
from backend.simulation import SimulationReplay


# Default DB path for save/load. The file isn't created until /save is called.
DB_PATH = os.environ.get("MIND_DB", "data/mind.db")


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
            state["loop"] = MindPersistence.load(DB_PATH)
            print(f"[mind] restored from {DB_PATH}")
        except Exception as exc:  # corrupted DB → start fresh
            print(f"[mind] failed to restore from {DB_PATH}: {exc}; starting fresh")
            state["loop"] = construct_mind()
    else:
        state["loop"] = construct_mind()
        print(f"[mind] new mind constructed (no save at {DB_PATH})")
    yield
    # Best-effort save on shutdown.
    try:
        os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
        MindPersistence(DB_PATH).save(state["loop"], now=time.time())
        print(f"[mind] saved to {DB_PATH} on shutdown")
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

    return {
        "type":         "cycle",
        "stimulus_id":  c.stimulus_id,
        "now":          now,
        "active_set":   {str(cid): float(v) for cid, v in c.spread.active_set.items()},
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
