"""Persistence (Phase 4.1).

Single SQLite file. Atomic save: either every component's state commits or
nothing does (one transaction, one COMMIT). Load restores a fully-formed
MainLoop in the same construction order the live system uses, with
skip_boot=True on Identity and Expression so the auto-allocations
(self/unknown concepts, seed templates) are NOT duplicated — they live
in the persisted ConceptGraph and are restored along with everything else.

Schema is inspectable with any sqlite3 client; no pickle, no JSON for
core state (only for small bounded lists that are unambiguously tabular,
like an OtherModel's believed_to_know set).

Caveats:
  - HabitOverlay (F's habit-tracking) is not yet a stateful structure
    in our codebase; nothing to persist for F today. When F grows habit
    state, it adds a `habits_*` table and the existing schema_version
    bumps.
  - Encoder swap journaling (synthesis T9) is deferred along with the
    swap mechanism itself; for now `encoder_id` lives only on Stimuli
    in flight, not on persisted ConceptNodes (per the locked Phase 1
    decision to defer the field on ConceptNode).
"""
from __future__ import annotations

import json
import sqlite3
from typing import Any

import numpy as np

from backend.affect import AffectStack
from backend.attention import Attention
from backend.config import D_REP, N_AFF
from backend.expression import Expression, StyleState, TemplateMeta
from backend.graph import (
    AffectTrace,
    ConceptGraph,
    ConceptNode,
    Edge,
    EdgeType,
)
from backend.identity import (
    ExpressionCalibration,
    Identity,
    IdentitySpine,
    NarrativeAnchor,
    OtherModel,
    PinReason,
    PinRecord,
)
from backend.input import AgentRecord, InputPipeline
from backend.main_loop import MainLoop
from backend.predict import PredictionEngine
from backend.simulation import (
    ActionDescriptor,
    ActionKind,
    CommittedDecisionRecord,
    ReplayEntry,
    SimulationReplay,
    WorldModelMetadata,
)


SCHEMA_VERSION = 1


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS mind_meta (
    key TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS affect_layers (
    layer_name TEXT PRIMARY KEY,
    vector BLOB NOT NULL,
    t REAL NOT NULL,
    half_life REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS affect_w (
    row_idx INTEGER PRIMARY KEY,
    vector BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS affect_meta (
    key TEXT PRIMARY KEY,
    value REAL
);

CREATE TABLE IF NOT EXISTS concept_nodes (
    concept_id INTEGER PRIMARY KEY,
    name TEXT,
    embedding BLOB NOT NULL,
    activation_count INTEGER,
    last_activated REAL,
    created_at REAL,
    surprise_at_birth REAL,
    affect_birth BLOB,
    affect_running BLOB,
    affect_last_update REAL
);

CREATE TABLE IF NOT EXISTS concept_edges (
    source_id INTEGER,
    target_id INTEGER,
    edge_type TEXT,
    weight REAL,
    created_at REAL,
    last_activated REAL,
    activation_count INTEGER,
    PRIMARY KEY (source_id, target_id, edge_type)
);

CREATE TABLE IF NOT EXISTS concept_pins (
    concept_id INTEGER PRIMARY KEY,
    reason TEXT
);

CREATE TABLE IF NOT EXISTS concept_meta (
    key TEXT PRIMARY KEY,
    value INTEGER
);

CREATE TABLE IF NOT EXISTS predict_meta (
    key TEXT PRIMARY KEY,
    value INTEGER
);

CREATE TABLE IF NOT EXISTS predict_layer_stats (
    layer TEXT PRIMARY KEY,
    count INTEGER,
    mean REAL,
    M2 REAL
);

CREATE TABLE IF NOT EXISTS replay_entries (
    entry_id TEXT PRIMARY KEY,
    original_tick INTEGER,
    original_t REAL,
    actual_repr BLOB,
    affect_snapshot BLOB,
    layer TEXT,
    priority REAL,
    replay_count INTEGER,
    last_replayed_t REAL,
    surprise_at_birth REAL
);

CREATE TABLE IF NOT EXISTS world_model_meta (
    key TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS committed_decisions (
    rowid_seq INTEGER PRIMARY KEY AUTOINCREMENT,
    decision_id TEXT,
    action TEXT,
    target_concept_id INTEGER,
    score REAL,
    chain_id INTEGER,
    t REAL,
    runner_up_action TEXT,
    runner_up_target INTEGER,
    runner_up_score REAL,
    affect_at_decision BLOB
);

CREATE TABLE IF NOT EXISTS identity_spine (
    key TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS identity_pinned (
    concept_id INTEGER PRIMARY KEY,
    reason TEXT,
    pinned_at REAL,
    last_touched_t REAL,
    note TEXT
);

CREATE TABLE IF NOT EXISTS identity_anchors (
    anchor_id TEXT PRIMARY KEY,
    t REAL,
    core_concepts TEXT,
    affect_at_event BLOB,
    note TEXT
);

CREATE TABLE IF NOT EXISTS identity_others (
    agent_concept_id INTEGER PRIMARY KEY,
    handle TEXT,
    believed_to_know TEXT
);

CREATE TABLE IF NOT EXISTS expression_style (
    key TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS expression_templates (
    concept_id INTEGER PRIMARY KEY,
    template_id TEXT,
    pattern TEXT,
    pragmatic_function TEXT,
    slot_keys TEXT
);

CREATE TABLE IF NOT EXISTS agents (
    agent_concept_id INTEGER PRIMARY KEY,
    handle TEXT UNIQUE,
    registered_at REAL,
    last_seen_t REAL
);

CREATE TABLE IF NOT EXISTS input_meta (
    key TEXT PRIMARY KEY,
    value INTEGER
);

CREATE TABLE IF NOT EXISTS main_loop_meta (
    key TEXT PRIMARY KEY,
    value REAL
);
"""


_TABLES_TO_WIPE = [
    "mind_meta", "affect_layers", "affect_w", "affect_meta",
    "concept_nodes", "concept_edges", "concept_pins", "concept_meta",
    "predict_meta", "predict_layer_stats",
    "replay_entries", "world_model_meta", "committed_decisions",
    "identity_spine", "identity_pinned", "identity_anchors", "identity_others",
    "expression_style", "expression_templates",
    "agents", "input_meta", "main_loop_meta",
]


def _vec_to_blob(v: np.ndarray) -> bytes:
    return v.astype(np.float32, copy=False).tobytes()


def _blob_to_vec(b: bytes, expected_len: int | None = None) -> np.ndarray:
    arr = np.frombuffer(b, dtype=np.float32).copy()
    if expected_len is not None and arr.size != expected_len:
        raise ValueError(f"vector length mismatch: expected {expected_len}, got {arr.size}")
    return arr


class MindPersistence:
    """Atomic save/load for the full mind. One SQLite file per mind."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(SCHEMA_SQL)
            conn.commit()

    # =========================================================
    # SAVE — single transaction, atomic commit
    # =========================================================

    def save(self, mind: MainLoop, now: float) -> None:
        """Snapshot the entire mind state to the DB. Either everything
        commits or nothing does — partial saves are impossible.
        """
        # isolation_level=None puts us in autocommit, then we explicitly
        # BEGIN/COMMIT for the atomic save. If anything raises before
        # COMMIT, the transaction rolls back.
        conn = sqlite3.connect(self.db_path, isolation_level=None)
        try:
            cur = conn.cursor()
            cur.execute("BEGIN IMMEDIATE")

            # Wipe — save is a snapshot, not append.
            for table in _TABLES_TO_WIPE:
                cur.execute(f"DELETE FROM {table}")

            self._save_mind_meta(cur, mind, now)
            self._save_affect(cur, mind.affect)
            self._save_graph(cur, mind.graph)
            self._save_predict(cur, mind.predict_engine)
            self._save_simulation(cur, mind.simulation)
            self._save_identity(cur, mind.identity)
            self._save_expression(cur, mind.expression)
            self._save_input(cur, mind.input_pipeline)
            self._save_main_loop(cur, mind)

            cur.execute("COMMIT")
        except Exception:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise
        finally:
            conn.close()

    def _save_mind_meta(self, cur: sqlite3.Cursor, mind: MainLoop, now: float) -> None:
        spine = mind.identity.spine
        rows = [
            ("schema_version",  str(SCHEMA_VERSION)),
            ("birth_seed",      str(spine.birth_seed)),
            ("birth_time",      str(spine.birth_time)),
            ("mind_uuid",       spine.mind_uuid),
            ("last_persisted_t", str(now)),
        ]
        cur.executemany("INSERT INTO mind_meta (key, value) VALUES (?, ?)", rows)

    def _save_affect(self, cur: sqlite3.Cursor, a: AffectStack) -> None:
        # Five layers
        layer_rows = [
            ("reaction",    _vec_to_blob(a.reaction.vector),    a.reaction.t,    a.reaction.half_life),
            ("working",     _vec_to_blob(a.working.vector),     a.working.t,     a.working.half_life),
            ("mood",        _vec_to_blob(a.mood.vector),        a.mood.t,        a.mood.half_life),
            ("disposition", _vec_to_blob(a.disposition.vector), a.disposition.t, a.disposition.half_life),
            ("character",   _vec_to_blob(a.character.vector),   a.character.t,   a.character.half_life),
        ]
        cur.executemany(
            "INSERT INTO affect_layers (layer_name, vector, t, half_life) VALUES (?, ?, ?, ?)",
            layer_rows,
        )
        # W matrix: one row per N_AFF row
        w_rows = [(i, _vec_to_blob(a.W[i])) for i in range(a.W.shape[0])]
        cur.executemany("INSERT INTO affect_w (row_idx, vector) VALUES (?, ?)", w_rows)
        # Affect-level scalar metadata
        cur.executemany(
            "INSERT INTO affect_meta (key, value) VALUES (?, ?)",
            [
                ("W_update_count",         float(a.W_update_count)),
                ("nudge_gain_multiplier",  float(a._nudge_gain_multiplier)),
            ],
        )

    def _save_graph(self, cur: sqlite3.Cursor, g: ConceptGraph) -> None:
        node_rows = []
        for node in g.nodes.values():
            tr = node.affect_trace
            node_rows.append((
                node.concept_id,
                node.name,
                _vec_to_blob(node.embedding),
                node.activation_count,
                node.last_activated,
                node.created_at,
                node.surprise_at_birth,
                _vec_to_blob(tr.birth_state),
                _vec_to_blob(tr.running_state),
                tr.last_affect_update,
            ))
        cur.executemany(
            "INSERT INTO concept_nodes "
            "(concept_id, name, embedding, activation_count, last_activated, "
            " created_at, surprise_at_birth, affect_birth, affect_running, "
            " affect_last_update) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            node_rows,
        )
        edge_rows = [
            (e.source_id, e.target_id, e.type.value, e.weight,
             e.created_at, e.last_activated, e.activation_count)
            for e in g._edges.values()
        ]
        cur.executemany(
            "INSERT INTO concept_edges "
            "(source_id, target_id, edge_type, weight, created_at, "
            " last_activated, activation_count) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            edge_rows,
        )
        pin_rows = [(cid, reason) for cid, reason in g._pins.items()]
        cur.executemany("INSERT INTO concept_pins (concept_id, reason) VALUES (?, ?)", pin_rows)
        cur.execute(
            "INSERT INTO concept_meta (key, value) VALUES (?, ?)",
            ("next_concept_id", g._next_concept_id),
        )

    def _save_predict(self, cur: sqlite3.Cursor, p: PredictionEngine) -> None:
        cur.executemany(
            "INSERT INTO predict_meta (key, value) VALUES (?, ?)",
            [
                ("tick",              p._tick),
                ("observation_count", p.observation_count),
                ("surprise_count",    p.surprise_count),
            ],
        )
        for layer_name, stats in p._stats.items():
            cur.execute(
                "INSERT INTO predict_layer_stats (layer, count, mean, M2) VALUES (?, ?, ?, ?)",
                (layer_name, stats.count, stats.mean, stats.M2),
            )

    def _save_simulation(self, cur: sqlite3.Cursor, sim: SimulationReplay) -> None:
        # Replay buffer
        entry_rows = []
        for e in sim._buffer.values():
            entry_rows.append((
                e.entry_id, e.original_tick, e.original_t,
                _vec_to_blob(e.actual_repr), _vec_to_blob(e.affect_snapshot),
                e.layer, e.priority, e.replay_count, e.last_replayed_t,
                e.surprise_at_birth,
            ))
        cur.executemany(
            "INSERT INTO replay_entries "
            "(entry_id, original_tick, original_t, actual_repr, affect_snapshot, "
            " layer, priority, replay_count, last_replayed_t, surprise_at_birth) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            entry_rows,
        )
        # World model metadata. Field set matches the actual WorldModelMetadata
        # dataclass in simulation.py (synthesis-flagged extras like
        # simulation_quality_ema land when D's full scoring loop arrives).
        wm = sim.world_model
        wm_rows = [
            ("homeostatic_target_ema_blob", _vec_to_blob(wm.homeostatic_target_ema).hex()),
            ("homeostatic_alpha", str(wm.homeostatic_alpha)),
            ("last_replay_run_t", "" if wm.last_replay_run_t is None else str(wm.last_replay_run_t)),
            ("replays_this_window", str(wm.replays_this_window)),
            ("window_started_t", "" if wm.window_started_t is None else str(wm.window_started_t)),
        ]
        cur.executemany("INSERT INTO world_model_meta (key, value) VALUES (?, ?)", wm_rows)

        # Decision history
        dec_rows = []
        for record in sim._decisions:
            ru = record.runner_up
            dec_rows.append((
                record.decision_id,
                record.chosen.action.value,
                record.chosen.target_concept_id,
                record.chosen.score,
                record.chosen.chain_id,
                record.t,
                ru.action.value if ru else None,
                ru.target_concept_id if ru else None,
                ru.score if ru else None,
                _vec_to_blob(record.affect_at_decision),
            ))
        cur.executemany(
            "INSERT INTO committed_decisions "
            "(decision_id, action, target_concept_id, score, chain_id, t, "
            " runner_up_action, runner_up_target, runner_up_score, affect_at_decision) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            dec_rows,
        )

    def _save_identity(self, cur: sqlite3.Cursor, ident: Identity) -> None:
        s = ident.spine
        spine_kv = [
            ("birth_seed", str(s.birth_seed)),
            ("birth_time", str(s.birth_time)),
            ("mind_uuid",  s.mind_uuid),
            ("self_concept_id",    str(s.self_concept_id)),
            ("unknown_concept_id", str(s.unknown_concept_id)),
            ("character_baseline_blob", _vec_to_blob(s.character_baseline).hex()),
            ("honesty_bias", str(s.expression_calibration.honesty_bias)),
        ]
        cur.executemany("INSERT INTO identity_spine (key, value) VALUES (?, ?)", spine_kv)

        pin_rows = [
            (r.concept_id, r.reason.value, r.pinned_at, r.last_touched_t, r.note)
            for r in s.pinned_concepts.values()
        ]
        cur.executemany(
            "INSERT INTO identity_pinned "
            "(concept_id, reason, pinned_at, last_touched_t, note) "
            "VALUES (?, ?, ?, ?, ?)",
            pin_rows,
        )

        anchor_rows = [
            (a.anchor_id, a.t, json.dumps(a.core_concepts),
             _vec_to_blob(a.affect_at_event), a.note)
            for a in s.narrative_anchors
        ]
        cur.executemany(
            "INSERT INTO identity_anchors "
            "(anchor_id, t, core_concepts, affect_at_event, note) "
            "VALUES (?, ?, ?, ?, ?)",
            anchor_rows,
        )

        other_rows = [
            (o.agent_concept_id, o.handle, json.dumps(sorted(o.believed_to_know)))
            for o in s.others.values()
        ]
        cur.executemany(
            "INSERT INTO identity_others "
            "(agent_concept_id, handle, believed_to_know) "
            "VALUES (?, ?, ?)",
            other_rows,
        )

    def _save_expression(self, cur: sqlite3.Cursor, gx: Expression) -> None:
        st = gx.style
        style_kv = [
            ("verbosity_mean",        str(st.verbosity_mean)),
            ("modality_preference",   json.dumps(list(st.modality_preference))),
            ("revision_temperament",  str(st.revision_temperament)),
            ("template_familiarity",  json.dumps({str(k): v for k, v in st.template_familiarity.items()})),
            ("modality_leak",         json.dumps(st.modality_leak)),
        ]
        cur.executemany("INSERT INTO expression_style (key, value) VALUES (?, ?)", style_kv)

        tpl_rows = [
            (m.concept_id, m.template_id, m.pattern, m.pragmatic_function, json.dumps(m.slot_keys))
            for m in gx._templates.values()
        ]
        cur.executemany(
            "INSERT INTO expression_templates "
            "(concept_id, template_id, pattern, pragmatic_function, slot_keys) "
            "VALUES (?, ?, ?, ?, ?)",
            tpl_rows,
        )

    def _save_input(self, cur: sqlite3.Cursor, h: InputPipeline) -> None:
        agent_rows = [
            (r.agent_concept_id, r.handle, r.registered_at, r.last_seen_t)
            for r in h._agents.values()
        ]
        cur.executemany(
            "INSERT INTO agents (agent_concept_id, handle, registered_at, last_seen_t) "
            "VALUES (?, ?, ?, ?)",
            agent_rows,
        )
        cur.executemany(
            "INSERT INTO input_meta (key, value) VALUES (?, ?)",
            [("stimulus_id", h._stimulus_id), ("tick", h._tick)],
        )

    def _save_main_loop(self, cur: sqlite3.Cursor, mind: MainLoop) -> None:
        cur.executemany(
            "INSERT INTO main_loop_meta (key, value) VALUES (?, ?)",
            [
                ("cycle_count",         float(mind.cycle_count)),
                ("last_observation_t",  float(mind.last_observation_t)),
            ],
        )

    # =========================================================
    # LOAD — construct components + restore state
    # =========================================================

    @classmethod
    def load(cls, db_path: str) -> MainLoop:
        instance = cls(db_path)
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        try:
            cur = conn.cursor()

            mind_meta = {r["key"]: r["value"] for r in cur.execute("SELECT key, value FROM mind_meta")}
            if not mind_meta:
                raise RuntimeError(f"no saved mind found in {db_path}")
            schema = int(mind_meta.get("schema_version", "1"))
            if schema != SCHEMA_VERSION:
                raise RuntimeError(
                    f"schema version mismatch: file v{schema}, code v{SCHEMA_VERSION}"
                )
            birth_seed = int(mind_meta["birth_seed"])
            birth_time = float(mind_meta["birth_time"])

            # 1. AffectStack
            affect = AffectStack(birth_seed=birth_seed, t_birth=birth_time)
            instance._restore_affect(cur, affect)

            # 2. ConceptGraph
            graph = ConceptGraph()
            instance._restore_graph(cur, graph)

            # 3. PredictionEngine
            predict = PredictionEngine(affect=affect, graph=graph)
            instance._restore_predict(cur, predict)

            # 4. SimulationReplay (auto-wires predict.replay_target = self)
            simulation = SimulationReplay(affect=affect, graph=graph, predict_engine=predict)
            instance._restore_simulation(cur, simulation)

            # 5. Identity (skip_boot=True; spine restored from DB)
            identity = Identity(
                affect=affect, graph=graph, predict_engine=predict, simulation=simulation,
                birth_seed=birth_seed, birth_time=birth_time, skip_boot=True,
            )
            instance._restore_identity(cur, identity)

            # 6. InputPipeline
            input_pipeline = InputPipeline(
                affect=affect, graph=graph, predict_engine=predict, identity=identity,
            )
            instance._restore_input(cur, input_pipeline)

            # 7. Attention (no persistent state in Phase 4 minimal)
            attention = Attention(affect=affect, graph=graph)

            # 8. Expression (skip_boot=True; templates restored from DB)
            expression = Expression(
                affect=affect, graph=graph, predict_engine=predict,
                identity=identity, input_pipeline=input_pipeline, skip_boot=True,
            )
            instance._restore_expression(cur, expression)

            # 9. MainLoop
            loop = MainLoop(
                affect=affect, graph=graph, predict_engine=predict, simulation=simulation,
                identity=identity, attention=attention, expression=expression,
                input_pipeline=input_pipeline,
            )
            instance._restore_main_loop(cur, loop)

            return loop
        finally:
            conn.close()

    def _restore_affect(self, cur: sqlite3.Cursor, a: AffectStack) -> None:
        layer_map = {"reaction": a.reaction, "working": a.working, "mood": a.mood,
                     "disposition": a.disposition, "character": a.character}
        for row in cur.execute("SELECT layer_name, vector, t, half_life FROM affect_layers"):
            layer = layer_map[row["layer_name"]]
            layer.vector = _blob_to_vec(row["vector"], expected_len=N_AFF)
            layer.t = float(row["t"])
            layer.half_life = float(row["half_life"])

        rows = list(cur.execute("SELECT row_idx, vector FROM affect_w ORDER BY row_idx"))
        if rows:
            W = np.zeros((len(rows), D_REP), dtype=np.float32)
            for r in rows:
                W[r["row_idx"]] = _blob_to_vec(r["vector"], expected_len=D_REP)
            a.W = W
            a._W_pinv_cache = None   # invalidate cached pinv

        meta = {r["key"]: r["value"] for r in cur.execute("SELECT key, value FROM affect_meta")}
        if "W_update_count" in meta:
            a.W_update_count = int(meta["W_update_count"])
        if "nudge_gain_multiplier" in meta:
            a._nudge_gain_multiplier = float(meta["nudge_gain_multiplier"])

    def _restore_graph(self, cur: sqlite3.Cursor, g: ConceptGraph) -> None:
        for r in cur.execute("SELECT * FROM concept_nodes"):
            trace = AffectTrace(
                birth_state=_blob_to_vec(r["affect_birth"], expected_len=N_AFF),
                running_state=_blob_to_vec(r["affect_running"], expected_len=N_AFF),
                last_affect_update=float(r["affect_last_update"]),
            )
            node = ConceptNode(
                concept_id=int(r["concept_id"]),
                name=r["name"] or "",
                embedding=_blob_to_vec(r["embedding"], expected_len=D_REP),
                affect_trace=trace,
                activation_count=int(r["activation_count"]),
                last_activated=float(r["last_activated"]),
                created_at=float(r["created_at"]),
                surprise_at_birth=float(r["surprise_at_birth"]),
            )
            g.nodes[node.concept_id] = node
        g._matrix_dirty = True

        for r in cur.execute("SELECT * FROM concept_edges"):
            edge = Edge(
                source_id=int(r["source_id"]),
                target_id=int(r["target_id"]),
                type=EdgeType(r["edge_type"]),
                weight=float(r["weight"]),
                created_at=float(r["created_at"]),
                last_activated=float(r["last_activated"]),
                activation_count=int(r["activation_count"]),
            )
            key = (edge.source_id, edge.target_id, edge.type)
            g._edges[key] = edge
            g._out_edges.setdefault(edge.source_id, []).append(edge)
            g._in_edges.setdefault(edge.target_id, []).append(edge)

        for r in cur.execute("SELECT * FROM concept_pins"):
            g._pins[int(r["concept_id"])] = r["reason"] or ""

        meta = {r["key"]: r["value"] for r in cur.execute("SELECT * FROM concept_meta")}
        if "next_concept_id" in meta:
            g._next_concept_id = int(meta["next_concept_id"])

    def _restore_predict(self, cur: sqlite3.Cursor, p: PredictionEngine) -> None:
        meta = {r["key"]: r["value"] for r in cur.execute("SELECT * FROM predict_meta")}
        if "tick" in meta:              p._tick = int(meta["tick"])
        if "observation_count" in meta: p.observation_count = int(meta["observation_count"])
        if "surprise_count" in meta:    p.surprise_count = int(meta["surprise_count"])
        for r in cur.execute("SELECT * FROM predict_layer_stats"):
            stats = p._stats.get(r["layer"])
            if stats is not None:
                stats.count = int(r["count"])
                stats.mean  = float(r["mean"])
                stats.M2    = float(r["M2"])

    def _restore_simulation(self, cur: sqlite3.Cursor, sim: SimulationReplay) -> None:
        for r in cur.execute("SELECT * FROM replay_entries"):
            entry = ReplayEntry(
                entry_id=r["entry_id"],
                original_tick=int(r["original_tick"]),
                original_t=float(r["original_t"]),
                actual_repr=_blob_to_vec(r["actual_repr"], expected_len=D_REP),
                affect_snapshot=_blob_to_vec(r["affect_snapshot"], expected_len=N_AFF),
                layer=r["layer"],
                priority=float(r["priority"]),
                replay_count=int(r["replay_count"]),
                last_replayed_t=None if r["last_replayed_t"] is None else float(r["last_replayed_t"]),
                surprise_at_birth=float(r["surprise_at_birth"]),
            )
            sim._buffer[entry.entry_id] = entry

        wm_kv = {r["key"]: r["value"] for r in cur.execute("SELECT * FROM world_model_meta")}
        if "homeostatic_target_ema_blob" in wm_kv:
            sim.world_model.homeostatic_target_ema = _blob_to_vec(
                bytes.fromhex(wm_kv["homeostatic_target_ema_blob"]), expected_len=N_AFF
            )
        if "homeostatic_alpha" in wm_kv:
            sim.world_model.homeostatic_alpha = float(wm_kv["homeostatic_alpha"])
        if "last_replay_run_t" in wm_kv:
            sim.world_model.last_replay_run_t = (
                None if wm_kv["last_replay_run_t"] == "" else float(wm_kv["last_replay_run_t"])
            )
        if "replays_this_window" in wm_kv:
            sim.world_model.replays_this_window = int(wm_kv["replays_this_window"])
        if "window_started_t" in wm_kv:
            sim.world_model.window_started_t = (
                None if wm_kv["window_started_t"] == "" else float(wm_kv["window_started_t"])
            )

        for r in cur.execute(
            "SELECT * FROM committed_decisions ORDER BY rowid_seq"
        ):
            chosen = ActionDescriptor(
                action=ActionKind(r["action"]),
                target_concept_id=r["target_concept_id"],
                score=float(r["score"]),
                chain_id=int(r["chain_id"]),
                decision_id=r["decision_id"],
            )
            runner_up = None
            if r["runner_up_action"] is not None:
                runner_up = ActionDescriptor(
                    action=ActionKind(r["runner_up_action"]),
                    target_concept_id=r["runner_up_target"],
                    score=float(r["runner_up_score"]),
                    chain_id=chosen.chain_id,
                    decision_id="",
                )
            sim._decisions.append(CommittedDecisionRecord(
                decision_id=r["decision_id"],
                chosen=chosen,
                runner_up=runner_up,
                chain_summary=[],
                affect_at_decision=_blob_to_vec(r["affect_at_decision"], expected_len=N_AFF),
                t=float(r["t"]),
            ))

    def _restore_identity(self, cur: sqlite3.Cursor, ident: Identity) -> None:
        kv = {r["key"]: r["value"] for r in cur.execute("SELECT * FROM identity_spine")}

        spine = IdentitySpine(
            birth_seed=int(kv["birth_seed"]),
            birth_time=float(kv["birth_time"]),
            mind_uuid=kv["mind_uuid"],
            self_concept_id=int(kv["self_concept_id"]),
            unknown_concept_id=int(kv["unknown_concept_id"]),
            character_baseline=_blob_to_vec(
                bytes.fromhex(kv["character_baseline_blob"]), expected_len=N_AFF,
            ),
            expression_calibration=ExpressionCalibration(
                honesty_bias=float(kv.get("honesty_bias", "0.7")),
            ),
        )

        for r in cur.execute("SELECT * FROM identity_pinned"):
            spine.pinned_concepts[int(r["concept_id"])] = PinRecord(
                concept_id=int(r["concept_id"]),
                reason=PinReason(r["reason"]),
                pinned_at=float(r["pinned_at"]),
                last_touched_t=float(r["last_touched_t"]),
                note=r["note"] or "",
            )

        for r in cur.execute("SELECT * FROM identity_anchors"):
            spine.narrative_anchors.append(NarrativeAnchor(
                anchor_id=r["anchor_id"],
                t=float(r["t"]),
                core_concepts=json.loads(r["core_concepts"]),
                affect_at_event=_blob_to_vec(r["affect_at_event"], expected_len=N_AFF),
                note=r["note"] or "",
            ))

        for r in cur.execute("SELECT * FROM identity_others"):
            spine.others[int(r["agent_concept_id"])] = OtherModel(
                agent_concept_id=int(r["agent_concept_id"]),
                handle=r["handle"],
                believed_to_know=set(json.loads(r["believed_to_know"])),
            )

        ident.spine = spine

    def _restore_expression(self, cur: sqlite3.Cursor, gx: Expression) -> None:
        kv = {r["key"]: r["value"] for r in cur.execute("SELECT * FROM expression_style")}
        if "verbosity_mean" in kv:        gx.style.verbosity_mean = float(kv["verbosity_mean"])
        if "modality_preference" in kv:   gx.style.modality_preference = json.loads(kv["modality_preference"])
        if "revision_temperament" in kv:  gx.style.revision_temperament = float(kv["revision_temperament"])
        if "template_familiarity" in kv:
            raw = json.loads(kv["template_familiarity"])
            gx.style.template_familiarity = {int(k): float(v) for k, v in raw.items()}
        if "modality_leak" in kv:         gx.style.modality_leak = json.loads(kv["modality_leak"])

        for r in cur.execute("SELECT * FROM expression_templates"):
            meta = TemplateMeta(
                template_id=r["template_id"],
                concept_id=int(r["concept_id"]),
                pattern=r["pattern"],
                pragmatic_function=r["pragmatic_function"],
                slot_keys=json.loads(r["slot_keys"]),
            )
            gx._templates[meta.concept_id] = meta
            gx._template_by_id[meta.template_id] = meta

    def _restore_input(self, cur: sqlite3.Cursor, h: InputPipeline) -> None:
        for r in cur.execute("SELECT * FROM agents"):
            rec = AgentRecord(
                agent_concept_id=int(r["agent_concept_id"]),
                handle=r["handle"],
                registered_at=float(r["registered_at"]),
                last_seen_t=float(r["last_seen_t"]),
            )
            h._agents[rec.agent_concept_id] = rec
            h._agent_handle_index[rec.handle] = rec.agent_concept_id

        meta = {r["key"]: r["value"] for r in cur.execute("SELECT * FROM input_meta")}
        if "stimulus_id" in meta: h._stimulus_id = int(meta["stimulus_id"])
        if "tick" in meta:         h._tick = int(meta["tick"])

    def _restore_main_loop(self, cur: sqlite3.Cursor, loop: MainLoop) -> None:
        meta = {r["key"]: r["value"] for r in cur.execute("SELECT * FROM main_loop_meta")}
        if "cycle_count" in meta:
            loop.cycle_count = int(meta["cycle_count"])
        if "last_observation_t" in meta:
            loop.last_observation_t = float(meta["last_observation_t"])
