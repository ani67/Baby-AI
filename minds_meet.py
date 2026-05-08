"""Two minds, separately trained, encountering each other.

Usage
-----
    python3 minds_meet.py first second
    python3 minds_meet.py first second --turns 20 --seed-prompt "introduce yourself"

Conversation protocol
---------------------
Turn 0:  mind A is prompted (default: "introduce yourself"). A's emitted
         surface is the first utterance.
Turn N:  the receiving mind ingests the previous utterance with the OTHER
         mind registered as an agent (so refers_to chains anchor cleanly,
         and so each mind can introspect "what concepts came from whom"
         after the conversation), then runs a force_respond=True cycle.
         The cycle's emitted surface is the next utterance.

Each turn captures (per the speaker's own state, after the cycle ran):
    * was_surprised:       did the input from the other mind cross
                           B's surprise threshold for the speaker?
    * was_new_write:       did a new concept get written this turn?
    * concept_id:          if so, which one (and its name).
    * arousal_after:       speaker's reaction-layer norm after the cycle.
    * composite_after:     speaker's full composite affect, post-cycle.
    * decision_type:       ChosenCandidate / Revision / Suppression / silent.

Outputs
-------
A human-readable transcript on stdout.
A JSONL log at data/conversations/{a}__{b}__{ts}.jsonl with one record
per turn — replayable, diff-able.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass

import numpy as np

# Allow running from repo root.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.identity import (              # noqa: E402
    ChosenCandidate,
    RevisionRequest,
    SuppressionRequest,
)
from backend.main_loop import MainLoop       # noqa: E402
from backend.mind_paths import MindPaths     # noqa: E402
from backend.persistence import MindPersistence  # noqa: E402


SEED_PROMPT_DEFAULT = "introduce yourself"
DEFAULT_TURNS = 20

CONVERSATION_DIR = "data/conversations"


# ============================================================
# Setup helpers
# ============================================================

def load_mind(name: str) -> tuple[MindPaths, MainLoop]:
    paths = MindPaths(name)
    if not os.path.exists(paths.db):
        raise FileNotFoundError(
            f"no mind at {paths.db} — run `python3 run_curriculum.py --mind {name} ...` first"
        )
    loop = MindPersistence.load(paths.db)
    # Re-point the in-memory Expression at this mind's per-mind LM paths
    # in case the persisted state was written by an older single-mind build.
    loop.expression._lm_weights_path = paths.language_head
    loop.expression._lm_vocab_path = paths.vocab
    loop.expression._language_head = None
    loop.expression._lm_vocab = None
    loop.expression._lm_load_attempted = False
    return paths, loop


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(a @ b / (na * nb))


# ============================================================
# Dataclasses for the transcript
# ============================================================

@dataclass
class TurnRecord:
    turn: int
    speaker: str
    listener: str
    heard_utterance: str          # what this mind ingested at the start of the turn
    spoken_utterance: str         # what this mind emitted (may be empty if silent)
    was_surprised: bool           # did `heard_utterance` cross surprise threshold?
    was_new_write: bool           # was a concept written from `heard_utterance`?
    concept_id: int | None        # if a new write, the concept's id
    concept_name: str             # truncated name of the written/matched concept
    decision_type: str            # ChosenCandidate / RevisionRequest / SuppressionRequest / silent
    arousal_after: float
    composite_after: list[float]
    composite_norm_after: float


# ============================================================
# One turn: the listening mind ingests + responds
# ============================================================

def take_turn(
    turn: int,
    speaker_loop: MainLoop,
    speaker_name: str,
    listener_name: str,
    heard_utterance: str,
    other_agent_id: int,
) -> TurnRecord:
    """Run one turn on `speaker_loop`. The mind ingests `heard_utterance`
    (attributed to the OTHER mind via `other_agent_id`), runs a
    force_respond=True cycle, emits a response.
    """
    now = time.time()
    ingest = speaker_loop.input_pipeline.ingest_text(
        heard_utterance, agent_id=other_agent_id, now=now,
    )
    cycle = speaker_loop.cycle(ingest, now=now + 1e-3, force_respond=True)

    decision = cycle.expression_decision
    if isinstance(decision, ChosenCandidate):
        decision_type = "ChosenCandidate"
    elif isinstance(decision, RevisionRequest):
        decision_type = "RevisionRequest"
    elif isinstance(decision, SuppressionRequest):
        decision_type = "SuppressionRequest"
    else:
        decision_type = "silent"

    spoken = cycle.emitted_surface or ""

    composite = speaker_loop.affect.composite(time.time())
    arousal = speaker_loop.affect.current_arousal(time.time())

    cid = ingest.gap.concept_id
    cname = ""
    if cid is not None and cid in speaker_loop.graph.nodes:
        cname = speaker_loop.graph.nodes[cid].name[:60]

    return TurnRecord(
        turn=turn,
        speaker=speaker_name,
        listener=listener_name,
        heard_utterance=heard_utterance,
        spoken_utterance=spoken,
        was_surprised=bool(ingest.gap.is_surprise),
        was_new_write=bool(ingest.gap.was_new_write),
        concept_id=cid,
        concept_name=cname,
        decision_type=decision_type,
        arousal_after=float(arousal),
        composite_after=[float(x) for x in composite],
        composite_norm_after=float(np.linalg.norm(composite)),
    )


# ============================================================
# Pretty printer for the transcript
# ============================================================

def _hr(width: int = 78) -> str:
    return "─" * width


def print_header(name_a: str, name_b: str, turns: int, seed_prompt: str) -> None:
    print()
    print("═" * 78)
    print(f"  {name_a}  ↔  {name_b}".center(78))
    print(f"  {turns} turns · seed prompt: {seed_prompt!r}".center(78))
    print("═" * 78)


def print_turn(record: TurnRecord, surprise_marker: str = "✦") -> None:
    width = 78
    head = f"  Turn {record.turn:>2d}  ·  {record.speaker} hears {record.listener}"
    print()
    print(_hr(width))
    print(head)
    print(_hr(width))

    # What the speaker heard.
    print(f"  ⟵  heard: {record.heard_utterance!r}")
    if record.was_surprised:
        marker = surprise_marker
        notes = ["surprise crossed threshold"]
    else:
        marker = " "
        notes = ["no surprise"]
    if record.was_new_write:
        notes.append(f"new concept #{record.concept_id} ({record.concept_name!r})")
    elif record.concept_id is not None:
        notes.append(f"matched existing concept #{record.concept_id}")
    print(f"      {marker}  " + "  ·  ".join(notes))

    # What the speaker said.
    if record.spoken_utterance:
        print(f"  ⟶   said:  {record.spoken_utterance!r}")
    else:
        print(f"  ⟶   said:  (silent — {record.decision_type})")

    # Affect snapshot, dimension-by-dimension would be too noisy; norm + arousal
    # gives the reader the felt change in one line.
    print(f"      affect: ‖composite‖ = {record.composite_norm_after:.3f}    "
          f"arousal = {record.arousal_after:.3f}    decision = {record.decision_type}")


def print_summary(
    records: list[TurnRecord],
    name_a: str,
    name_b: str,
    a_loop: MainLoop,
    b_loop: MainLoop,
    nodes_before_a: int,
    nodes_before_b: int,
    edges_before_a: int,
    edges_before_b: int,
) -> None:
    width = 78

    print()
    print("═" * width)
    print("  SUMMARY".center(width))
    print("═" * width)

    # Count surprises by speaker (i.e., the speaker was surprised by what they heard).
    surprised = {name_a: 0, name_b: 0}
    new_writes = {name_a: 0, name_b: 0}
    for r in records:
        if r.was_surprised:
            surprised[r.speaker] += 1
        if r.was_new_write:
            new_writes[r.speaker] += 1

    print()
    print(f"  total turns        : {len(records)}")
    print(f"  {name_a:>20s} surprised by {name_b}: {surprised[name_a]:>3d} times  "
          f"({new_writes[name_a]} new concepts written)")
    print(f"  {name_b:>20s} surprised by {name_a}: {surprised[name_b]:>3d} times  "
          f"({new_writes[name_b]} new concepts written)")

    # Graph growth from baseline (passed in by the caller — nodes/edges before
    # the conversation started).
    print()
    print(f"  {name_a:>20s} graph: {a_loop.graph.node_count - nodes_before_a:+d} nodes,  "
          f"{a_loop.graph.edge_count - edges_before_a:+d} edges")
    print(f"  {name_b:>20s} graph: {b_loop.graph.node_count - nodes_before_b:+d} nodes,  "
          f"{b_loop.graph.edge_count - edges_before_b:+d} edges")

    # Affect convergence trajectory.
    if records:
        a_records = [r for r in records if r.speaker == name_a]
        b_records = [r for r in records if r.speaker == name_b]
        # Take aligned pairs by turn position; the easiest is to sample a few
        # checkpoints throughout the conversation.
        checkpoints = [0, len(records) // 4, len(records) // 2,
                       3 * len(records) // 4, len(records) - 1]
        checkpoints = sorted(set(c for c in checkpoints if 0 <= c < len(records)))

        print()
        print("  affect convergence (cosine(A.composite, B.composite)):")
        # Build the running affect of each mind from records.
        a_affect_at: dict[int, np.ndarray] = {}
        b_affect_at: dict[int, np.ndarray] = {}
        for r in records:
            arr = np.asarray(r.composite_after, dtype=np.float32)
            if r.speaker == name_a:
                a_affect_at[r.turn] = arr
            else:
                b_affect_at[r.turn] = arr

        last_a = np.zeros(12, dtype=np.float32)
        last_b = np.zeros(12, dtype=np.float32)
        for cp in checkpoints:
            for k, v in sorted(a_affect_at.items()):
                if k <= cp:
                    last_a = v
            for k, v in sorted(b_affect_at.items()):
                if k <= cp:
                    last_b = v
            sim = cosine(last_a, last_b)
            print(f"    turn {cp:>3d}  cos = {sim:+.3f}    "
                  f"‖A‖={np.linalg.norm(last_a):.3f}  ‖B‖={np.linalg.norm(last_b):.3f}")

    # Concepts written this conversation that came from the other mind. We
    # detect this via the graph's `refers_to` edges from the other agent's
    # concept — every world-source ingest that resulted in a write or match
    # leaves such an edge.
    def from_other(loop: MainLoop, other_agent_handle: str) -> list[tuple[int, str]]:
        h = loop.input_pipeline
        agent_cid = h.agent_id_for(other_agent_handle)
        if agent_cid is None:
            return []
        from backend.graph import EdgeType
        out: list[tuple[int, str]] = []
        for e in loop.graph.get_edges(agent_cid, type=EdgeType.REFERS_TO, direction="out"):
            n = loop.graph.nodes.get(e.target_id)
            if n is None:
                continue
            out.append((e.target_id, n.name[:60]))
        return out

    a_from_b = from_other(a_loop, name_b)
    b_from_a = from_other(b_loop, name_a)
    print()
    print(f"  {name_a:>20s} concepts referred from {name_b}: {len(a_from_b)}")
    for cid, cname in a_from_b[:10]:
        print(f"      cid={cid:<5d}  {cname!r}")
    print()
    print(f"  {name_b:>20s} concepts referred from {name_a}: {len(b_from_a)}")
    for cid, cname in b_from_a[:10]:
        print(f"      cid={cid:<5d}  {cname!r}")
    print()
    print("═" * width)


# ============================================================
# Driver
# ============================================================

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("name_a", help="first mind name")
    ap.add_argument("name_b", help="second mind name")
    ap.add_argument("--turns", type=int, default=DEFAULT_TURNS)
    ap.add_argument("--seed-prompt", default=SEED_PROMPT_DEFAULT)
    ap.add_argument("--save", action="store_true",
                    help="persist both minds' state changes after the conversation")
    args = ap.parse_args()

    paths_a, loop_a = load_mind(args.name_a)
    paths_b, loop_b = load_mind(args.name_b)

    # Each mind registers the OTHER as an agent in its own input pipeline.
    agent_a_in_b = loop_b.input_pipeline.register_agent(args.name_a, now=time.time())
    agent_b_in_a = loop_a.input_pipeline.register_agent(args.name_b, now=time.time())

    # Capture baselines so the summary can report deltas honestly.
    nodes_before_a = loop_a.graph.node_count
    nodes_before_b = loop_b.graph.node_count
    edges_before_a = loop_a.graph.edge_count
    edges_before_b = loop_b.graph.edge_count

    print_header(args.name_a, args.name_b, args.turns, args.seed_prompt)

    # Conversation log.
    records: list[TurnRecord] = []
    os.makedirs(CONVERSATION_DIR, exist_ok=True)
    log_path = os.path.join(
        CONVERSATION_DIR,
        f"{args.name_a}__{args.name_b}__{int(time.time())}.jsonl",
    )

    # Turn 0: A receives the seed prompt and speaks. The seed comes from
    # outside both minds (it's the experimenter's prompt), so we attribute
    # it to A's agent_b record — A treats it as if B said it. This gives B
    # a coherent agent_handle to be associated with from the very first
    # exchange.
    record = take_turn(
        turn=0,
        speaker_loop=loop_a,
        speaker_name=args.name_a,
        listener_name=args.name_b,
        heard_utterance=args.seed_prompt,
        other_agent_id=agent_b_in_a,
    )
    records.append(record)
    print_turn(record)
    last_utterance = record.spoken_utterance

    # Subsequent turns: alternate.
    speakers = [
        (loop_b, args.name_b, args.name_a, agent_a_in_b),
        (loop_a, args.name_a, args.name_b, agent_b_in_a),
    ]
    side = 0
    for t in range(1, args.turns):
        speaker_loop, speaker_name, listener_name, other_agent_id = speakers[side]
        if not last_utterance:
            # If a mind went silent, give the next mind a sentinel so the
            # conversation can recover rather than stall.
            last_utterance = "(silence)"
        record = take_turn(
            turn=t,
            speaker_loop=speaker_loop,
            speaker_name=speaker_name,
            listener_name=listener_name,
            heard_utterance=last_utterance,
            other_agent_id=other_agent_id,
        )
        records.append(record)
        print_turn(record)
        last_utterance = record.spoken_utterance
        side = 1 - side

    # Write JSONL log.
    with open(log_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps({
                "turn": r.turn, "speaker": r.speaker, "listener": r.listener,
                "heard": r.heard_utterance, "said": r.spoken_utterance,
                "was_surprised": r.was_surprised, "was_new_write": r.was_new_write,
                "concept_id": r.concept_id, "concept_name": r.concept_name,
                "decision_type": r.decision_type,
                "arousal_after": r.arousal_after,
                "composite_norm_after": r.composite_norm_after,
                "composite_after": r.composite_after,
            }) + "\n")
    print(f"\n  → transcript: {log_path}")

    print_summary(
        records, args.name_a, args.name_b, loop_a, loop_b,
        nodes_before_a=nodes_before_a, nodes_before_b=nodes_before_b,
        edges_before_a=edges_before_a, edges_before_b=edges_before_b,
    )

    if args.save:
        MindPersistence(paths_a.db).save(loop_a, now=time.time())
        MindPersistence(paths_b.db).save(loop_b, now=time.time())
        print(f"  saved both minds back to disk")

    return 0


if __name__ == "__main__":
    sys.exit(main())
