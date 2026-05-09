"""Input Pipeline + World Interface (Component H).

H is the boundary that makes the rest of the mind work. Bytes / arrays /
internal vectors come in, encoded representations come out, and every
modality enters the same predict→observe handshake. There is no special
path for any modality — internal thought is processed identically to
external input.

What exists in this Phase 3.5 H-complete:
  - Encoder registry (text, image, audio) with versioned IDs and a
    deterministic registry lookup. New encoders register; old ones
    persist for restoration of older concepts (encoder versioning is
    structural, swap rotation lands later).
  - Three real encoders (no pretrained weights, no stubs):
      • text  : char-trigram BoW with hashed signed buckets → 256-d unit
      • image : 16×16 mean-pool grayscale → 256-d unit
      • audio : 256-band linear-magnitude FFT → 256-d unit
  - Stimulus dataclass with full provenance (stimulus_id, tick,
    t_arrival, t_encoded, modality, source, agent_id, encoder_id,
    cycle_depth, parent_stimulus_id).
  - Self/world boundary: Source enum is deterministic from Modality.
    External (TEXT/IMAGE/AUDIO) → WORLD; INTERNAL_* → SELF. The bit
    rides on every Stimulus and is preserved downstream.
  - ingest_text / ingest_image / ingest_audio — the world entry points.
  - inject_internal — the self entry point. G's output-echo and (later)
    D's replay/simulation re-enter the same predict→observe pipeline
    through this rather than calling B directly.
  - emit_output — G commits a chosen surface here. Two side effects
    (locked per T6): the surface is logged to the world, and the same
    surface is re-encoded and re-injected as INTERNAL_OUTPUT_ECHO so
    the system overhears itself.
  - Agent registry: register_agent allocates an agent-concept via the
    same anchor mechanism E uses for self/unknown; attribute_to_agent
    auto-tags any concept written/matched on a WORLD ingest with a
    `refers_to` edge from the speaker's agent-concept (locked per T8).

Backwards compatibility: ingest_text returns an IngestResult that
preserves the Phase 1/2 fields (stimulus_id, text, representation_norm,
prediction, gap) as properties forwarding to the embedded Stimulus.
phase1.py continues to pass without modification.

Deferred:
  - Real bytes-decoding (PIL/wave) — Phase 3 H takes ndarray inputs for
    image and audio. Decoding from raw bytes is a future concern.
  - Encoder swap (HOT_REPROJECT and friends) — registry tracks versions
    but the rebuild pass is later.
  - IngestQueue / async tick path — Phase 3 H is synchronous; the
    MainLoop drives ingest directly.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import numpy as np

from backend.affect import AffectStack
from backend.config import D_REP
from backend.graph import ConceptGraph, Edge, EdgeType
from backend.predict import Prediction, PredictionEngine, PredictionGap


# ============================================================
# Modality / Source — the self/world boundary, deterministic
# ============================================================

class Modality(Enum):
    TEXT                = "text"
    IMAGE               = "image"
    AUDIO               = "audio"
    INTERNAL_REPLAY     = "internal_replay"
    INTERNAL_SIMULATION = "internal_simulation"
    INTERNAL_THOUGHT    = "internal_thought"
    INTERNAL_OUTPUT_ECHO = "internal_output_echo"


class Source(Enum):
    WORLD = "world"
    SELF  = "self"


_SOURCE_FROM_MODALITY: dict[Modality, Source] = {
    Modality.TEXT:                 Source.WORLD,
    Modality.IMAGE:                Source.WORLD,
    Modality.AUDIO:                Source.WORLD,
    Modality.INTERNAL_REPLAY:      Source.SELF,
    Modality.INTERNAL_SIMULATION:  Source.SELF,
    Modality.INTERNAL_THOUGHT:     Source.SELF,
    Modality.INTERNAL_OUTPUT_ECHO: Source.SELF,
}


def source_for(modality: Modality) -> Source:
    return _SOURCE_FROM_MODALITY[modality]


# ============================================================
# Encoders — text encoders live in backend/encoders.py to keep this
# file focused on the H runtime; image and audio encoders are inlined
# below since they're small and have no large dependencies.
# ============================================================

from backend.encoders import (
    ENCODER_TEXT_CHAR_TRIGRAM,
    ENCODER_TEXT_GLOVE,
    encode_text_char_trigram,
    encode_text_glove,
    glove_is_available,
)

# Phase-5 active text encoder is GloVe when available, else fall back to
# the Phase 1 char-trigram encoder. Both stay in the registry; only the
# active one is used for new ingests.
ENCODER_TEXT_ID  = ENCODER_TEXT_GLOVE if glove_is_available() else ENCODER_TEXT_CHAR_TRIGRAM
ENCODER_IMAGE_ID = "image:patch-stats-v1"
ENCODER_AUDIO_ID = "audio:fft-stats-v1"


def encode_text(text: str, dim: int = D_REP) -> np.ndarray:
    """Active text encoder. Phase 5: GloVe-meanpool-PCA256 if its binary
    is on disk; Phase 1 char-trigram BoW otherwise. Both produce a
    deterministic L2-normalized float32[D_REP] (or zero vector for empty
    / all-unknown input).
    """
    if ENCODER_TEXT_ID == ENCODER_TEXT_GLOVE:
        return encode_text_glove(text, dim=dim)
    return encode_text_char_trigram(text, dim=dim)


def encode_text_batch(texts: list[str], dim: int = D_REP) -> np.ndarray:
    """Batch text encoder. Returns (B, D) np.ndarray of L2-normalized
    rows. Used by encode_corpus.py to amortize the GPU dispatch (post
    Section 5: GloVe matrix lives on MPS, the batch is a single matmul).
    Falls back to a per-item loop today; same shape contract holds.
    """
    from backend.encoders import encode_text_glove_batch, encode_text_char_trigram_batch
    if ENCODER_TEXT_ID == ENCODER_TEXT_GLOVE:
        return encode_text_glove_batch(texts, dim=dim)
    return encode_text_char_trigram_batch(texts, dim=dim)


def encode_image(image: np.ndarray, dim: int = D_REP) -> np.ndarray:
    """Patch-stats encoder: mean-pool to 16×16, flatten, L2-normalize.

    Accepts grayscale (H, W) or RGB (H, W, 3). RGB is converted to
    luminance first (standard 0.299/0.587/0.114). The resulting 256-d
    unit vector is deterministic per input. dim must be a perfect square
    so the mean-pool grid divides evenly; D_REP=256 = 16×16 is the
    canonical case.
    """
    side = int(round(dim ** 0.5))
    if side * side != dim:
        raise ValueError(f"encode_image needs dim = perfect square, got {dim}")

    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[2] == 3:
        # RGB → luminance.
        arr = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    elif arr.ndim != 2:
        raise ValueError(f"encode_image needs (H,W) or (H,W,3), got shape {arr.shape}")

    h, w = arr.shape
    if h < side or w < side:
        raise ValueError(f"encode_image needs at least {side}×{side}, got {h}×{w}")

    # Mean-pool by integer-block grouping. We crop to the largest h, w that
    # divide evenly by `side` so the pool is exact (no fractional blocks).
    bh, bw = h // side, w // side
    arr = arr[:bh * side, :bw * side]
    arr = arr.reshape(side, bh, side, bw).mean(axis=(1, 3))
    vec = arr.flatten().astype(np.float32)
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm >= 1e-12 else vec


def encode_audio(waveform: np.ndarray, dim: int = D_REP) -> np.ndarray:
    """Linear-band FFT-magnitude encoder: rfft → bin into `dim` bands →
    L2-normalize. Real, no pretrained weights, deterministic.

    Accepts a 1-D float waveform. Stereo / multi-channel audio should be
    averaged to mono before passing in.
    """
    arr = np.asarray(waveform, dtype=np.float32).ravel()
    if arr.size < 2:
        return np.zeros(dim, dtype=np.float32)
    spec = np.abs(np.fft.rfft(arr))
    if spec.size < dim:
        # Pad with zeros so binning produces a deterministic dim-length vector.
        spec = np.concatenate([spec, np.zeros(dim - spec.size, dtype=np.float32)])
    # Linear binning: equal-width bands across the spectrum.
    edges = np.linspace(0, spec.size, dim + 1, dtype=np.int64)
    vec = np.zeros(dim, dtype=np.float32)
    for i in range(dim):
        lo, hi = edges[i], max(edges[i] + 1, edges[i + 1])
        vec[i] = float(spec[lo:hi].mean())
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm >= 1e-12 else vec


@dataclass
class EncoderEntry:
    encoder_id: str
    modality: Modality
    encode: Callable[..., np.ndarray]   # (input, dim=D_REP) → ndarray[D_REP]


class EncoderRegistry:
    """Deterministic registry of encoders, keyed by encoder_id.

    Phase 3 H ships three: text, image, audio. The registry is the
    single source of truth for which encoder produced any given concept's
    embedding (synthesis-locked encoder_id field, deferred on ConceptNode
    but tracked on Stimulus). Encoder swap and HOT_REPROJECT land later
    by adding new entries with bumped version IDs and walking the graph.
    """

    def __init__(self) -> None:
        self._by_id: dict[str, EncoderEntry] = {}
        self._active_per_modality: dict[Modality, str] = {}

        # Both text encoders are registered. The active one is whichever
        # ENCODER_TEXT_ID points at — GloVe when its binary is on disk,
        # char-trigram otherwise. The inactive one stays addressable by
        # encoder_id so concepts written under it remain explainable.
        if ENCODER_TEXT_ID == ENCODER_TEXT_GLOVE:
            self.register(EncoderEntry(ENCODER_TEXT_GLOVE, Modality.TEXT, encode_text_glove), active=True)
            self.register(EncoderEntry(ENCODER_TEXT_CHAR_TRIGRAM, Modality.TEXT, encode_text_char_trigram), active=False)
        else:
            self.register(EncoderEntry(ENCODER_TEXT_CHAR_TRIGRAM, Modality.TEXT, encode_text_char_trigram), active=True)

        self.register(EncoderEntry(ENCODER_IMAGE_ID, Modality.IMAGE, encode_image), active=True)
        self.register(EncoderEntry(ENCODER_AUDIO_ID, Modality.AUDIO, encode_audio), active=True)

    def register(self, entry: EncoderEntry, active: bool = False) -> None:
        if entry.encoder_id in self._by_id:
            raise ValueError(f"encoder_id collision: {entry.encoder_id}")
        self._by_id[entry.encoder_id] = entry
        if active:
            self._active_per_modality[entry.modality] = entry.encoder_id

    def active_for(self, modality: Modality) -> EncoderEntry:
        eid = self._active_per_modality.get(modality)
        if eid is None:
            raise KeyError(f"no active encoder for modality {modality.value}")
        return self._by_id[eid]

    def by_id(self, encoder_id: str) -> EncoderEntry:
        if encoder_id not in self._by_id:
            raise KeyError(f"no encoder with id {encoder_id!r}")
        return self._by_id[encoder_id]

    def known_ids(self) -> list[str]:
        return sorted(self._by_id.keys())


# ============================================================
# Stimulus + provenance
# ============================================================

@dataclass
class ProvenanceRecord:
    """Per-stimulus origin trail. Phase 3 minimal: enough for cycle-depth
    accounting and parent-of-echo tracking. Audit chains across replay/
    simulation extend this when those callers wire through inject_internal.
    """
    origin: str = "world"            # "world" | "output_echo" | "replay" | "simulation" | "self"
    parent_stimulus_id: int | None = None
    cycle_depth: int = 0


@dataclass
class Stimulus:
    """One ingest event. Built by H, consumed by everything downstream."""
    stimulus_id: int
    tick: int
    t_arrival: float
    t_encoded: float
    modality: Modality
    source: Source
    agent_id: int | None
    representation: np.ndarray            # float32[D_REP]
    representation_norm: float
    encoder_id: str
    name_hint: str
    provenance: ProvenanceRecord


@dataclass
class IngestResult:
    """Returned by every ingest_* call. Carries the Stimulus, the
    prediction made for it, and the gap that fell out. `refers_to_edge`
    is populated when a WORLD ingest with an agent_id auto-tagged the
    matched/written concept (synthesis-locked T8).

    Phase 1/2 backward-compat: stimulus_id, text, representation_norm
    are exposed as properties forwarding to the Stimulus.
    """
    stimulus: Stimulus
    prediction: Prediction
    gap: PredictionGap
    refers_to_edge: Edge | None = None

    @property
    def stimulus_id(self) -> int: return self.stimulus.stimulus_id
    @property
    def text(self) -> str: return self.stimulus.name_hint
    @property
    def representation_norm(self) -> float: return self.stimulus.representation_norm


# ============================================================
# Agents
# ============================================================

@dataclass
class AgentRecord:
    agent_concept_id: int
    handle: str
    registered_at: float
    last_seen_t: float


# ============================================================
# InputPipeline — the H runtime
# ============================================================

class InputPipeline:
    """Component H. Synchronous: callers invoke ingest_* directly and
    receive an IngestResult after the predict→observe handshake completes.

    Construction:
      - affect, graph, predict_engine: the substrate.
      - identity (optional): when wired, agent registration uses E's
        pin_concept (so OTHER_AGENT pins are recorded in the spine);
        attribute_to_agent uses E.unknown_concept_id as the default
        anchor. When None, H still works (Phase 1 paths) but agent
        attribution is disabled.
    """

    def __init__(
        self,
        *,
        affect: AffectStack,
        graph: ConceptGraph,
        predict_engine: PredictionEngine,
        identity: "Any | None" = None,    # forward-typed Identity
    ) -> None:
        self.affect = affect
        self.graph = graph
        self.predict_engine = predict_engine
        self.identity = identity

        self.encoders = EncoderRegistry()

        self._stimulus_id: int = 0
        self._tick: int = 0

        # Agent registry. Keyed by agent_concept_id (the concept allocated
        # in C for the agent). Handle-name index for fast lookup.
        self._agents: dict[int, AgentRecord] = {}
        self._agent_handle_index: dict[str, int] = {}

        # Recently emitted output surfaces — tail log for diagnostics.
        # Real "delivery to the world" is an external concern; H records
        # the emission and re-injects it for the self-overhearing loop.
        self.emitted_log: list[tuple[float, str, str]] = []   # (t, modality, surface)

    # ---- agents ----

    def register_agent(self, handle: str, now: float = 0.0) -> int:
        """Register an external agent. Allocates an agent-concept in C,
        pins it via E if available, registers in the agent table.
        Idempotent on handle: re-registering returns the existing id.
        """
        if handle in self._agent_handle_index:
            return self._agent_handle_index[handle]

        # Allocate via the same anchor mechanism E uses for self/unknown.
        # Deterministic embedding so re-registration with the same handle
        # under the same conditions produces a stable identity.
        rng = np.random.default_rng(abs(hash(("agent", handle))) % (2 ** 32))
        emb = rng.normal(0.0, 1.0, size=D_REP).astype(np.float32)
        n = float(np.linalg.norm(emb))
        if n > 1e-9:
            emb /= n
        composite = self.affect.composite(now)

        cid, _ = self.graph.write_on_surprise(
            representation=emb,
            surprise=0.0,
            current_affect=composite,
            name_hint=f"agent:{handle}",
            now=now,
        )

        # Pin: prefer E's PinReason record; fall back to C's bare pin.
        if self.identity is not None:
            from backend.identity import PinReason
            self.identity.pin_concept(cid, PinReason.OTHER_AGENT, now=now, note=f"agent {handle}")
        else:
            self.graph.pin(cid, reason="other_agent")

        self._agents[cid] = AgentRecord(
            agent_concept_id=cid,
            handle=handle,
            registered_at=now,
            last_seen_t=now,
        )
        self._agent_handle_index[handle] = cid
        return cid

    def agent_id_for(self, handle: str) -> int | None:
        return self._agent_handle_index.get(handle)

    @property
    def agent_count(self) -> int:
        return len(self._agents)

    def query_shared_knowledge(self, agent_id: int, k: int = 10) -> list[int]:
        """Concepts the agent has been linked to via refers_to. Phase 3
        minimal: returns refers_to-edge targets ordered by edge weight.
        Caller (E, D) uses this for audience-belief queries.
        """
        if agent_id not in self._agents:
            return []
        edges = self.graph.get_edges(agent_id, type=EdgeType.REFERS_TO, direction="out")
        edges.sort(key=lambda e: -e.weight)
        return [e.target_id for e in edges[:k]]

    # ---- ingest paths ----

    def ingest_text(
        self,
        text: str,
        now: float,
        agent_id: int | None = None,
        representation: np.ndarray | None = None,
        encoder_id: str | None = None,
    ) -> IngestResult:
        """Ingest a text. Phase 5: an optional pre-encoded representation
        can be passed in (from data/encoded_corpus.db) to skip encoding
        entirely. When supplied, encoder_id should also be supplied so
        the Stimulus carries the correct provenance; if omitted, the
        active text encoder's id is stamped.
        """
        if representation is None:
            encoder = self.encoders.active_for(Modality.TEXT)
            rep = encoder.encode(text, dim=D_REP)
            stamped_id = encoder.encoder_id
        else:
            if representation.shape != (D_REP,):
                raise ValueError(
                    f"pre-encoded representation must be ({D_REP},), got {representation.shape}"
                )
            rep = representation.astype(np.float32, copy=False)
            stamped_id = encoder_id or self.encoders.active_for(Modality.TEXT).encoder_id
        return self._dispatch(
            representation=rep,
            modality=Modality.TEXT,
            encoder_id=stamped_id,
            name_hint=text[:48],
            agent_id=agent_id,
            now=now,
            origin="world",
        )

    def ingest_image(
        self,
        image: np.ndarray,
        now: float,
        agent_id: int | None = None,
        name_hint: str = "",
    ) -> IngestResult:
        encoder = self.encoders.active_for(Modality.IMAGE)
        rep = encoder.encode(image, dim=D_REP)
        return self._dispatch(
            representation=rep,
            modality=Modality.IMAGE,
            encoder_id=encoder.encoder_id,
            name_hint=name_hint or f"image[{image.shape}]",
            agent_id=agent_id,
            now=now,
            origin="world",
        )

    def ingest_audio(
        self,
        waveform: np.ndarray,
        now: float,
        agent_id: int | None = None,
        name_hint: str = "",
    ) -> IngestResult:
        encoder = self.encoders.active_for(Modality.AUDIO)
        rep = encoder.encode(waveform, dim=D_REP)
        return self._dispatch(
            representation=rep,
            modality=Modality.AUDIO,
            encoder_id=encoder.encoder_id,
            name_hint=name_hint or f"audio[{waveform.size} samples]",
            agent_id=agent_id,
            now=now,
            origin="world",
        )

    def inject_internal(
        self,
        payload: np.ndarray,
        modality: Modality,
        now: float,
        *,
        layer: str = "INPUT",
        name_hint: str = "",
        agent_id: int | None = None,
        parent_stimulus_id: int | None = None,
        replay_origin: bool = False,
    ) -> IngestResult:
        """Self-side entry into the predict→observe pipeline. Validates
        dim and norm, builds a Stimulus with source=SELF, runs through
        the same dispatch path as world ingests.

        Used by G's emit_output (which auto-injects the surface as
        INTERNAL_OUTPUT_ECHO so the system overhears itself), and by any
        future caller (D's replay, E's self-thought) that wants the
        Stimulus provenance trail rather than calling B directly.
        """
        if source_for(modality) is not Source.SELF:
            raise ValueError(f"inject_internal requires INTERNAL_* modality, got {modality}")
        if payload.shape != (D_REP,):
            raise ValueError(f"payload must be ({D_REP},), got {payload.shape}")
        return self._dispatch(
            representation=payload.astype(np.float32, copy=True),
            modality=modality,
            encoder_id="<internal>",
            name_hint=name_hint or modality.value,
            agent_id=agent_id,
            now=now,
            origin=_origin_for_internal(modality),
            parent_stimulus_id=parent_stimulus_id,
            cycle_depth=1 if parent_stimulus_id is not None else 0,
            layer=layer,
            replay_origin=replay_origin,
        )

    # ---- emit (G calls this after E commits a candidate) ----

    def emit_output(
        self,
        modality: Modality,
        surface: str,
        now: float,
        *,
        parent_stimulus_id: int | None = None,
    ) -> IngestResult:
        """G commits a chosen surface here. Two side effects (T6 locked):
          1. The surface is appended to `emitted_log` (the world delivery
             surface; Phase 3 has no frontend, so logging is the world).
          2. The same surface is re-encoded and re-injected as
             INTERNAL_OUTPUT_ECHO so B fires INPUT-layer surprise on the
             system's own output. This is what makes "the gap is a choice"
             observable — the system feels what it actually said.
        """
        if modality is not Modality.TEXT:
            raise NotImplementedError("Phase 3 H supports text emission only")
        self.emitted_log.append((float(now), modality.value, surface))

        encoder = self.encoders.active_for(modality)
        echo_rep = encoder.encode(surface, dim=D_REP)
        return self.inject_internal(
            payload=echo_rep,
            modality=Modality.INTERNAL_OUTPUT_ECHO,
            now=now,
            name_hint=f"echo:{surface[:40]}",
            parent_stimulus_id=parent_stimulus_id,
        )

    # ---- internal: the single dispatch path all ingests share ----

    def _dispatch(
        self,
        *,
        representation: np.ndarray,
        modality: Modality,
        encoder_id: str,
        name_hint: str,
        agent_id: int | None,
        now: float,
        origin: str,
        parent_stimulus_id: int | None = None,
        cycle_depth: int = 0,
        layer: str = "INPUT",
        replay_origin: bool = False,
    ) -> IngestResult:
        self._stimulus_id += 1
        self._tick += 1

        norm = float(np.linalg.norm(representation))
        stim = Stimulus(
            stimulus_id=self._stimulus_id,
            tick=self._tick,
            t_arrival=float(now),
            t_encoded=float(now),
            modality=modality,
            source=source_for(modality),
            agent_id=agent_id,
            representation=representation,
            representation_norm=norm,
            encoder_id=encoder_id,
            name_hint=name_hint,
            provenance=ProvenanceRecord(
                origin=origin,
                parent_stimulus_id=parent_stimulus_id,
                cycle_depth=cycle_depth,
            ),
        )

        # Predict → Observe through B. B routes side-effects to A
        # (always) and C (only on surprise), and auto-pushes to D's
        # replay buffer for non-replay surprises.
        prediction = self.predict_engine.predict(representation, layer=layer)
        gap = self.predict_engine.observe(
            prediction=prediction,
            actual=representation,
            now=now,
            name_hint=name_hint,
            replay_origin=replay_origin,
        )

        # Auto-tag with refers_to from agent_concept → matched/written concept,
        # but only for WORLD stimuli (synthesis T8 — incoming utterances).
        # Internal stimuli (replay, output-echo) carry no agent attribution.
        refers_edge: Edge | None = None
        if stim.source is Source.WORLD and gap.concept_id is not None:
            anchor = self._anchor_agent_id(agent_id)
            if anchor is not None and anchor != gap.concept_id:
                # Idempotent add; strengthen-on-repeat would otherwise grow
                # weight but Phase 3 minimal keeps the relationship simple.
                refers_edge = self.graph.add_edge(
                    source_id=anchor,
                    target_id=gap.concept_id,
                    type=EdgeType.REFERS_TO,
                    weight=0.5,
                    now=now,
                )
                # Update agent's last_seen.
                rec = self._agents.get(anchor)
                if rec is not None:
                    rec.last_seen_t = float(now)

        return IngestResult(stimulus=stim, prediction=prediction, gap=gap, refers_to_edge=refers_edge)

    def _anchor_agent_id(self, agent_id: int | None) -> int | None:
        """Resolve the agent_concept_id to attach refers_to edges from.
        Falls back to E's unknown_concept_id when no explicit agent and E
        is wired. Returns None if neither is available — callers then
        skip auto-tagging entirely (Phase 1 path).
        """
        if agent_id is not None:
            return agent_id
        if self.identity is not None:
            return self.identity.spine.unknown_concept_id
        return None


def _origin_for_internal(modality: Modality) -> str:
    return {
        Modality.INTERNAL_REPLAY:      "replay",
        Modality.INTERNAL_SIMULATION:  "simulation",
        Modality.INTERNAL_THOUGHT:     "self",
        Modality.INTERNAL_OUTPUT_ECHO: "output_echo",
    }.get(modality, "self")
