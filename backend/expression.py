"""Expression Layer (Component G). Phase 3 minimal — text only.

G is weight-free. It owns a small StyleState (EMAs and preferences) and
a seed inventory of templates that live as concept nodes in C. Generation
happens by selecting templates, slot-filling them from the F-active set,
re-encoding each candidate via H, and calling B at the OUTPUT layer to
fire the output-trigger affect injection. E then decides which candidate
(if any) to commit.

What exists in Phase 3 minimal:
  - StyleState: lexical_register, verbosity_mean, modality_preference,
    revision_temperament, template_familiarity. EMA-updated by commit
    feedback.
  - SeedTemplate + a small inventory loaded at boot. Each template is
    created in C as a ConceptNode (abstraction_level=1) and pinned by G
    via E so the forget loop can never reap them. G keeps a side-table
    mapping template concept_id → metadata (pattern, pragmatic_function,
    slot keys), avoiding a ConceptNode field addition for a G-private read.
  - ExpressionIntent (caller-supplied input).
  - generate_candidates(intent) → list[CandidateExpression]: for each
    template, picks top active concepts as fillers, renders text, encodes
    via H to get surface_repr, packages a CandidateExpression for E.
  - render_text(template, fillers) → str.
  - request_expression(intent) → ExpressionDecision: full pipeline —
    generate candidates, fire OUTPUT trigger per candidate via B.observe,
    snapshot E's private state, call E.decide_expression, return decision.
  - attach_surface_to_concept(cid, surface_text, modality, affect, now):
    encodes the surface, find-or-creates a surface-form concept, lays an
    `expresses` edge from cid to it. Real implementation; not a stub.
  - style_snapshot(): read-only view for E and frontend.

Deferred (no caller in Phase 3 G):
  - Image and audio renderers. Synthesis allows stubs at v1; we omit
    them entirely so future work has nothing pretending-to-work to delete.
  - Style evolution via on_commit feedback (the hook exists; full EMA
    update on the daily-half-life cadence lands when there is a daily
    cadence to drive it).
  - simulate_audience_response integration (D's audience-prediction skeleton
    exists; G uses internal_repr/surface_repr distance only for now).
  - Modality leakiness scalar L_m as post-commit affect amplifier (text
    is the only modality so leakiness ≡ leakiness_text=0.20 and is folded
    into the score via the existing expression_gap; richer use waits for
    image/audio).
  - REVISION loop. E may return RevisionRequest; G surfaces it but does
    not auto-iterate (would require E to ask G for "revised candidates"
    given prior feedback, a Phase 3+ refinement).
"""
from __future__ import annotations

import os
import sys
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from backend.affect import AffectStack
from backend.config import D_REP, N_AFF
from backend.graph import ConceptGraph, EdgeType

if TYPE_CHECKING:
    # Late-bound to avoid the circular Expression ←→ Attention import.
    from backend.attention import Attention
from backend.identity import (
    CandidateExpression,
    ChosenCandidate,
    ExpressionDecision,
    Identity,
    PinReason,
    RevisionRequest,
    SuppressionRequest,
)
from backend.input import InputPipeline, encode_text
from backend.predict import PredictionEngine

# Language head is optional — imported eagerly so the type/log surface is
# available, but actual model loading is lazy and silent (returns None on
# failure so G can fall back to templates).
try:
    import torch
    from backend.language_head import (
        CONDITIONED_DECODER_FILENAME,
        CONDITIONED_DECODER_V2_FILENAME,
        ConditionedDecoder,
        DEFAULT_VOCAB_PATH,
        DEFAULT_WEIGHTS_PATH,
        LanguageHead,
        Vocab,
        build_conditioning_vector,
        is_available as language_head_available,
        is_conditioned_decoder_available,
        load_checkpoint,
        render_tokens,
    )
    _TORCH_OK = True
except Exception:
    _TORCH_OK = False
    LanguageHead = None         # type: ignore
    ConditionedDecoder = None   # type: ignore
    Vocab = None                # type: ignore


# Phase 3 minimal G constants. Synthesis symbol table values.
N_CANDIDATES_DEFAULT = 4
MODALITY_LEAK_TEXT   = 0.20
DEFAULT_VERBOSITY    = 12.0


# Seed template inventory. Each entry: pattern with one {phrase} slot, plus
# a pragmatic-function tag. The pattern is rendered by simple substitution
# of the top active concept's `name`. This is honestly text-shaped output
# from the system's stored memories — not parsing/grammar work, which is a
# different concern. When a Phase 4+ parser/lexicon exists, slot-filling
# can fan out across multi-slot templates with sub-concept fillers.
_SEED_TEMPLATES: list[tuple[str, str, str]] = [
    # (template_id, pattern, pragmatic_function)
    ("tpl.raw",      "{phrase}",                "statement"),
    ("tpl.recall",   "i remember {phrase}",     "recall"),
    ("tpl.question", "is {phrase}?",            "question"),
    ("tpl.affirm",   "yes, {phrase}",           "affirm"),
    ("tpl.deny",     "no, {phrase}",            "deny"),
    ("tpl.partial",  "{phrase}, perhaps",       "hedged"),
]


@dataclass
class TemplateMeta:
    """G-side metadata for a template concept. Kept alongside the
    ConceptNode to avoid adding a `pragmatic_function` field to C's
    schema for a G-private read.
    """
    template_id: str
    concept_id: int
    pattern: str
    pragmatic_function: str
    slot_keys: list[str]


@dataclass
class StyleState:
    """G's only persistent state. EMAs and preferences that drift with
    commit feedback. Phase 3 minimal: structure exists, evolution rule
    arrives with the on_commit hook in a later phase.
    """
    verbosity_mean: float = DEFAULT_VERBOSITY
    modality_preference: list[float] = field(
        default_factory=lambda: [0.6, 0.2, 0.2]   # [text, image, audio]
    )
    revision_temperament: float = 0.5
    template_familiarity: dict[int, float] = field(default_factory=dict)
    # Modality leak scalars — synthesis-locked. text is the only one
    # exercised in Phase 3 minimal.
    modality_leak: dict[str, float] = field(
        default_factory=lambda: {"text": MODALITY_LEAK_TEXT, "image": 0.50, "audio": 0.85}
    )


@dataclass
class ExpressionIntent:
    """Caller-supplied input to request_expression. The main loop / a test
    harness builds these from the F-active set and current affect.
    """
    intent_id: str
    internal_repr: np.ndarray             # D_REP — what the system "wanted to say"
    active_concepts: dict[int, float]     # F-active set, cid → activation
    now: float
    audience_concept_id: int | None = None


class Expression:
    """Component G — text-only expression for Phase 3 minimal."""

    def __init__(
        self,
        *,
        affect: AffectStack,
        graph: ConceptGraph,
        predict_engine: PredictionEngine,
        identity: Identity,
        input_pipeline: InputPipeline,
        skip_boot: bool = False,
        lm_weights_path: str | None = None,
        lm_vocab_path: str | None = None,
    ) -> None:
        self.affect = affect
        self.graph = graph
        self.predict_engine = predict_engine
        self.identity = identity
        self.input_pipeline = input_pipeline

        self.style = StyleState()
        # template_concept_id → TemplateMeta. Bidirectional lookup as needed.
        self._templates: dict[int, TemplateMeta] = {}
        self._template_by_id: dict[str, TemplateMeta] = {}

        # Language head — Phase 5. Loaded lazily so construction stays fast
        # and the file-on-disk check happens at runtime (training may finish
        # while the API is up, in which case the next request picks it up).
        # Per-mind paths can be passed in (Phase 6 multi-mind); when None,
        # falls back to the global defaults so legacy / single-mind setups
        # keep working unchanged.
        self._lm_weights_path = lm_weights_path or DEFAULT_WEIGHTS_PATH if _TORCH_OK else None
        self._lm_vocab_path   = lm_vocab_path   or DEFAULT_VOCAB_PATH   if _TORCH_OK else None
        # Phase 7: GPT-2 conditioned decoder lives next to the LSTM head
        # in each mind's dir. Tried first; falls back silently to LSTM,
        # which falls back silently to templates.
        #
        # The decoder filename used at load time is selected by
        # _try_load_conditioned_decoder: A3's `language_head_gpt2_v2.pt`
        # is preferred when present (longer fine-tune at lr=1e-5),
        # otherwise A2's `language_head_gpt2.pt`. _cd_weights_path holds
        # the path of the v1 file (compat with code that reads it directly,
        # e.g. test harnesses). _cd_root is the dir we search both files in.
        self._cd_root: str | None = None
        self._cd_weights_path: str | None = None
        if lm_weights_path:
            self._cd_root = os.path.dirname(lm_weights_path) or "."
            self._cd_weights_path = os.path.join(
                self._cd_root, CONDITIONED_DECODER_FILENAME,
            )
        self._language_head: "LanguageHead | None" = None
        self._conditioned_decoder: "ConditionedDecoder | None" = None
        self._lm_vocab: "Vocab | None" = None
        self._lm_device: str = (
            "mps" if (_TORCH_OK and torch.backends.mps.is_available())
            else "cpu"
        )

        # Phase 7 workstream B: late-bound Attention reference. MainLoop
        # injects this in its constructor (and on persistence reload).
        # generate_extended needs to spread from the self-overhearing echo;
        # rather than push attention through Expression's already-long ctor,
        # we accept that the runtime owner (MainLoop) wires it post-init.
        # Setting to None means generate_extended falls back to single-shot
        # behavior (no echo-driven re-attention between sentences).
        self.attention: "Attention | None" = None
        self._lm_load_attempted: bool = False
        self._cd_load_attempted: bool = False

        if not skip_boot:
            self._load_seed_templates()

    # ---- late-bound Attention reference (Phase 7 workstream B) ----

    def set_attention(self, attention: "Attention") -> None:
        """MainLoop calls this in __init__ (and after persistence reload)
        so generate_extended can request a PROCESSING-mode spread from the
        echo-written concept between sentences. Optional — when unset,
        generate_extended falls back to no echo-driven re-attention.
        """
        self.attention = attention

    # ---- language head (lazy load + on-demand reload) ----

    def _try_load_language_head(self) -> bool:
        """Idempotent: attempts to load on first call only. Returns True
        if a head is currently loaded. Silent failure on missing files
        or torch issues — the template path remains as fallback."""
        if self._language_head is not None:
            return True
        if self._lm_load_attempted:
            return False
        self._lm_load_attempted = True
        if not _TORCH_OK:
            return False
        if not (self._lm_weights_path and self._lm_vocab_path
                and os.path.exists(self._lm_weights_path)
                and os.path.exists(self._lm_vocab_path)):
            return False
        try:
            self._language_head = load_checkpoint(self._lm_weights_path, device=self._lm_device)
            self._lm_vocab = Vocab.load(self._lm_vocab_path)
            return True
        except Exception as exc:
            print(f"[expression] language head load failed: {exc}")
            self._language_head = None
            self._lm_vocab = None
            return False

    def _try_load_conditioned_decoder(self) -> bool:
        """Phase 7: try loading GPT-2 conditioned decoder. Idempotent.
        Silent failure if transformers isn't installed or the delta
        weights file is missing — Expression falls back to the LSTM head.

        Load priority (Phase 7 A3):
          1. <root>/language_head_gpt2_v2.pt   (lr=1e-5, +30 epochs)
          2. <root>/language_head_gpt2.pt      (lr=2e-5, 20 epochs)
        """
        if self._conditioned_decoder is not None:
            return True
        if self._cd_load_attempted:
            return False
        self._cd_load_attempted = True
        if not _TORCH_OK:
            return False
        if not self._cd_root:
            return False

        # v2-first preference. Each is independently checked for
        # `is_conditioned_decoder_available` (which validates that
        # transformers is importable and the file exists).
        candidates: list[tuple[str, str]] = []
        v2_path = os.path.join(self._cd_root, CONDITIONED_DECODER_V2_FILENAME)
        if is_conditioned_decoder_available(v2_path):
            candidates.append(("v2", v2_path))
        if self._cd_weights_path and is_conditioned_decoder_available(self._cd_weights_path):
            candidates.append(("v1", self._cd_weights_path))

        if not candidates:
            return False

        for label, path in candidates:
            try:
                print(f"[expression] loading {label} conditioned decoder from {path} …")
                cd = ConditionedDecoder()
                cd.load_delta(path)
                cd.to(self._lm_device)
                cd.eval()
                self._conditioned_decoder = cd
                print(f"[expression] conditioned decoder ready ({label}, device={self._lm_device})")
                return True
            except Exception as exc:
                print(f"[expression] {label} conditioned decoder load failed: {exc}")
                self._conditioned_decoder = None
                # Fall through to next candidate.
        return False

    def reload_language_head(self) -> bool:
        """Force a reload — useful when training finishes mid-session.
        Tries the conditioned decoder first; if that's missing, the
        LSTM head; if that too, returns False (template fallback only)."""
        self._language_head = None
        self._conditioned_decoder = None
        self._lm_vocab = None
        self._lm_load_attempted = False
        self._cd_load_attempted = False
        if self._try_load_conditioned_decoder():
            return True
        return self._try_load_language_head()

    # ---- template inventory ----

    def _load_seed_templates(self) -> None:
        """Create one ConceptNode per seed template, pin via E, record metadata.

        Each template gets a deterministic embedding seeded from its id,
        L2-normalized. The concept's affect_trace.birth_state is the
        current composite at boot. Templates have abstraction_level
        conceptually = 1; we'd write that field if C had it. For now
        we mark templates by their presence in G's _templates map.
        """
        composite = self.affect.composite(self.identity.spine.birth_time)
        for template_id, pattern, pragmatic_function in _SEED_TEMPLATES:
            seed = abs(hash(template_id))
            rng = np.random.default_rng(seed)
            emb = rng.normal(0.0, 1.0, size=D_REP).astype(np.float32)
            n = float(np.linalg.norm(emb))
            if n > 1e-9:
                emb /= n

            cid, _ = self.graph.write_on_surprise(
                representation=emb,
                surprise=0.0,
                current_affect=composite,
                name_hint=template_id,
                now=self.identity.spine.birth_time,
            )
            # Pin via E so the eventual forget loop respects it.
            self.identity.pin_concept(
                cid,
                PinReason.EXPRESSION_HABIT,
                now=self.identity.spine.birth_time,
                note=f"seed template: {pragmatic_function}",
            )

            slot_keys = _extract_slots(pattern)
            meta = TemplateMeta(
                template_id=template_id,
                concept_id=cid,
                pattern=pattern,
                pragmatic_function=pragmatic_function,
                slot_keys=slot_keys,
            )
            self._templates[cid] = meta
            self._template_by_id[template_id] = meta
            # Initialize familiarity uniformly so first-time template choice
            # is balanced.
            self.style.template_familiarity[cid] = 1.0

    @property
    def template_count(self) -> int:
        return len(self._templates)

    # ---- generation ----

    def render_text(self, template: TemplateMeta, fillers: dict[str, str]) -> str:
        """Substitute {slot} markers in the template pattern. Missing
        fillers cause this template to be skipped upstream — render_text
        itself is strict; missing keys raise KeyError so the caller knows.
        """
        return template.pattern.format(**fillers)

    def generate_candidates(
        self,
        intent: ExpressionIntent,
    ) -> list[CandidateExpression]:
        """Build candidates by pairing top active concepts with templates,
        OR by sampling from the language head when one is loaded.

        Phase 5: when the language head is available, this method
        delegates to `_generate_lm_candidates` which produces N
        independently-sampled completions from the trained GRU,
        conditioned on (current affect composite, top-5 active concept
        embeddings). Each completion becomes a CandidateExpression and
        is scored by E identically to a template candidate — the
        expression-gap math is unchanged.

        Phase 4 template path remains as fallback. With auto-linked
        similar_to edges feeding F.spread, the template path produces
        candidates from the active set's name fillers; with the LM,
        the head produces novel English from the same conditioning
        signal.
        """
        if not intent.active_concepts:
            return []

        # Phase 7: prefer the GPT-2 conditioned decoder when available.
        if self._try_load_conditioned_decoder():
            cd_candidates = self._generate_cd_candidates(intent)
            if cd_candidates:
                return cd_candidates
            # Fall through to LSTM head if the conditioned decoder
            # produced nothing (network blip, all-empty, etc.)

        if self._try_load_language_head():
            lm_candidates = self._generate_lm_candidates(intent)
            if lm_candidates:
                return lm_candidates
            # If LM yielded nothing usable (all empty / all <unk>), fall
            # through to templates rather than emit nothing.

        # Top active concepts by activation, drop templates and missing nodes.
        ranked_active = sorted(intent.active_concepts.items(), key=lambda kv: -kv[1])
        usable: list[tuple[int, str]] = []
        for cid, _ in ranked_active:
            if cid in self._templates:
                continue                     # don't fill a template with another template
            node = self.graph.nodes.get(cid)
            if node is None or not node.name:
                continue
            usable.append((cid, node.name))
        if not usable:
            return []

        # Templates ordered by familiarity (highest first).
        ordered_templates = sorted(
            self._templates.values(),
            key=lambda m: -self.style.template_familiarity.get(m.concept_id, 0.0),
        )

        TOP_FILLERS = 3
        fillers = usable[:TOP_FILLERS]

        # Build (template × filler) candidates round-robin over fillers so
        # E sees a variety even when total caps out. Order: filler 0 + tpl 0,
        # filler 1 + tpl 0, filler 2 + tpl 0, filler 0 + tpl 1, ...
        candidates: list[CandidateExpression] = []
        for tpl_idx, meta in enumerate(ordered_templates):
            for filler_idx, (_, filler_name) in enumerate(fillers):
                if len(candidates) >= N_CANDIDATES_DEFAULT:
                    break
                try:
                    surface_text = self.render_text(meta, {"phrase": filler_name})
                except KeyError:
                    continue
                surface_repr = encode_text(surface_text, dim=D_REP)
                candidates.append(CandidateExpression(
                    candidate_id=uuid.uuid4().hex,
                    intent_id=intent.intent_id,
                    internal_repr=intent.internal_repr.astype(np.float32, copy=True),
                    surface_repr=surface_repr,
                    surface_text=surface_text,
                    modality="text",
                ))
            if len(candidates) >= N_CANDIDATES_DEFAULT:
                break
        return candidates

    def _generate_cd_candidates(
        self,
        intent: ExpressionIntent,
    ) -> list[CandidateExpression]:
        """Generate via the GPT-2 conditioned decoder. Same conditioning
        contract as the LSTM head: composite affect + top-5 active concept
        embeddings. N samples at temperature 0.7, top-p 0.9."""
        if self._conditioned_decoder is None:
            return []

        composite = self.affect.composite(intent.now)
        ranked = sorted(intent.active_concepts.items(), key=lambda kv: -kv[1])
        embeddings: list[np.ndarray] = []
        for cid, _ in ranked:
            if cid in self._templates:
                continue
            node = self.graph.nodes.get(cid)
            if node is None:
                continue
            embeddings.append(node.embedding.astype(np.float32, copy=False))
            if len(embeddings) >= 5:
                break

        cond_np = build_conditioning_vector(composite, embeddings)
        cond_t = torch.tensor(cond_np, dtype=torch.float32).unsqueeze(0)

        candidates: list[CandidateExpression] = []
        seen: set[str] = set()
        attempts = max(N_CANDIDATES_DEFAULT, 6)
        for _ in range(attempts):
            text = self._conditioned_decoder.generate(cond_t).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            surface_repr = encode_text(text, dim=D_REP)
            candidates.append(CandidateExpression(
                candidate_id=uuid.uuid4().hex,
                intent_id=intent.intent_id,
                internal_repr=intent.internal_repr.astype(np.float32, copy=True),
                surface_repr=surface_repr,
                surface_text=text,
                modality="text",
            ))
            if len(candidates) >= N_CANDIDATES_DEFAULT:
                break
        return candidates

    def _generate_lm_candidates(
        self,
        intent: ExpressionIntent,
    ) -> list[CandidateExpression]:
        """Produce N independently-sampled completions from the language
        head. Conditioning = composite affect (12d) + top-5 active concept
        embeddings flat-packed (5×256d). Each call to LanguageHead.generate
        is multinomial at temperature 0.7 so the N samples differ.
        """
        if self._language_head is None or self._lm_vocab is None:
            return []

        # Composite affect right now — same source the cycle uses.
        composite = self.affect.composite(intent.now)

        # Top-5 active concept embeddings (zero-pad if fewer).
        ranked = sorted(intent.active_concepts.items(), key=lambda kv: -kv[1])
        embeddings: list[np.ndarray] = []
        for cid, _ in ranked:
            if cid in self._templates:
                continue
            node = self.graph.nodes.get(cid)
            if node is None:
                continue
            embeddings.append(node.embedding.astype(np.float32, copy=False))
            if len(embeddings) >= 5:
                break

        cond_np = build_conditioning_vector(composite, embeddings)
        cond_t = torch.tensor(cond_np, dtype=torch.float32).unsqueeze(0)

        candidates: list[CandidateExpression] = []
        seen_text: set[str] = set()
        # Sample slightly more than N_CANDIDATES_DEFAULT in case dedup or
        # all-<unk> generations cull some.
        attempts = max(N_CANDIDATES_DEFAULT, 6)
        for _ in range(attempts):
            ids = self._language_head.generate(cond_t)
            text = render_tokens(ids, self._lm_vocab).strip()
            if not text:
                continue
            if text in seen_text:
                continue
            seen_text.add(text)
            surface_repr = encode_text(text, dim=D_REP)
            candidates.append(CandidateExpression(
                candidate_id=uuid.uuid4().hex,
                intent_id=intent.intent_id,
                internal_repr=intent.internal_repr.astype(np.float32, copy=True),
                surface_repr=surface_repr,
                surface_text=text,
                modality="text",
            ))
            if len(candidates) >= N_CANDIDATES_DEFAULT:
                break
        return candidates

    # ---- request pipeline ----

    def request_expression(self, intent: ExpressionIntent) -> ExpressionDecision:
        """The full pipeline. Generate candidates → fire OUTPUT trigger per
        candidate (B.predict + B.observe at OUTPUT layer; A.inject(OUTPUT)
        feels each candidate) → snapshot E's private state → E decides.
        """
        candidates = self.generate_candidates(intent)
        if not candidates:
            # Empty active set or no usable fillers; let E return its
            # canonical "no candidates" SuppressionRequest.
            ps = self.identity.snapshot_private_state(intent.now, intent.active_concepts)
            return self.identity.decide_expression([], ps)

        # Fire OUTPUT trigger for each candidate. B.observe routes through
        # A.inject(OUTPUT) and may write a self-overhearing concept if the
        # candidate's surface diverges enough from the intended_repr to
        # cross surprise threshold (correct architectural behavior — the
        # system can surprise itself with what it almost said).
        for candidate in candidates:
            prediction = self.predict_engine.predict(intent.internal_repr, layer="OUTPUT")
            self.predict_engine.observe(
                prediction=prediction,
                actual=candidate.surface_repr,
                now=intent.now,
                name_hint=f"output:{candidate.surface_text[:32]}",
            )

        # E's decision.
        private_state = self.identity.snapshot_private_state(intent.now, intent.active_concepts)
        return self.identity.decide_expression(candidates, private_state)

    # ---- iterative expression — Phase 7 workstream B ----

    def generate_extended(
        self,
        intent: ExpressionIntent,
        budget: int,
        *,
        parent_stimulus_id: int | None = None,
    ) -> list[str]:
        """Generate up to `budget` consecutive sentences with self-overhearing.

        After each ChosenCandidate the just-said sentence re-enters via
        H.emit_output (which delivers + injects INTERNAL_OUTPUT_ECHO). If
        the echo wrote/matched a concept, F.attend(PROCESSING) seeds from
        it at 0.5 and the spread merges into the next intent at 0.3 weight.
        The original active set is preserved as the dominant signal — echo
        adds, never replaces.

        Stop conditions:
          - budget exhausted (returned `len(sentences) == budget`)
          - RevisionRequest    (the system tried, couldn't, stopped)
          - SuppressionRequest (nothing left to say)
          - empty echo + budget==1 fallback (single-shot)

        Falls back to single-shot single-sentence behavior when:
          - budget <= 1 (always run exactly request_expression once)
          - self.attention is None (no spread reference; can't echo-attend)
        """
        if budget <= 0:
            return []

        sentences: list[str] = []
        current_intent = intent

        from backend.attention import AttentionPhase
        from backend.input import Modality

        debug = os.environ.get("MIND_GEN_EXTENDED_DEBUG") == "1"
        if debug:
            sys.stderr.write(
                f"      [gen_ext START]  budget={budget}  "
                f"active={len(intent.active_concepts)}  "
                f"attention={'set' if self.attention is not None else 'None'}\n"
            )
            sys.stderr.flush()

        for _i in range(budget):
            decision = self.request_expression(current_intent)

            if debug:
                t = type(decision).__name__
                gap = (
                    decision.expression_gap
                    if isinstance(decision, ChosenCandidate)
                    else getattr(decision, "best_gap", None)
                )
                sys.stderr.write(
                    f"      [gen_ext iter {_i+1}/{budget}]  decision={t}  "
                    f"best_gap={gap}  active={len(current_intent.active_concepts)}\n"
                )
                sys.stderr.flush()

            if isinstance(decision, ChosenCandidate):
                surface = decision.candidate.surface_text
                sentences.append(surface)
                if debug:
                    sys.stderr.write(f"        chose: {surface!r}\n")
                    sys.stderr.flush()

                # If we have no attention reference, we still emit + echo
                # (so the system feels what it said), but we cannot spread
                # from the echo to update the active set. The next iteration
                # uses the same intent — typically generates the same
                # sentence which the dedup at the surface-text level should
                # break out of, but to stay clean we just stop.
                echo_result = self.input_pipeline.emit_output(
                    modality=Modality.TEXT,
                    surface=surface,
                    now=intent.now,
                    parent_stimulus_id=parent_stimulus_id,
                )

                if self.attention is None:
                    if debug:
                        sys.stderr.write("        break: attention is None\n")
                        sys.stderr.flush()
                    break

                # Echo seeding. If write_on_surprise fired, the echo bound
                # to a concept and we use that. Otherwise (fluent self-talk
                # is below the surprise threshold), the echo still has a
                # representation — seed from its cosine-nearest existing
                # concept. This mirrors MainLoop.cycle's fallback for
                # non-surprise input. Only when neither path produces a
                # seed (echo_cid None AND graph empty / no near match) do
                # we break.
                echo_cid: int | None = echo_result.gap.concept_id
                seed_activation = 0.5
                if echo_cid is None:
                    nearest = self.graph.nearest(echo_result.stimulus.representation)
                    if nearest is not None:
                        cid, sim = nearest
                        if sim > 0.0:
                            echo_cid = int(cid)
                            seed_activation = float(sim)
                if echo_cid is None:
                    if debug:
                        sys.stderr.write(
                            f"        break: no echo seed  "
                            f"(echo gap mag={float(echo_result.gap.magnitude):.4f}, "
                            f"is_surprise={echo_result.gap.is_surprise}, "
                            f"no near match)\n"
                        )
                        sys.stderr.flush()
                    break

                # Spread from the echo seed at the chosen activation strength.
                spread_result, _ = self.attention.attend(
                    phase=AttentionPhase.PROCESSING,
                    raw_seeds={echo_cid: seed_activation},
                    now=intent.now,
                )

                # Merge echo activation at 0.3 weight into the original
                # active set. Original concepts retain their full weight
                # so the system stays on topic; the echo nudges, doesn't
                # dictate. (If the echo reaches a concept already in the
                # active set, weights additively combine — the topic
                # tightens around what was said.)
                merged: dict[int, float] = dict(current_intent.active_concepts)
                for cid, act in spread_result.active_set.items():
                    merged[cid] = merged.get(cid, 0.0) + float(act) * 0.3

                # Recompute internal_repr from the merged active set's
                # weighted centroid so each sentence is evaluated against
                # an *evolved* intent rather than the original fixed
                # representation. Without this, every iteration after the
                # first compares its candidates against a centroid that
                # the conversation has already moved past — gaps balloon
                # and E suppresses the rest of the budget. With this,
                # the intent drifts with what the system is overhearing
                # itself say.
                evolved_repr = current_intent.internal_repr
                if merged:
                    cids_in_graph = [c for c in merged if c in self.graph.nodes]
                    if cids_in_graph:
                        embeddings = np.stack([
                            self.graph.nodes[c].embedding for c in cids_in_graph
                        ])
                        weights = np.array(
                            [merged[c] for c in cids_in_graph], dtype=np.float32,
                        )
                        if float(weights.sum()) > 1e-9:
                            centroid = np.average(embeddings, axis=0, weights=weights)
                            n = float(np.linalg.norm(centroid))
                            if n > 1e-9:
                                evolved_repr = (centroid / n).astype(np.float32)

                current_intent = ExpressionIntent(
                    intent_id=intent.intent_id,
                    internal_repr=evolved_repr,
                    active_concepts=merged,
                    now=intent.now,
                    audience_concept_id=intent.audience_concept_id,
                )

            elif isinstance(decision, RevisionRequest):
                # Tried, couldn't — stop. The sentences emitted so far are
                # what the system can honestly stand behind.
                break

            elif isinstance(decision, SuppressionRequest):
                # Nothing left to say — stop.
                break

            else:
                # Unknown decision shape; defensive break (would only fire
                # if the decision taxonomy is extended without updating G).
                break

        return sentences

    # ---- surface-form binding ----

    def attach_surface_to_concept(
        self,
        concept_id: int,
        surface_text: str,
        modality: str,
        affect_at_event: np.ndarray,
        now: float,
    ) -> "Edge":  # noqa: F821 — backend.graph.Edge
        """Bind a surface form to a concept via an `expresses` edge.

        Phase 3 minimal: the surface form itself becomes (or matches) a
        surface-token concept in C. The edge from `concept_id` to that
        surface-token concept records "this is one way this concept has
        been said." On repeat surface forms, find_or_match in C deduplicates,
        and add_edge is idempotent — repeat attach calls strengthen rather
        than duplicate.
        """
        if concept_id not in self.graph.nodes:
            raise KeyError(f"cannot attach to missing concept {concept_id}")

        surface_repr = encode_text(surface_text, dim=D_REP)
        # find_or_create the surface-token concept.
        surface_cid, _ = self.graph.write_on_surprise(
            representation=surface_repr,
            surprise=0.0,
            current_affect=affect_at_event,
            name_hint=surface_text[:48],
            now=now,
        )
        edge = self.graph.add_edge(
            source_id=concept_id,
            target_id=surface_cid,
            type=EdgeType.EXPRESSES,
            weight=0.5,
            now=now,
        )
        # Re-attach of the same surface should bump weight, not just be a no-op.
        # add_edge is idempotent; strengthen here so style/familiarity grows.
        self.graph.strengthen_edge(concept_id, surface_cid, EdgeType.EXPRESSES, now=now)
        return edge

    # ---- introspection ----

    def style_snapshot(self) -> dict:
        """Read-only view for E and frontend. Defensive copy of mutables."""
        return {
            "verbosity_mean":       self.style.verbosity_mean,
            "modality_preference":  list(self.style.modality_preference),
            "revision_temperament": self.style.revision_temperament,
            "template_familiarity": dict(self.style.template_familiarity),
            "modality_leak":        dict(self.style.modality_leak),
            "template_count":       self.template_count,
        }


def _extract_slots(pattern: str) -> list[str]:
    """Tiny parser: pull `{name}` tokens out of a format-style string.
    Used only to record which slots a template advertises. Not robust to
    nested braces — seed templates avoid them.
    """
    slots: list[str] = []
    i = 0
    while i < len(pattern):
        if pattern[i] == "{":
            j = pattern.find("}", i + 1)
            if j == -1:
                break
            slots.append(pattern[i + 1:j])
            i = j + 1
        else:
            i += 1
    return slots
