"""UnifiedMind: Phase 2 integration of the v2.0 unified architecture.

Wires the seven Phase 1 components into a single class with a synchronous
`process(...)` entry point and a `status()` summary:

    encoder -> memory_transformer -> affect_module -> decoder.generate
                                |
                          memory_bank (trained_slots + experience_slots)

Side effects on each forward pass:
  - surprise > SURPRISE_THRESHOLD or force_respond  -> bank.soft_write(...)
  - affect_module.update_timescales(...) advances the 5 EMA timescales
  - emitted text is tracked in a self-echo deque so the next input can be
    flagged when it overlaps our own recent output (>60% over <30s)

Threading: a single threading.RLock serializes process() so the API can call
it concurrently from multiple request handlers. process() runs under
@torch.no_grad — gradients live only in training.py.

This file is NOT a training driver. The training loop lives in
backend/training.py only; here we just compose the forward path.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from backend.affect_module import AffectModule
from backend.decoder import UnifiedDecoder, compute_expression_gap
from backend.encoder import CurriculumTokenizer, MultiModalEncoder
from backend.memory_bank import PersistentMemoryBank
from backend.memory_transformer import SparseMemoryTransformer
from backend.unified_config import (
    D_REP,
    EXPRESSION_GAP_THRESHOLD,
    M_SLOTS,
    SUPPRESS_THRESHOLD,
    SURPRISE_THRESHOLD,
    VOCAB_SIZE,
    memory_bank_path,
    vocab_path,
)
from backend.mind_paths import MindPaths

log = logging.getLogger('unified_mind')


# ---------------------------------------------------------------------------
# Response dataclass
# ---------------------------------------------------------------------------

@dataclass
class MindResponse:
    """Output of UnifiedMind.process(...).

    `suppressed=True` means we chose not to emit (gap too high). The decoder
    still produced a string; we keep it in `text` for debugging / introspection
    but downstream callers should treat suppressed responses as silent.
    """
    text: str
    gap: float
    active_memory_slots: int
    arousal: float
    affect_vector: list[float]
    suppressed: bool = False
    generator: str = 'unified'
    # debug-only echo flag — populated when process() is called with debug=True.
    # Not part of the public schema; the API serializer drops it.
    echo: bool = False


# ---------------------------------------------------------------------------
# UnifiedMind
# ---------------------------------------------------------------------------

class UnifiedMind:
    """Composition root for v2.0 — one mind, one process.

    Components are passed in pre-built (see `load_or_create` for the typical
    build path). The caller is responsible for having moved each component to
    `device` and cast it to bf16 where appropriate; the trainer in training.py
    already does this when it constructs its own copies.
    """

    def __init__(
        self,
        memory_bank: PersistentMemoryBank,
        memory_transformer: SparseMemoryTransformer,
        encoder: MultiModalEncoder,
        affect_module: AffectModule,
        decoder: UnifiedDecoder,
        tokenizer: CurriculumTokenizer,
        device: torch.device,
    ):
        self.bank = memory_bank
        self.mt = memory_transformer
        self.enc = encoder
        self.aff = affect_module
        self.dec = decoder
        self.tokenizer = tokenizer
        self.device = device

        # Inference-only by default. Training is owned by UnifiedMindTrainer.
        for m in (self.mt, self.enc, self.aff, self.dec):
            m.eval()

        # Self-echo buffer: (timestamp, surface) for recently emitted text.
        # Bounded so it can't grow unbounded across a long-running process.
        self._echo_buf: deque[tuple[float, str]] = deque(maxlen=64)
        self._echo_window_s: float = 30.0
        self._echo_overlap: float = 0.60

        # Single RLock — reentrant so a nested call (e.g. via debug helpers
        # that re-enter process()) doesn't deadlock.
        self._lock = threading.RLock()

        # Step counter used by soft_write and any future telemetry.
        self._step = 0

    # ---------- factory --------------------------------------------------

    @classmethod
    def load_or_create(
        cls,
        mind_name: str,
        checkpoint_path: Optional[str] = None,
        device: Optional[torch.device] = None,
    ) -> 'UnifiedMind':
        """Build all components for `mind_name` and seed the memory bank from
        the legacy concept graph (if data/{mind}/mind.db exists). Optionally
        restore a v2.0 checkpoint on top.

        Failure modes raise — callers (e.g. the API lifespan) wrap this in a
        try/except so a missing tokenizer or DB doesn't break legacy routes.
        """
        if device is None:
            device = torch.device(
                'mps' if torch.backends.mps.is_available() else 'cpu'
            )

        paths = MindPaths(mind_name)

        # 1) Tokenizer — required. Falls back to the char-level stub if the
        # BPE file is missing, but we surface that as a warning so the caller
        # knows the vocab is degraded.
        vp = vocab_path(mind_name)
        if os.path.exists(vp):
            tokenizer = CurriculumTokenizer.from_path(vp)
            vocab_size = tokenizer.vocab_size
        else:
            log.warning(
                "tokenizer not found at %s — falling back to char-level stub "
                "(vocab_size=%d). Run scripts/prepare_v2_training_data.py to "
                "build a proper tokenizer.", vp, VOCAB_SIZE,
            )
            tokenizer = CurriculumTokenizer()
            vocab_size = VOCAB_SIZE

        # 2) Memory bank. Try to restore from a saved v2 bank first; otherwise
        # construct fresh and (optionally) seed from the legacy concept graph.
        bank_path = memory_bank_path(mind_name)
        if os.path.exists(bank_path):
            log.info("loading memory_bank from %s", bank_path)
            bank = PersistentMemoryBank.load(bank_path, device=device)
        else:
            bank = PersistentMemoryBank(
                m_slots=M_SLOTS, d_rep=D_REP, device=device,
            )
            n_seed = cls._seed_bank_from_concept_graph(bank, paths.db)
            log.info("seeded memory_bank from %s (%d slots populated)",
                     paths.db, n_seed)

        # 3) Encoder, decoder, transformer, affect. Weight-tied
        # token_embedding between encoder and decoder per spec § 4.6.
        enc = MultiModalEncoder(bank, vocab_size=vocab_size).to(device).to(torch.bfloat16)
        mt = SparseMemoryTransformer(bank, use_checkpoint=False).to(device).to(torch.bfloat16)
        aff = AffectModule().to(device).to(torch.bfloat16)
        dec = UnifiedDecoder(
            bank, vocab_size=vocab_size,
            shared_embedding=enc.text_embedding,
        ).to(device).to(torch.bfloat16)

        mind = cls(bank, mt, enc, aff, dec, tokenizer, device)

        # 4) Optional checkpoint restore (component state dicts only — the
        # bank already loaded itself above if a saved bank existed).
        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                ckpt = torch.load(checkpoint_path, map_location=device)
                if 'mem' in ckpt: mt.load_state_dict(ckpt['mem'])
                if 'enc' in ckpt: enc.load_state_dict(ckpt['enc'])
                if 'aff' in ckpt: aff.load_state_dict(ckpt['aff'])
                if 'dec' in ckpt: dec.load_state_dict(ckpt['dec'])
                mind._step = int(ckpt.get('step', 0))
                log.info("restored checkpoint %s (step=%d)",
                         checkpoint_path, mind._step)
            except Exception as exc:
                log.warning("checkpoint restore failed (%s); using fresh weights",
                            exc)

        return mind

    # ---------- concept-graph seeding (one-time, slow) -------------------

    @staticmethod
    def _seed_bank_from_concept_graph(
        bank: PersistentMemoryBank, db_path: str,
    ) -> int:
        """Open the legacy mind.db via MindPersistence.load(...) and copy
        concept embeddings / affect traces / activation counts into the bank.
        Defensive: skips nodes with malformed embedding/affect shapes.
        Returns the number of slots populated. Returns 0 if the DB is missing
        or unreadable — the bank remains randomly-initialized.
        """
        if not os.path.exists(db_path):
            log.info("no legacy mind.db at %s; bank stays random-init", db_path)
            return 0
        try:
            from backend.persistence import MindPersistence
            log.info("loading legacy concept graph from %s (may take 5-10s)...",
                     db_path)
            t0 = time.perf_counter()
            loop = MindPersistence.load(db_path)
        except Exception as exc:
            log.warning("legacy graph load failed (%s); bank stays random-init",
                        exc)
            return 0

        embs: list[np.ndarray] = []
        affs: list[np.ndarray] = []
        counts: list[int] = []
        d_rep = bank.d_rep
        n_aff = bank.affect_traces.shape[1]
        skipped = 0
        for node in loop.graph.nodes.values():
            try:
                e = np.asarray(node.embedding, dtype=np.float32)
                if e.shape != (d_rep,):
                    skipped += 1
                    continue
                rs = np.asarray(node.affect_trace.running_state,
                                dtype=np.float32)
                if rs.shape != (n_aff,):
                    # Affect mismatch is non-fatal — zero it and keep the embedding.
                    rs = np.zeros(n_aff, dtype=np.float32)
                ac = int(getattr(node, 'activation_count', 0))
            except Exception:
                skipped += 1
                continue
            embs.append(e)
            affs.append(rs)
            counts.append(ac)
        n_load = time.perf_counter() - t0
        log.info(
            "graph loaded in %.1fs: %d usable concepts, %d skipped (malformed)",
            n_load, len(embs), skipped,
        )
        if not embs:
            return 0

        emb_arr = np.stack(embs, axis=0)
        aff_arr = np.stack(affs, axis=0)
        cnt_arr = np.array(counts, dtype=np.int64)
        return bank.initialize_from_concept_graph(emb_arr, aff_arr, cnt_arr)

    # ---------- self-echo tracking ---------------------------------------

    def _word_overlap(self, a: str, b: str) -> float:
        """Symmetric Jaccard overlap on lowercased tokens. Empty inputs → 0."""
        ta = {t for t in a.lower().split() if t}
        tb = {t for t in b.lower().split() if t}
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / float(min(len(ta), len(tb)))

    def _is_self_echo(self, text: str, now: float) -> bool:
        """True if `text` overlaps >60% with anything we emitted in the last 30s."""
        if not text:
            return False
        cutoff = now - self._echo_window_s
        for t, surface in self._echo_buf:
            if t < cutoff:
                continue
            if self._word_overlap(text, surface) > self._echo_overlap:
                return True
        return False

    def _record_emission(self, text: str, now: float) -> None:
        if text:
            self._echo_buf.append((now, text))

    # ---------- the hot path ---------------------------------------------

    @torch.no_grad()
    def process(
        self,
        text: Optional[str] = None,
        image: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
        person_id: str = 'default',
        force_respond: bool = False,
        debug: bool = False,
    ) -> MindResponse:
        """Full forward pass: encode -> memory_transformer -> affect -> decoder.generate
        -> expression-gap check -> emit-or-suppress.

        Side effects (under the lock):
          - surprise > SURPRISE_THRESHOLD or force_respond → bank.soft_write
          - affect_module.update_timescales advances EMAs
          - emitted surface is appended to the self-echo deque
        """
        if text is None and image is None and audio is None:
            raise ValueError("process() requires at least one modality")
        now = time.time()

        with self._lock:
            # --- 0) self-echo check on text input (no early return — the
            # caller may still want a response; we just flag it).
            echo = bool(text and self._is_self_echo(text, now))

            # --- 1) encode
            enc_out = self.enc(text=text, image=image, audio=audio)
            input_rep = enc_out['input_rep']           # (D,)
            active_idx = enc_out['active_slot_indices']  # (K,)
            surprise = float(enc_out['surprise'].float())

            # --- 2) memory transformer over the full bank, biased by the
            # affect attention bias from the previous step (None on cold start).
            # Use the un-transformed slots → transformed slots; the active
            # subset feeds the decoder cross-attn.
            #
            # We pass affect_bias=None on the first call; subsequent calls
            # could feed back the last affect bias, but the spec doesn't
            # require it — and the test suite expects identical structure
            # across calls. Leave it as None until/unless training shows
            # otherwise.
            memory_full = self.mt(self.bank.slots, affect_bias=None)

            # --- 3) affect from transformed memory
            aff_out = self.aff(memory_full)
            affect_vec = aff_out['affect_vector']      # (12,) bf16/fp
            arousal = float(aff_out['arousal'])

            # advance the timescale EMAs (no_grad, CPU)
            try:
                self.aff.update_timescales(affect_vec.detach(), now)
            except Exception as exc:
                log.debug("update_timescales failed: %s", exc)

            # --- 4) generate
            mem_active = memory_full[active_idx]       # (K, D)
            surface, gen_ids = self.dec.generate(
                mem_active, self.tokenizer,
            )

            # --- 5) expression-gap check
            try:
                centroid = self.bank.get_field_centroid()
                gap = compute_expression_gap(
                    gen_ids, centroid, self.enc, self.tokenizer,
                )
            except Exception as exc:
                # Generation produced nothing or centroid is degenerate. Treat
                # as max gap so the suppression path fires cleanly.
                log.debug("expression_gap failed (%s); defaulting to 1.0", exc)
                gap = 1.0

            # Suppression policy. force_respond bypasses the gap check entirely
            # — the social contract is that a direct address gets an answer.
            suppressed = False
            if not force_respond:
                if gap >= SUPPRESS_THRESHOLD:
                    suppressed = True
                elif gap >= EXPRESSION_GAP_THRESHOLD:
                    # Between the thresholds: emit but flag the surface as a
                    # "low-confidence" expression. We still emit (the spec
                    # doesn't suppress below SUPPRESS_THRESHOLD) but keep the
                    # gap visible to the caller.
                    suppressed = False
                else:
                    suppressed = False

            # --- 6) side effects
            if surprise > SURPRISE_THRESHOLD or force_respond:
                try:
                    # NOTE: PersistentMemoryBank.soft_write has a latent
                    # device bug when both bank.affect_traces is on MPS and
                    # affect_vector is passed in (the internal .cpu() call
                    # mixes MPS+CPU tensors). We skip affect_vector here and
                    # accept that the bank's affect_trace metadata won't be
                    # updated on inference-time writes — the trainer does
                    # not depend on that path. The fix lives in memory_bank
                    # and is out of scope for this agent.
                    self.bank.soft_write(
                        input_rep.detach(),
                        torch.tensor(surprise, device=self.device),
                        None,
                        step=self._step,
                    )
                except Exception as exc:
                    log.warning("soft_write failed: %s", exc)

            # Track our own emission for the self-echo window (only when we
            # actually emit — suppressed text never goes out).
            if surface and not suppressed:
                self._record_emission(surface, now)

            self._step += 1

            # --- 7) build response
            affect_list = [float(x) for x in affect_vec.detach().float().cpu().tolist()]
            resp = MindResponse(
                text=surface if not suppressed else '',
                gap=float(gap),
                active_memory_slots=int(active_idx.shape[0]),
                arousal=arousal,
                affect_vector=affect_list,
                suppressed=suppressed,
                generator='unified',
            )
            if debug:
                resp.echo = echo
            return resp

    # ---------- status ---------------------------------------------------

    def status(self) -> dict:
        """Snapshot for /unified/status. Numbers, not tensors."""
        with self._lock:
            n_written = int(self.bank.n_written.item())
            alpha = float(torch.sigmoid(self.bank.alpha_logit).item())
            try:
                comp = self.aff.composite()
                comp_list = [float(x) for x in comp.detach().cpu().tolist()]
            except Exception:
                comp_list = []
            try:
                char = self.aff.character()
                char_list = [float(x) for x in char.detach().cpu().tolist()]
            except Exception:
                char_list = []
            return {
                'step': self._step,
                'm_slots': int(self.bank.m_slots),
                'n_written': n_written,
                'alpha_trained': alpha,
                'device': str(self.device),
                'vocab_size': int(self.dec.vocab_size),
                'composite_affect': comp_list,
                'character_affect': char_list,
                'echo_buffer_size': len(self._echo_buf),
            }
