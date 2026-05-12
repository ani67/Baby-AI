#!/usr/bin/env python3
"""train_unified.py — driver for v2.0 unified joint training (Wave-3).

Builds the six components in spec order, optionally seeds the memory bank
from the existing concept graph, and runs `UnifiedMindTrainer.train_step`
over a streaming jsonl dataset.

USAGE
-----
    # Full training (after `scripts/prepare_v2_training_data.py` has run):
    python3 scripts/train_unified.py --mind first --steps 100000

    # Quick smoke run on a tiny bank, no graph init, 10 steps:
    python3 scripts/train_unified.py --smoke

NOTES
-----
- The training data must already exist at data/{mind}/v2_train.jsonl.
- The tokenizer must exist at data/{mind}/v2_vocab.json. Build both via:
      python3 scripts/prepare_v2_training_data.py --mind {mind}
- Checkpoints written under data/{mind}/v2_checkpoints/ every
  cfg.checkpoint_every steps.
- Memory-bank state separately saved to data/{mind}/v2_memory_bank.pt.
"""
import argparse
import json
import logging
import sqlite3
import sys
import time
from pathlib import Path

# project root on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from backend.affect_module import AffectModule
from backend.decoder import UnifiedDecoder
from backend.encoder import CurriculumTokenizer, MultiModalEncoder
from backend.memory_bank import PersistentMemoryBank
from backend.memory_transformer import SparseMemoryTransformer
from backend.training import TrainingConfig, UnifiedMindTrainer
from backend.unified_config import (
    D_REP,
    M_SLOTS,
    checkpoint_dir,
    memory_bank_path,
    vocab_path,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s %(message)s',
)
log = logging.getLogger('train_unified')

MAX_TOKENS = 64  # max target_ids length per example


# ----------------------------------------------------------------------------
# concept graph -> memory bank seeding
# ----------------------------------------------------------------------------

def _decode_emb(blob: bytes, d_rep: int) -> np.ndarray:
    arr = np.frombuffer(blob, dtype=np.float32)
    if arr.size != d_rep:
        # silently skip wrong-dim rows (legacy migrations may leave stragglers)
        return None
    return arr.copy()


def seed_bank_from_concept_graph(
    bank: PersistentMemoryBank,
    db_path: str,
    max_slots: int,
) -> int:
    """Pull concept embeddings (and affect/activation traces) from mind.db
    into the bank's first N slots. Returns the number of slots populated."""
    if not Path(db_path).exists():
        log.info(f"no concept graph at {db_path}; skipping seed")
        return 0

    conn = sqlite3.connect(db_path)
    try:
        cur = conn.execute(
            "SELECT embedding, affect_running, activation_count "
            "FROM concept_nodes "
            "WHERE embedding IS NOT NULL "
            "ORDER BY activation_count DESC "
            "LIMIT ?",
            (max_slots,),
        )
        embs, affects, counts = [], [], []
        for emb_blob, aff_blob, act in cur.fetchall():
            arr = _decode_emb(emb_blob, bank.d_rep)
            if arr is None:
                continue
            embs.append(arr)
            counts.append(int(act or 0))
            if aff_blob:
                af = np.frombuffer(aff_blob, dtype=np.float32)
                if af.size == 12:
                    affects.append(af.copy())
                else:
                    affects.append(np.zeros(12, dtype=np.float32))
            else:
                affects.append(np.zeros(12, dtype=np.float32))
    finally:
        conn.close()

    if not embs:
        log.info("concept graph yielded zero usable rows; skipping seed")
        return 0

    emb_arr = np.stack(embs).astype(np.float32)
    aff_arr = np.stack(affects).astype(np.float32)
    cnt_arr = np.asarray(counts, dtype=np.int64)
    n = bank.initialize_from_concept_graph(emb_arr, aff_arr, cnt_arr)
    log.info(f"seeded bank from concept graph: {n} slots from {db_path}")
    return n


# ----------------------------------------------------------------------------
# streaming dataset
# ----------------------------------------------------------------------------

def stream_examples(path: str):
    """Yield decoded records forever, looping back to the start of the file."""
    while True:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except Exception:
                    continue


def example_to_batch(rec: dict, tokenizer: CurriculumTokenizer, device) -> dict:
    """Convert a jsonl record to the dict accepted by train_step."""
    text = rec.get('text') or ''
    batch = {'text': text}

    # target_ids: encode the text and pad/truncate to MAX_TOKENS
    if text:
        ids = tokenizer.encode(text)
        if len(ids) > MAX_TOKENS:
            ids = ids[:MAX_TOKENS]
        else:
            ids = ids + [0] * (MAX_TOKENS - len(ids))
        batch['target_ids'] = torch.tensor([ids], device=device, dtype=torch.long)

    if 'is_surprising' in rec:
        batch['is_surprising'] = torch.tensor(
            [float(rec['is_surprising'])], device=device,
        )

    if 'affect' in rec and rec['affect']:
        aff = np.asarray(rec['affect'], dtype=np.float32)
        if aff.size == 12:
            batch['next_affect'] = torch.tensor(aff, device=device)

    return batch


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def build_components(
    mind: str,
    smoke: bool,
    device,
    init_from_graph: bool,
):
    """Construct all six components in spec order. Returns (bank, mt, enc,
    aff, dec, tokenizer, vocab_size)."""
    # 1) tokenizer (required for vocab size)
    if smoke:
        # tiny vocab for smoke; tokenizer-less encoder (CurriculumTokenizer
        # without a path uses char-mod fallback)
        tokenizer = CurriculumTokenizer()
        # encoder fallback returns ids up to 1023; pad bank size accordingly
        vocab_size = 1024
    else:
        vp = vocab_path(mind)
        if not Path(vp).exists():
            raise FileNotFoundError(
                f"tokenizer not found at {vp}. "
                f"Run: python3 scripts/prepare_v2_training_data.py --mind {mind}"
            )
        tokenizer = CurriculumTokenizer.from_path(vp)
        vocab_size = tokenizer.vocab_size

    # 2) memory bank
    m_slots = 2048 if smoke else M_SLOTS
    log.info(f"building memory_bank (m_slots={m_slots}, d_rep={D_REP})")
    bank = PersistentMemoryBank(m_slots=m_slots, d_rep=D_REP, device=device)

    if init_from_graph and not smoke:
        db = f"data/{mind}/mind.db"
        seed_bank_from_concept_graph(bank, db, max_slots=m_slots)
    else:
        log.info("skipping concept-graph init")

    # 3) sparse memory transformer — gradient checkpointing disabled in smoke
    log.info("building memory_transformer")
    use_ckpt = not smoke
    mt = SparseMemoryTransformer(bank, use_checkpoint=use_ckpt)

    # 4) encoder (text embedding lives here, will be shared with decoder)
    log.info(f"building encoder (vocab_size={vocab_size})")
    enc = MultiModalEncoder(bank, vocab_size=vocab_size, tokenizer=tokenizer)

    # 5) affect module
    log.info("building affect_module")
    aff = AffectModule()

    # 6) decoder — weight-tied with encoder.text_embedding
    log.info("building decoder (weight-tied to encoder.text_embedding)")
    dec = UnifiedDecoder(
        bank, vocab_size=vocab_size,
        shared_embedding=enc.text_embedding,
    )
    # sanity: weight tying
    assert dec.lm_head.weight is dec.token_embedding.weight is enc.text_embedding.weight

    return bank, mt, enc, aff, dec, tokenizer, vocab_size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mind', default='first')
    ap.add_argument('--steps', type=int, default=1000)
    ap.add_argument('--resume', type=str, default=None,
                    help='checkpoint .pt to resume from')
    ap.add_argument('--smoke', action='store_true',
                    help='10-step smoke run with a tiny bank, no graph init')
    ap.add_argument('--no-init-from-graph', action='store_true',
                    help='do not seed the memory bank from mind.db')
    ap.add_argument('--log-every', type=int, default=None,
                    help='override TrainingConfig.log_every (default 50)')
    ap.add_argument('--device', default='auto',
                    choices=['auto', 'cuda', 'mps', 'cpu'],
                    help='device override; auto picks cuda > mps > cpu')
    args = ap.parse_args()

    # Auto device detection: CUDA on cloud (A100/H100), MPS on M1 dev,
    # CPU last resort. Same code runs on all three; the only difference
    # is throughput.
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
        if device.type == 'cuda':
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

    log.info(f"device={device}")

    smoke = args.smoke
    steps = 10 if smoke else args.steps
    init_from_graph = (not args.no_init_from_graph) and (not smoke)

    bank, mt, enc, aff, dec, tokenizer, vocab_size = build_components(
        args.mind, smoke=smoke, device=device, init_from_graph=init_from_graph,
    )

    # config
    cfg_kwargs = dict(
        max_steps=steps,
        vocab_size=vocab_size,
        grad_accum=2 if smoke else TrainingConfig.grad_accum,
        gradnorm_every=10_000_000 if smoke else TrainingConfig.gradnorm_every,
    )
    if args.log_every is not None:
        cfg_kwargs['log_every'] = args.log_every
    cfg = TrainingConfig(**cfg_kwargs)
    log.info(f"config: max_steps={cfg.max_steps} grad_accum={cfg.grad_accum} "
             f"vocab_size={cfg.vocab_size}")

    trainer = UnifiedMindTrainer(cfg, bank, mt, enc, aff, dec, device)

    if args.resume:
        log.info(f"resuming from {args.resume}")
        trainer.load(args.resume)

    # data
    if smoke:
        # in-memory micro corpus — no disk dependency
        micro = [
            {'text': 'hello world test', 'is_surprising': 1.0},
            {'text': 'the cat sat on the mat', 'is_surprising': 0.0},
            {'text': 'surprise is the only teacher', 'is_surprising': 1.0},
        ]
        def gen():
            i = 0
            while True:
                yield micro[i % len(micro)]
                i += 1
        data_iter = gen()
    else:
        train_path = f"data/{args.mind}/v2_train.jsonl"
        if not Path(train_path).exists():
            raise FileNotFoundError(
                f"training data not found at {train_path}. "
                f"Run: python3 scripts/prepare_v2_training_data.py --mind {args.mind}"
            )
        log.info(f"streaming examples from {train_path}")
        data_iter = stream_examples(train_path)

    # training loop
    t0 = time.time()
    last_log = t0
    log.info(f"begin training — target steps={cfg.max_steps}")
    final_metrics = None

    while trainer.step < cfg.max_steps:
        rec = next(data_iter)
        batch = example_to_batch(rec, tokenizer, device)
        metrics = trainer.train_step(batch)
        final_metrics = metrics

        if trainer.step % cfg.log_every == 0 or trainer.step <= 3 or trainer.step == cfg.max_steps:
            dt = time.time() - last_log
            last_log = time.time()
            ml = ' '.join(
                f"{t}={metrics[t]:.4f}"
                for t in ('mask', 'lm', 'align', 'affect', 'surp')
            )
            gn = ''
            if 'grad_norms' in metrics:
                gn = ' grad=' + '/'.join(
                    f"{k}:{v:.4g}" for k, v in metrics['grad_norms'].items()
                )
            log.info(
                f"step={trainer.step:6d} total={metrics['total_loss']:.4f} "
                f"{ml} dt={dt:.1f}s{gn}"
            )

        if (not smoke) and trainer.step % cfg.checkpoint_every == 0:
            log.info(f"checkpoint at step {trainer.step}")
            trainer.save(args.mind)

    if not smoke and trainer.step > 0:
        log.info("final checkpoint")
        trainer.save(args.mind, tag='_final')

    dt = time.time() - t0
    log.info(f"done — {trainer.step} steps in {dt:.1f}s "
             f"({trainer.step / max(dt, 1e-3):.2f} steps/s)")
    if final_metrics is not None:
        log.info(f"final losses: {final_metrics}")
    return final_metrics


if __name__ == '__main__':
    main()
