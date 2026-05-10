"""Train the native head v2 — larger transformer + filtered corpus.

Reads:
  data/<mind>/surprised_sentences_filtered.jsonl   (the >=8-word corpus)
  data/<mind>/language_head_vocab.json              (existing v1 vocab)

Writes:
  data/<mind>/native_head_v2.pt

Standard teacher-forced LM loss on next-token prediction. Adam, grad-clip 5.0.
Conditioning per example uses the affect snapshot logged at write time and
the concept embeddings stored in the journal record (when present); if the
record lacks `concept_embeddings`, we fall back to zero-padded slots.

Why not re-derive top-K concepts from the live mind: the v1 trainer does
that (it loads mind.db and queries the cosine matrix). For v2 we keep the
trainer self-contained on the journal so it doesn't need the mind state
loaded — the journal already carries enough conditioning context for the
head to learn from. If the journal lacks `concept_embeddings`, training
still runs (cond = affect + zeros), just with weaker conditioning signal.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Allow running from repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.language_head import (              # noqa: E402
    BOS_ID,
    EOS_ID,
    PAD_ID,
    TOP_K_CONCEPTS,
    Vocab,
    build_conditioning_vector,
    tokenize,
)
from backend.native_head import (                 # noqa: E402
    DEFAULT_V2_FILENAME,
    NativeHead,
    NativeHeadConfig,
    save_checkpoint_v2,
)


DEFAULT_EPOCHS    = 150
DEFAULT_BATCH     = 128
DEFAULT_LR        = 3e-4
MAX_SEQ_LEN_TRAIN = 64        # match NativeHeadConfig.max_seq_len


# ============================================================
# Build training tuples from the filtered journal
# ============================================================


def build_examples_from_journal(
    corpus_path: str,
    vocab: Vocab,
) -> list[tuple[list[int], np.ndarray]]:
    """Read the filtered jsonl. Each record is one (token_ids, cond_vec).

    Skips records whose tokenized length < 2. If `concept_embeddings`
    is present (list of [256]-float lists), we use up to TOP_K_CONCEPTS
    of them; otherwise the cond is affect + zero-padded concept slots.
    """
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(corpus_path)

    out: list[tuple[list[int], np.ndarray]] = []
    skipped = {"too_short": 0, "no_sentence": 0}
    with open(corpus_path, "r", encoding="utf-8") as f:
        for raw in f:
            if not raw.strip():
                continue
            rec = json.loads(raw)
            sentence = rec.get("sentence", "")
            if not sentence:
                skipped["no_sentence"] += 1
                continue
            # Soft word-length filter: skip only fragments under 3 words.
            # Phase 8+ corpus is the unfiltered 163K-line journal; we
            # want short phrases (3-7 words) to contribute vocabulary
            # and pattern fragments, while still excluding the truly
            # trivial (1-2 token) records that mostly tokenize back
            # to a single word + punctuation.
            if len(sentence.split()) < 3:
                skipped["too_short"] += 1
                continue
            tokens = tokenize(sentence)
            if len(tokens) < 2:
                skipped["too_short"] += 1
                continue
            ids = vocab.encode(tokens)[: MAX_SEQ_LEN_TRAIN - 2]
            ids = [BOS_ID] + ids + [EOS_ID]

            affect = np.asarray(rec.get("affect", []), dtype=np.float32)
            if affect.shape != (12,):
                affect = np.zeros(12, dtype=np.float32)

            top_embs: list[np.ndarray] = []
            ce_raw = rec.get("concept_embeddings", []) or []
            for emb in ce_raw[:TOP_K_CONCEPTS]:
                arr = np.asarray(emb, dtype=np.float32)
                if arr.shape == (256,):
                    top_embs.append(arr)

            cond = build_conditioning_vector(affect, top_embs)
            out.append((ids, cond))

    print(f"[train_v2] built {len(out):,} examples (skipped {skipped})")
    return out


class SeqDataset(Dataset):
    def __init__(self, examples: list[tuple[list[int], np.ndarray]]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        ids, cond = self.examples[idx]
        return torch.tensor(ids, dtype=torch.long), torch.tensor(cond, dtype=torch.float32)


def collate(batch):
    """Pad to max length in batch. Returns (tokens, cond, mask)."""
    max_len = max(t.size(0) for t, _ in batch)
    tokens = torch.full((len(batch), max_len), PAD_ID, dtype=torch.long)
    cond = torch.stack([c for _, c in batch])
    mask = torch.zeros((len(batch), max_len), dtype=torch.float32)
    for i, (t, _) in enumerate(batch):
        tokens[i, : t.size(0)] = t
        mask[i, : t.size(0)] = 1.0
    return tokens, cond, mask


# ============================================================
# Train loop
# ============================================================


def train(
    examples: list[tuple[list[int], np.ndarray]],
    vocab: Vocab,
    epochs: int,
    device: str,
    batch_size: int,
    lr: float,
    save_path: str | None = None,
    save_every: int = 10,
) -> NativeHead:
    if not examples:
        raise RuntimeError("no training examples — has the filtered corpus been written?")

    cfg = NativeHeadConfig(vocab_size=vocab.size, max_seq_len=MAX_SEQ_LEN_TRAIN)
    model = NativeHead(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train_v2] model params: {n_params/1e6:.2f}M  vocab={vocab.size}  "
          f"d_model={cfg.d_model}  n_layers={cfg.n_layers}  device={device}  "
          f"batch={batch_size}  lr={lr:g}", flush=True)

    dataset = SeqDataset(examples)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)

    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_ID, reduction="mean")

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.perf_counter()
        total_loss = 0.0
        n_batches = 0
        for tokens, cond, _mask in loader:
            tokens = tokens.to(device)
            cond = cond.to(device)
            # Standard teacher-forced LM: predict tokens[:, t+1] given tokens[:, :t+1].
            logits = model(tokens[:, :-1], cond)             # (B, T-1, V)
            target = tokens[:, 1:]                           # (B, T-1)
            loss = loss_fn(logits.reshape(-1, vocab.size), target.reshape(-1))
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optim.step()
            total_loss += loss.item()
            n_batches += 1
        avg = total_loss / max(1, n_batches)
        elapsed = time.perf_counter() - t0
        # Periodic durable save: every `save_every` epochs and on the last
        # epoch. An external kill mid-run no longer wastes all the work
        # already done — the next run can warm-start (not implemented yet,
        # but the file at least reflects recent training state).
        ck_marker = ""
        if save_path and (epoch % save_every == 0 or epoch == epochs):
            save_checkpoint_v2(model, save_path)
            ck_marker = "  [ckpt]"
        print(f"  epoch {epoch}/{epochs}  loss={avg:.4f}  ({elapsed:.1f}s){ck_marker}",
              flush=True)
        # Inter-epoch cooldown — lets the M1 GPU drop temperature
        # between epochs so a 150-epoch MPS run doesn't thermally
        # throttle (or trigger a thermal shutdown). Adds ~5 min total
        # to a 150-epoch run, in exchange for stable wall-clock and
        # noticeably less heat.
        if epoch < epochs:
            time.sleep(2)

    return model


# ============================================================
# Entry point
# ============================================================


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mind", default="default",
                    help="mind name (paths under data/{mind}/)")
    ap.add_argument("--corpus", default=None,
                    help="path to surprised-sentence jsonl. Default: "
                         "data/{mind}/surprised_sentences.jsonl "
                         "(unfiltered 163K-line journal — pass "
                         "surprised_sentences_filtered.jsonl to use the "
                         "stale >=8-word v0.8 corpus)")
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    ap.add_argument("--lr", type=float, default=DEFAULT_LR)
    ap.add_argument("--device", default=None,
                    help="cpu | mps | cuda — default mps if available")
    args = ap.parse_args()

    mind_root = os.path.join("data", args.mind)
    if not os.path.isdir(mind_root):
        raise FileNotFoundError(f"mind dir not found: {mind_root}")

    corpus_path = args.corpus or os.path.join(mind_root, "surprised_sentences.jsonl")
    vocab_path = os.path.join(mind_root, "language_head_vocab.json")
    save_path = os.path.join(mind_root, DEFAULT_V2_FILENAME)

    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    if device == "mps":
        # libomp conflict guard — torch + numpy on M1 each link their own
        # libomp; >1 thread × >1 lib crashes mid-batch.
        torch.set_num_threads(1)
        os.environ["OMP_NUM_THREADS"] = "1"

    print(f"[train_v2] mind={args.mind}  corpus={corpus_path}")
    print(f"[train_v2] vocab={vocab_path}")
    print(f"[train_v2] save_to={save_path}")

    vocab = Vocab.load(vocab_path)
    print(f"[train_v2] vocab size: {vocab.size}")

    examples = build_examples_from_journal(corpus_path, vocab)
    if not examples:
        raise RuntimeError("no training examples — corpus is empty after parsing")

    t_train = time.perf_counter()
    model = train(
        examples=examples,
        vocab=vocab,
        epochs=args.epochs,
        device=device,
        batch_size=args.batch_size,
        lr=args.lr,
        save_path=save_path,
        save_every=10,
    )
    total = time.perf_counter() - t_train
    print(f"[train_v2] training complete in {total:.1f}s ({total/60:.1f} min)",
          flush=True)

    # Final save (the loop's last-epoch save also covers this; redundant
    # but cheap and explicit).
    save_checkpoint_v2(model, save_path)
    size_mb = os.path.getsize(save_path) / 1024 / 1024
    print(f"[train_v2] saved {save_path}  ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
