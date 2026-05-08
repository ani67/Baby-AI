"""Train the language head on the mind's surprised-sentence corpus.

Reads:
  data/surprised_sentences.jsonl  (one record per surprised sentence)
  data/last_book.txt              (the full kept-sentence list, for vocab)
  data/mind.db                    (current mind state, for top-5 conditioning)

Writes:
  data/language_head.pt
  data/language_head_vocab.json

Architecture is the LanguageHead defined in backend/language_head.py.
Training is autoregressive cross-entropy with token-level mask. The
conditioning vector for each example is the affect snapshot logged at
write time + the top-5 concepts in the graph closest to the example's
written concept_id.

5 epochs by default. Adam lr=1e-3. Batch size 32. Should finish in
under 10 minutes on M1 even with thousands of surprised sentences.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Allow running from repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.language_head import (              # noqa: E402
    BOS_ID,
    COND_DIM,
    EOS_ID,
    LanguageHead,
    PAD_ID,
    SPECIAL_TOKENS,
    TOP_K_CONCEPTS,
    UNK_ID,
    Vocab,
    build_conditioning_vector,
    save_checkpoint,
    tokenize,
)
from backend.mind_paths import MindPaths         # noqa: E402
from backend.persistence import MindPersistence  # noqa: E402

VOCAB_LIMIT     = 8000          # top-N words from the book + 4 specials = 8004
DEFAULT_EPOCHS  = 5
BATCH_SIZE      = 32
LEARNING_RATE   = 1e-3
MAX_SEQ_LEN     = 30            # tokens (sentence + bos + eos); longer truncated


# ============================================================
# Vocab from book text
# ============================================================

def build_vocab(book_path: str, limit: int = VOCAB_LIMIT) -> Vocab:
    if not os.path.exists(book_path):
        raise FileNotFoundError(book_path)
    counts: Counter[str] = Counter()
    with open(book_path, "r", encoding="utf-8") as f:
        for line in f:
            counts.update(tokenize(line))
    most_common = [w for w, _ in counts.most_common(limit)]
    id_to_token = list(SPECIAL_TOKENS) + most_common
    return Vocab(token_to_id={t: i for i, t in enumerate(id_to_token)},
                 id_to_token=id_to_token)


# ============================================================
# Build training tuples (sentence_token_ids, conditioning_vector)
# ============================================================

def build_examples(
    surprised_path: str,
    db_path: str,
    vocab: Vocab,
) -> list[tuple[list[int], np.ndarray]]:
    """For each surprised-sentence record, build (token_ids, cond_vec).

    cond_vec packs (affect_at_write, top-5 concept embeddings nearest to
    the written concept). Concepts come from the current mind state —
    the LM trains on the conditioning a freshly-loaded mind would see
    on the same input.
    """
    if not os.path.exists(surprised_path):
        raise FileNotFoundError(surprised_path)
    if not os.path.exists(db_path):
        raise FileNotFoundError(db_path)

    print(f"loading mind from {db_path} for top-{TOP_K_CONCEPTS} lookups…")
    loop = MindPersistence.load(db_path)
    g = loop.graph
    # Make sure the cosine matrix is built.
    g._rebuild_matrix()

    out: list[tuple[list[int], np.ndarray]] = []
    skipped = {"too_short": 0, "no_concept": 0, "no_embedding": 0}
    with open(surprised_path, "r", encoding="utf-8") as f:
        for raw in f:
            if not raw.strip():
                continue
            rec = json.loads(raw)
            sentence = rec.get("sentence", "")
            tokens = tokenize(sentence)
            if len(tokens) < 2:
                skipped["too_short"] += 1
                continue
            ids = vocab.encode(tokens)[: MAX_SEQ_LEN - 2]
            ids = [BOS_ID] + ids + [EOS_ID]

            cid = rec.get("concept_id")
            affect = np.asarray(rec.get("affect", []), dtype=np.float32)
            if affect.shape != (12,):
                affect = np.zeros(12, dtype=np.float32)

            if cid is None or cid not in g.nodes:
                skipped["no_concept"] += 1
                # Build cond with affect + zeros.
                cond = build_conditioning_vector(affect, [])
                out.append((ids, cond))
                continue

            # Find top-K cosine-nearest concepts to the written one.
            base_emb = g.nodes[cid].embedding
            sims = g._cosine_to_all(base_emb)
            order = np.argsort(-sims)
            top_embs: list[np.ndarray] = []
            for idx in order:
                peer_id = g._matrix_ids[int(idx)]
                if peer_id == cid:
                    continue
                peer = g.nodes.get(peer_id)
                if peer is None:
                    continue
                top_embs.append(peer.embedding)
                if len(top_embs) >= TOP_K_CONCEPTS:
                    break
            cond = build_conditioning_vector(affect, top_embs)
            out.append((ids, cond))
    print(f"  built {len(out):,} training examples (skipped {skipped})")
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
# Training
# ============================================================

def train(
    examples: list[tuple[list[int], np.ndarray]],
    vocab: Vocab,
    epochs: int,
    device: str,
) -> LanguageHead:
    if not examples:
        raise RuntimeError("no training examples — has the book been ingested yet?")

    dataset = SeqDataset(examples)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)

    model = LanguageHead(vocab_size=vocab.size).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params / 1e6:.2f} M params, vocab={vocab.size}, device={device}")

    optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_ID, reduction="mean")

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.perf_counter()
        total_loss = 0.0
        n_batches = 0
        for tokens, cond, mask in loader:
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
        print(f"  epoch {epoch}/{epochs}  loss={avg:.4f}  ({time.perf_counter() - t0:.1f}s)")

    return model


# ============================================================
# Entry point
# ============================================================

def run_training(
    paths: MindPaths,
    epochs: int,
    device: str | None = None,
    vocab_limit: int = VOCAB_LIMIT,
) -> None:
    """Programmatic entry point used by run_curriculum.py's `train_lm` step."""
    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")

    print(f"[train_lm] mind={paths.mind_name}  device={device}")
    print("[train_lm] building vocab …")
    vocab = build_vocab(paths.book_text_log, limit=vocab_limit)
    print(f"  vocab size: {vocab.size} (incl. {len(SPECIAL_TOKENS)} special tokens)")
    vocab.save(paths.vocab)

    print("[train_lm] building training examples …")
    examples = build_examples(paths.surprised_log, paths.db, vocab)

    print(f"[train_lm] training {epochs} epoch(s) on {len(examples):,} examples …")
    model = train(examples, vocab, epochs=epochs, device=device)

    save_checkpoint(model, paths.language_head)
    size_mb = os.path.getsize(paths.language_head) / 1024 / 1024
    print(f"[train_lm] saved {paths.language_head}  ({size_mb:.1f} MB)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mind", default="default",
                    help="mind name (paths under data/{mind}/)")
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--device", default=None,
                    help="cpu | mps | cuda — default mps if available")
    ap.add_argument("--vocab-limit", type=int, default=VOCAB_LIMIT)
    args = ap.parse_args()

    paths = MindPaths(args.mind)
    paths.ensure_dirs()
    run_training(
        paths=paths,
        epochs=args.epochs,
        device=args.device,
        vocab_limit=args.vocab_limit,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
