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

# ============================================================
# Phase 7: GPT-2 conditioned-decoder fine-tuner
# ============================================================

CD_BATCH_SIZE   = 8
# Spec said 1e-4 but at this scale GPT-2 fine-tuning is unstable — loss
# climbs after ~2 epochs. 2e-5 with warmup keeps training stable while
# still letting condition_proj absorb the corpus over 10 epochs.
CD_LR           = 2e-5
CD_WD           = 0.01
CD_EPOCHS       = 10
CD_MAX_TOKENS   = 64
CD_WARMUP_FRAC  = 0.10           # fraction of total steps for linear warmup


def build_cd_examples(
    surprised_path: str,
    db_path: str,
    tokenizer,
):
    """For each surprised-sentence record, produce (token_ids,
    attention_mask, cond_vec). Tokenization uses GPT-2's tokenizer; we
    add a trailing EOS so the model learns to stop at sentence boundary."""
    if not os.path.exists(surprised_path):
        raise FileNotFoundError(surprised_path)
    if not os.path.exists(db_path):
        raise FileNotFoundError(db_path)

    print(f"[cd-train] loading mind from {db_path} for top-{TOP_K_CONCEPTS} concept lookups…")
    loop = MindPersistence.load(db_path)
    g = loop.graph
    g._rebuild_matrix()

    examples: list[tuple[list[int], np.ndarray]] = []
    with open(surprised_path, "r", encoding="utf-8") as f:
        for raw in f:
            if not raw.strip():
                continue
            rec = json.loads(raw)
            sentence = rec.get("sentence", "").strip()
            if not sentence:
                continue
            cid = rec.get("concept_id")
            affect = np.asarray(rec.get("affect", []), dtype=np.float32)
            if affect.shape != (12,):
                affect = np.zeros(12, dtype=np.float32)

            # Top-K nearest concept embeddings from the current mind.
            top_embs: list[np.ndarray] = []
            if cid is not None and cid in g.nodes:
                base_emb = g.nodes[cid].embedding
                sims = g._cosine_to_all(base_emb)
                order = np.argsort(-sims)
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

            # GPT-2 tokenization. Append EOS so the model has a stop signal.
            ids = tokenizer.encode(
                sentence + tokenizer.eos_token,
                add_special_tokens=False,
            )[:CD_MAX_TOKENS]
            if len(ids) < 2:
                continue
            examples.append((ids, cond))

    return examples, loop  # return loop too so caller can keep it alive (avoids re-loading)


def collate_cd(batch, pad_token_id: int):
    """Pad token_ids to max length. Returns tokens, attention_mask, cond."""
    max_len = max(len(t) for t, _ in batch)
    tokens = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    cond = torch.stack([torch.tensor(c, dtype=torch.float32) for _, c in batch])
    for i, (ids, _) in enumerate(batch):
        tokens[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        mask[i, : len(ids)] = 1
    return tokens, mask, cond


def run_training_cd(
    paths: MindPaths,
    epochs: int = CD_EPOCHS,
    device: str | None = None,
    batch_size: int = CD_BATCH_SIZE,
    resume_from: str | None = None,
    save_to: str | None = None,
    lr_override: float | None = None,
    skip_warmup: bool = False,
) -> None:
    """Fine-tune the GPT-2 conditioned decoder on this mind's surprise corpus.

    Phase 7 A2 default usage:
      run_training_cd(paths, epochs=20)
    saves the best-loss delta to <root>/language_head_gpt2.pt and uses
    CD_LR with CD_WARMUP_FRAC linear warmup.

    Phase 7 A3 second-pass usage:
      run_training_cd(
          paths, epochs=30, resume_from=<...>/language_head_gpt2.pt,
          save_to=<...>/language_head_gpt2_v2.pt,
          lr_override=1e-5, skip_warmup=True,
      )
    The continuation path warmup is skipped because the model already saw
    the data; constant lr=1e-5 from step 0.
    """
    from backend.language_head import (        # noqa: E402
        CONDITIONED_DECODER_FILENAME,
        ConditionedDecoder,
    )

    # Phase 7 stability fix: force CPU. MPS produced corrupt zero-loss
    # weights mid-A3 (saw norm=inf in a transformer ln_1). On a 16 GB M1
    # under memory pressure MPS also silently falls back to CPU-class
    # speeds anyway. CPU is reliable and fast enough for batch=8.
    # `device` arg is honored if explicitly passed — only the auto-pick
    # default changes.
    if device is None:
        device = "cpu"
    lr = lr_override if lr_override is not None else CD_LR
    print(f"[cd-train] mind={paths.mind_name}  device={device}  lr={lr:g}")

    print("[cd-train] constructing ConditionedDecoder (loads GPT-2 small) …")
    model = ConditionedDecoder()                      # imports transformers; downloads weights
    if resume_from:
        if not os.path.exists(resume_from):
            raise FileNotFoundError(f"resume_from missing: {resume_from}")
        print(f"[cd-train] resuming from {resume_from} …")
        model.load_delta(resume_from)
    model.to(device)

    tokenizer = model.tokenizer
    examples, _loop = build_cd_examples(paths.surprised_log, paths.db, tokenizer)
    if not examples:
        raise RuntimeError("no training examples — has the curriculum been run yet?")
    print(f"[cd-train] {len(examples):,} training examples")

    # Trainable params: condition_proj + unfrozen GPT-2 layers + ln_f + lm_head
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[cd-train] trainable: {n_train / 1e6:.1f}M / {n_total / 1e6:.1f}M params")

    # Build a simple iterable from examples; we use plain shuffling
    # because the dataset is small and DataLoader's overhead isn't worth
    # it for ~1000 examples.
    pad_id = tokenizer.pad_token_id

    optim_obj = torch.optim.AdamW(trainable, lr=lr, weight_decay=CD_WD)

    # Linear warmup over the first 10% of steps then constant. The first-
    # pass training (A2) uses warmup; the A3 continuation skips warmup
    # because the model has already adapted to the data.
    n_batches_per_epoch = max(1, len(examples) // batch_size + (1 if len(examples) % batch_size else 0))
    total_steps = epochs * n_batches_per_epoch
    if skip_warmup:
        warmup_steps = 0
    else:
        warmup_steps = max(1, int(total_steps * CD_WARMUP_FRAC))

    def lr_at_step(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return lr * (step + 1) / warmup_steps
        return lr

    import random
    step = 0
    best_loss = float("inf")
    saved_path = save_to or os.path.join(paths.root, CONDITIONED_DECODER_FILENAME)
    print(f"[cd-train] best-loss delta will be saved to {saved_path}")
    for epoch in range(1, epochs + 1):
        model.train()
        random.shuffle(examples)
        t0 = time.perf_counter()
        total_loss = 0.0
        n_seen = 0
        for bi in range(0, len(examples), batch_size):
            for pg in optim_obj.param_groups:
                pg["lr"] = lr_at_step(step)
            batch = examples[bi: bi + batch_size]
            tokens, mask, cond = collate_cd(batch, pad_id)
            tokens = tokens.to(device); mask = mask.to(device); cond = cond.to(device)
            loss = model.forward_with_prefix(cond, tokens, mask)
            optim_obj.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optim_obj.step()
            total_loss += float(loss.item())
            n_seen += 1
            step += 1
        avg = total_loss / max(1, n_seen)
        # Save best-loss checkpoint so a destabilizing late epoch doesn't
        # wipe out a good earlier model.
        if avg < best_loss:
            best_loss = avg
            model.save_delta(saved_path)
            best_marker = " ← best, saved"
        else:
            best_marker = ""
        print(f"  epoch {epoch}/{epochs}  loss={avg:.4f}  lr={lr_at_step(step):.2e}  ({time.perf_counter() - t0:.1f}s){best_marker}")

    # Best-loss delta is already on disk (saved per-epoch above).
    if os.path.exists(saved_path):
        size_mb = os.path.getsize(saved_path) / 1024 / 1024
        print(f"[cd-train] best-loss delta at {saved_path}  ({size_mb:.1f} MB, loss={best_loss:.4f})")


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
    ap.add_argument("--epochs", type=int, default=None,
                    help="default: 5 for LSTM, 10 for GPT-2 conditioned decoder")
    ap.add_argument("--device", default=None,
                    help="cpu | mps | cuda — default mps if available")
    ap.add_argument("--vocab-limit", type=int, default=VOCAB_LIMIT)
    ap.add_argument("--gpt2", action="store_true",
                    help="train the GPT-2 conditioned decoder instead of the "
                         "Phase-5 LSTM head (Phase 7).")
    ap.add_argument("--batch-size", type=int, default=CD_BATCH_SIZE,
                    help="batch size for --gpt2 mode")
    ap.add_argument("--resume-from", default=None,
                    help="(--gpt2 only) load existing delta weights from this "
                         "path before training. Used by Phase 7 A3 to continue "
                         "fine-tuning at a lower lr from the A2 checkpoint.")
    ap.add_argument("--save-to", default=None,
                    help="(--gpt2 only) override the default save path "
                         "(<mind_root>/language_head_gpt2.pt). Used by Phase 7 "
                         "A3 to write the v2 checkpoint to language_head_gpt2_v2.pt.")
    ap.add_argument("--lr", type=float, default=None,
                    help="(--gpt2 only) override the default lr (CD_LR=2e-5). "
                         "Phase 7 A3 uses 1e-5 for the second-pass continuation.")
    ap.add_argument("--skip-warmup", action="store_true",
                    help="(--gpt2 only) skip the linear-warmup schedule. "
                         "Used for second-pass continuation training where the "
                         "model has already adapted to the data.")
    args = ap.parse_args()

    paths = MindPaths(args.mind)
    paths.ensure_dirs()

    if args.gpt2:
        run_training_cd(
            paths=paths,
            epochs=args.epochs or CD_EPOCHS,
            device=args.device,
            batch_size=args.batch_size,
            resume_from=args.resume_from,
            save_to=args.save_to,
            lr_override=args.lr,
            skip_warmup=args.skip_warmup,
        )
    else:
        run_training(
            paths=paths,
            epochs=args.epochs or DEFAULT_EPOCHS,
            device=args.device,
            vocab_limit=args.vocab_limit,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
