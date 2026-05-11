# Unified Mind Transformer — v2.0 Spec

**Date:** 2026-05-11
**Status:** active build — Phase 1 implementation

This is the authoritative spec for the v2.0 unified architecture. The v0.9/v1.1
systems (graph.py, wave_field.py, native_head.py, affect.py) remain on disk for
API back-compat. New work goes in new files listed below.

This spec supersedes any prior draft. The four critical corrections from the
prior draft are called out inline with `**CORRECTION**` markers.

---

## 1. Architecture overview

```
INPUT (text / vision / audio)
     │
     ▼
[Encoder]  modality-specific projection → D=512 input tokens
     │
     ▼  cross-attn: input queries, memory keys/values
[MemoryBank]  M=65536 slots × D=512
   trained_slots (nn.Parameter, lr-updated)
   experience_slots (buffer, soft-write-updated)
   combined_slots = α · trained + (1-α) · experience
     │
     ▼
[MemoryTransformer]  N=4 layers of sparse self-attn over combined_slots
   each slot attends to top-K=64 neighbors (gather-scatter on MPS)
   affect_bias added to attention scores per head
     │
     ▼  pool → (D,)
[AffectModule]
   memory pool → 12-dim affect vector
   5 timescale EMA buffers (reaction/working/mood/disposition/character)
   affect → per-head attention bias (fed back next step)
     │
     ▼
[Decoder]  autoregressive token generation
   self-attn (causal) + cross-attn over top-K active memory slots
   weight-tied LM head with encoder text embedding
   self-overhearing: emitted tokens → encoder next step
     │
     ▼
OUTPUT
```

Training: joint, end-to-end, GradNorm-balanced. 5 losses:
- L_mask (JEPA-style masked memory reconstruction)
- L_lm (next-token cross-entropy)
- L_align (cross-modal InfoNCE when both modalities present)
- L_affect (predict next affect from current state)
- L_surp (binary surprise classification)

---

## 2. Shared constants

All components import from `backend/unified_config.py` (new file, written by
Agent 1):

```python
# backend/unified_config.py
"""Shared constants for v2.0 unified architecture."""
import torch

# representation
D_REP        = 512
N_HEADS      = 8
D_HEAD       = D_REP // N_HEADS  # 64

# memory bank
M_SLOTS      = 65536
TOP_K_NBR    = 64       # sparse attention neighborhood
TOP_K_ACTIVE = 256      # decoder cross-attention pool

# affect
N_AFF        = 12

# transformer depths
N_MEM_LAYERS = 4
N_ENC_LAYERS = 5     # PC hierarchy levels
N_DEC_LAYERS = 4

# vocab (set after tokenizer built)
VOCAB_SIZE   = 16384

# training
BATCH_SIZE       = 4
GRAD_ACCUM       = 8        # effective batch 32
MAX_STEPS        = 100_000
WARMUP_STEPS     = 1_000
GRADNORM_EVERY   = 200      # **CORRECTION 4** (not every step)

# soft-write
SURPRISE_THRESHOLD = 0.3
DRIFT_RATE_BASE    = 0.01
MAX_DRIFT_RATE     = 0.05

# expression gap
EXPRESSION_GAP_THRESHOLD = 0.70
SUPPRESS_THRESHOLD       = 0.91

# memory mixing
INITIAL_TRAINED_ALPHA = 0.3   # learned, starts here

# checkpoint paths (subbed in by mind_paths)
def checkpoint_dir(mind_name: str) -> str:
    return f"data/{mind_name}/v2_checkpoints"

def memory_bank_path(mind_name: str) -> str:
    return f"data/{mind_name}/v2_memory_bank.pt"

def vocab_path(mind_name: str) -> str:
    return f"data/{mind_name}/v2_vocab.json"
```

---

## 3. MPS rules — mandatory for every component

Every agent applies these. Code that violates is rejected at reconciliation.

1. **Device.** `torch.device('mps' if torch.backends.mps.is_available() else 'cpu')`.
2. **Dtype.** All model params + activations in `torch.bfloat16`. Loss computation
   in `float32` (cast before `loss_fn(...)`).
3. **No in-place on autograd-tracked tensors.** `x = x + y`, never `x += y` on a
   tensor that has `requires_grad=True`.
4. **Contiguous before MPS ops.** `x.contiguous()` if shape was permuted.
5. **No `torch.sparse`** on MPS. Use gather-scatter instead.
6. **`torch.mps.empty_cache()`** after each training step (guarded by device check).
7. **No dynamic shapes** inside transformer layers. Fix all sizes at init; pad to
   fixed length.
8. **Chunk anything that produces a (M, M) or (M, L) tensor where M=65536.** Max
   chunk size of 4096 along the larger dim.
9. **`pin_memory=False`** in DataLoader (MPS doesn't support it).
10. **`empty_cache` syntax:** `if device.type == 'mps': torch.mps.empty_cache()`
    (statement, not ternary).

---

## 4. Component specs

Each component below has:
- **File** — where it lives
- **Replaces** — what legacy code it supersedes (legacy not modified)
- **Interface** — public API
- **Implementation notes** — non-obvious details
- **Test** — minimal acceptance test

---

### 4.1 PersistentMemoryBank — Agent 1

**File:** `backend/memory_bank.py`
**Also writes:** `backend/unified_config.py` (the shared constants above)
**Replaces:** concept graph storage role (graph.py kept for legacy API)

**CORRECTION 1:** Memory is split into two tensors. `trained_slots` is an
`nn.Parameter` updated by the optimizer. `experience_slots` is a buffer updated
by `soft_write`. The "active" memory is a learned convex combination. This
resolves the "in-place write on Parameter" autograd conflict.

```python
class PersistentMemoryBank(nn.Module):
    def __init__(self, m_slots=M_SLOTS, d_rep=D_REP, device=None):
        super().__init__()
        self.m_slots = m_slots
        self.d_rep = d_rep
        self.device = device or torch.device(
            'mps' if torch.backends.mps.is_available() else 'cpu'
        )

        # trained component — gradient-updated
        self.trained_slots = nn.Parameter(
            torch.zeros(m_slots, d_rep, dtype=torch.bfloat16)
        )

        # experience component — soft-write-updated, NOT a Parameter
        self.register_buffer(
            'experience_slots',
            torch.zeros(m_slots, d_rep, dtype=torch.bfloat16)
        )

        # learned mixing weight (scalar, sigmoid-bounded)
        self.alpha_logit = nn.Parameter(
            torch.tensor(math.log(INITIAL_TRAINED_ALPHA /
                                  (1 - INITIAL_TRAINED_ALPHA)))
        )

        # metadata buffers (not differentiable)
        self.register_buffer('activation_count',
                             torch.zeros(m_slots, dtype=torch.long))
        self.register_buffer('last_written',
                             torch.full((m_slots,), -1, dtype=torch.long))
        self.register_buffer('surprise_at_write',
                             torch.zeros(m_slots, dtype=torch.float32))
        self.register_buffer('affect_traces',
                             torch.zeros(m_slots, N_AFF, dtype=torch.float32))
        self.register_buffer('n_written',
                             torch.tensor(0, dtype=torch.long))

        self.to(self.device)

    @property
    def slots(self) -> torch.Tensor:
        """The 'active' memory: convex combination of trained + experience."""
        alpha = torch.sigmoid(self.alpha_logit)
        # bfloat16 mixing, F.normalize'd
        mixed = (alpha * self.trained_slots
                 + (1 - alpha) * self.experience_slots)
        return F.normalize(mixed, dim=-1)

    def initialize_from_concept_graph(
        self,
        concept_embeddings: np.ndarray,         # (N, 512)
        concept_affect_traces: np.ndarray=None, # (N, 12)
        concept_activation_counts: np.ndarray=None,  # (N,)
    ) -> int:
        """Populate first N slots from existing concepts. Both trained and
        experience start from these embeddings."""
        N = min(len(concept_embeddings), self.m_slots)
        emb = torch.tensor(concept_embeddings[:N], dtype=torch.bfloat16,
                           device=self.device)
        emb = F.normalize(emb, dim=-1)
        with torch.no_grad():
            self.trained_slots.data[:N] = emb
            self.experience_slots[:N] = emb
            self.n_written.fill_(N)
            if concept_affect_traces is not None:
                self.affect_traces[:N] = torch.tensor(
                    concept_affect_traces[:N], dtype=torch.float32,
                    device=self.device)
            if concept_activation_counts is not None:
                self.activation_count[:N] = torch.tensor(
                    concept_activation_counts[:N], dtype=torch.long,
                    device=self.device)
        return N

    @torch.no_grad()
    def soft_write(
        self,
        representation: torch.Tensor,    # (D,) or (B, D)
        surprise_magnitude: torch.Tensor,
        affect_vector: torch.Tensor=None,  # (12,)
        step: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Drift experience_slots toward `representation` at most-similar slot."""
        if representation.dim() == 1:
            representation = representation.unsqueeze(0)
        B = representation.shape[0]
        rep_norm = F.normalize(
            representation.to(torch.bfloat16), dim=-1)

        sims, top_idx = self.search(rep_norm, k=1)
        best_slot = top_idx[:, 0]  # (B,)

        if isinstance(surprise_magnitude, (int, float)):
            surprise_magnitude = torch.tensor(
                [surprise_magnitude] * B, device=self.device)
        elif surprise_magnitude.dim() == 0:
            surprise_magnitude = surprise_magnitude.unsqueeze(0).expand(B)

        drift = torch.clamp(
            surprise_magnitude.float() * DRIFT_RATE_BASE,
            max=MAX_DRIFT_RATE)

        for b in range(B):
            idx = int(best_slot[b])
            dr = float(drift[b])
            old = self.experience_slots[idx]
            new = old + dr * (rep_norm[b] - old)
            self.experience_slots[idx] = F.normalize(new, dim=-1)
            self.activation_count[idx] += 1
            self.last_written[idx] = step
            self.surprise_at_write[idx] = float(surprise_magnitude[b])
            if affect_vector is not None:
                self.affect_traces[idx] = (
                    0.9 * self.affect_traces[idx]
                    + 0.1 * affect_vector.detach().cpu().float())

        return best_slot, drift

    @torch.no_grad()
    def search(
        self,
        queries: torch.Tensor,   # (B, D)
        k: int = TOP_K_NBR,
        q_chunk: int = 1024,     # **CORRECTION 3**: chunk Q too
        m_chunk: int = 4096,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Top-k cosine-similarity search. Chunked over BOTH queries and
        memory to avoid (B, M) tensor allocation when B is large."""
        if queries.dim() == 1:
            queries = queries.unsqueeze(0)
        q_norm = F.normalize(queries, dim=-1)
        slots = self.slots  # (M, D)
        slots_norm = F.normalize(slots, dim=-1)
        B = q_norm.shape[0]
        k = min(k, self.m_slots)

        all_top_sims = torch.empty(
            B, k, device=self.device, dtype=q_norm.dtype)
        all_top_idx = torch.empty(
            B, k, device=self.device, dtype=torch.long)

        for qs in range(0, B, q_chunk):
            qe = min(qs + q_chunk, B)
            q_part = q_norm[qs:qe]  # (q_chunk, D)
            # collect topk across m_chunks in a streaming heap-like fashion
            running_sims = torch.full(
                (qe - qs, k), -1.0,
                device=self.device, dtype=q_norm.dtype)
            running_idx = torch.zeros(
                (qe - qs, k), device=self.device, dtype=torch.long)
            for ms in range(0, self.m_slots, m_chunk):
                me = min(ms + m_chunk, self.m_slots)
                sims = q_part @ slots_norm[ms:me].T  # (q_chunk, m_chunk)
                # combine with running
                combined_sims = torch.cat([running_sims, sims], dim=-1)
                combined_idx = torch.cat([
                    running_idx,
                    torch.arange(ms, me, device=self.device)
                         .unsqueeze(0).expand(qe - qs, -1)
                ], dim=-1)
                top_v, top_i = combined_sims.topk(k, dim=-1)
                running_sims = top_v
                running_idx = torch.gather(combined_idx, -1, top_i)
            all_top_sims[qs:qe] = running_sims
            all_top_idx[qs:qe] = running_idx

        return all_top_sims, all_top_idx

    def get_top_active(self, k: int = 50) -> list[tuple[int, float]]:
        scores = self.activation_count.float()
        top_v, top_i = scores.topk(min(k, self.m_slots))
        return [(int(i), float(v)) for i, v in zip(top_i, top_v)]

    def get_field_centroid(self, weights: torch.Tensor = None) -> torch.Tensor:
        if weights is None:
            counts = self.activation_count.float()
            weights = F.softmax(counts, dim=0)
        slots = self.slots.float()
        c = (weights.unsqueeze(-1) * slots).sum(0)
        return F.normalize(c, dim=-1)

    def save(self, path: str):
        torch.save({
            'trained_slots': self.trained_slots.data.cpu(),
            'experience_slots': self.experience_slots.cpu(),
            'alpha_logit': self.alpha_logit.data.cpu(),
            'activation_count': self.activation_count.cpu(),
            'last_written': self.last_written.cpu(),
            'surprise_at_write': self.surprise_at_write.cpu(),
            'affect_traces': self.affect_traces.cpu(),
            'n_written': self.n_written.cpu(),
            'm_slots': self.m_slots, 'd_rep': self.d_rep,
        }, path)

    @classmethod
    def load(cls, path: str, device=None) -> 'PersistentMemoryBank':
        d = torch.load(path, map_location='cpu')
        bank = cls(m_slots=d['m_slots'], d_rep=d['d_rep'], device=device)
        with torch.no_grad():
            bank.trained_slots.data = d['trained_slots'].to(bank.device)
            bank.experience_slots = d['experience_slots'].to(bank.device)
            bank.alpha_logit.data = d['alpha_logit'].to(bank.device)
            bank.activation_count = d['activation_count'].to(bank.device)
            bank.last_written = d['last_written'].to(bank.device)
            bank.surprise_at_write = d['surprise_at_write'].to(bank.device)
            bank.affect_traces = d['affect_traces'].to(bank.device)
            bank.n_written = d['n_written'].to(bank.device)
        return bank
```

**Test (`tests/test_memory_bank.py`):**

```python
import numpy as np, torch
from backend.memory_bank import PersistentMemoryBank
def test_bank():
    bank = PersistentMemoryBank(m_slots=2048, d_rep=512)
    emb = np.random.randn(1500, 512).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    assert bank.initialize_from_concept_graph(emb) == 1500
    rep = torch.randn(512, device=bank.device)
    slot, dr = bank.soft_write(rep, torch.tensor(0.8))
    assert slot.shape == (1,) and 0 < float(dr[0]) <= MAX_DRIFT_RATE
    sims, idx = bank.search(rep.unsqueeze(0), k=10)
    assert sims.shape == (1, 10) and idx.shape == (1, 10)
    # search chunks: verify large-Q path
    queries = torch.randn(3000, 512, device=bank.device)
    sims, idx = bank.search(queries, k=5)
    assert sims.shape == (3000, 5)
    # combined slots are normalized
    s = bank.slots
    norms = s.norm(dim=-1).float()
    assert (norms - 1).abs().max() < 0.05
    # gradients flow through trained_slots via .slots
    loss = bank.slots.sum()
    loss.backward()
    assert bank.trained_slots.grad is not None
    print("memory_bank OK")
if __name__ == '__main__': test_bank()
```

---

### 4.2 AffectModule — Agent 2

**File:** `backend/affect_module.py`
**Replaces:** AffectStack in the training/forward path (legacy `affect.py`
kept for v1.1 API back-compat). Reuses HALF_LIFE_* values from `backend/config.py`.

Standalone — no dependencies on other v2.0 files (only `unified_config.py` for
constants, which Agent 1 writes first; Agent 2 may inline the constants needed
if `unified_config.py` is not yet available at the moment Agent 2 starts).

```python
"""AffectModule: differentiable affect integrated into the transformer."""
import math, time
import torch, torch.nn as nn, torch.nn.functional as F

N_AFF = 12
N_HEADS = 8
D_REP = 512

HALF_LIVES_S = {
    'reaction':    2.0,
    'working':     180.0,
    'mood':        7200.0,
    'disposition': 1_209_600.0,
    'character':   6_307_200.0,
}
COMPOSITE_WEIGHTS = [0.30, 0.30, 0.20, 0.15, 0.05]


class AffectTimescales(nn.Module):
    """5 EMA buffers, not differentiable."""
    def __init__(self, n_aff=N_AFF):
        super().__init__()
        for name in HALF_LIVES_S:
            self.register_buffer(f'{name}_state',
                                 torch.zeros(n_aff, dtype=torch.float32))
        self.register_buffer('last_t', torch.tensor(0.0, dtype=torch.float64))

    @torch.no_grad()
    def update(self, affect_vec: torch.Tensor, now: float):
        dt = max(now - float(self.last_t.item()), 0.0)
        vec = affect_vec.detach().float().cpu()
        for name, hl in HALF_LIVES_S.items():
            alpha = 1.0 - math.exp(-dt * math.log(2) / hl)
            alpha = max(0.0, min(1.0, alpha))
            state = getattr(self, f'{name}_state')
            new = (1 - alpha) * state + alpha * vec
            setattr(self, f'{name}_state', new)
        self.last_t = torch.tensor(now, dtype=torch.float64)

    def composite(self) -> torch.Tensor:
        states = [getattr(self, f'{n}_state') for n in HALF_LIVES_S]
        return sum(w * s for w, s in zip(COMPOSITE_WEIGHTS, states))


class AffectModule(nn.Module):
    def __init__(self, d_model=D_REP, n_aff=N_AFF, n_heads=N_HEADS):
        super().__init__()
        self.n_heads = n_heads
        self.affect_net = nn.Sequential(
            nn.Linear(d_model, 256), nn.GELU(),
            nn.Linear(256, 64), nn.GELU(),
            nn.Linear(64, n_aff), nn.Tanh(),
        )
        self.bias_proj = nn.Linear(n_aff, n_heads)
        self.valence_head = nn.Sequential(
            nn.Linear(n_aff, 16), nn.GELU(),
            nn.Linear(16, 1), nn.Tanh(),
        )
        self.timescales = AffectTimescales(n_aff)

    def forward(
        self,
        memory_state: torch.Tensor,   # (M, D) — already mixed/combined slots
        memory_weights: torch.Tensor = None,
    ) -> dict:
        if memory_weights is not None:
            w = F.softmax(memory_weights.float(), dim=0)
            pooled = (w.unsqueeze(-1).to(memory_state.dtype)
                      * memory_state).sum(0)
        else:
            pooled = memory_state.mean(0)
        pooled_f = pooled.float()
        aff = self.affect_net(pooled_f)
        bias = self.bias_proj(aff).to(memory_state.dtype)
        valence = self.valence_head(aff).squeeze()
        arousal = torch.clamp(aff.norm() / math.sqrt(N_AFF), 0, 1)
        return {'affect_vector': aff, 'attention_bias': bias,
                'valence': valence, 'arousal': arousal}

    def update_timescales(self, aff: torch.Tensor, now: float):
        self.timescales.update(aff, now)

    def composite(self) -> torch.Tensor:
        return self.timescales.composite()

    def character(self) -> torch.Tensor:
        return self.timescales.character_state.clone()
```

**Test (`tests/test_affect_module.py`):**

```python
import torch, time
from backend.affect_module import AffectModule
def test_affect():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    m = AffectModule().to(device).to(torch.bfloat16)
    mem = torch.randn(256, 512, device=device, dtype=torch.bfloat16)
    r = m(mem)
    assert r['affect_vector'].shape == (12,)
    assert r['attention_bias'].shape == (8,)
    assert 0 <= float(r['arousal']) <= 1
    r['affect_vector'].sum().backward()
    m.update_timescales(r['affect_vector'].detach(), time.time())
    assert m.composite().shape == (12,)
    print("affect_module OK")
if __name__ == '__main__': test_affect()
```

---

### 4.3 Data pipeline — Agent 3

**File:** `scripts/prepare_v2_training_data.py`
**Replaces:** nothing (new)
**Depends on:** nothing in v2.0 (operates on existing `data/{mind}/surprised_sentences.jsonl`)

Prepares train/val/test splits and trains a BPE tokenizer.

```python
#!/usr/bin/env python3
"""Prepare v2.0 training data from existing curriculum corpus.

Reads data/{mind}/surprised_sentences.jsonl (one record per line, with affect
vector). Produces:
  - data/{mind}/v2_train.jsonl
  - data/{mind}/v2_val.jsonl
  - data/{mind}/v2_test.jsonl
  - data/{mind}/v2_vocab.json (BPE tokenizer, 16384 tokens)

Optionally adds negative (non-surprising) examples from data/encoded_corpus.db.
"""
import argparse, json, sys, os, sqlite3, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def build_bpe(corpus_path: str, save_path: str, vocab_size: int = 16384):
    try:
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers
    except ImportError:
        os.system('pip install tokenizers --break-system-packages -q')
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers
    tok = Tokenizer(models.BPE(unk_token='<unk>'))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=['<pad>', '<bos>', '<eos>', '<unk>'],
        min_frequency=2,
    )
    # build temp plaintext file (one sentence per line) for trainer
    txt_path = save_path + '.tmp.txt'
    with open(corpus_path) as fin, open(txt_path, 'w') as fout:
        for line in fin:
            try:
                rec = json.loads(line)
                s = rec.get('sentence', '').replace('\n', ' ').strip()
                if s:
                    fout.write(s + '\n')
            except Exception:
                continue
    tok.train([txt_path], trainer)
    tok.save(save_path)
    os.remove(txt_path)
    print(f"Tokenizer: {tok.get_vocab_size()} tokens → {save_path}")
    return tok


def prepare_splits(corpus_path: str, out_dir: str, max_examples: int = None):
    examples = []
    affect_n = 0
    with open(corpus_path) as f:
        for i, line in enumerate(f):
            if max_examples and i >= max_examples:
                break
            try:
                rec = json.loads(line)
                s = rec.get('sentence', '').strip()
                if len(s.split()) < 3:
                    continue
                ex = {'text': s, 'is_surprising': 1.0}
                if rec.get('affect'):
                    ex['affect'] = rec['affect']
                    affect_n += 1
                examples.append(ex)
            except Exception:
                continue
    random.seed(42)
    random.shuffle(examples)
    n = len(examples); n_tr = int(n * 0.9); n_va = int(n * 0.05)
    splits = {
        'train': examples[:n_tr],
        'val':   examples[n_tr:n_tr + n_va],
        'test':  examples[n_tr + n_va:],
    }
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    for name, exs in splits.items():
        p = out / f'v2_{name}.jsonl'
        with open(p, 'w') as f:
            for ex in exs:
                f.write(json.dumps(ex) + '\n')
        print(f"{name}: {len(exs)} → {p}")
    print(f"affect labels: {affect_n}/{n}")
    return {'train': n_tr, 'val': n_va, 'test': n - n_tr - n_va,
            'affect': affect_n}


def add_negatives(encoded_db: str, train_path: str, n: int = 50_000):
    if not Path(encoded_db).exists():
        print(f"no encoded_corpus.db at {encoded_db}; skipping negatives")
        return 0
    conn = sqlite3.connect(encoded_db)
    try:
        rows = conn.execute(
            "SELECT sentence FROM encoded_sentences "
            "WHERE level='sentence' ORDER BY RANDOM() LIMIT ?", (n,)
        ).fetchall()
    finally:
        conn.close()
    added = 0
    with open(train_path, 'a') as f:
        for (s,) in rows:
            if not s or len(s.split()) < 3:
                continue
            f.write(json.dumps({'text': s, 'is_surprising': 0.0}) + '\n')
            added += 1
    print(f"+{added} negatives → {train_path}")
    return added


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--mind', default='first')
    ap.add_argument('--max-examples', type=int, default=None)
    ap.add_argument('--skip-negatives', action='store_true')
    args = ap.parse_args()

    from backend.mind_paths import MindPaths
    paths = MindPaths(args.mind)

    corpus = paths.surprised_log
    out_dir = paths.root
    vocab_out = f"{paths.root}/v2_vocab.json"

    stats = prepare_splits(corpus, out_dir, args.max_examples)
    build_bpe(corpus, vocab_out)
    if not args.skip_negatives:
        add_negatives('data/encoded_corpus.db',
                      f'{out_dir}/v2_train.jsonl')
    print("done")
```

**Verify:**

```bash
python3 scripts/prepare_v2_training_data.py --mind first --max-examples 5000
ls -la data/first/v2_*.jsonl data/first/v2_vocab.json
```

---

### 4.4 SparseMemoryTransformer — Agent 4

**File:** `backend/memory_transformer.py`
**Replaces:** `wave_field.py` (legacy kept). 4 attention layers = 4 wave steps.
**Depends on:** Agent 1's `PersistentMemoryBank` and `backend/unified_config.py`.

**CORRECTION 2 + 3:** Within the memory transformer's *self*-attention, each
memory slot attends only to its top-K neighbors. The neighbor index uses the
chunked-in-both-dims `bank.search()` from Agent 1. This is the right direction:
small K=64 reduces (M, K) attention matrix to manageable size on M1.

```python
"""SparseMemoryTransformer: sparse self-attn over the memory bank.

Replaces WaveField. Each transformer layer = one wave step.
Sparse attention: each slot attends to top-K=64 neighbors via gather-scatter.
Affect bias added to attention scores per head.
"""
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from backend.memory_bank import PersistentMemoryBank
from backend.unified_config import (
    D_REP, N_HEADS, N_MEM_LAYERS, TOP_K_NBR, DROPOUT := 0.1
)


class SparseSelfAttention(nn.Module):
    def __init__(self, d_model=D_REP, n_heads=N_HEADS, top_k=TOP_K_NBR,
                 dropout=DROPOUT):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.top_k = top_k
        self.scale = self.d_head ** -0.5
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,                       # (M, D) full memory state
        memory_bank: PersistentMemoryBank,     # for neighbor search
        affect_bias: torch.Tensor = None,      # (n_heads,)
        chunk_size: int = 4096,
    ) -> torch.Tensor:
        M, D = x.shape
        device = x.device
        Q = self.q_proj(x).view(M, self.n_heads, self.d_head)
        K = self.k_proj(x).view(M, self.n_heads, self.d_head)
        V = self.v_proj(x).view(M, self.n_heads, self.d_head)

        out = torch.zeros_like(x)
        for s in range(0, M, chunk_size):
            e = min(s + chunk_size, M)
            q_c = Q[s:e]                       # (chunk, H, d_head)
            chunk_len = e - s

            # neighbor search: query with the raw x[s:e], not projected.
            # we want geometric neighbors in representation space.
            with torch.no_grad():
                _, top_idx = memory_bank.search(
                    x[s:e].detach(), k=self.top_k,
                    q_chunk=min(chunk_len, 1024))
            # top_idx: (chunk, K)

            flat = top_idx.reshape(-1)              # (chunk*K,)
            k_g = K[flat].view(chunk_len, self.top_k, self.n_heads,
                               self.d_head)
            v_g = V[flat].view(chunk_len, self.top_k, self.n_heads,
                               self.d_head)

            # scores: (chunk, H, K)
            # q_c: (chunk, H, d_head); k_g: (chunk, K, H, d_head)
            scores = torch.einsum('chd,ckhd->chk', q_c, k_g) * self.scale
            if affect_bias is not None:
                scores = scores + affect_bias.view(1, -1, 1)
            attn = F.softmax(scores, dim=-1)
            attn = self.dropout(attn)
            # weighted sum: (chunk, H, K) × (chunk, K, H, d_head) → (chunk, H, d_head)
            o = torch.einsum('chk,ckhd->chd', attn, v_g)
            out[s:e] = o.reshape(chunk_len, D)

        return self.out_proj(out)


class MemTransformerLayer(nn.Module):
    def __init__(self, d_model=D_REP, n_heads=N_HEADS, d_ff=D_REP * 4):
        super().__init__()
        self.attn = SparseSelfAttention(d_model, n_heads)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(DROPOUT),
            nn.Linear(d_ff, d_model), nn.Dropout(DROPOUT))
        self.n1 = nn.LayerNorm(d_model)
        self.n2 = nn.LayerNorm(d_model)

    def forward(self, x, bank, affect_bias=None):
        x = x + self.attn(self.n1(x), bank, affect_bias)
        x = x + self.ff(self.n2(x))
        return x


class SparseMemoryTransformer(nn.Module):
    def __init__(self, memory_bank: PersistentMemoryBank,
                 n_layers=N_MEM_LAYERS, use_checkpoint=True):
        super().__init__()
        self.bank = memory_bank
        self.use_checkpoint = use_checkpoint
        self.layers = nn.ModuleList(
            [MemTransformerLayer() for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(D_REP)

    def forward(self, memory_state=None, affect_bias=None) -> torch.Tensor:
        if memory_state is None:
            memory_state = self.bank.slots
        x = memory_state
        for layer in self.layers:
            if self.use_checkpoint and self.training:
                x = checkpoint(layer, x, self.bank, affect_bias,
                               use_reentrant=False)
            else:
                x = layer(x, self.bank, affect_bias)
        return self.final_norm(x)
```

Note: `DROPOUT := 0.1` in the import line is invalid Python; agents must put
`DROPOUT = 0.1` at top-of-module or import it from `unified_config`.

**Test (`tests/test_memory_transformer.py`):**

```python
import numpy as np, torch
from backend.memory_bank import PersistentMemoryBank
from backend.memory_transformer import SparseMemoryTransformer
def test_xfmr():
    bank = PersistentMemoryBank(m_slots=1024, d_rep=512)
    emb = np.random.randn(800, 512).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    bank.initialize_from_concept_graph(emb)
    t = SparseMemoryTransformer(bank, n_layers=2,
                                use_checkpoint=False).to(bank.device).to(torch.bfloat16)
    out = t()
    assert out.shape == (1024, 512)
    bias = torch.zeros(8, device=bank.device, dtype=torch.bfloat16)
    out2 = t(affect_bias=bias + 2.0)
    diff = (out2 - out).abs().mean()
    assert diff > 0
    out.sum().backward()
    print("memory_transformer OK")
if __name__ == '__main__': test_xfmr()
```

---

### 4.5 MultiModalEncoder — Agent 5

**File:** `backend/encoder.py`
**Replaces:** `fusion.py` and the GloVe encoding role of `encoders.py`.
**Depends on:** Agent 1's `PersistentMemoryBank` and `backend/unified_config.py`.

**CORRECTION 2:** Cross-attention direction inverted. Input tokens are
**queries**. Top-K active memory slots are **keys/values**. This produces
attention matrices of size (L_input, K=256), which is tiny. The encoder's
output is a memory-update tensor written back via scatter to the active slots.

```python
"""MultiModalEncoder: perceiver-style encoder.

Cross-attention DIRECTION (CORRECTION 2):
  - input tokens (L) query
  - top-K=256 active memory slots key/value
  - output: per-input-token contextualized representations (L, D)
  - we then scatter-update the K active slots from input tokens that attended to them

This is the right way around. Previous direction (M queries → L keys)
produced (65536, 64) attention matrices, OOM on M1.
"""
import torch, torch.nn as nn, torch.nn.functional as F
import logging
from typing import Optional
from backend.memory_bank import PersistentMemoryBank
from backend.unified_config import (
    D_REP, N_HEADS, N_ENC_LAYERS, TOP_K_ACTIVE, VOCAB_SIZE,
)

log = logging.getLogger('encoder')


class CurriculumTokenizer:
    PAD, BOS, EOS, UNK = 0, 1, 2, 3
    def __init__(self, path: str = None):
        self._tok = None
        if path:
            self.load(path)

    def load(self, path: str):
        from tokenizers import Tokenizer
        self._tok = Tokenizer.from_file(path)
        self.vocab_size = self._tok.get_vocab_size()

    @classmethod
    def from_path(cls, path: str):
        obj = cls(); obj.load(path); return obj

    def encode(self, text: str) -> list[int]:
        if self._tok is None:
            return [self.BOS] + [ord(c) % 256 for c in text[:256]] + [self.EOS]
        ids = self._tok.encode(text).ids
        return [self.BOS] + ids + [self.EOS]

    def decode(self, ids: list[int]) -> str:
        if self._tok is None:
            return ''.join(chr(i) for i in ids
                           if 32 <= i < 127)
        keep = [i for i in ids if i not in {self.PAD, self.BOS, self.EOS}]
        return self._tok.decode(keep)


class InputCrossAttn(nn.Module):
    """Input attends to top-K memory slots. One PC level."""
    def __init__(self, d_model=D_REP, n_heads=N_HEADS):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        # top-down predictor: predict input from memory pool
        self.predictor = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, d_model))
        self.n_in = nn.LayerNorm(d_model)
        self.n_mem = nn.LayerNorm(d_model)
        self.n_ff = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(),
            nn.Linear(d_model * 4, d_model))

    def forward(self, inputs: torch.Tensor, mem_active: torch.Tensor):
        """
        inputs: (L, D)
        mem_active: (K, D) top-K active memory slots
        returns:
          updated_inputs: (L, D)
          pc_error: (D,) prediction error magnitude (mean over input tokens)
        """
        L = inputs.shape[0]
        K = mem_active.shape[0]
        x = self.n_in(inputs)
        m = self.n_mem(mem_active)
        Q = self.q_proj(x).view(L, self.n_heads, self.d_head)
        Kp = self.k_proj(m).view(K, self.n_heads, self.d_head)
        V = self.v_proj(m).view(K, self.n_heads, self.d_head)
        # (L, H, d) × (K, H, d) → (L, H, K)
        scores = torch.einsum('lhd,khd->lhk', Q, Kp) * self.scale
        attn = F.softmax(scores, dim=-1)
        out = torch.einsum('lhk,khd->lhd', attn, V).reshape(L, -1)
        out = self.out_proj(out)
        updated = inputs + out
        updated = updated + self.ff(self.n_ff(updated))

        # PC prediction error: top-down from memory pool predicts input
        mem_pool = mem_active.mean(0)              # (D,)
        predicted = self.predictor(mem_pool)       # (D,)
        actual = inputs.mean(0)                    # (D,)
        pc_error = (actual - predicted).float()
        return updated, pc_error


class MultiModalEncoder(nn.Module):
    def __init__(
        self,
        memory_bank: PersistentMemoryBank,
        vocab_size: int = VOCAB_SIZE,
        n_layers: int = N_ENC_LAYERS,
        d_model: int = D_REP,
        tokenizer: CurriculumTokenizer = None,
        top_k_active: int = TOP_K_ACTIVE,
    ):
        super().__init__()
        self.bank = memory_bank
        self.tokenizer = tokenizer
        self.top_k = top_k_active

        self.text_embedding = nn.Embedding(vocab_size, d_model)
        self.text_pos_embedding = nn.Embedding(2048, d_model)

        # vision/audio projections (frozen pretrained encoders are lazy-loaded
        # in encode_vision / encode_audio; stub for now)
        D_CLIP = 512
        D_WHISPER = 512
        self.vision_proj = nn.Linear(D_CLIP, d_model)
        self.audio_proj = nn.Linear(D_WHISPER, d_model)
        self._clip = None
        self._whisper = None

        self.layers = nn.ModuleList([InputCrossAttn() for _ in range(n_layers)])

    def encode_text(self, text: str) -> torch.Tensor:
        device = next(self.parameters()).device
        if self.tokenizer:
            ids = self.tokenizer.encode(text)
        else:
            ids = [1] + [ord(c) % 1024 for c in text[:256]] + [2]
        ids_t = torch.tensor(ids, device=device, dtype=torch.long)
        positions = torch.arange(len(ids_t), device=device)
        emb = self.text_embedding(ids_t) + self.text_pos_embedding(positions)
        return emb.to(torch.bfloat16)  # (L, D)

    def encode_vision(self, image: torch.Tensor) -> torch.Tensor:
        device = next(self.parameters()).device
        if self._clip is None:
            try:
                import clip
                self._clip, _ = clip.load('ViT-B/32', device=device)
                for p in self._clip.parameters(): p.requires_grad_(False)
            except Exception:
                return torch.zeros(1, D_REP, device=device,
                                   dtype=torch.bfloat16)
        if image.dim() == 3:
            image = image.unsqueeze(0)
        with torch.no_grad():
            feat = self._clip.encode_image(image).to(torch.bfloat16)
        return self.vision_proj(feat)

    def encode_audio(self, audio: torch.Tensor) -> torch.Tensor:
        device = next(self.parameters()).device
        # Whisper integration deferred; return zero tokens
        return torch.zeros(1, D_REP, device=device, dtype=torch.bfloat16)

    def forward(
        self,
        text: str = None,
        image: torch.Tensor = None,
        audio: torch.Tensor = None,
    ) -> dict:
        device = next(self.parameters()).device
        tokens = []
        if text is not None:
            tokens.append(self.encode_text(text))
        if image is not None:
            tokens.append(self.encode_vision(image))
        if audio is not None:
            tokens.append(self.encode_audio(audio))
        if not tokens:
            raise ValueError("at least one modality required")
        inputs = torch.cat(tokens, dim=0)  # (L, D)

        # get top-K active memory slots
        counts = self.bank.activation_count.float()
        if (counts > 0).sum() < self.top_k:
            # not enough active slots yet — pad with first-N
            top_idx = torch.arange(min(self.top_k, self.bank.m_slots),
                                   device=self.bank.device)
        else:
            _, top_idx = counts.topk(self.top_k)
        mem_active = self.bank.slots[top_idx]  # (K, D)

        pc_errors = []
        x = inputs
        for layer in self.layers:
            x, err = layer(x, mem_active)
            pc_errors.append(err)

        # surprise = mean error magnitude across layers
        surprise = torch.stack([e.norm() for e in pc_errors]).mean()

        # memory_delta to scatter back: pool input contributions, project,
        # write proportional to attention. Simplification: aggregate input
        # rep, return it; the soft_write step lives in UnifiedMind.process().
        input_rep = x.mean(0)  # (D,)

        return {
            'updated_inputs': x,                 # (L, D)
            'input_rep': input_rep,              # (D,) for soft-write
            'active_slot_indices': top_idx,      # (K,)
            'mem_active': mem_active,            # (K, D)
            'pc_errors': pc_errors,
            'surprise': surprise,
        }
```

**Test (`tests/test_encoder.py`):**

```python
import numpy as np, torch
from backend.memory_bank import PersistentMemoryBank
from backend.encoder import MultiModalEncoder
def test_enc():
    bank = PersistentMemoryBank(m_slots=512, d_rep=512)
    emb = np.random.randn(300, 512).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    bank.initialize_from_concept_graph(emb)
    enc = MultiModalEncoder(bank, vocab_size=1024, n_layers=2).to(bank.device).to(torch.bfloat16)
    r = enc(text="justice and natural selection")
    assert r['updated_inputs'].dim() == 2
    assert r['input_rep'].shape == (512,)
    assert len(r['pc_errors']) == 2
    r['surprise'].backward()
    print("encoder OK")
if __name__ == '__main__': test_enc()
```

---

### 4.6 UnifiedDecoder — Agent 6

**File:** `backend/decoder.py`
**Replaces:** `native_head.py` and `expression_graph.py` (legacy kept)
**Depends on:** Agent 1's `PersistentMemoryBank`, Agent 5's `MultiModalEncoder` and
`CurriculumTokenizer`, Agent 1's `unified_config.py`.

Standard autoregressive decoder. Cross-attn over top-K=256 active memory slots
(same pool as encoder). Weight tying with encoder.text_embedding via the LM
head.

```python
"""UnifiedDecoder: expression via cross-attention to memory."""
import torch, torch.nn as nn, torch.nn.functional as F
from typing import Optional
from backend.memory_bank import PersistentMemoryBank
from backend.unified_config import D_REP, N_HEADS, N_DEC_LAYERS, TOP_K_ACTIVE

MAX_GEN_LEN = 64
D_FF = D_REP * 4


class CausalSelfAttn(nn.Module):
    def __init__(self, d_model=D_REP, n_heads=N_HEADS):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self._mask_cache: dict[int, torch.Tensor] = {}

    def _mask(self, T, device):
        if T not in self._mask_cache:
            self._mask_cache[T] = torch.tril(
                torch.ones(T, T, device=device, dtype=torch.bool))
        return self._mask_cache[T]

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(2)
        q = q.transpose(1, 2); k = k.transpose(1, 2); v = v.transpose(1, 2)
        sc = (q @ k.transpose(-2, -1)) * self.scale
        sc = sc.masked_fill(~self._mask(T, x.device), float('-inf'))
        a = F.softmax(sc, dim=-1)
        return self.out((a @ v).transpose(1, 2).reshape(B, T, D))


class MemoryCrossAttn(nn.Module):
    def __init__(self, d_model=D_REP, n_heads=N_HEADS):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, mem_active: torch.Tensor):
        # x: (B, T, D); mem_active: (K, D)
        B, T, D = x.shape; K = mem_active.shape[0]
        Q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        Kp = self.k_proj(mem_active).view(1, K, self.n_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(mem_active).view(1, K, self.n_heads, self.d_head).transpose(1, 2)
        sc = Q @ Kp.transpose(-2, -1) * self.scale
        a = F.softmax(sc, dim=-1)
        return self.out_proj((a @ V).transpose(1, 2).reshape(B, T, D))


class DecoderLayer(nn.Module):
    def __init__(self, d_model=D_REP):
        super().__init__()
        self.self_attn = CausalSelfAttn(d_model)
        self.cross_attn = MemoryCrossAttn(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, D_FF), nn.GELU(),
            nn.Linear(D_FF, d_model))
        self.n1 = nn.LayerNorm(d_model)
        self.n2 = nn.LayerNorm(d_model)
        self.n3 = nn.LayerNorm(d_model)

    def forward(self, x, mem_active):
        x = x + self.self_attn(self.n1(x))
        x = x + self.cross_attn(self.n2(x), mem_active)
        x = x + self.ff(self.n3(x))
        return x


class UnifiedDecoder(nn.Module):
    def __init__(
        self,
        memory_bank: PersistentMemoryBank,
        vocab_size: int,
        shared_embedding: Optional[nn.Embedding] = None,
        n_layers: int = N_DEC_LAYERS,
        d_model: int = D_REP,
    ):
        super().__init__()
        self.bank = memory_bank
        self.vocab_size = vocab_size
        self.token_embedding = shared_embedding or nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(MAX_GEN_LEN + 2, d_model)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(d_model)
        # weight-tied LM head
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, input_ids: torch.Tensor, mem_active: torch.Tensor):
        B, T = input_ids.shape
        device = input_ids.device
        pos = torch.arange(T, device=device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.pos_embedding(pos)
        x = x.to(torch.bfloat16)
        for layer in self.layers:
            x = layer(x, mem_active)
        x = self.final_norm(x)
        logits = self.lm_head(x.float())
        return logits, x

    @torch.no_grad()
    def generate(
        self,
        mem_active: torch.Tensor,
        tokenizer,
        max_new_tokens: int = 40,
        temperature: float = 0.8,
        top_p: float = 0.9,
        bos_id: int = 1,
        eos_id: int = 2,
    ) -> tuple[str, torch.Tensor]:
        device = mem_active.device
        ids = torch.tensor([[bos_id]], device=device, dtype=torch.long)
        gen: list[int] = []
        for _ in range(max_new_tokens):
            logits, _ = self.forward(ids, mem_active)
            nl = logits[0, -1, :] / max(temperature, 1e-3)
            sv, si = torch.sort(nl, descending=True)
            cp = torch.cumsum(F.softmax(sv, dim=-1), dim=-1)
            rm = cp > top_p
            rm[1:] = rm[:-1].clone()
            rm[0] = False
            nl[si[rm]] = float('-inf')
            probs = F.softmax(nl, dim=-1)
            nxt = int(torch.multinomial(probs, 1))
            if nxt == eos_id: break
            gen.append(nxt)
            ids = torch.cat([ids, torch.tensor([[nxt]], device=device)], dim=1)
        return tokenizer.decode(gen), torch.tensor(gen, device=device)


def compute_expression_gap(
    gen_ids: torch.Tensor,
    bank_centroid: torch.Tensor,
    encoder,
    tokenizer,
) -> float:
    surface = tokenizer.decode(gen_ids.tolist())
    with torch.no_grad():
        r = encoder(text=surface)
        gen_rep = F.normalize(r['input_rep'].float(), dim=-1)
        c = F.normalize(bank_centroid.float(), dim=-1)
        return 1.0 - float(torch.dot(gen_rep, c))
```

**Test (`tests/test_decoder.py`):**

```python
import numpy as np, torch
from backend.memory_bank import PersistentMemoryBank
from backend.decoder import UnifiedDecoder
def test_dec():
    bank = PersistentMemoryBank(m_slots=256, d_rep=512)
    emb = np.random.randn(200, 512).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    bank.initialize_from_concept_graph(emb)
    V = 1024
    d = UnifiedDecoder(bank, vocab_size=V, n_layers=2).to(bank.device).to(torch.bfloat16)
    assert d.lm_head.weight is d.token_embedding.weight   # weight tying
    mem_active = bank.slots[:128]                          # (K, D)
    ids = torch.randint(0, V, (2, 16), device=bank.device)
    logits, _ = d(ids, mem_active)
    assert logits.shape == (2, 16, V)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, V),
        torch.randint(0, V, (2 * 16,), device=bank.device))
    loss.backward()
    print("decoder OK")
if __name__ == '__main__': test_dec()
```

---

### 4.7 Training infrastructure — Agent 7

**File:** `backend/training.py` + `scripts/train_unified.py`
**Replaces:** nothing (new)
**Depends on:** all of 1–6.

**CORRECTION 4:** GradNorm only updates every 200 steps (`GRADNORM_EVERY`).
Between updates, weights are held constant. Skip per-loss `torch.autograd.grad`
on intermediate steps — just do one combined backward.

```python
"""UnifiedMindTrainer + train script.

GradNorm balances 5 losses every 200 steps. Between updates, the previously
computed weights are used as constants.
"""
import torch, torch.nn as nn, torch.nn.functional as F
from torch.optim import AdamW
import time, json, logging, math
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

from backend.unified_config import (
    D_REP, N_HEADS, M_SLOTS, BATCH_SIZE, GRAD_ACCUM, MAX_STEPS,
    WARMUP_STEPS, GRADNORM_EVERY,
)

log = logging.getLogger('training')


@dataclass
class TrainingConfig:
    max_steps: int = MAX_STEPS
    batch_size: int = BATCH_SIZE
    grad_accum: int = GRAD_ACCUM
    warmup_steps: int = WARMUP_STEPS
    lr_mem: float = 1e-4
    lr_enc: float = 1e-4
    lr_dec: float = 1e-4
    lr_affect: float = 3e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    w_init: tuple = (1.0, 1.0, 0.5, 0.3, 0.2)   # mask, lm, align, affect, surp
    gradnorm_alpha: float = 1.5
    gradnorm_lr: float = 1e-2
    gradnorm_every: int = GRADNORM_EVERY
    checkpoint_every: int = 1000
    log_every: int = 50
    empty_cache_every: int = 50
    vocab_size: int = 16384


class GradNorm(nn.Module):
    """Per-task loss weights, updated every N steps."""
    def __init__(self, n_tasks: int, alpha: float, init_weights: tuple):
        super().__init__()
        self.alpha = alpha
        self.log_w = nn.Parameter(torch.log(torch.tensor(init_weights)))
        self.register_buffer('initial_losses', torch.zeros(n_tasks))
        self.register_buffer('initialized', torch.tensor(False))

    @property
    def w(self):
        return torch.exp(self.log_w)

    @torch.no_grad()
    def init_losses(self, losses: list[torch.Tensor]):
        if not bool(self.initialized):
            for i, l in enumerate(losses):
                self.initial_losses[i] = float(l)
            self.initialized.fill_(True)

    def gradnorm_step(
        self,
        losses: list[torch.Tensor],
        shared_params: list[torch.Tensor],
        gradnorm_opt: torch.optim.Optimizer,
    ):
        """Run GradNorm update. Call every gradnorm_every steps."""
        gradnorm_opt.zero_grad()
        weights = self.w
        grad_norms = []
        for i, loss in enumerate(losses):
            grads = torch.autograd.grad(
                weights[i] * loss, shared_params,
                retain_graph=True, allow_unused=True)
            n2 = sum(float((g**2).sum()) for g in grads if g is not None)
            grad_norms.append(n2 ** 0.5)
        gn = torch.tensor(grad_norms, dtype=torch.float32,
                          device=self.log_w.device)
        mean = gn.mean()
        loss_ratios = torch.stack([
            l.detach().float() / (self.initial_losses[i] + 1e-8)
            for i, l in enumerate(losses)
        ])
        rel = loss_ratios / (loss_ratios.mean() + 1e-8)
        target = (mean * (rel ** self.alpha)).detach()
        gn_loss = F.l1_loss(gn, target)
        gn_loss.backward()
        gradnorm_opt.step()


class UnifiedMindTrainer:
    def __init__(self, cfg, memory_bank, memory_transformer, encoder,
                 affect_module, decoder, device):
        self.cfg = cfg
        self.device = device
        self.bank = memory_bank
        self.mt = memory_transformer
        self.enc = encoder
        self.aff = affect_module
        self.dec = decoder
        for m in (memory_transformer, encoder, affect_module, decoder):
            m.to(device).to(torch.bfloat16)
        # memory_bank itself uses bfloat16 already; alpha_logit stays fp32

        self.opts = {
            'mem': AdamW([p for p in memory_transformer.parameters()
                          if p.requires_grad],
                         lr=cfg.lr_mem, weight_decay=cfg.weight_decay),
            'enc': AdamW([p for p in encoder.parameters()
                          if p.requires_grad],
                         lr=cfg.lr_enc, weight_decay=cfg.weight_decay),
            'aff': AdamW(affect_module.parameters(),
                         lr=cfg.lr_affect, weight_decay=cfg.weight_decay),
            'dec': AdamW(decoder.parameters(),
                         lr=cfg.lr_dec, weight_decay=cfg.weight_decay),
            'bank': AdamW([memory_bank.trained_slots, memory_bank.alpha_logit],
                          lr=cfg.lr_mem, weight_decay=0.0),
        }
        self.balancer = GradNorm(5, cfg.gradnorm_alpha, cfg.w_init).to(device)
        self.gn_opt = AdamW(self.balancer.parameters(), lr=cfg.gradnorm_lr)
        self.step = 0
        self.best = float('inf')

    def compute_losses(self, batch: dict) -> tuple[dict, dict]:
        device = self.device
        cfg = self.cfg
        text = batch.get('text', '')

        # 1) encode
        enc_out = self.enc(text=text, image=batch.get('image'))
        active_idx = enc_out['active_slot_indices']
        mem_active = enc_out['mem_active']

        # 2) memory transformer on full memory
        memory_full = self.mt(self.bank.slots)
        # active subset for downstream
        mem_active_xfm = memory_full[active_idx]

        # 3) affect from memory
        aff_out = self.aff(memory_full)

        losses = {}

        # L_mask: mask 15% of active memory, reconstruct
        M_active = mem_active_xfm.shape[0]
        n_mask = max(1, int(M_active * 0.15))
        mask_idx = torch.randperm(M_active, device=device)[:n_mask]
        original = mem_active_xfm[mask_idx].detach().float()
        masked = mem_active_xfm.clone()
        masked[mask_idx] = 0.0
        # one-layer recovery: re-run MT layer over masked
        recovered = self.mt.layers[0](masked, self.bank, aff_out['attention_bias'])
        losses['mask'] = F.mse_loss(recovered[mask_idx].float(), original)

        # L_lm: next-token cross-entropy
        if 'target_ids' in batch:
            tgt = batch['target_ids'].to(device).long()    # (B, T)
            bos = torch.full((tgt.shape[0], 1), 1,
                             device=device, dtype=torch.long)
            inp = torch.cat([bos, tgt[:, :-1]], dim=1)
            logits, _ = self.dec(inp, mem_active_xfm)
            losses['lm'] = F.cross_entropy(
                logits.reshape(-1, cfg.vocab_size),
                tgt.reshape(-1),
                ignore_index=0)
        else:
            losses['lm'] = torch.tensor(0.0, device=device)

        # L_align: cross-modal alignment (when both modalities)
        if batch.get('image') is not None and text:
            tr = self.enc(text=text)['input_rep']
            vr = self.enc(image=batch['image'])['input_rep']
            tr_n = F.normalize(tr.float(), dim=-1)
            vr_n = F.normalize(vr.float(), dim=-1)
            losses['align'] = F.mse_loss(tr_n, vr_n)
        else:
            losses['align'] = torch.tensor(0.0, device=device)

        # L_affect: predict affect (if labels) or self-supervised consistency
        if 'next_affect' in batch:
            tgt_aff = batch['next_affect'].to(device).float().squeeze(0)
            losses['affect'] = F.mse_loss(aff_out['affect_vector'].float(),
                                          tgt_aff)
        else:
            surp = enc_out['surprise'].float()
            mag = aff_out['affect_vector'].float().norm()
            losses['affect'] = F.mse_loss(mag.unsqueeze(0), surp.unsqueeze(0))

        # L_surp: surprise binary classification
        if 'is_surprising' in batch:
            label = batch['is_surprising'].to(device).float().squeeze()
            pred = enc_out['surprise'].float().unsqueeze(0)
            losses['surp'] = F.binary_cross_entropy_with_logits(
                pred.expand_as(label.unsqueeze(0)), label.unsqueeze(0))
        else:
            losses['surp'] = torch.tensor(0.0, device=device)

        return losses, enc_out

    def train_step(self, batch: dict) -> dict:
        cfg = self.cfg; device = self.device
        for opt in self.opts.values(): opt.zero_grad()

        per_loss_running = {}
        total_running = 0.0

        for _ in range(cfg.grad_accum):
            losses, enc_out = self.compute_losses(batch)
            tasks = ['mask', 'lm', 'align', 'affect', 'surp']
            task_losses = [losses[t] for t in tasks]
            self.balancer.init_losses(task_losses)

            # Combined backward with current weights (no per-loss grad walk)
            w = self.balancer.w.detach()  # **CORRECTION 4**: detach during accum
            total = sum(weight * l for weight, l in zip(w, task_losses))
            total = total / cfg.grad_accum
            total.backward()

            for t in tasks:
                per_loss_running[t] = per_loss_running.get(t, 0.0) + float(losses[t]) / cfg.grad_accum
            total_running += float(total) * cfg.grad_accum / cfg.grad_accum

        # clip + step
        all_params = []
        for m in (self.mt, self.enc, self.aff, self.dec):
            all_params.extend(p for p in m.parameters() if p.requires_grad)
        all_params.extend([self.bank.trained_slots, self.bank.alpha_logit])
        torch.nn.utils.clip_grad_norm_(all_params, cfg.max_grad_norm)
        for opt in self.opts.values(): opt.step()

        self.step += 1

        # GradNorm weight update every N steps (one-shot, fresh forward)
        if self.step % cfg.gradnorm_every == 0:
            try:
                losses, _ = self.compute_losses(batch)
                tasks = ['mask', 'lm', 'align', 'affect', 'surp']
                task_losses = [losses[t] for t in tasks]
                # shared params: memory transformer's first layer
                shared = list(self.mt.layers[0].parameters())
                self.balancer.gradnorm_step(task_losses, shared, self.gn_opt)
            except Exception as e:
                log.warning(f"GradNorm update skipped: {e}")

        if self.step % cfg.empty_cache_every == 0:
            if device.type == 'mps':
                torch.mps.empty_cache()

        return {'total_loss': total_running, **per_loss_running,
                'weights': self.balancer.w.detach().tolist()}

    def save(self, mind_name: str, tag: str = ''):
        from backend.unified_config import checkpoint_dir, memory_bank_path
        d = Path(checkpoint_dir(mind_name)); d.mkdir(parents=True, exist_ok=True)
        torch.save({
            'step': self.step,
            'mem': self.mt.state_dict(),
            'enc': self.enc.state_dict(),
            'aff': self.aff.state_dict(),
            'dec': self.dec.state_dict(),
            'balancer': self.balancer.state_dict(),
            'opts': {k: v.state_dict() for k, v in self.opts.items()},
            'best': self.best,
            'cfg': asdict(self.cfg),
        }, d / f'ckpt_{self.step:08d}{tag}.pt')
        self.bank.save(memory_bank_path(mind_name))
```

And `scripts/train_unified.py` — see the spec doc for the full driver script
(builds tokenizer if missing, initializes memory_bank from concept graph,
constructs all components with weight tying, iterates DataLoader, calls
trainer.train_step, checkpoints).

**Smoke test (no training):**

```bash
python3 -c "
from backend.training import TrainingConfig, UnifiedMindTrainer
print('TrainingConfig:', TrainingConfig())
print('UnifiedMindTrainer imports OK')
"
```

---

## 5. Reconciliation (after all 7 agents complete)

1. Run each component test in isolation:
   ```bash
   python3 tests/test_memory_bank.py
   python3 tests/test_affect_module.py
   python3 tests/test_memory_transformer.py
   python3 tests/test_encoder.py
   python3 tests/test_decoder.py
   python3 -c "from backend.training import TrainingConfig; print('OK')"
   ```
2. Legacy regressions still pass: `python3 phase1.py && python3 phase3.py`
3. Agent 8 wires `backend/unified_mind.py` + `/unified/*` API endpoints.
4. Agent 8's integration test runs end-to-end (text in → response out, no crash).
5. Tag v2.0 only after 1–4 all pass.

## 6. Things agents must NOT do

- Modify legacy files (graph.py, wave_field.py, native_head.py, affect.py,
  fusion.py, predictive_coding.py).
- Touch files owned by another agent.
- Introduce circular imports (memory_bank → no deps; affect_module → no deps;
  others → only depend on already-completed components).
- Skip MPS rules from §3.
- Add new abstractions, helpers, or "improvements" not in the spec — match the
  spec literally and the reconciliation runs clean.
