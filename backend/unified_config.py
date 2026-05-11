"""Shared constants for v2.0 unified architecture."""
import torch

# MPS memory cap — leave headroom for OS and other processes on 16GB M1.
# 0.6 × 16GB ≈ 9.6GB hard cap for this process. Applied at import time so
# any code that imports unified_config picks it up before allocating.
if torch.backends.mps.is_available():
    try:
        torch.mps.set_per_process_memory_fraction(0.6)
    except Exception:
        pass  # older PyTorch may not support this

# representation
D_REP        = 512
N_HEADS      = 8
D_HEAD       = D_REP // N_HEADS  # 64

# memory bank — initially halved from 65536 to 32768 after the first M1
# OOM (2026-05-11). 32768 then 16384 still OOM'd at step 1: with the
# 0.6 MPS fraction the process ends up with ~7.1GB usable, and ~5GB
# is permanently held by bank slots + AdamW state + transformer
# activations + memory-bank index tensor (q_chunk=1024 × m_chunk=4096
# × int64 = 32MB per chunk × cached). Dropped to 8192 — small enough
# that search needs only 2 m-chunks and the index tensor stays modest.
# initialize_from_concept_graph slices top-M_SLOTS by activation count,
# so all 56K concepts are still considered; only the most-activated kept.
M_SLOTS      = 8192
TOP_K_NBR    = 64       # sparse attention neighborhood
TOP_K_ACTIVE = 256      # decoder cross-attention pool

# Attention mode in SparseSelfAttention: dense when M_SLOTS ≤ threshold
# (single matmul, no kernel-launch overhead from gather-scatter), sparse
# (top-K=64 gather-scatter) when M_SLOTS exceeds it. Dense at M=8192:
# (H=8, M=8192, M=8192) bf16 ≈ 128MB transient — fits on M1 with current
# fraction cap. Past 16384 the (M, M) matrix becomes prohibitive.
SPARSE_ATTENTION_THRESHOLD = 16384

# affect
N_AFF        = 12

# transformer depths — shrunk for training; raise after weights stabilize.
N_MEM_LAYERS = 2     # was 4
N_ENC_LAYERS = 3     # was 5 — PC hierarchy levels
N_DEC_LAYERS = 4

# vocab (set after tokenizer built)
VOCAB_SIZE   = 16384

# training
BATCH_SIZE       = 2        # was 4
GRAD_ACCUM       = 4        # was 8 — effective batch now 8 (down from 32)
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
