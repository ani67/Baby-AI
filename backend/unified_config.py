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
