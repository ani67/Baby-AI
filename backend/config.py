"""Locked constants from doc/mind/SYNTHESIS.md (Symbol Table).

Phase 1 uses the subset relevant to the A+B+C+H minimal slice.
Constants not used in Phase 1 are not declared here.
"""

# Dimensions
N_AFF = 12      # affect-vector dim
D_REP = 256     # representation-space dim

# Affect timescales (half-life seconds)
HALF_LIFE_REACTION    = 2.0
HALF_LIFE_WORKING     = 180.0
HALF_LIFE_MOOD        = 7200.0
HALF_LIFE_DISPOSITION = 1.21e6
HALF_LIFE_CHARACTER   = 6.3e7

# Composite weighting across the five layers
# Order: reaction, working, mood, disposition, character
COMPOSITE_WEIGHTS = (0.30, 0.30, 0.20, 0.15, 0.05)

# Layer-to-layer nudge (lower → upper). Index i is from layer i to layer i+1.
NUDGE_GAINS      = (0.05, 0.05, 0.04, 0.02)
NUDGE_THRESHOLDS = (0.10, 0.10, 0.20, 0.35)

# Per-injection-point gain
INJECTION_GAIN_INPUT      = 1.0
INJECTION_GAIN_PROCESSING = 0.6
INJECTION_GAIN_OUTPUT     = 0.8

# Surprise (B)
MIN_THRESHOLD          = 1.5    # z-score multiplier: threshold = mean + 1.5·stddev
COLD_START_N           = 30     # min observations before Welford z-scoring engages
COLD_START_MAGNITUDE_FLOOR = 0.10  # raw-magnitude floor during cold-start
STDDEV_FLOOR           = 1e-9   # guard against division by zero in z-score

# Concept graph (C)
R_MATCH = 0.92  # cosine similarity threshold for find_or_match dedup
