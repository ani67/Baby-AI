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

# Hard ceiling on every affect-layer L2 norm. Without this, dense
# ingestion (thousands of cycles/sec, half-lives much longer than the
# inter-cycle interval) lets the upward nudge chain accumulate without
# bound — the upper layers were observed at 1e8+ after 3K cycles. With
# this ceiling, the system has bounded affect under any input regime.
MAX_LAYER_NORM = 10.0

# Surprise (B)
MIN_THRESHOLD          = 1.5    # z-score multiplier: threshold = mean + 1.5·stddev
COLD_START_N           = 30     # min observations before Welford z-scoring engages
COLD_START_MAGNITUDE_FLOOR = 0.10  # raw-magnitude floor during cold-start
STDDEV_FLOOR           = 1e-9   # guard against division by zero in z-score

# Ingestion-mode surprise (B) — Phase 7 conditioning fix.
# During curriculum/book ingestion the mind reads thousands of sentences in
# bulk. The default 1.5σ threshold yields ~5% surprise rate which leaves the
# language-head training corpus too small to override GPT-2's web-text
# priors. PredictionEngine.set_ingestion_mode(True) swaps these in for the
# duration of a book step; the warm-state Welford branch uses 1.0σ instead
# of 1.5σ, and the cold-start floor drops from 0.10 to 0.06. Surprise rate
# rises ~3× to ~15%, tripling the training corpus per book step.
INGESTION_MIN_THRESHOLD              = 1.0
INGESTION_COLD_START_MAGNITUDE_FLOOR = 0.06

# Concept graph (C)
R_MATCH = 0.92  # cosine similarity threshold for find_or_match dedup

# Auto-link top-K (Phase 7 perf fix). On every fresh concept write, lay
# similar_to edges to the K cosine-nearest existing concepts above
# AUTO_LINK_MIN_COSINE. K=3 was the v0.4 default but adds 3 FAISS searches
# per write; at K=1 the graph still builds horizontal density (sleep's
# replay path strengthens what the spread re-traverses) at a third of the
# per-write cost.
AUTO_LINK_K          = 1
AUTO_LINK_MIN_COSINE = 0.25

# LM training cadence during interleaved ingestion (Phase 7 perf fix).
# v0.4 fired GPT-2 retraining every 500 surprises which at the 25%
# ingestion-mode rate is every ~2,000 sentences — too frequent. Each
# train spends 5-10 minutes that doesn't earn enough delta to justify
# the interruption. 2000 surprises means trains land at every ~8K
# ingested sentences instead, plus a hard floor: don't train until at
# least LM_TRAIN_MIN_CORPUS surprised sentences exist on disk.
TRAIN_LM_EVERY_N_SURPRISES = 2000
LM_TRAIN_MIN_CORPUS        = 500

# Processing loop (F) — Phase 5
# Between INPUT attend and action selection, F runs PROCESSING_TICKS
# internal spread iterations. Each tick decays the existing actives by
# PROCESSING_DECAY and re-spreads, so the input concept's dominance
# fades from 1.0 toward ~0.5 as neighbors accumulate activation. After
# the loop completes, any concept above PROCESSING_MAX_ACTIVATION is
# scaled down (proportionally) so the final active set treats input
# as evidence rather than the entirety of mental state.
PROCESSING_TICKS           = 4
PROCESSING_DECAY           = 0.85   # per-tick global multiplier; 1.0^4·0.85^4 ≈ 0.52
PROCESSING_MAX_ACTIVATION  = 0.5    # post-loop cap so the seed never dominates expression

# Parallel ingestion (Phase 7 perf — N readers + 1 writer architecture).
# Readers preprocess + encode + propose diffs into a multiprocessing
# queue; writer consumes diffs and applies via find_or_match dedup.
PARALLEL_INGESTION_SNAPSHOT_INTERVAL = 500    # writer pushes FAISS snapshot every N writes
PARALLEL_INGESTION_LOCAL_DEDUP_WINDOW = 60.0  # seconds — reader-side dedup window

# Concept-graph ceiling + post-prune target (synthesis "forgetting is
# curation" finally in code). Without this every write_on_surprise
# grew the graph unbounded — the v0.6 curriculum reached 90K+ nodes
# with no curation. The forget loop is now sleep-triggered: when
# node_count > CONCEPT_CEILING after abstraction formation, prune the
# weakest non-pinned non-abstraction-incident concepts down to
# PRUNE_TO.
#
# Sized for the multi-resolution curriculum that lands ~7.7× more
# items per source than sentence-only ingestion. 50K caps roughly
# match human active vocabulary (~50K words) with room across the
# 4 resolution levels:
#   ~8-10K word concepts
#   ~6-8K phrase concepts
#   ~15-20K sentence concepts
#   ~5-7K paragraph concepts
#   ~2-3K abstraction parents (sqrt(N) ≈ 224 k-means clusters)
#   ~5-8K cross-domain bridges
# The 5K headroom (CEILING - PRUNE_TO) lets concepts arrive between
# sleep cycles without immediately re-tripping the threshold.
#
# Pinned concepts (E's narrative anchors, self/unknown, expression
# templates) and concepts incident on any IS_A edge (abstraction
# parents AND members) are immune.
CONCEPT_CEILING = 50000
PRUNE_TO        = 45000
