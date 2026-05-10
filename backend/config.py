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

# Valence deadband. Below |valence| ≤ 0.3 the alignment between an incoming
# gap and the current composite affect is treated as noise — modulation off.
# Above it, sign decides direction (negative = aversive → inject reverses;
# positive = approach → inject amplifies). Without the deadband, every tiny
# correlation would flip behavior on every observation.
VALENCE_THRESHOLD      = 0.3

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

# Auto-link top-K. On every fresh concept write, lay similar_to edges
# to the K cosine-nearest existing concepts above AUTO_LINK_MIN_COSINE.
#
# Was lowered to K=1 during the FAISS era to amortize per-write search
# cost. The numpy-backed index makes a top-3 search nearly free
# (~0.06 ms at N=7K, ~1 ms at N=50K), and the v0.7b run revealed the
# downside: 3.7 mean edges/node leaves the spread function with no
# paths to follow — active sets stayed at 1-3 even on a 73K mind.
# At K=3 mean degree climbs to ~9-11, giving spreading activation real
# bridges across domains.
AUTO_LINK_K          = 3
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
# Periodic mind-save during long parallel runs. Without this, a SIGTERM
# (e.g. user Ctrl+C → curriculum stop) on a multi-hour run that hangs
# in mp.Queue cleanup leaves mind.db at the last book-step boundary.
# 10K writes is roughly 30–60 s of writer time at 200–400/s, so a worst-
# case crash drops <1 minute of work.
PARALLEL_INGESTION_SAVE_INTERVAL      = 10_000
# How often (in writer items) to check whether node_count > CONCEPT_CEILING
# and, if so, fire a short sleep — this is the only path that runs
# abstraction formation + ceiling-pruning during a parallel ingestion
# run. Sequential interleaved runs already sleep on their own cadence;
# the parallel path has no curriculum step boundaries, so it has to do
# this itself or the graph grows unbounded.
PARALLEL_INGESTION_SLEEP_CHECK_EVERY  = 10_000
PARALLEL_INGESTION_SLEEP_DURATION     = 10.0
# Writer-side batch size for the parallel ingestion path. The writer
# collects up to this many diffs from the queue, runs ONE batched
# nearest-neighbor search (predict_batch) over the whole batch, then
# applies per-item via loop.cycle. Set to 1 to disable batching and
# fall back to per-item dequeue (used by benchmarks).
WRITER_BATCH_SIZE                     = 64

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
#
# Phase 8 (post-v0.8) update — moved from size-based to quality-based
# pruning at sleep. After a deep curriculum pass produced ~88K IS_A
# members against 2.5K parents, the IS_A-incidence protection inflated
# to 99% of the graph, leaving the ceiling-prune candidate set <1% of
# nodes — sleep cycles could remove only a few hundred at a time
# while ingestion was writing thousands per cycle. Quality-based
# pruning (graph.prune_weak_concepts) evaluates every concept on its
# own merit (activation_count, edge degree, affect magnitude,
# surprise_at_birth) and drops those that fail every criterion.
# CONCEPT_CEILING is kept only as an emergency brake — if the graph
# blows past 200K despite quality pruning, fall back to size-based
# trim to PRUNE_TO.
CONCEPT_CEILING = 200_000   # emergency brake only
PRUNE_TO        = 180_000   # if emergency fires
QUALITY_PRUNE_ON_SLEEP = True

# Attention arousal cap. F.attend feeds arousal into the spread gate;
# at saturating arousal (0.9+) the gate collapses to a tiny top-k and
# the active set narrows to 1-3 concepts. That's appropriate for
# fight-or-flight, but it means a mind freshly emerging from a
# 1.4M-item ingestion run (where reaction/working/mood layers have all
# accumulated) cannot think broadly until arousal decays — even when
# the underlying graph has 73K concepts ready to spread through.
#
# This cap preserves the *felt* arousal (composite_affect, the affect
# engine, expression's tone-shaping all see the real value) but prevents
# saturated arousal from collapsing perception. The mind still feels
# intense; it just doesn't go blind from it.
ATTENTION_AROUSAL_CAP = 0.6

# Expression gating thresholds. The original 0.4 / 0.8 values were
# calibrated against the LSTM language head. GPT-2 at loss 0.40
# generates more fluent text but encodes to different GloVe positions,
# so the gap-to-input distance distribution shifted. With these values
# and honesty_bias=0.7 the effective thresholds become 0.91 (revise)
# and 1.17 (suppress) — the LM has real room to express, while the
# suppression mechanism still catches genuinely misaligned outputs.
EXPRESSION_REVISE_THRESHOLD   = 0.70
EXPRESSION_SUPPRESS_THRESHOLD = 0.90
EXPRESSION_HONESTY_BIAS       = 0.7
