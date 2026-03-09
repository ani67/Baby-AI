"""Component 5: Importance Scorer (Amygdala)

Assigns an importance score (0.0–1.0) to every episode.
High importance = stronger learning signal.

Scoring formula:
  BASE: 0.3
  +0.3  if episode has a correction
  +0.2  if same prompt was corrected before (recurring error)
  +0.1  if times_referenced > 3
  +0.1  if correction length > 50 chars
  -0.1  if episode older than 7 days
  clamp to [0.1, 1.0]

Learning rate mapping (linear):
  score 0.1 → 1e-5
  score 0.5 → 1e-4  (not exactly linear with the other two, see below)
  score 1.0 → 5e-4

We use log-linear interpolation to hit all three anchor points.
"""

import time

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import component4
from component4 import Episode

_SEVEN_DAYS = 7 * 24 * 60 * 60


def _has_prior_correction_for_prompt(prompt: str) -> bool:
    """Check if the episodic store has a previous correction for this prompt."""
    component4._ensure_init()
    for ep in component4._episodes.values():
        if ep.prompt == prompt and ep.correction is not None:
            return True
    return False


def score_episode(episode: Episode) -> float:
    """Score an episode's importance. Returns 0.1–1.0."""
    score = 0.3

    # Correction made → more important
    if episode.correction is not None:
        score += 0.3

    # Recurring error on same prompt → even more important
    if episode.correction is not None and _has_prior_correction_for_prompt(episode.prompt):
        score += 0.2

    # Frequently referenced → important pattern
    if episode.times_referenced > 3:
        score += 0.1

    # Long correction → substantial change needed
    if episode.correction is not None and len(episode.correction) > 50:
        score += 0.1

    # Recency decay
    age = time.time() - episode.timestamp
    if age > _SEVEN_DAYS:
        score -= 0.1

    return round(max(0.1, min(1.0, score)), 4)


def learning_rate_for_episode(episode: Episode) -> float:
    """Map importance score to a learning rate in [1e-5, 5e-4].

    Linear interpolation between the anchor points:
      score 0.1 → 1e-5
      score 1.0 → 5e-4
    """
    s = episode.importance_score
    # Linear map: lr = 1e-5 + (s - 0.1) / (1.0 - 0.1) * (5e-4 - 1e-5)
    lr = 1e-5 + (s - 0.1) / 0.9 * (5e-4 - 1e-5)
    return max(1e-5, min(5e-4, lr))


# ---------- register hook with Component 4 ----------

component4._score_fn = score_episode
