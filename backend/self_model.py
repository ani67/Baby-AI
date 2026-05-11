"""
SelfModel: the mind's model of itself and others.

Tracks:
  what did I just express?     (self-echo detection)
  what do I currently feel?    (affect snapshot)
  what will I feel next?       (self-prediction)
  what does this person want?  (theory of mind)

The self-echo flag solves the problem where the mind treated its
own output as external input indistinguishably. With self-echo:
the mind knows when it's hearing itself. The gap between
'what I said' and 'how I currently feel' is the architecture
of regret and satisfaction.

Layer: IDENTITY. Private internal state distinct from expressed output.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# 30 seconds: longer than a single utterance, shorter than a topic shift.
# Anything older than this is not the mind hearing itself anymore.
SELF_ECHO_WINDOW = 30.0
SELF_ECHO_MIN_WORDS = 2
SELF_ECHO_OVERLAP_THRESHOLD = 0.6


@dataclass
class OtherModel:
    """Per-person interaction stats for theory-of-mind queries.

    Theory of mind is shallow here on purpose. We track only what is
    cheap and informative: how often this person has appeared, what
    surprised us coming from them, and how long they take to respond.
    Deeper modeling is for later; this is the seed.
    """
    person_id: str
    interaction_count: int = 0
    recent_topics: list = field(default_factory=list)
    surprise_history: list = field(default_factory=list)
    response_latency: list = field(default_factory=list)

    def update(self, their_input_rep: np.ndarray, surprise_score: float,
               latency: float) -> None:
        self.interaction_count += 1
        self.response_latency.append(float(latency))
        if surprise_score > 0.5:
            self.surprise_history.append(np.asarray(their_input_rep).copy())
        # Bounded windows — the mind forgets old stats aggressively.
        self.response_latency = self.response_latency[-20:]
        self.surprise_history = self.surprise_history[-50:]


class SelfPredictor(nn.Module):
    """Predicts next affect from current affect + concept centroid.

    Small online-trained MLP. The point is not accuracy — it is that
    the mind has a model of how it will feel a moment from now, and
    when that model is wrong, the mind learns something about itself.

    predict() caches the full input vector so learn() can do a proper
    forward+backward pass against the actual outcome without us having
    to re-pass concept_centroid through the API.
    """

    def __init__(self, affect_dim: int = 12, concept_dim: int = 512,
                 hidden: int = 64):
        super().__init__()
        self.affect_dim = affect_dim
        self.concept_dim = concept_dim
        self.net = nn.Sequential(
            nn.Linear(affect_dim + concept_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, affect_dim),
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

        # Cached state from the most recent predict() call.
        self._last_input: Optional[np.ndarray] = None
        self._last_prediction: Optional[np.ndarray] = None

    def predict(self, affect: np.ndarray,
                concept_centroid: np.ndarray) -> np.ndarray:
        """Predict the next-step affect vector. Caches input for learn()."""
        full_input = np.concatenate([affect, concept_centroid]).astype(np.float32)
        self._last_input = full_input
        x = torch.tensor(full_input, dtype=torch.float32)
        with torch.no_grad():
            pred = self.net(x)
        self._last_prediction = pred.numpy().copy()
        return self._last_prediction

    def learn(self, actual_next_affect: np.ndarray) -> Optional[float]:
        """One Adam step using the cached input from the last predict()."""
        if self._last_input is None:
            return None
        x = torch.tensor(self._last_input, dtype=torch.float32)
        pred = self.net(x)
        actual = torch.tensor(
            np.asarray(actual_next_affect, dtype=np.float32),
            dtype=torch.float32,
        )
        loss = F.mse_loss(pred, actual)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())


class SelfModel:
    """The mind's model of itself.

    Holds:
      - a short rolling window of recent outputs (for self-echo detection)
      - an affect history (for narrative anchors and prediction grounding)
      - a learned next-affect predictor (online Adam)
      - per-person OtherModel instances (theory of mind)
    """

    def __init__(self, affect_dim: int = 12, concept_dim: int = 512):
        self.affect_dim = affect_dim
        self.concept_dim = concept_dim

        self._recent_outputs: deque = deque(maxlen=100)
        self._affect_history: deque = deque(maxlen=1000)
        self._current_affect: Optional[np.ndarray] = None

        self.predictor = SelfPredictor(affect_dim, concept_dim)
        self._prediction_losses: list[float] = []
        self._predicted_next_affect: Optional[np.ndarray] = None

        self._others: dict[str, OtherModel] = {}
        self._narrative_anchors: list[int] = []

        # Observability counters.
        self.echo_detections = 0
        self.self_corrections = 0

    # ---- self-echo ---------------------------------------------------------

    def register_output(self, surface_text: str,
                        surface_encoding: np.ndarray, now: float) -> None:
        """Record something the mind just emitted so we can later detect
        if it returns to us as 'input'."""
        self._recent_outputs.append({
            't': now,
            'text': surface_text.lower().strip(),
            'encoding': np.asarray(surface_encoding).copy(),
        })

    def is_self_echo(self, text: str, now: float) -> bool:
        """True iff the given text overlaps a recent emission within the
        self-echo window. Word-overlap based; cheap and good enough."""
        if len(text.split()) < SELF_ECHO_MIN_WORDS:
            return False
        input_words = set(text.lower().strip().split())
        for output in self._recent_outputs:
            if now - output['t'] > SELF_ECHO_WINDOW:
                continue
            output_words = set(output['text'].split())
            if not output_words:
                continue
            overlap = len(output_words & input_words) / len(output_words)
            if overlap > SELF_ECHO_OVERLAP_THRESHOLD:
                self.echo_detections += 1
                return True
        return False

    def self_echo_resonance(self, text: str, text_encoding: np.ndarray,
                            current_affect: np.ndarray, now: float) -> float:
        """How much does the echoed phrase still resonate with us?

        Returns cosine similarity in [-1, 1] between the echoed text's
        encoding and the original emission encoding. 1.0 returned when
        the text is not an echo (treat as fully novel — no resonance check
        applies). Used by the mind to feel regret vs. satisfaction about
        what it said.
        """
        if not self.is_self_echo(text, now):
            return 1.0
        for output in reversed(self._recent_outputs):
            if now - output['t'] > SELF_ECHO_WINDOW:
                continue
            if output['encoding'] is not None and current_affect is not None:
                a = text_encoding / (np.linalg.norm(text_encoding) + 1e-9)
                b = output['encoding'] / (np.linalg.norm(output['encoding']) + 1e-9)
                return float(np.dot(a, b))
        return 0.5

    # ---- self-prediction ---------------------------------------------------

    def update(self, current_affect: np.ndarray,
               concept_centroid: np.ndarray, now: float) -> None:
        """Step the self-prediction loop.

        1. If we made a prediction last step, learn from how wrong it was.
        2. Make a fresh prediction for the next step.
        3. Snapshot current affect into history.
        """
        if self._current_affect is not None:
            loss = self.predictor.learn(current_affect)
            if loss is not None:
                self._prediction_losses.append(float(loss))

        self._predicted_next_affect = self.predictor.predict(
            current_affect, concept_centroid,
        )
        self._current_affect = current_affect.copy()
        self._affect_history.append({'t': now, 'affect': current_affect.copy()})

    # ---- theory of mind ----------------------------------------------------

    def update_other(self, person_id: str, their_input_rep: np.ndarray,
                     surprise_score: float, latency: float) -> None:
        if person_id not in self._others:
            self._others[person_id] = OtherModel(person_id)
        self._others[person_id].update(their_input_rep, surprise_score, latency)

    def what_does_person_want(self, person_id: str) -> dict:
        if person_id not in self._others:
            return {'known': False}
        other = self._others[person_id]
        return {
            'known': True,
            'interaction_count': other.interaction_count,
            'avg_response_latency': (
                float(np.mean(other.response_latency))
                if other.response_latency else 0.0
            ),
            'surprised_by_n_things': len(other.surprise_history),
        }

    # ---- introspection -----------------------------------------------------

    @property
    def predicted_next_affect(self) -> Optional[np.ndarray]:
        return self._predicted_next_affect

    @property
    def self_prediction_loss(self) -> float:
        """Rolling mean of recent prediction losses; 0.0 when untrained."""
        if not self._prediction_losses:
            return 0.0
        return float(np.mean(self._prediction_losses[-100:]))
