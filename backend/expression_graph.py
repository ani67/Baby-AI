"""
Graph traversal expression: the mouth that works like the mind.

Instead of a separate language model conditioned on concept embeddings,
generation IS spreading activation through word-concept nodes.

Every word the mind has met as a single-token concept is a graph node.
Generation: walk through word-nodes following wave-field activation.
Each emitted word shifts the active set; the next word follows from
the shifted state.

Currently runs alongside the native head as an alternative.
Will replace it when graph density is sufficient.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch


if TYPE_CHECKING:
    from backend.wave_field import WaveField


_TOKEN_RE = re.compile(r"[a-zA-Z']+|[.,!?;:]")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class GraphTraversalExpression:
    """Expression via spreading activation through word-concept nodes."""

    STOP_WORDS = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were',
        'to', 'of', 'and', 'in', 'that', 'it',
    }
    EOS_WORDS = {'.', '!', '?'}

    def __init__(self, graph, wave_field: "WaveField"):
        self.graph = graph
        self.wave_field = wave_field

        self.max_words = 40
        self.repetition_penalty = 1.5
        self.min_activation_threshold = 0.01

        self._identify_word_nodes()

    def _identify_word_nodes(self):
        """Find graph concepts whose name is a single alphabetic word > 2 chars."""
        self.word_nodes: dict[str, int] = {}
        self.word_node_ids: set[int] = set()
        for cid, node in self.graph.nodes.items():
            name = (node.name or '').strip().lower()
            tokens = _tokenize(name)
            if len(tokens) == 1 and len(tokens[0]) > 2 and tokens[0].isalpha():
                word = tokens[0]
                if word not in self.word_nodes:
                    self.word_nodes[word] = cid
                    self.word_node_ids.add(cid)
        print(f"[graph_expression] {len(self.word_nodes)} word-concept nodes "
              f"identified", flush=True)

    def _read_word_activations(self) -> dict[str, float]:
        """Snapshot current wave-field activation over word nodes only."""
        out: dict[str, float] = {}
        act = self.wave_field.activation
        n_to_idx = self.wave_field._node_to_idx
        for word, cid in self.word_nodes.items():
            idx = n_to_idx.get(cid)
            if idx is None:
                continue
            if idx < len(act):
                out[word] = float(act[idx])
        return out

    def generate(self, wave_field_activation: Optional[torch.Tensor] = None,
                 affect: Optional[np.ndarray] = None,
                 max_words: Optional[int] = None) -> str:
        """Walk word-nodes following wave activation; inject each chosen
        word back into the field to shift the active set; next word
        follows from the new state."""
        max_words = max_words or self.max_words
        if not self.word_nodes:
            return ""

        generated_words: list[str] = []
        generated_ids: list[int] = []
        word_activations = self._read_word_activations()
        if not word_activations:
            return ""

        for step in range(max_words):
            scored: dict[str, float] = {}
            for word, activation in word_activations.items():
                penalty = 1.0
                cid = self.word_nodes.get(word)
                if cid is not None and cid in generated_ids:
                    penalty = self.repetition_penalty ** 2
                if word in self.STOP_WORDS and step > 0:
                    penalty *= 2.0
                scored[word] = activation / penalty

            if not scored:
                break
            best_word, best_score = max(scored.items(), key=lambda kv: kv[1])
            if best_score < self.min_activation_threshold:
                break

            if best_word in self.EOS_WORDS and len(generated_words) > 3:
                generated_words.append(best_word)
                break

            generated_words.append(best_word)
            best_cid = self.word_nodes[best_word]
            generated_ids.append(best_cid)

            # Inject the chosen word back: the act of choosing shifts the
            # active set; next word follows from the shifted state.
            try:
                self.wave_field.inject([best_cid], strength=0.3,
                                       mode='velocity')
                self.wave_field.step_n(5)
            except Exception:
                pass

            word_activations = self._read_word_activations()

        return ' '.join(generated_words)

    def generate_and_score(self, wave_field_activation=None, affect=None,
                           internal_repr=None, n_candidates: int = 4) -> list[dict]:
        """Generate multiple candidates at different temperatures.
        (Temperature is plumbing — the simple walker is greedy; we
        re-seed wave state between candidates and report multiple
        deterministic generations after small re-injections.)"""
        candidates: list[dict] = []

        saved_act = self.wave_field.activation.clone()
        saved_vel = self.wave_field.velocity.clone()

        for i in range(n_candidates):
            self.wave_field.activation = saved_act.clone()
            self.wave_field.velocity = saved_vel.clone()
            # nudge field slightly per candidate for variation
            if i > 0:
                # add small random injection
                try:
                    import random
                    sample_cids = random.sample(
                        list(self.word_nodes.values()),
                        min(2, len(self.word_nodes)),
                    )
                    self.wave_field.inject(sample_cids, strength=0.2 * i,
                                           mode='velocity')
                    self.wave_field.step_n(3)
                except Exception:
                    pass

            surface = self.generate()
            if not surface.strip():
                continue
            candidates.append({
                'surface': surface,
                'candidate_idx': i,
                'generator': 'graph_traversal',
            })

        self.wave_field.activation = saved_act
        self.wave_field.velocity = saved_vel
        return candidates
