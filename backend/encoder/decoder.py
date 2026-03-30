"""
Grounded decoder: CLIP-bootstrapped word embeddings + nearest-neighbor retrieval.

Each word has a 512-dim embedding in CLIP space, initialized from CLIP's own
text encoder. Decoding = find the words closest to the model's output vector.
Grounded by construction: you can only say words near what you see.

Training: teacher descriptions nudge word embeddings toward the contexts they
appear in, refining CLIP's initial alignment with the baby's own representations.
"""

import math

import torch
import torch.nn.functional as F

from .vocab import Vocabulary

# Content word filter — skip these when extracting from teacher text
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can", "need",
    "i", "me", "my", "we", "our", "you", "your", "he", "him", "his",
    "she", "her", "it", "its", "they", "them", "their",
    "this", "that", "these", "those", "which", "who", "whom",
    "in", "on", "at", "to", "for", "of", "with", "by", "from",
    "up", "out", "if", "or", "and", "but", "not", "no", "so",
    "as", "about", "into", "over", "after", "before", "between",
    "through", "during", "until", "than", "then", "there", "here",
    "very", "just", "also", "still", "even", "only", "more", "most",
    "some", "any", "all", "each", "every", "both", "few", "many",
    "much", "such", "what", "when", "where", "how", "why",
}

# Number of special tokens to skip in similarity search
_NUM_SPECIAL = len(Vocabulary.SPECIAL_TOKENS)


class BigramTable:
    """Sparse bigram transition probabilities learned from teacher text."""

    def __init__(self):
        self._counts: dict[str, dict[str, int]] = {}  # word -> {next_word: count}
        self._totals: dict[str, int] = {}

    def observe(self, words: list[str]) -> None:
        """Record bigram transitions from a word sequence."""
        for i in range(len(words) - 1):
            w1, w2 = words[i], words[i + 1]
            if w1 not in self._counts:
                self._counts[w1] = {}
            self._counts[w1][w2] = self._counts[w1].get(w2, 0) + 1
            self._totals[w1] = self._totals.get(w1, 0) + 1

    def transition_prob(self, prev_word: str, next_word: str) -> float:
        """P(next_word | prev_word). Returns 1.0 if no data (uniform prior)."""
        if prev_word not in self._counts:
            return 1.0
        count = self._counts[prev_word].get(next_word, 0)
        return max(count / self._totals[prev_word], 0.01)  # floor at 1% to avoid zeroing


class GroundedDecoder:
    def __init__(self, text_encoder=None, vocab_size: int = 2048, db_path: str | None = None):
        self.vocab = Vocabulary(max_size=8192)
        self._text_encoder = text_encoder
        self._alpha_teacher = 0.005   # nudge toward teacher context
        self.bigrams = BigramTable()
        self._train_call_count = 0

        # Bootstrap word embeddings from images (better separated) + CLIP text fallback
        n_words = len(self.vocab.word_to_id)
        self.word_embeddings = torch.zeros(n_words, 512)

        if text_encoder is not None:
            self._bootstrap_embeddings(db_path=db_path)

    def _bootstrap_embeddings(self, db_path: str | None = None):
        """Bootstrap word embeddings. Uses mean of actual images per category
        from the embedding cache (much better separated than CLIP text).
        Falls back to CLIP text phrases for words without image data."""
        import sqlite3
        import struct

        words = []
        indices = []
        for idx in range(_NUM_SPECIAL, len(self.vocab.id_to_word)):
            word = self.vocab.id_to_word[idx]
            words.append(word)
            indices.append(idx)

        if not words:
            return

        # Try image-mean embeddings from embedding cache (dog-bus: 0.62 vs 0.85 text)
        image_embs = {}
        if db_path:
            try:
                conn = sqlite3.connect(db_path)
                for word in words:
                    rows = conn.execute(
                        "SELECT image_emb FROM embedding_cache WHERE category=? ORDER BY RANDOM() LIMIT 20",
                        (word,),
                    ).fetchall()
                    if len(rows) >= 5:
                        vecs = []
                        for r in rows:
                            n = len(r[0]) // 4
                            vals = struct.unpack(f"{n}f", r[0])
                            vecs.append(torch.tensor(vals, dtype=torch.float32))
                        image_embs[word] = F.normalize(torch.stack(vecs).mean(dim=0), dim=0)
                conn.close()
            except Exception:
                pass

        # Encode remaining words via CLIP text (phrase template)
        text_words = [w for w in words if w not in image_embs]
        text_vecs = {}
        if text_words and self._text_encoder is not None:
            for i in range(0, len(text_words), 64):
                batch = [f"a photo of a {w}" for w in text_words[i:i + 64]]
                vecs = self._text_encoder.encode_batch(batch)
                for j, w in enumerate(text_words[i:i + 64]):
                    text_vecs[w] = vecs[j]

        # Fill embedding matrix
        for i, idx in enumerate(indices):
            word = words[i]
            if word in image_embs:
                self.word_embeddings[idx] = image_embs[word]
            elif word in text_vecs:
                self.word_embeddings[idx] = text_vecs[word]

        print(
            f"[decoder] bootstrapped {len(words)} words "
            f"({len(image_embs)} from images, {len(text_vecs)} from text)",
            flush=True,
        )

    def decode(
        self,
        vector: torch.Tensor,
        max_words: int = 4,
        model_step: int = 0,
    ) -> str:
        """
        Find the nearest words to the output vector.
        Developmental staging: fewer words at early steps.
        """
        # Developmental staging (overrides caller's max_words)
        if model_step < 5000:
            max_words = 1
        elif model_step < 20000:
            max_words = 2
        else:
            max_words = min(max_words, 4)

        v = F.normalize(vector.detach(), dim=-1)

        # Cosine similarity to all word embeddings
        sims = v @ self.word_embeddings.T  # (vocab_size,)

        # Mask special tokens
        sims[:_NUM_SPECIAL] = -1.0

        # Get top candidates (more than we need, for suppression)
        k = min(max_words * 3, len(sims) - _NUM_SPECIAL)
        top_sims, top_ids = torch.topk(sims, k)

        # Select words with suppression (skip words too similar to already-picked)
        selected = []
        selected_vecs = []
        for i in range(len(top_ids)):
            if len(selected) >= max_words:
                break
            sim_val = top_sims[i].item()
            if sim_val < 0.10:  # minimum similarity threshold
                break
            idx = top_ids[i].item()
            emb = self.word_embeddings[idx]

            # Suppress near-duplicates (e.g., "dog" and "dogs")
            too_similar = False
            for prev_vec in selected_vecs:
                if torch.dot(emb, prev_vec).item() > 0.85:
                    too_similar = True
                    break
            if too_similar:
                continue

            selected.append(self.vocab.id_to_word[idx])
            selected_vecs.append(emb)

        # Always return at least the top-1 word (even if below threshold)
        if not selected and k > 0:
            selected.append(self.vocab.id_to_word[top_ids[0].item()])
        return " ".join(selected)

    def encode_sequence(
        self,
        text: str,
        brain,
        text_encoder=None,
    ) -> torch.Tensor:
        """
        Process an input sentence token by token through the brain.

        For each word with a known embedding: feed it through brain.forward().
        The activation buffer accumulates context across tokens — by the end,
        the brain's state represents the full sentence meaning.

        Words not in vocabulary fall back to CLIP text encoder (full phrase).
        This bridges known words (fast, grounded) and unknown phrases (slow, CLIP).

        Returns the brain's final prediction after processing all tokens.
        """
        # Tokenize: split into words, look up embeddings
        words = text.lower().split()
        token_vectors = []
        unknown_words = []

        for word in words:
            clean = word.strip(".,!?;:\"'()-[]{}").lower()
            if not clean:
                continue
            idx = self.vocab.word_to_id.get(clean)
            if idx is not None and idx >= _NUM_SPECIAL:
                emb = self.word_embeddings[idx]
                if emb.norm() > 1e-6:
                    token_vectors.append(emb)
                    continue
            unknown_words.append(clean)

        # Fall back to CLIP for unknown words or if no known words found
        if not token_vectors and text_encoder is not None:
            clip_vec = text_encoder.encode(text)
            return clip_vec

        # Process each token through the brain sequentially.
        # The buffer builds up context — "the big red dog" becomes:
        #   forward("the") → buffer primed
        #   forward("big") → buffer = the + big
        #   forward("red") → buffer = the + big + red
        #   forward("dog") → buffer = the + big + red + dog ← final state
        prediction = torch.zeros(512)
        for vec in token_vectors:
            prediction, _ = brain.forward(vec)

        # If there were unknown words, also encode them via CLIP and blend
        if unknown_words and text_encoder is not None:
            clip_vec = text_encoder.encode(" ".join(unknown_words))
            blend_pred, _ = brain.forward(clip_vec)
            prediction = F.normalize(prediction + blend_pred, dim=0)

        return prediction

    def _score_with_bigrams(self, sims: torch.Tensor, prev_word: str | None) -> torch.Tensor:
        """Apply bigram transition probabilities to similarity scores."""
        if prev_word is None:
            return sims
        scored = sims.clone()
        for idx in range(len(scored)):
            word = self.vocab.id_to_word.get(idx, "")
            scored[idx] *= self.bigrams.transition_prob(prev_word, word)
        return scored

    def generate(
        self,
        initial_vector: torch.Tensor,
        brain,
        max_tokens: int = 12,
        model_step: int = 0,
        temperature: float = 0.7,
    ) -> str:
        """
        Autoregressive generation: predict one word at a time, feed it back
        through the brain, repeat. The brain's activation buffer carries
        context forward — each generated word primes the next.

        Uses temperature sampling + bigram rescoring for diversity and coherence.
        """
        tokens = []
        hidden = initial_vector
        seen = set()  # suppress repeats
        prev_word = None

        for _ in range(max_tokens):
            # Decode from current brain state
            v = F.normalize(hidden.detach(), dim=-1)
            sims = v @ self.word_embeddings.T
            sims[:_NUM_SPECIAL] = -1.0

            # Suppress already-generated words
            for word in seen:
                idx = self.vocab.word_to_id.get(word)
                if idx is not None:
                    sims[idx] = -1.0

            # Apply bigram rescoring
            sims = self._score_with_bigrams(sims, prev_word)

            # Temperature sampling instead of argmax
            probs = F.softmax(sims / temperature, dim=0)
            best_idx = torch.multinomial(probs, 1).item()
            best_sim = sims[best_idx].item()

            if best_sim < 0.05:  # confidence too low — stop
                break

            word = self.vocab.id_to_word[best_idx]
            tokens.append(word)
            seen.add(word)
            prev_word = word

            # Feed generated word's embedding back through brain
            word_emb = self.word_embeddings[best_idx]
            hidden, _ = brain.forward(word_emb)

        return " ".join(tokens) if tokens else self.decode(initial_vector, max_words=1, model_step=model_step)

    def generate_beam(
        self,
        initial_vector: torch.Tensor,
        brain,
        beam_width: int = 3,
        max_tokens: int = 12,
        model_step: int = 0,
        temperature: float = 0.7,
    ) -> str:
        """Beam search: keep top-k partial sequences, rescore at each step."""
        # Each beam: (tokens, cumulative_log_score, hidden_state, buffer_snapshot, seen_set)
        initial_buffer = brain.activation_buffer.clone()
        beams = [
            ([], 0.0, initial_vector, initial_buffer, set())
        ]

        completed = []

        for _ in range(max_tokens):
            candidates = []

            for tokens, score, hidden, buf_snapshot, seen in beams:
                # Restore brain buffer state for this beam
                brain.activation_buffer = buf_snapshot.clone()

                v = F.normalize(hidden.detach(), dim=-1)
                sims = v @ self.word_embeddings.T
                sims[:_NUM_SPECIAL] = -1.0

                # Suppress already-generated words
                for word in seen:
                    idx = self.vocab.word_to_id.get(word)
                    if idx is not None:
                        sims[idx] = -1.0

                # Apply bigram rescoring
                prev_word = tokens[-1] if tokens else None
                sims = self._score_with_bigrams(sims, prev_word)

                # Get top-k candidates for expansion
                k = min(beam_width * 2, len(sims) - _NUM_SPECIAL)
                top_sims, top_ids = torch.topk(sims, k)

                for i in range(k):
                    sim_val = top_sims[i].item()
                    idx = top_ids[i].item()

                    if sim_val < 0.05:
                        # This beam is done — no more good tokens
                        if tokens:
                            completed.append((tokens, score))
                        break

                    word = self.vocab.id_to_word[idx]
                    log_sim = math.log(max(sim_val, 1e-8))
                    new_score = score + log_sim
                    new_tokens = tokens + [word]
                    new_seen = seen | {word}

                    # Run brain forward to get next hidden state
                    brain.activation_buffer = buf_snapshot.clone()
                    word_emb = self.word_embeddings[idx]
                    new_hidden, _ = brain.forward(word_emb)
                    new_buf = brain.activation_buffer.clone()

                    candidates.append(
                        (new_tokens, new_score, new_hidden, new_buf, new_seen)
                    )

            if not candidates:
                break

            # Keep top beam_width candidates (by cumulative log score)
            candidates.sort(key=lambda c: c[1], reverse=True)
            beams = candidates[:beam_width]

        # Add remaining active beams to completed
        for tokens, score, _, _, _ in beams:
            if tokens:
                completed.append((tokens, score))

        if not completed:
            return self.decode(initial_vector, max_words=1, model_step=model_step)

        # Pick best completed beam
        completed.sort(key=lambda c: c[1], reverse=True)
        best_tokens = completed[0][0]

        # Restore brain buffer to the best beam's final state (find it in beams)
        # Use the first beam's buffer as best approximation
        if beams:
            brain.activation_buffer = beams[0][3].clone()

        return " ".join(best_tokens)

    def train_step(
        self,
        output_vector: torch.Tensor,
        teacher_text: str,
        brain=None,
        text_encoder=None,
        model_step: int = 0,
    ) -> None:
        """
        Nudge word embeddings toward the contexts they appear in.
        Extracts content words from teacher text, moves their embeddings
        toward both the teacher vector and the model's output vector.
        """
        if not teacher_text:
            return

        # Extract content words
        content_words = []
        for word in teacher_text.lower().split():
            clean = word.strip(".,!?;:\"'()-[]{}").lower()
            if len(clean) > 2 and clean.isalpha() and clean not in _STOPWORDS:
                if clean in self.vocab.word_to_id:
                    content_words.append(clean)
                else:
                    # Try to grow vocabulary
                    self.vocab.add_word(clean)
                    if clean in self.vocab.word_to_id:
                        # New word added — bootstrap its embedding
                        self._bootstrap_new_word(clean)
                        content_words.append(clean)

        if not content_words:
            return

        # Learn bigram transitions from teacher text
        self.bigrams.observe(content_words)

        teacher_vec = F.normalize(output_vector.detach(), dim=-1)

        # Nudge each content word's embedding toward the teacher context
        with torch.no_grad():
            for word in content_words:
                idx = self.vocab.word_to_id[word]
                if idx < _NUM_SPECIAL:
                    continue
                emb = self.word_embeddings[idx]
                if emb.norm() < 1e-6:
                    continue
                # Nudge toward teacher/model output context
                emb = emb + self._alpha_teacher * (teacher_vec - emb)
                self.word_embeddings[idx] = F.normalize(emb, dim=-1)

        # Sequence-level training: generate, re-encode, compare to target
        # Only every 10th call (expensive)
        self._train_call_count += 1
        if self._train_call_count % 10 == 0 and brain is not None:
            generated = self.generate(output_vector, brain, max_tokens=6, model_step=model_step)
            gen_vector = self.encode_sequence(generated, brain, text_encoder)
            if gen_vector is not None:
                gap = F.normalize(output_vector, dim=0) - F.normalize(gen_vector, dim=0)
                with torch.no_grad():
                    for word in generated.split():
                        idx = self.vocab.word_to_id.get(word.lower())
                        if idx is not None and idx >= _NUM_SPECIAL:
                            self.word_embeddings[idx] += 0.002 * gap
                            self.word_embeddings[idx] = F.normalize(self.word_embeddings[idx], dim=0)

    def _bootstrap_new_word(self, word: str):
        """Encode a newly-added word via CLIP and append to embeddings."""
        if self._text_encoder is None:
            return
        idx = self.vocab.word_to_id.get(word)
        if idx is None:
            return
        # Expand embedding matrix if needed
        if idx >= self.word_embeddings.shape[0]:
            extra = torch.zeros(idx - self.word_embeddings.shape[0] + 1, 512)
            self.word_embeddings = torch.cat([self.word_embeddings, extra], dim=0)
        vec = self._text_encoder.encode(f"a photo of a {word}")
        self.word_embeddings[idx] = vec

    def state_dict(self) -> dict:
        return {"word_embeddings": self.word_embeddings.clone()}

    def load_state_dict(self, d: dict):
        if "word_embeddings" in d:
            self.word_embeddings = d["word_embeddings"]
            print(f"[decoder] restored {self.word_embeddings.shape[0]} word embeddings", flush=True)

    def save_embeddings(self, path: str):
        torch.save(self.state_dict(), path)

    def load_embeddings(self, path: str):
        self.load_state_dict(torch.load(path, weights_only=True))
