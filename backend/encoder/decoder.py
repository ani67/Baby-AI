"""
Grounded decoder: CLIP-bootstrapped word embeddings + learned projection.

Each word has a frozen 512-dim embedding in CLIP space, initialized from CLIP's
text encoder. Decoding = project brain output into CLIP space via a learned
linear layer, then find the nearest words by cosine similarity.

Training: a projection layer learns to map brain vectors into CLIP embedding
space. Word embeddings stay frozen to prevent collapse.
"""

import math

import torch
import torch.nn as nn
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
        self.bigrams = BigramTable()
        self._train_call_count = 0

        # Brain-native decode cache: neuron_idx -> [(word, similarity), ...] top-3
        self._neuron_word_cache: dict[int, list[tuple[str, float]]] = {}
        self._neuron_word_cache_call_count = 0
        self._neuron_word_cache_version = -1  # forces rebuild on first call

        # Bootstrap word embeddings from images (better separated) + CLIP text fallback
        n_words = len(self.vocab.word_to_id)
        self.word_embeddings = torch.zeros(n_words, 512)

        if text_encoder is not None:
            self._bootstrap_embeddings(db_path=db_path)

        # Freeze word embeddings — they stay in CLIP space where they are well-separated
        self.word_embeddings.requires_grad_(False)

        # Learned projection: brain output → CLIP embedding space
        self.projection = nn.Linear(512, 512)
        # Near-identity init so decode() works immediately
        with torch.no_grad():
            self.projection.weight.copy_(torch.eye(512) + 0.01 * torch.randn(512, 512))
            self.projection.bias.zero_()
        self.proj_optimizer = torch.optim.Adam(self.projection.parameters(), lr=0.001)

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

        # Project brain output into CLIP space, then find nearest words
        projected = self.projection(vector.detach())
        v = F.normalize(projected, dim=-1)

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

    def _rebuild_neuron_word_cache(self, brain) -> None:
        """Precompute top-3 closest words for each active neuron."""
        n = brain.n
        if n == 0:
            self._neuron_word_cache = {}
            return

        # Normalize word embeddings once (skip specials and zero vectors)
        we = self.word_embeddings[_NUM_SPECIAL:]
        norms = we.norm(dim=1, keepdim=True).clamp(min=1e-8)
        we_norm = we / norms
        valid_mask = norms.squeeze(1) > 1e-6

        cache: dict[int, list[tuple[str, float]]] = {}
        # Process neurons in batches of 256 for efficiency
        active_mask = ~brain.dormant[:n]
        active_idx = active_mask.nonzero().squeeze(1)
        if len(active_idx) == 0:
            self._neuron_word_cache = cache
            return

        for batch_start in range(0, len(active_idx), 256):
            batch_idx = active_idx[batch_start:batch_start + 256]
            batch_weights = F.normalize(brain.weights[batch_idx], dim=1)
            # (batch, vocab) cosine similarity
            sims = batch_weights @ we_norm.T  # works because both are normalized
            # Mask invalid word embeddings
            sims[:, ~valid_mask] = -1.0
            top_sims, top_ids = sims.topk(3, dim=1)

            for i, neuron_idx in enumerate(batch_idx.tolist()):
                entries = []
                for j in range(3):
                    sim_val = top_sims[i, j].item()
                    if sim_val < 0.05:
                        break
                    word_idx = top_ids[i, j].item() + _NUM_SPECIAL
                    word = self.vocab.id_to_word.get(word_idx, "")
                    if word:
                        entries.append((word, sim_val))
                if entries:
                    cache[neuron_idx] = entries

        self._neuron_word_cache = cache

    def decode_from_brain(self, brain, max_words: int = 4) -> str:
        """
        Brain-native word selection: use fired neurons and edge structure
        to pick words, instead of dictionary cosine-similarity lookup.

        Algorithm:
        1. Get fired neurons and their activation scores
        2. For each fired neuron, find closest words (cached)
        3. Follow edges from top-fired neurons to connected neurons
        4. Score candidates: neuron_activation * edge_strength * word_similarity
        5. Deduplicate, sort, return top words
        """
        if brain._last_fired is None or brain._last_scores is None:
            return ""

        n = brain.n
        fired = brain._last_fired[:n]
        scores = brain._last_scores[:n]
        fired_idx = fired.nonzero().squeeze(1)

        if len(fired_idx) == 0:
            return ""

        # Rebuild cache every 1000 calls or when neuron count changes
        self._neuron_word_cache_call_count += 1
        if (self._neuron_word_cache_call_count - self._neuron_word_cache_version >= 1000
                or not self._neuron_word_cache):
            self._rebuild_neuron_word_cache(brain)
            self._neuron_word_cache_version = self._neuron_word_cache_call_count

        cache = self._neuron_word_cache

        # Candidate scores: word -> max score
        candidates: dict[str, float] = {}

        # Sort fired neurons by score, take top-10 for edge traversal
        fired_scores = scores[fired_idx]
        if len(fired_idx) > 10:
            top_k_vals, top_k_local = fired_scores.topk(10)
            top_fired = fired_idx[top_k_local]
            top_scores = top_k_vals
        else:
            top_fired = fired_idx
            top_scores = fired_scores

        # Score words from directly-fired neurons (edge_strength = 1.0)
        for i, nidx in enumerate(top_fired.tolist()):
            activation = top_scores[i].item()
            if nidx in cache:
                for word, sim in cache[nidx]:
                    score = activation * sim  # edge_strength=1.0 for self
                    if word not in candidates or score > candidates[word]:
                        candidates[word] = score

        # Follow edges from top-fired neurons to connected neurons
        edge_strengths = brain._edge_strengths
        if edge_strengths:
            top_set = set(top_fired.tolist())
            # Collect connected neurons and their edge strengths
            connected: dict[int, float] = {}  # neighbor_idx -> max(activation * edge_str)
            for (src, dst), estr in edge_strengths.items():
                if src in top_set and dst < n and not fired[dst]:
                    src_act = scores[src].item()
                    combined = src_act * estr
                    if dst not in connected or combined > connected[dst]:
                        connected[dst] = combined
                elif dst in top_set and src < n and not fired[src]:
                    dst_act = scores[dst].item()
                    combined = dst_act * estr
                    if src not in connected or combined > connected[src]:
                        connected[src] = combined

            # Score words from edge-connected neurons
            for nidx, edge_score in connected.items():
                if nidx in cache:
                    for word, sim in cache[nidx]:
                        score = edge_score * sim
                        if word not in candidates or score > candidates[word]:
                            candidates[word] = score

        if not candidates:
            return ""

        # Sort by score descending, return top max_words
        sorted_words = sorted(candidates.items(), key=lambda x: x[1], reverse=True)

        # Deduplicate near-synonyms using embedding similarity
        selected = []
        selected_vecs = []
        for word, _ in sorted_words:
            if len(selected) >= max_words:
                break
            idx = self.vocab.word_to_id.get(word)
            if idx is None:
                continue
            emb = self.word_embeddings[idx]
            too_similar = False
            for prev_vec in selected_vecs:
                if torch.dot(emb, prev_vec).item() > 0.85:
                    too_similar = True
                    break
            if too_similar:
                continue
            selected.append(word)
            selected_vecs.append(emb)

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
        use_brain_native: bool = False,
    ) -> str:
        """
        Autoregressive generation: predict one word at a time, feed it back
        through the brain, repeat. The brain's activation buffer carries
        context forward — each generated word primes the next.

        Uses temperature sampling + bigram rescoring for diversity and coherence.
        When use_brain_native=True, the first word is selected via brain neuron/edge
        structure instead of CLIP cosine similarity.
        """
        tokens = []
        hidden = initial_vector
        seen = set()  # suppress repeats
        prev_word = None

        # Brain-native first word: use neuron firing + edge structure
        if use_brain_native:
            first = self.decode_from_brain(brain, max_words=1)
            if first:
                word = first.split()[0]
                tokens.append(word)
                seen.add(word)
                prev_word = word
                idx = self.vocab.word_to_id.get(word)
                if idx is not None:
                    word_emb = self.word_embeddings[idx]
                    hidden, _ = brain.forward(word_emb)

        for _ in range(max_tokens):
            # Decode from current brain state — project into CLIP space first
            projected = self.projection(hidden.detach())
            v = F.normalize(projected, dim=-1)
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

    def generate_wave(
        self,
        input_text: str,
        brain,
        text_encoder=None,
        max_tokens: int = 6,
    ) -> str:
        """
        Superimposed wave generation: interleave forward and reflect passes.

        Phase 1 (input wave): process each input word through forward + reflect,
        building context and strengthening edges simultaneously.

        Phase 2 (output wave): generate words from accumulated brain state,
        feeding each chosen word back through forward + reflect to build
        output context incrementally.

        The reflect pass propagates prediction error backward through edges,
        so the brain refines its internal representation at each step rather
        than relying on a single forward pass.
        """
        has_reflect = hasattr(brain, "reflect") and callable(brain.reflect)

        # Phase 1: Input wave — process each word with forward + reflect
        words = input_text.lower().split()
        prediction = torch.zeros(512)
        for word in words:
            clean = word.strip(".,!?;:\"'()-[]{}").lower()
            if not clean:
                continue
            # Look up grounded embedding first, fall back to CLIP
            idx = self.vocab.word_to_id.get(clean)
            if idx is not None and idx >= _NUM_SPECIAL:
                emb = self.word_embeddings[idx]
                if emb.norm() > 1e-6:
                    vec = emb
                elif text_encoder is not None:
                    vec = text_encoder.encode(clean)
                else:
                    continue
            elif text_encoder is not None:
                vec = text_encoder.encode(clean)
            else:
                continue

            pred, _ = brain.forward(vec)
            prediction = pred
            if has_reflect:
                error = vec - pred
                brain.reflect(error)

        # Phase 2: Output wave — generate words using accumulated brain state
        output_words = []
        seen = set()

        for _ in range(max_tokens):
            # Use brain's last prediction (reflects full input + generated context)
            pred = getattr(brain, "_last_prediction", prediction)
            if pred is None or pred.norm() < 1e-8:
                break

            # Project into CLIP space and find nearest words
            projected = self.projection(pred.detach())
            v = F.normalize(projected, dim=-1)
            sims = v @ self.word_embeddings.T
            sims[:_NUM_SPECIAL] = -1.0

            # Suppress already-generated words
            for w in seen:
                w_idx = self.vocab.word_to_id.get(w)
                if w_idx is not None:
                    sims[w_idx] = -1.0

            # Top-K filtering (only consider top 20 words)
            k = min(20, len(sims) - _NUM_SPECIAL)
            topk_vals, topk_idx = sims.topk(k)

            if topk_vals[0].item() < 0.05:
                break  # nothing relevant, stop

            # Pick best unseen word
            chosen_word = None
            chosen_idx = None
            for i in range(len(topk_idx)):
                w = self.vocab.id_to_word.get(topk_idx[i].item())
                if w and w not in seen:
                    chosen_word = w
                    chosen_idx = topk_idx[i].item()
                    break

            if chosen_word is None:
                break

            output_words.append(chosen_word)
            seen.add(chosen_word)

            # Feed the chosen word back through forward + reflect
            word_emb = self.word_embeddings[chosen_idx]
            pred, _ = brain.forward(word_emb)
            prediction = pred
            if has_reflect:
                error = word_emb - pred
                brain.reflect(error)

        if not output_words:
            return self.decode(prediction, max_words=1, model_step=0)

        return " ".join(output_words)

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

                projected = self.projection(hidden.detach())
                v = F.normalize(projected, dim=-1)
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
        Train the projection layer to map brain outputs into CLIP embedding space.
        Word embeddings stay frozen — only the projection learns.
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
                        self._bootstrap_new_word(clean)
                        content_words.append(clean)

        if not content_words:
            return

        # Learn bigram transitions from teacher text
        self.bigrams.observe(content_words)

        # --- Projection training: maximize similarity to correct word embeddings ---
        projected = self.projection(output_vector.detach())
        projected_norm = F.normalize(projected, dim=-1)

        loss = torch.tensor(0.0)
        n_targets = 0
        for word in content_words:
            idx = self.vocab.word_to_id[word]
            if idx < _NUM_SPECIAL:
                continue
            target_emb = self.word_embeddings[idx]
            if target_emb.norm() < 1e-6:
                continue
            target_norm = F.normalize(target_emb, dim=-1)
            cos_sim = torch.dot(projected_norm, target_norm)
            loss = loss + (1.0 - cos_sim)
            n_targets += 1

        if n_targets > 0:
            loss = loss / n_targets
            self.proj_optimizer.zero_grad()
            loss.backward()
            self.proj_optimizer.step()

        # Sequence-level training: generate, re-encode, compare to target
        # Every 3rd call
        self._train_call_count += 1
        if self._train_call_count % 3 == 0 and brain is not None:
            generated = self.generate(output_vector, brain, max_tokens=6, model_step=model_step)
            gen_vector = self.encode_sequence(generated, brain, text_encoder)
            if gen_vector is not None:
                proj_gen = self.projection(gen_vector.detach())
                proj_target = self.projection(output_vector.detach())
                seq_loss = 1.0 - F.cosine_similarity(
                    proj_gen.unsqueeze(0), proj_target.unsqueeze(0),
                )
                seq_loss = 0.005 * seq_loss.squeeze()
                self.proj_optimizer.zero_grad()
                seq_loss.backward()
                self.proj_optimizer.step()

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
            self.word_embeddings = torch.cat([self.word_embeddings, extra], dim=0).detach()
            self.word_embeddings.requires_grad_(False)
        vec = self._text_encoder.encode(f"a photo of a {word}")
        self.word_embeddings[idx] = vec

    def state_dict(self) -> dict:
        return {
            "word_embeddings": self.word_embeddings.clone(),
            "projection": self.projection.state_dict(),
            "proj_optimizer": self.proj_optimizer.state_dict(),
        }

    def load_state_dict(self, d: dict):
        if "word_embeddings" in d:
            self.word_embeddings = d["word_embeddings"].detach()
            self.word_embeddings.requires_grad_(False)
            print(f"[decoder] restored {self.word_embeddings.shape[0]} word embeddings", flush=True)
        if "projection" in d:
            self.projection.load_state_dict(d["projection"])
            print("[decoder] restored projection layer", flush=True)
        if "proj_optimizer" in d:
            self.proj_optimizer.load_state_dict(d["proj_optimizer"])

    def save_embeddings(self, path: str):
        torch.save(self.state_dict(), path)

    def load_embeddings(self, path: str):
        self.load_state_dict(torch.load(path, weights_only=False))
