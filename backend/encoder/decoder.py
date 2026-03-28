"""
Grounded decoder: CLIP-bootstrapped word embeddings + nearest-neighbor retrieval.

Each word has a 512-dim embedding in CLIP space, initialized from CLIP's own
text encoder. Decoding = find the words closest to the model's output vector.
Grounded by construction: you can only say words near what you see.

Training: teacher descriptions nudge word embeddings toward the contexts they
appear in, refining CLIP's initial alignment with the baby's own representations.
"""

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


class GroundedDecoder:
    def __init__(self, text_encoder=None, vocab_size: int = 2048):
        self.vocab = Vocabulary(max_size=8192)
        self._text_encoder = text_encoder
        self._alpha_teacher = 0.005   # nudge toward teacher context
        self._alpha_model = 0.001     # nudge toward model output

        # Bootstrap word embeddings from CLIP
        n_words = len(self.vocab.word_to_id)
        self.word_embeddings = torch.zeros(n_words, 512)

        if text_encoder is not None:
            self._bootstrap_embeddings()

    def _bootstrap_embeddings(self):
        """Encode every vocabulary word via CLIP. Special tokens stay zero."""
        words = []
        indices = []
        for idx in range(_NUM_SPECIAL, len(self.vocab.id_to_word)):
            word = self.vocab.id_to_word[idx]
            words.append(word)
            indices.append(idx)

        if not words:
            return

        # Batch encode in chunks of 64 to avoid memory issues
        all_vecs = []
        for i in range(0, len(words), 64):
            batch = words[i:i + 64]
            vecs = self._text_encoder.encode_batch(batch)
            all_vecs.append(vecs)

        embeddings = torch.cat(all_vecs, dim=0)  # (N, 512)
        for i, idx in enumerate(indices):
            self.word_embeddings[idx] = embeddings[i]

        print(f"[decoder] bootstrapped {len(words)} word embeddings from CLIP", flush=True)

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
        # Developmental staging
        if model_step < 5000:
            max_words = 1
        elif model_step < 20000:
            max_words = 2
        elif model_step < 50000:
            max_words = 3

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

    def train_step(
        self,
        output_vector: torch.Tensor,
        teacher_text: str,
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
        vec = self._text_encoder.encode(word)
        self.word_embeddings[idx] = vec

    def state_dict(self) -> dict:
        return {"word_embeddings": self.word_embeddings.clone()}

    def load_state_dict(self, d: dict):
        if "word_embeddings" in d:
            self.word_embeddings = d["word_embeddings"]
            print(f"[decoder] restored {self.word_embeddings.shape[0]} word embeddings", flush=True)
