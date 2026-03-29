"""
Native text encoder: learned embeddings + context MLP.

Replaces CLIP text encoding with a lightweight native encoder that shares
word embeddings with the GroundedDecoder. Trained via distillation from CLIP
outputs, then gradually takes over as the primary text encoder.

Architecture:
    word → embedding(512) → MLP(512→512→512) → mean pool → L2 normalize

The MLP contextualizes each word embedding before pooling, giving the encoder
capacity to represent word-order effects and compositional meaning beyond
simple bag-of-words averaging.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vocab import Vocabulary

# Skip special tokens when tokenizing
_NUM_SPECIAL = len(Vocabulary.SPECIAL_TOKENS)


class NativeTextEncoder:
    """
    Learned text encoder sharing vocabulary and embeddings with GroundedDecoder.

    Parameters
    ----------
    vocab : Vocabulary
        Shared vocabulary instance (same object as decoder's).
    word_embeddings : torch.Tensor
        Shared word embedding matrix (same tensor as decoder's).
        Shape: (vocab_size, 512). Updates from decoder training are
        visible here automatically.
    """

    def __init__(self, vocab: Vocabulary, word_embeddings: torch.Tensor):
        self.vocab = vocab
        self.word_embeddings = word_embeddings  # shared reference, not a copy

        # 2-layer context MLP (~525K params)
        self.context_mlp = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
        )

        # Optimizer created lazily on first distill_step
        self._optimizer: torch.optim.Adam | None = None

    # ------------------------------------------------------------------
    # Tokenization
    # ------------------------------------------------------------------

    def _tokenize(self, text: str) -> list[int]:
        """Split text into words, return vocab indices. Unknown words → UNK(3)."""
        unk_id = Vocabulary.SPECIAL_TOKENS["<UNK>"]
        ids = []
        for word in text.lower().split():
            clean = word.strip(".,!?;:\"'()-[]{}")
            if not clean:
                continue
            idx = self.vocab.word_to_id.get(clean, unk_id)
            ids.append(idx)
        return ids

    # ------------------------------------------------------------------
    # Encode
    # ------------------------------------------------------------------

    def encode(self, text: str) -> torch.Tensor:
        """Single text → (512,) L2-normalized vector."""
        ids = self._tokenize(text)
        if not ids:
            # Empty input → normalized zero vector (all dims equal)
            return F.normalize(torch.ones(512), dim=0)

        # Gather embeddings: (N, 512)
        embs = torch.stack([self.word_embeddings[i] for i in ids])

        # Contextualize each word through MLP
        contextualized = self.context_mlp(embs)  # (N, 512)

        # Mean pool → single vector
        pooled = contextualized.mean(dim=0)  # (512,)

        return F.normalize(pooled, dim=0)

    def encode_batch(self, texts: list[str]) -> torch.Tensor:
        """Batch of texts → (N, 512) L2-normalized vectors."""
        return torch.stack([self.encode(t) for t in texts])

    # ------------------------------------------------------------------
    # Blended encoding (smooth CLIP → native transition)
    # ------------------------------------------------------------------

    def encode_blended(
        self, text: str, clip_vector: torch.Tensor, alpha: float
    ) -> torch.Tensor:
        """
        Blend native and CLIP encodings.

        Returns: alpha * native + (1 - alpha) * clip, L2-normalized.
        alpha=0 → pure CLIP, alpha=1 → pure native.
        """
        native = self.encode(text)
        blended = alpha * native + (1.0 - alpha) * clip_vector
        return F.normalize(blended, dim=0)

    # ------------------------------------------------------------------
    # Distillation training
    # ------------------------------------------------------------------

    def distill_step(self, text: str, clip_target: torch.Tensor) -> float:
        """
        Train the native encoder to match CLIP's output for the same text.

        Uses cosine embedding loss: the native output should point in the
        same direction as the CLIP target. Backprop flows through the MLP
        and into the shared word embeddings.

        Returns the loss value (lower is better, 0 = perfect match).
        """
        # Lazy optimizer creation (covers MLP params + shared embeddings)
        if self._optimizer is None:
            self._optimizer = torch.optim.Adam(
                self.context_mlp.parameters(), lr=0.001
            )

        self.context_mlp.train()

        native = self.encode(text)
        target = F.normalize(clip_target.detach(), dim=0)

        # Cosine loss: 1 - cos_sim (range [0, 2], 0 = identical)
        loss = 1.0 - F.cosine_similarity(
            native.unsqueeze(0), target.unsqueeze(0)
        )

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        return loss.item()

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        """Return serializable state (MLP weights + optimizer)."""
        d = {"context_mlp": self.context_mlp.state_dict()}
        if self._optimizer is not None:
            d["optimizer"] = self._optimizer.state_dict()
        return d

    def load_state_dict(self, d: dict) -> None:
        """Restore MLP weights and optimizer state."""
        if "context_mlp" in d:
            self.context_mlp.load_state_dict(d["context_mlp"])
            print("[native_text] restored context MLP weights", flush=True)
        if "optimizer" in d:
            if self._optimizer is None:
                self._optimizer = torch.optim.Adam(
                    self.context_mlp.parameters(), lr=0.001
                )
            self._optimizer.load_state_dict(d["optimizer"])
