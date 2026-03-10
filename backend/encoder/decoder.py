import torch
import torch.nn as nn
import torch.nn.functional as F

from .vocab import Vocabulary


class TextDecoder:
    def __init__(
        self,
        vocab_size: int = 2048,
        hidden_dim: int = 512,
    ):
        self.hidden_dim = hidden_dim
        self.vocab = Vocabulary(max_size=8192)
        actual_vocab_size = len(self.vocab.word_to_id)
        self.projection = nn.Linear(hidden_dim, actual_vocab_size)

    def decode(
        self,
        vector: torch.Tensor,
        max_words: int = 30,
        temperature: float = 0.7,
    ) -> str:
        """
        Autoregressive decoding over the small vocabulary.
        Generates up to max_words tokens.
        Stops at end-of-sentence token or max_words.
        """
        end_id = Vocabulary.SPECIAL_TOKENS["<END>"]
        pad_id = Vocabulary.SPECIAL_TOKENS["<PAD>"]

        current = vector.detach().clone()
        ids: list[int] = []

        for _ in range(max_words):
            logits = self.projection(current)
            # Mask special tokens except <END>
            logits[pad_id] = float("-inf")
            logits[Vocabulary.SPECIAL_TOKENS["<START>"]] = float("-inf")
            logits[Vocabulary.SPECIAL_TOKENS["<UNK>"]] = float("-inf")

            probs = F.softmax(logits / temperature, dim=-1)
            token_id = torch.multinomial(probs, 1).item()

            if token_id == end_id:
                break
            ids.append(token_id)

            # Shift the input vector slightly based on the chosen token
            # This is a simple autoregressive signal via the projection weights
            token_embedding = self.projection.weight[token_id]
            current = F.normalize(current + 0.1 * token_embedding, dim=-1)

        return self.vocab.decode(ids)

    def train_step(
        self,
        output_vector: torch.Tensor,
        target_text: str,
    ) -> float:
        """
        Single supervised step — trains the projection layer only.
        Returns the loss value.
        """
        target_ids = self.vocab.encode(target_text)
        if not target_ids:
            return 0.0

        # Train to predict first target word from the vector
        logits = self.projection(output_vector)
        target_tensor = torch.tensor(target_ids[0], dtype=torch.long)
        loss = F.cross_entropy(logits.unsqueeze(0), target_tensor.unsqueeze(0))

        # Backward pass on projection only
        self.projection.zero_grad()
        loss.backward()
        with torch.no_grad():
            for p in self.projection.parameters():
                if p.grad is not None:
                    p.data -= 0.01 * p.grad

        return loss.item()
