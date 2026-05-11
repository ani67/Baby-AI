"""The language head — a small conditional GRU.

Architecture
------------
A 2-layer GRU language model. At generation time it is conditioned on
a concatenation of the current affect composite (12 dims) and the
top-5 active concept embeddings (5 × 256 = 1280 dims), totaling 1292d.
The conditioning vector is projected to the GRU's initial hidden state
across both layers.

The head is the mind's mouth, not the mind itself. It learns fluent
English from the corpus the mind was surprised by; the mind chooses
WHEN to speak (D's action selection) and WHAT to speak ABOUT (the
active set built by F.processing_loop). The language head only
decides HOW to phrase it.

Inference: temperature 0.7, max 20 tokens, multinomial sampling.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Special tokens (lock these IDs — vocab files reference them).
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]

# Architecture constants — locked so train + load match exactly.
EMBED_DIM       = 256
HIDDEN_DIM      = 256
NUM_LAYERS      = 2
COND_DIM        = 12 + 5 * 256          # affect[12] + top-5 concept embeddings
TOP_K_CONCEPTS  = 5
GENERATION_MAX  = 20
GENERATION_TEMP = 0.7


@dataclass
class Vocab:
    """Token ↔ id dictionary plus the reverse list for fast id→token."""
    token_to_id: dict[str, int]
    id_to_token: list[str]

    @property
    def size(self) -> int:
        return len(self.id_to_token)

    def encode(self, tokens: list[str]) -> list[int]:
        unk = UNK_ID
        return [self.token_to_id.get(t, unk) for t in tokens]

    def decode(self, ids: list[int]) -> list[str]:
        return [self.id_to_token[i] if 0 <= i < self.size else "<unk>" for i in ids]

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"id_to_token": self.id_to_token}, f)

    @classmethod
    def load(cls, path: str) -> "Vocab":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        ids = data["id_to_token"]
        return cls(token_to_id={t: i for i, t in enumerate(ids)}, id_to_token=ids)


class LanguageHead(nn.Module):
    """Conditional GRU language model. Tiny — ~5M params at vocab=8K."""

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, EMBED_DIM, padding_idx=PAD_ID)
        # Project the 1292-d conditioning vector into NUM_LAYERS × HIDDEN_DIM
        # initial-hidden tensor for the GRU stack.
        self.cond_proj = nn.Linear(COND_DIM, NUM_LAYERS * HIDDEN_DIM)
        self.gru = nn.GRU(EMBED_DIM, HIDDEN_DIM, num_layers=NUM_LAYERS, batch_first=True)
        self.out = nn.Linear(HIDDEN_DIM, vocab_size)

    def init_hidden(self, cond: torch.Tensor) -> torch.Tensor:
        """cond: (B, COND_DIM) → h0: (NUM_LAYERS, B, HIDDEN_DIM)."""
        B = cond.shape[0]
        h = self.cond_proj(cond).view(B, NUM_LAYERS, HIDDEN_DIM)
        # GRU expects (num_layers, batch, hidden).
        return torch.tanh(h.permute(1, 0, 2).contiguous())

    def forward(self, tokens: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """tokens: (B, T)  cond: (B, COND_DIM)  →  logits: (B, T, V)."""
        h0 = self.init_hidden(cond)
        emb = self.embed(tokens)
        out, _ = self.gru(emb, h0)
        return self.out(out)

    @torch.no_grad()
    def generate(
        self,
        cond: torch.Tensor,
        max_len: int = GENERATION_MAX,
        temperature: float = GENERATION_TEMP,
        repetition_penalty: float = 1.5,
        blocked_start_ids: list[int] | None = None,
    ) -> list[int]:
        """Greedy-multinomial decode from <bos> until <eos> or max_len.

        cond: shape (1, COND_DIM) — single example only (no batched gen).
        Returns the list of generated ids (excludes the <bos> seed and
        the terminal <eos>).

        repetition_penalty: divides logit value of any already-generated
            token (clamped to a floor of 1.0). Was 1.0 (no penalty) in
            the v0.5 LSTM era; bumped to 1.5 after v0.7b training showed
            opening-phrase collapse ("what is …" repeated across all
            candidates regardless of prompt).

        blocked_start_ids: token ids whose logits are -inf at position 0
            (only). Used to suppress overfit prefixes. Mid-sequence the
            same tokens are allowed — only the *opening* is blocked.
        """
        self.eval()
        device = next(self.parameters()).device
        h = self.init_hidden(cond.to(device))
        x = torch.tensor([[BOS_ID]], device=device, dtype=torch.long)
        produced: list[int] = []
        blocked_set = set(blocked_start_ids or [])
        for _ in range(max_len):
            emb = self.embed(x)
            out, h = self.gru(emb, h)
            logits = self.out(out[:, -1, :]) / max(temperature, 1e-6)

            # Repetition penalty: divide previously-generated tokens'
            # logits down (or multiply for negative logits, since dividing
            # a negative number makes it less negative i.e. *more* likely
            # — the standard fix is to multiply when logit < 0).
            if repetition_penalty and repetition_penalty != 1.0 and produced:
                seen = set(produced)
                row = logits[0]
                for tok in seen:
                    v = row[tok].item()
                    row[tok] = v / repetition_penalty if v > 0 else v * repetition_penalty
                logits[0] = row

            # Position-0 prefix block: zero out logits for overfit
            # opening tokens. Only at the first decode step.
            if not produced and blocked_set:
                for tok in blocked_set:
                    if 0 <= tok < logits.shape[-1]:
                        logits[0, tok] = float("-inf")

            probs = torch.softmax(logits, dim=-1)
            tok = int(torch.multinomial(probs[0], num_samples=1).item())
            if tok in (EOS_ID, PAD_ID):
                break
            if tok == UNK_ID:
                # Skip <unk> — re-sample from the top of the distribution.
                top_ids = torch.topk(probs[0], k=10).indices.tolist()
                top_ids = [t for t in top_ids if t not in (PAD_ID, UNK_ID, BOS_ID, EOS_ID)]
                if not top_ids:
                    break
                tok = top_ids[0]
            produced.append(tok)
            x = torch.tensor([[tok]], device=device, dtype=torch.long)
        return produced


def build_conditioning_vector(
    affect_composite: np.ndarray,
    concept_embeddings: list[np.ndarray],
) -> np.ndarray:
    """Pack (affect, top-5 concept embeddings) into a fixed COND_DIM vector.

    Pads with zeros if fewer than TOP_K_CONCEPTS embeddings are supplied.
    Returns float32[COND_DIM].
    """
    if affect_composite.shape != (12,):
        raise ValueError(f"affect must be (12,), got {affect_composite.shape}")
    out = np.zeros(COND_DIM, dtype=np.float32)
    out[:12] = affect_composite.astype(np.float32, copy=False)
    for i, emb in enumerate(concept_embeddings[:TOP_K_CONCEPTS]):
        # v1.0 migration: the GPT-2 / native v2 conditioning was trained
        # at 256-dim. The post-migration graph is 512-dim but the
        # migration's top-half-identity projection preserves the
        # original 256 dims exactly — so truncating 512 → 256 gives
        # the LM head the same vector it was trained on. Strict 256
        # is still accepted unchanged.
        if emb.shape == (512,):
            emb = emb[:256]
        elif emb.shape != (256,):
            raise ValueError(f"concept embedding must be (256,) or (512,), got {emb.shape}")
        out[12 + i * 256: 12 + (i + 1) * 256] = emb.astype(np.float32, copy=False)
    return out


# ============================================================
# Tokenization (matches scripts/build_vocab.py)
# ============================================================

import re

_TOKEN_RE = re.compile(r"[a-z][a-z'-]*[a-z]|[a-z]")


def tokenize(text: str) -> list[str]:
    """Lowercase + simple word-boundary tokenization. Identical to the
    encoder's tokenization so vocab built from book text aligns with
    what the LM was trained on.
    """
    return _TOKEN_RE.findall(text.lower())


def render_tokens(ids: list[int], vocab: Vocab) -> str:
    return " ".join(vocab.decode(ids))


# ============================================================
# Disk paths + load/save helpers
# ============================================================

DEFAULT_WEIGHTS_PATH = "data/language_head.pt"
DEFAULT_VOCAB_PATH   = "data/language_head_vocab.json"


def save_checkpoint(model: LanguageHead, path: str = DEFAULT_WEIGHTS_PATH) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"vocab_size": model.vocab_size, "state_dict": model.state_dict()}, path)


def load_checkpoint(path: str = DEFAULT_WEIGHTS_PATH, device: str = "cpu") -> LanguageHead:
    obj = torch.load(path, map_location=device, weights_only=True)
    model = LanguageHead(vocab_size=int(obj["vocab_size"]))
    model.load_state_dict(obj["state_dict"])
    model.to(device)
    model.eval()
    return model


def is_available() -> bool:
    return os.path.exists(DEFAULT_WEIGHTS_PATH) and os.path.exists(DEFAULT_VOCAB_PATH)


# ============================================================
# Phase 7: GPT-2 small as a conditioned decoder
# ============================================================
#
# The Phase 5 LSTM language head learned grammar from the mind's
# surprised-sentence corpus, but with only ~5M params and ~few-thousand
# training examples it produced word-soup. GPT-2 small (117M params)
# already speaks fluent English; we just need to steer it.
#
# Steering mechanism: project the mind's conditioning vector
# (12-d affect + 5×256-d top concept embeddings = 1292 dims) into GPT-2's
# hidden dimension (768) and inject it as a virtual prefix token's
# embedding. GPT-2's attention sees it as the first token of context.
#
# Fine-tuning surface: only condition_proj and the top 4 transformer
# blocks (layers 8-11). The bottom 8 + token embeddings + lm_head stay
# frozen. This preserves GPT-2's English while letting the top of the
# stack learn to use the soft prefix.

CONDITIONED_DECODER_FILENAME    = "language_head_gpt2.pt"
CONDITIONED_DECODER_V2_FILENAME = "language_head_gpt2_v2.pt"   # Phase 7 A3
GPT2_HIDDEN_DIM = 768
GPT2_NUM_LAYERS = 12
GPT2_FROZEN_BOTTOM = 8                 # layers 0-7 frozen; 8-11 fine-tune
GPT2_BASE_MODEL = "gpt2"

# Phase 7 A2: multi-token soft prefix.
#
# v0.4 used a single virtual token: condition_proj projected the 1292-d
# conditioning vector to one 768-d "embedding" prepended as the first
# token of the GPT-2 context. One token is too narrow a channel — GPT-2's
# 117M-param web-text priors swamp the conditioning at decode time, so
# answers drift toward generic web language.
#
# v0.5 projects to FIVE virtual tokens. Each gets its own 768-d slot in
# the prefix sequence; GPT-2's attention sees five contextual cues to
# lean on instead of one. Trainable params on condition_proj go from
# ~1M (768·1292) to ~5M (5·768·1292) — still negligible vs the 29.3M
# trainable from the top-4 transformer layers.
N_PREFIX_TOKENS = 5


def _try_import_transformers():
    """Attempt the imports lazily — keeps Expression's silent-fallback
    contract clean when the dependency isn't installed."""
    try:
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        return GPT2LMHeadModel, GPT2Tokenizer
    except Exception:
        return None, None


class ConditionedDecoder(nn.Module):
    """GPT-2 small steered by a soft prefix derived from the mind's
    composite affect + top-K active-concept embeddings.

    At inference: concept_proj(cond) → tanh → unsqueeze to (1, 1, 768)
    → fed as `inputs_embeds` to GPT-2 transformer → past_key_values
    capture the prefix → autoregressive sampling continues from there.

    At training: prefix prepended to the surprised sentence's token
    embeddings; standard causal-LM cross-entropy on the sentence's tokens
    (the prefix position is masked out via labels=-100). Optimizer only
    touches `condition_proj` + the top 4 GPT-2 layers.
    """

    def __init__(self, base_model_name: str = GPT2_BASE_MODEL) -> None:
        super().__init__()
        GPT2LMHeadModel, GPT2Tokenizer = _try_import_transformers()
        if GPT2LMHeadModel is None:
            raise RuntimeError("transformers is not installed")
        self.tokenizer = GPT2Tokenizer.from_pretrained(base_model_name)
        # GPT-2 has no pad token by default; reuse EOS for safety.
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.base_model = GPT2LMHeadModel.from_pretrained(base_model_name)
        self.base_model.config.pad_token_id = self.tokenizer.pad_token_id

        # Phase 7 A2: project to N_PREFIX_TOKENS × GPT2_HIDDEN_DIM, then
        # reshape to (B, N_PREFIX_TOKENS, GPT2_HIDDEN_DIM) at forward time.
        self.condition_proj = nn.Linear(
            COND_DIM, N_PREFIX_TOKENS * GPT2_HIDDEN_DIM,
        )

        self._freeze_partial()

    def _freeze_partial(self) -> None:
        """Freeze token embeddings, position embeddings, and the bottom
        GPT2_FROZEN_BOTTOM (=8) transformer layers. Leave the top 4
        layers + final LayerNorm + lm_head trainable, plus condition_proj."""
        # Token + position embeddings frozen.
        for p in self.base_model.transformer.wte.parameters():
            p.requires_grad = False
        for p in self.base_model.transformer.wpe.parameters():
            p.requires_grad = False
        # Bottom 8 layers frozen, top 4 trainable.
        n_layers = len(self.base_model.transformer.h)
        for i in range(n_layers):
            trainable = (i >= GPT2_FROZEN_BOTTOM)
            for p in self.base_model.transformer.h[i].parameters():
                p.requires_grad = trainable
        # Final LayerNorm + lm_head left trainable (default).
        # condition_proj is trainable by default.

    # ---- shared: build the multi-token soft prefix ----

    def _build_prefix(self, cond: torch.Tensor) -> torch.Tensor:
        """cond: (B, COND_DIM) → prefix embeds: (B, N_PREFIX_TOKENS, H).

        Linear projection followed by tanh squashing (matches the v0.4
        single-token formulation in spirit) reshaped to N_PREFIX_TOKENS
        independent virtual-token slots.
        """
        B = cond.shape[0]
        flat = self.condition_proj(cond)                              # (B, N*H)
        return torch.tanh(flat.view(B, N_PREFIX_TOKENS, GPT2_HIDDEN_DIM))

    # ---- training helper: forward pass with prefix ----

    def forward_with_prefix(
        self,
        cond: torch.Tensor,            # (B, COND_DIM)
        token_ids: torch.Tensor,       # (B, T)  the sentence to learn
        attention_mask: torch.Tensor,  # (B, T)
    ) -> torch.Tensor:
        """Forward pass with N_PREFIX_TOKENS soft prefix slots prepended.
        Returns the loss (causal-LM CE; labels masked at every prefix
        position so the model trains only on real-token next-step
        predictions)."""
        B, T = token_ids.shape
        # N virtual tokens of conditioning prefix.
        prefix = self._build_prefix(cond)                                  # (B, N, H)
        # Embed the sentence tokens.
        token_embeds = self.base_model.transformer.wte(token_ids)          # (B, T, H)
        inputs_embeds = torch.cat([prefix, token_embeds], dim=1)           # (B, N+T, H)
        # Attention mask: N ones for prefix, then the existing mask.
        prefix_mask = torch.ones(
            (B, N_PREFIX_TOKENS),
            dtype=attention_mask.dtype, device=attention_mask.device,
        )
        full_mask = torch.cat([prefix_mask, attention_mask], dim=1)        # (B, N+T)
        # Labels: -100 for every prefix slot, token_ids for the rest.
        # GPT-2 shifts internally for CE — labels[i] is predicted from i-1.
        prefix_labels = torch.full(
            (B, N_PREFIX_TOKENS), -100,
            dtype=token_ids.dtype, device=token_ids.device,
        )
        labels = torch.cat([prefix_labels, token_ids], dim=1)
        # Mask out pad positions in labels too.
        labels = labels.masked_fill(full_mask == 0, -100)

        out = self.base_model(
            inputs_embeds=inputs_embeds,
            attention_mask=full_mask,
            labels=labels,
        )
        return out.loss

    # ---- inference: generate from cond ----

    @torch.no_grad()
    def generate(
        self,
        cond: torch.Tensor,                 # (1, COND_DIM)
        max_tokens: int = 30,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.3,
    ) -> str:
        """Sample autoregressively from the conditioned prefix. Stops at
        the first sentence-ending punctuation, EOS, or `max_tokens`."""
        self.eval()
        device = next(self.parameters()).device
        cond = cond.to(device)

        # N_PREFIX_TOKENS virtual tokens. The first forward pass populates
        # past_key_values across all of them so subsequent autoregressive
        # steps attend to the full conditioning context.
        prefix = self._build_prefix(cond)                            # (1, N, H)
        out = self.base_model(inputs_embeds=prefix, use_cache=True)
        past = out.past_key_values
        logits = out.logits[:, -1, :]                                # (1, V)

        eos_id = self.tokenizer.eos_token_id
        generated: list[int] = []

        for _ in range(max_tokens):
            scaled = logits.clone() / max(temperature, 1e-6)

            # Repetition penalty applied to already-emitted tokens.
            if repetition_penalty and repetition_penalty != 1.0:
                for tok in set(generated):
                    v = scaled[0, tok]
                    scaled[0, tok] = v / repetition_penalty if v > 0 else v * repetition_penalty

            probs = torch.softmax(scaled, dim=-1)

            # Top-p (nucleus) sampling.
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            mask = cumulative > top_p
            mask[..., 0] = False                                     # always keep at least the top
            sorted_probs = sorted_probs.masked_fill(mask, 0.0)
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)
            sample_pos = torch.multinomial(sorted_probs[0], num_samples=1).item()
            token_id = int(sorted_idx[0, sample_pos].item())

            if token_id == eos_id:
                break
            generated.append(token_id)

            # Sentence boundary stop.
            text_so_far = self.tokenizer.decode(generated, skip_special_tokens=True)
            if text_so_far and text_so_far.rstrip()[-1:] in ".!?":
                break

            next_input = torch.tensor([[token_id]], device=device, dtype=torch.long)
            out = self.base_model(input_ids=next_input, past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = out.logits[:, -1, :]

        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    # ---- save / load: only the trainable (delta) parameters ----

    def save_delta(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        delta: dict[str, torch.Tensor] = {}
        # condition_proj
        for k, v in self.condition_proj.state_dict().items():
            delta[f"condition_proj.{k}"] = v.cpu()
        # top 4 transformer layers
        n_layers = len(self.base_model.transformer.h)
        for i in range(GPT2_FROZEN_BOTTOM, n_layers):
            for k, v in self.base_model.transformer.h[i].state_dict().items():
                delta[f"transformer.h.{i}.{k}"] = v.cpu()
        # final LayerNorm
        for k, v in self.base_model.transformer.ln_f.state_dict().items():
            delta[f"transformer.ln_f.{k}"] = v.cpu()
        # lm_head (often tied to wte; saving anyway for safety on swap)
        for k, v in self.base_model.lm_head.state_dict().items():
            delta[f"lm_head.{k}"] = v.cpu()
        torch.save({"delta": delta, "base_model_name": GPT2_BASE_MODEL}, path)

    def load_delta(self, path: str) -> None:
        obj = torch.load(path, map_location="cpu", weights_only=True)
        delta = obj["delta"]
        full_state = self.state_dict()
        for k, v in delta.items():
            # Map our flat keys back to model state_dict keys.
            if k.startswith("condition_proj."):
                full_state[k] = v
            elif k.startswith("transformer.h."):
                full_state[f"base_model.{k}"] = v
            elif k.startswith("transformer.ln_f."):
                full_state[f"base_model.{k}"] = v
            elif k.startswith("lm_head."):
                full_state[f"base_model.{k}"] = v
        # strict=False because the frozen-base parameters aren't in `delta`
        self.load_state_dict(full_state, strict=False)


def is_conditioned_decoder_available(weights_path: str) -> bool:
    """Both the saved delta on disk AND the transformers package must
    be importable for ConditionedDecoder to be usable."""
    if not os.path.exists(weights_path):
        return False
    GPT2LMHeadModel, _ = _try_import_transformers()
    return GPT2LMHeadModel is not None
