"""Component 1: Base Inference

Loads Llama-3.2-3B-Instruct-4bit via MLX and provides a single
run_inference() function. Model and tokenizer are loaded once at
module import time.
"""

import mlx_lm
from mlx_lm.sample_utils import make_sampler

MODEL_NAME = "mlx-community/Llama-3.2-3B-Instruct-4bit"

# Load once at import — not per call
_model, _tokenizer = mlx_lm.load(MODEL_NAME)


def run_inference(
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> tuple[str, float, int]:
    """Run inference and return (response, tokens_per_second, token_count)."""
    response_text = ""
    tokens_per_second = 0.0
    token_count = 0

    sampler = make_sampler(temp=temperature)

    for resp in mlx_lm.stream_generate(
        _model,
        _tokenizer,
        prompt,
        max_tokens=max_tokens,
        sampler=sampler,
    ):
        response_text += resp.text
        tokens_per_second = resp.generation_tps
        token_count = resp.generation_tokens

    return response_text, tokens_per_second, token_count
