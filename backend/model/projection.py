"""
Learned projection layer: 512→512 residual linear transform.

Adapts frozen CLIP embeddings to what the neural graph can distinguish.
Trained by the distributed error signal (teacher - model_output) using
an outer-product update rule (equivalent to gradient descent on MSE).

Starts as identity and gradually deviates via warmup ramp.
"""

import torch
import torch.nn.functional as F


class LearnedProjection:
    def __init__(self, dim: int = 512, warmup_steps: int = 5000):
        self.dim = dim
        self.warmup_steps = warmup_steps
        self._delta = torch.zeros(dim, dim)
        self._momentum = torch.zeros(dim, dim)
        self._step = 0
        self._lr = 0.001
        self._momentum_decay = 0.9

    @property
    def alpha(self) -> float:
        """Ramp 0→1 over warmup_steps. At 0, projection = identity."""
        return min(1.0, self._step / max(self.warmup_steps, 1))

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """Project a 512-d vector. Returns L2-normalized result.

        P(x) = normalize(x + alpha * Delta @ x)
        At alpha=0 this is identity. Gradually deviates as training progresses.
        """
        if self.alpha < 1e-6:
            return F.normalize(x, dim=-1)
        projected = x + self.alpha * (x @ self._delta.T)
        return F.normalize(projected, dim=-1)

    def update(self, raw_input: torch.Tensor, error: torch.Tensor):
        """Update projection using distributed error signal.

        raw_input: un-projected CLIP vector (512,)
        error: teacher_projected - model_output (512,)
        """
        # Outer product = exact gradient of ||teacher - P(input)||^2 w.r.t. Delta
        grad = error.unsqueeze(0).T @ raw_input.unsqueeze(0)  # (512, 512)
        self._momentum = self._momentum_decay * self._momentum + (1 - self._momentum_decay) * grad
        self._delta = self._delta + self._lr * self.alpha * self._momentum
        self._step += 1
        # Periodic norm clamp to prevent explosion
        if self._step % 100 == 0:
            norm = self._delta.norm()
            if norm > 10.0:
                self._delta = self._delta * (10.0 / norm)
            if self._step % 5000 == 0:
                print(
                    f"[projection] step={self._step} alpha={self.alpha:.3f} "
                    f"delta_norm={self._delta.norm():.4f}",
                    flush=True,
                )

    def state_dict(self) -> dict:
        return {
            "projection_delta": self._delta.clone(),
            "projection_momentum": self._momentum.clone(),
            "projection_step": self._step,
        }

    def load_state_dict(self, d: dict):
        self._delta = d.get("projection_delta", torch.zeros(self.dim, self.dim))
        self._momentum = d.get("projection_momentum", torch.zeros(self.dim, self.dim))
        self._step = d.get("projection_step", 0)
        print(
            f"[projection] restored step={self._step} alpha={self.alpha:.3f} "
            f"delta_norm={self._delta.norm():.4f}",
            flush=True,
        )
