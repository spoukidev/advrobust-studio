"""Lightweight randomized smoothing (Cohen et al., 2019).

The smoothed classifier predicts the class most likely under Gaussian noise:

    g(x) = argmax_c  P_{d ~ N(0, sigma^2 I)} [ f(x + d) = c ]

We estimate it with ``n`` Monte-Carlo samples and return averaged softmax
probabilities (log-transformed so it plugs into cross-entropy). Because the
forward pass is a differentiable expectation, white-box attacks against it
are *EOT* attacks (expectation over transformation) automatically.

This is the "prediction" half of randomized smoothing only -- no certified
radius is computed (documented limitation).
"""
import torch
import torch.nn as nn


class RandomizedSmoothing(nn.Module):
    def __init__(self, base: nn.Module, sigma: float = 0.12, n: int = 32):
        super().__init__()
        self.base, self.sigma, self.n = base, sigma, n

    def forward(self, x):
        b = x.size(0)
        xs = x.repeat_interleave(self.n, 0)
        noisy = (xs + torch.randn_like(xs) * self.sigma).clamp(0, 1)
        probs = self.base(noisy).softmax(-1).view(b, self.n, -1).mean(1)
        return probs.clamp_min(1e-12).log()  # log-probs act as logits
