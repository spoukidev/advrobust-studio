"""Simplified Carlini & Wagner L2 attack (Carlini & Wagner, 2017).

We optimise in tanh-space so the image box constraint holds automatically:

    x_adv = 0.5 * (tanh(w) + 1)
    minimise  ||x_adv - x||_2^2  +  c * f(x_adv)
    f(x')   = max( Z(x')_y - max_{j != y} Z(x')_j , -kappa )   (untargeted)
    f(x')   = max( max_{j != t} Z(x')_j - Z(x')_t , -kappa )   (targeted)

Simplifications vs. the paper: a single fixed constant ``c`` (no binary
search) and Adam for a fixed number of steps. We keep the best (smallest-L2)
successful adversarial per example. If ``eps`` is given, the final
perturbation is additionally projected into the L2 ball so the result is a
valid budgeted attack comparable to PGD-L2.
"""
from __future__ import annotations

import torch

from ._common import eval_mode
from .pgd import project


def cw_l2(model, x: torch.Tensor, y: torch.Tensor, c: float = 1.0, kappa: float = 0.0,
          steps: int = 50, lr: float = 0.01, targeted: bool = False, eps: float | None = None,
          **_) -> torch.Tensor:
    x0 = x.clone().detach()
    n = x0.size(0)
    # inverse of 0.5*(tanh+1); clamp avoids atanh(+-1) = inf
    w = torch.atanh((x0 * 2 - 1).clamp(-1 + 1e-6, 1 - 1e-6)).requires_grad_(True)
    opt = torch.optim.Adam([w], lr=lr)
    best = x0.clone()
    best_l2 = torch.full((n,), float("inf"))

    with eval_mode(model):
        for _ in range(steps):
            x_adv = 0.5 * (torch.tanh(w) + 1)
            logits = model(x_adv)
            onehot = torch.nn.functional.one_hot(y, logits.size(1)).bool()
            real = logits[onehot]
            other = logits.masked_fill(onehot, -1e9).max(1).values
            f = (other - real) if targeted else (real - other)
            f = f.clamp(min=-kappa)
            l2 = (x_adv - x0).flatten(1).pow(2).sum(1)
            loss = (l2 + c * f).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()

            with torch.no_grad():
                pred = logits.argmax(1)
                success = (pred == y) if targeted else (pred != y)
                improved = success & (l2.sqrt() < best_l2)
                best_l2 = torch.where(improved, l2.sqrt(), best_l2)
                best[improved] = x_adv[improved].detach()

        # examples that never succeeded: return the last iterate (still a valid image)
        with torch.no_grad():
            last = 0.5 * (torch.tanh(w) + 1)
            failed = torch.isinf(best_l2)
            best[failed] = last[failed]
            if eps is not None:
                best = (x0 + project(best - x0, eps, "l2")).clamp(0, 1)
    return best.detach()
