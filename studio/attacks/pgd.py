"""Projected Gradient Descent (Madry et al., 2018), L-inf and L2.

Iterate ``steps`` times:

    L-inf:  x <- Proj_{||x - x0||_inf <= eps}( x + alpha * sign(g) )
    L2:     x <- Proj_{||x - x0||_2   <= eps}( x + alpha * g / ||g||_2 )

followed by clipping to the valid image box [0, 1]. A random start inside the
eps-ball makes the attack stronger and less sensitive to gradient masking.
"""
from __future__ import annotations

import torch

from ._common import attack_loss, eval_mode


def _flat_norm(t: torch.Tensor) -> torch.Tensor:
    """Per-example L2 norm, shaped for broadcasting back to [N, C, H, W]."""
    return t.flatten(1).norm(p=2, dim=1).clamp_min(1e-12).view(-1, *([1] * (t.dim() - 1)))


def project(delta: torch.Tensor, eps: float, norm: str) -> torch.Tensor:
    """Project a perturbation onto the eps-ball of the given norm."""
    if norm == "linf":
        return delta.clamp(-eps, eps)
    if norm == "l2":
        factor = (eps / _flat_norm(delta)).clamp(max=1.0)
        return delta * factor
    raise ValueError(norm)


def pgd(model, x: torch.Tensor, y: torch.Tensor, eps: float = 8 / 255, alpha: float | None = None,
        steps: int = 10, norm: str = "linf", targeted: bool = False, random_start: bool = True,
        **_) -> torch.Tensor:
    """Generic PGD.

    Args:
        eps:   budget (pixel units). Typical: 8/255 for L-inf, 0.5 for L2 on CIFAR.
        alpha: step size; default 2.5 * eps / steps (Madry et al.).
        steps: number of gradient iterations.
        norm:  "linf" or "l2".
    """
    alpha = alpha if alpha is not None else 2.5 * eps / steps
    x0 = x.clone().detach()

    with eval_mode(model):
        if random_start:
            if norm == "linf":
                delta = torch.empty_like(x0).uniform_(-eps, eps)
            else:
                d = torch.randn_like(x0)
                r = torch.rand(x0.size(0), *([1] * (x0.dim() - 1)), device=x0.device)
                delta = d / _flat_norm(d) * eps * r
            delta = ((x0 + delta).clamp(0, 1) - x0)
        else:
            delta = torch.zeros_like(x0)

        for _ in range(steps):
            delta.requires_grad_(True)
            loss = attack_loss(model(x0 + delta), y, targeted)
            (grad,) = torch.autograd.grad(loss, delta)
            with torch.no_grad():
                if norm == "linf":
                    delta = delta + alpha * grad.sign()
                else:
                    delta = delta + alpha * grad / _flat_norm(grad)
                delta = project(delta, eps, norm)
                # keep x0 + delta inside the image box
                delta = (x0 + delta).clamp(0, 1) - x0
        return (x0 + delta).detach()


def pgd_linf(model, x, y, eps=8 / 255, **kw):
    return pgd(model, x, y, eps=eps, norm="linf", **kw)


def pgd_l2(model, x, y, eps=0.5, **kw):
    return pgd(model, x, y, eps=eps, norm="l2", **kw)
