"""Fast Gradient Sign Method (Goodfellow et al., 2015).

    x_adv = clip_[0,1]( x + eps * sign( grad_x L(f(x), y) ) )

One gradient evaluation; the L-inf distance to x is at most eps by
construction (sign() is in {-1, 0, 1}).
"""
import torch

from ._common import attack_loss, eval_mode


def fgsm(model, x: torch.Tensor, y: torch.Tensor, eps: float = 8 / 255,
         targeted: bool = False, **_) -> torch.Tensor:
    """Return FGSM adversarial images.

    Args:
        model: classifier taking images in [0, 1].
        x: clean batch [N, C, H, W] in [0, 1].
        y: true labels (untargeted) or target labels (``targeted=True``).
        eps: L-inf budget in pixel units.
    """
    with eval_mode(model):
        x_adv = x.clone().detach().requires_grad_(True)
        loss = attack_loss(model(x_adv), y, targeted)
        (grad,) = torch.autograd.grad(loss, x_adv)
        # Untargeted: step *up* the loss. Targeted loss is already negated.
        x_adv = x + eps * grad.sign()
        return x_adv.clamp(0.0, 1.0).detach()
