"""Black-box transfer attack: craft on a surrogate, evaluate on a victim.

The attacker never queries the victim's gradients -- only its final
predictions are used for scoring. This measures *transferability*, the
practical black-box threat when an attacker can train or download a similar
model.
"""
from __future__ import annotations

import torch

from . import fgsm as _fgsm  # noqa: F401  (module import keeps dispatch simple)
from .pgd import pgd


@torch.no_grad()
def _predict(model, x, batch_size=256):
    model.eval()
    return torch.cat([model(x[i:i + batch_size]).argmax(1) for i in range(0, len(x), batch_size)])


def transfer_attack(surrogate, victim, x, y, attack=None, **attack_kw):
    """Craft adversarial examples on ``surrogate`` and score them on ``victim``.

    Returns a dict with the adversarial batch and accuracy numbers:
    ``surrogate_adv_acc`` (white-box) and ``victim_clean_acc`` /
    ``victim_adv_acc`` (black-box transfer), plus ``transfer_rate`` = fraction
    of victim-correct examples that the transferred perturbation flips.
    """
    attack = attack or pgd
    x_adv = attack(surrogate, x, y, **attack_kw)
    v_clean = _predict(victim, x)
    v_adv = _predict(victim, x_adv)
    s_adv = _predict(surrogate, x_adv)
    correct = v_clean == y
    flipped = (v_adv != y) & correct
    return {
        "x_adv": x_adv,
        "surrogate_adv_acc": (s_adv == y).float().mean().item(),
        "victim_clean_acc": correct.float().mean().item(),
        "victim_adv_acc": (v_adv == y).float().mean().item(),
        "transfer_rate": (flipped.sum() / correct.sum().clamp_min(1)).item(),
    }
