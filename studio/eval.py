"""Robustness metrics.

* clean accuracy
* robust accuracy @ eps (accuracy on adversarial inputs)
* attack success rate (ASR) = fraction of *originally correct* examples the
  attack flips (untargeted) / pushes to the target (targeted)
* mean / max L-inf and L2 perturbation size (sanity-checks the budget)
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import torch

from .attacks import ATTACKS


@dataclass
class AttackResult:
    model: str
    attack: str
    eps: float
    norm: str
    n: int
    clean_acc: float
    robust_acc: float
    asr: float
    mean_linf: float
    max_linf: float
    mean_l2: float
    max_l2: float
    seconds: float
    defense: str = "none"

    def to_dict(self):
        return asdict(self)


@torch.no_grad()
def predict(model, x: torch.Tensor, batch_size: int = 256) -> torch.Tensor:
    model.eval()
    return torch.cat([model(x[i:i + batch_size]).argmax(1) for i in range(0, len(x), batch_size)])


def accuracy(model, x, y, batch_size=256) -> float:
    return (predict(model, x, batch_size) == y).float().mean().item()


def perturbation_stats(x: torch.Tensor, x_adv: torch.Tensor) -> dict:
    d = (x_adv - x).flatten(1)
    linf, l2 = d.abs().max(1).values, d.norm(dim=1)
    return {"mean_linf": linf.mean().item(), "max_linf": linf.max().item(),
            "mean_l2": l2.mean().item(), "max_l2": l2.max().item()}


def generate(attack_fn, model, x, y, batch_size=250, **kw) -> torch.Tensor:
    return torch.cat([attack_fn(model, x[i:i + batch_size], y[i:i + batch_size], **kw)
                      for i in range(0, len(x), batch_size)])


def evaluate_attack(model, x, y, attack: str, eps: float, model_name: str = "model",
                    eval_model=None, defense: str = "none", batch_size: int = 250, **attack_kw) -> AttackResult:
    """Craft ``attack`` against ``model`` and score on ``eval_model`` (defaults to ``model``).

    Passing a defended ``model`` gives an *adaptive* white-box evaluation
    (gradients flow through the defense via BPDA/EOT). Passing the undefended
    model as ``model`` and the defended one as ``eval_model`` gives the weaker
    *defense-unaware* (gray-box) setting.
    """
    eval_model = eval_model or model
    norm = "l2" if attack in ("pgd_l2", "cw_l2") else "linf"
    t0 = time.time()
    clean_pred = predict(eval_model, x)
    if eps == 0:
        x_adv = x.clone()
    else:
        x_adv = generate(ATTACKS[attack], model, x, y, batch_size=batch_size, eps=eps, **attack_kw)
    adv_pred = predict(eval_model, x_adv)
    correct = clean_pred == y
    flipped = correct & (adv_pred != y)
    st = perturbation_stats(x, x_adv)
    return AttackResult(model=model_name, attack=attack, eps=float(eps), norm=norm, n=len(y),
                        clean_acc=correct.float().mean().item(),
                        robust_acc=(adv_pred == y).float().mean().item(),
                        asr=(flipped.sum() / correct.sum().clamp_min(1)).item(),
                        seconds=round(time.time() - t0, 2), defense=defense, **st)


def eps_curve(model, x, y, eps_list, attack="pgd_linf", model_name="model", **kw) -> list[dict]:
    """Robust accuracy as a function of epsilon."""
    return [evaluate_attack(model, x, y, attack, e, model_name=model_name, **kw).to_dict() for e in eps_list]
