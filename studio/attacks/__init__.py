"""Hand-written white-box attacks. All inputs/outputs are images in [0, 1]."""
from .fgsm import fgsm
from .pgd import pgd, pgd_linf, pgd_l2
from .cw import cw_l2
from .transfer import transfer_attack

ATTACKS = {
    "fgsm": fgsm,
    "pgd_linf": pgd_linf,
    "pgd_l2": pgd_l2,
    "cw_l2": cw_l2,
}

__all__ = ["fgsm", "pgd", "pgd_linf", "pgd_l2", "cw_l2", "transfer_attack", "ATTACKS", "run_attack"]


def run_attack(name: str, model, x, y, **kw):
    """Dispatch by name, e.g. ``run_attack("pgd_linf", model, x, y, eps=8/255)``."""
    return ATTACKS[name](model, x, y, **kw)
