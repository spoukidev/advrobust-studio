"""Perturbation-budget and correctness checks for every attack."""
import pytest
import torch
import torch.nn as nn

from studio.attacks import ATTACKS, cw_l2, fgsm, pgd_l2, pgd_linf, transfer_attack
from studio.attacks.pgd import project

TOL = 1e-6


@pytest.mark.parametrize("eps", [1 / 255, 4 / 255, 8 / 255, 16 / 255])
@pytest.mark.parametrize("attack", [fgsm, pgd_linf])
def test_linf_budget_respected(model, batch, attack, eps):
    x, y = batch
    x_adv = attack(model, x, y, eps=eps, steps=5) if attack is pgd_linf else attack(model, x, y, eps=eps)
    assert (x_adv - x).abs().max().item() <= eps + TOL


@pytest.mark.parametrize("eps", [0.1, 0.5, 1.0])
@pytest.mark.parametrize("attack", [pgd_l2, cw_l2])
def test_l2_budget_respected(model, batch, attack, eps):
    x, y = batch
    x_adv = attack(model, x, y, eps=eps, steps=5)
    assert (x_adv - x).flatten(1).norm(dim=1).max().item() <= eps + 1e-4


@pytest.mark.parametrize("name", list(ATTACKS))
def test_output_in_image_box_and_shape(model, batch, name):
    x, y = batch
    x_adv = ATTACKS[name](model, x, y, eps=8 / 255 if "linf" in name or name == "fgsm" else 0.5, steps=3)
    assert x_adv.shape == x.shape
    assert x_adv.min() >= 0 and x_adv.max() <= 1
    assert not x_adv.requires_grad


def test_targeted_pgd_respects_budget(model, batch):
    x, _ = batch
    t = torch.full((len(x),), 3)
    x_adv = pgd_linf(model, x, t, eps=8 / 255, steps=5, targeted=True)
    assert (x_adv - x).abs().max() <= 8 / 255 + TOL


def test_project_l2_and_linf():
    d = torch.randn(4, 3, 8, 8) * 10
    assert project(d, 0.3, "linf").abs().max() <= 0.3
    assert torch.all(project(d, 0.5, "l2").flatten(1).norm(dim=1) <= 0.5 + 1e-5)
    small = torch.full((1, 3, 2, 2), 1e-3)
    assert torch.allclose(project(small, 1.0, "l2"), small)  # inside the ball: unchanged


class _Linear(nn.Module):
    """Two-class linear model on a 1x1x1 'image': logit = (+w (x-0.25), -w (x-0.25))."""

    def forward(self, x):
        v = x.flatten(1)[:, :1] - 0.25
        return torch.cat([v, -v], 1) * 10


def test_fgsm_moves_against_true_class():
    m = _Linear()
    x = torch.full((1, 1, 1, 1), 0.5)
    y = torch.tensor([0])  # class 0 prefers large x -> attack must decrease x
    x_adv = fgsm(m, x, y, eps=0.1)
    assert torch.isclose(x_adv, torch.tensor(0.4)).all()


def test_pgd_reduces_accuracy_on_trained_linear_problem():
    """PGD must actually find adversarials on a model that is correct on clean data."""
    torch.manual_seed(0)
    m = _Linear()
    x = torch.rand(32, 1, 1, 1) * 0.1 + 0.5  # all ~0.55 -> class 0
    y = torch.zeros(32, dtype=torch.long)
    assert (m(x).argmax(1) == y).all()
    x_adv = pgd_linf(m, x, y, eps=0.7, steps=10)
    assert (m(x_adv).argmax(1) != y).all()


def test_attack_restores_train_mode(model, batch):
    model.train()
    pgd_linf(model, *batch, steps=1)
    assert model.training
    model.eval()


def test_transfer_attack_keys(model, batch):
    out = transfer_attack(model, model, *batch, eps=8 / 255, steps=2)
    for k in ("x_adv", "surrogate_adv_acc", "victim_clean_acc", "victim_adv_acc", "transfer_rate"):
        assert k in out
