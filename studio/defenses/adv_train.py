"""Short PGD adversarial training on CIFAR-10 (Madry et al., 2018).

    min_theta  E_(x,y) [ max_{||d||_inf <= eps}  L(f_theta(x + d), y) ]

The inner max is approximated with K-step PGD (``train_steps``). With
``mode="vanilla"`` the same loop does ordinary ERM training so both
checkpoints share every hyper-parameter except the inner attack -- a fair
comparison.
"""
from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn.functional as F

from ..attacks.pgd import pgd
from ..models import build_cifar_model


def _augment(x: torch.Tensor) -> torch.Tensor:
    """Random crop (pad 4) + horizontal flip, batched."""
    n = x.size(0)
    flip = torch.rand(n) < 0.5
    x = torch.where(flip.view(-1, 1, 1, 1), x.flip(3), x)
    padded = F.pad(x, (4, 4, 4, 4), mode="reflect")
    i, j = torch.randint(0, 9, (2,))
    return padded[:, :, i:i + 32, j:j + 32]


def train(x_train: torch.Tensor, y_train: torch.Tensor, arch: str = "cifar_cnn", mode: str = "vanilla",
          epochs: int = 10, batch_size: int = 256, lr: float = 0.1, eps: float = 8 / 255,
          train_steps: int = 3, seed: int = 0, log=print):
    """Train a CIFAR model.

    Args:
        x_train: uint8 [N,3,32,32]; y_train: long [N].
        mode: "vanilla" (standard training) or "pgd" (PGD adversarial training).
        train_steps: PGD iterations in the inner max (alpha = 2.5*eps/steps).
    Returns:
        (model, history) where history is a list of per-epoch dicts.
    """
    torch.manual_seed(seed)
    model = build_cifar_model(arch)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    steps_per_epoch = (len(x_train) + batch_size - 1) // batch_size
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * steps_per_epoch,
                                                pct_start=0.25)
    history = []
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(x_train))
        t0, tot, correct, loss_sum = time.time(), 0, 0, 0.0
        for k in range(steps_per_epoch):
            idx = perm[k * batch_size:(k + 1) * batch_size]
            xb = _augment(x_train[idx].float().div(255))
            yb = y_train[idx]
            if mode == "pgd":
                # pgd() switches the model to eval mode internally (BN uses running stats
                # while crafting) and restores train mode afterwards.
                xb = pgd(model, xb, yb, eps=eps, steps=train_steps, norm="linf")
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            loss_sum += loss.item() * len(yb)
            correct += (logits.argmax(1) == yb).sum().item()
            tot += len(yb)
        rec = {"epoch": ep + 1, "loss": loss_sum / tot, "train_acc": correct / tot, "sec": time.time() - t0}
        history.append(rec)
        log(f"[{mode}/{arch}] epoch {ep + 1}/{epochs} loss={rec['loss']:.3f} "
            f"train_acc={rec['train_acc']:.3f} ({rec['sec']:.0f}s)")
    return model.eval(), history


def save_checkpoint(model, path: str | Path, arch: str, meta: dict | None = None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"arch": arch, "state_dict": model.state_dict(), "meta": meta or {}}, path)
