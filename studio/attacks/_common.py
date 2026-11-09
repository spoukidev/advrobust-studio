import torch
import torch.nn.functional as F


def attack_loss(logits: torch.Tensor, y: torch.Tensor, targeted: bool) -> torch.Tensor:
    """Loss the attacker *ascends*.

    Untargeted: maximise cross-entropy w.r.t. the true label ``y``.
    Targeted:   minimise cross-entropy w.r.t. the target label, i.e. ascend
                its negative.
    """
    ce = F.cross_entropy(logits, y, reduction="sum")
    return -ce if targeted else ce


class eval_mode:
    """Context manager: put the model in eval mode, restore afterwards."""

    def __init__(self, model):
        self.model, self.was_training = model, model.training

    def __enter__(self):
        self.model.eval()
        return self.model

    def __exit__(self, *exc):
        self.model.train(self.was_training)
