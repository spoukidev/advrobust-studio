"""Feature squeezing by colour bit-depth reduction (Xu et al., 2018).

Quantising each channel to ``bits`` bits removes low-amplitude perturbations.
The forward pass is non-differentiable; we use a straight-through estimator
(BPDA, Athalye et al., 2018) in backward so an *adaptive* attacker can still
compute useful gradients -- this is how the defense should be evaluated.
"""
import torch
import torch.nn as nn


def bit_depth_reduce(x: torch.Tensor, bits: int = 4) -> torch.Tensor:
    levels = 2 ** bits - 1
    return torch.round(x * levels) / levels


class _STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, bits):
        return bit_depth_reduce(x, bits)

    @staticmethod
    def backward(ctx, g):
        return g, None  # identity gradient (BPDA)


class BitDepthReduce(nn.Module):
    def __init__(self, bits: int = 4):
        super().__init__()
        self.bits = bits

    def forward(self, x):
        return _STE.apply(x, self.bits)
