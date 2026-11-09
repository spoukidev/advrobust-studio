import torch

from studio.defenses import bit_depth_reduce, jpeg_compress, wrap_with_defense
from studio.attacks import pgd_linf


def test_bit_depth_levels():
    x = torch.rand(2, 3, 8, 8)
    q = bit_depth_reduce(x, 2)
    assert set((q * 3).round().unique().tolist()) <= {0.0, 1.0, 2.0, 3.0}


def test_jpeg_roundtrip_close():
    x = torch.rand(2, 3, 32, 32) * 0.1 + 0.45
    out = jpeg_compress(x, quality=95)
    assert out.shape == x.shape and (out - x).abs().mean() < 0.05


def test_defenses_forward_and_backward(model, batch):
    x, y = batch
    for name in ("none", "jpeg", "bit_depth", "smoothing"):
        d = wrap_with_defense(model, name, n=4)
        assert d(x).shape == (len(x), 10)
        # adaptive attack (BPDA / EOT) must run and respect the budget
        x_adv = pgd_linf(d, x, y, eps=8 / 255, steps=2)
        assert (x_adv - x).abs().max() <= 8 / 255 + 1e-6
