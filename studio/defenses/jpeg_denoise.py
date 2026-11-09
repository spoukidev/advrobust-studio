"""JPEG compression as an input-denoising defense (Dziugaite et al., 2016;
Guo et al., 2018).

Each image is round-tripped through an in-memory JPEG encoder (PIL). JPEG's
quantisation of high-frequency DCT coefficients tends to wipe out small
adversarial noise. Backward uses a straight-through (BPDA) identity gradient
so adaptive attacks remain possible.
"""
import io

import numpy as np
import torch
import torch.nn as nn
from PIL import Image


def jpeg_compress(x: torch.Tensor, quality: int = 75) -> torch.Tensor:
    out = []
    for img in x.detach().cpu():
        arr = (img.clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        rec = np.asarray(Image.open(buf).convert("RGB"), dtype=np.float32) / 255.0
        out.append(torch.from_numpy(rec).permute(2, 0, 1))
    return torch.stack(out).to(x.device, x.dtype)


class _JpegSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, quality):
        return jpeg_compress(x, quality)

    @staticmethod
    def backward(ctx, g):
        return g, None


class JpegCompress(nn.Module):
    def __init__(self, quality: int = 75):
        super().__init__()
        self.quality = quality

    def forward(self, x):
        return _JpegSTE.apply(x, self.quality)
