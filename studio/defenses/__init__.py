"""Defenses: input preprocessing (JPEG, bit-depth), randomized smoothing, and
adversarial training."""
from .bit_depth import BitDepthReduce, bit_depth_reduce
from .jpeg_denoise import JpegCompress, jpeg_compress
from .smoothing import RandomizedSmoothing


def wrap_with_defense(model, name: str | None, **kw):
    """Return ``model`` wrapped with a preprocessing/inference-time defense."""
    import torch.nn as nn

    if name in (None, "none"):
        return model
    if name == "jpeg":
        return nn.Sequential(JpegCompress(kw.get("quality", 75)), model).eval()
    if name == "bit_depth":
        return nn.Sequential(BitDepthReduce(kw.get("bits", 4)), model).eval()
    if name == "smoothing":
        return RandomizedSmoothing(model, sigma=kw.get("sigma", 0.12), n=kw.get("n", 32)).eval()
    raise ValueError(f"unknown defense {name}")


__all__ = ["BitDepthReduce", "bit_depth_reduce", "JpegCompress", "jpeg_compress",
           "RandomizedSmoothing", "wrap_with_defense"]
