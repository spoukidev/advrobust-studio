"""Model zoo and loaders.

Every model returned here takes **un-normalized** images in [0, 1] and applies
its own normalization as the first layer (``Normalize``). This keeps the
threat model honest: epsilon is measured in pixel space, exactly what an
attacker can change.

* ``cifar_cnn``     - compact VGG-style CNN (~0.3M params), CPU friendly.
* ``cifar_resnet``  - small ResNet-ish network (different architecture, used as
                      the transfer-attack *victim*).
* ``resnet18`` / ``efficientnet_b0`` - torchvision ImageNet-pretrained models.
* any ``.pt`` file  - a full pickled ``nn.Module`` or a state_dict for one of
                      the CIFAR architectures above.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class Normalize(nn.Module):
    """Per-channel (x - mean) / std as a differentiable layer."""

    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, -1, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std


class CifarCNN(nn.Module):
    """Compact VGG-style CNN: 3 conv blocks (32-64-128) + linear head."""

    def __init__(self, num_classes: int = 10, width: int = 32):
        super().__init__()
        w = width

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(block(3, w), block(w, 2 * w), block(2 * w, 4 * w))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4 * w, num_classes))

    def forward(self, x):
        return self.head(self.features(x))


class _BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.short = nn.Sequential()
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        out = F.relu(self.b1(self.c1(x)))
        out = self.b2(self.c2(out))
        return F.relu(out + self.short(x))


class CifarResNet(nn.Module):
    """Small ResNet (ResNet-8 style: 3 stages x 1 basic block, widths 16-32-64)."""

    def __init__(self, num_classes: int = 10, width: int = 16):
        super().__init__()
        w = width
        self.stem = nn.Sequential(nn.Conv2d(3, w, 3, 1, 1, bias=False), nn.BatchNorm2d(w), nn.ReLU(inplace=True))
        self.layers = nn.Sequential(_BasicBlock(w, w), _BasicBlock(w, 2 * w, 2), _BasicBlock(2 * w, 4 * w, 2))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4 * w, num_classes))

    def forward(self, x):
        return self.head(self.layers(self.stem(x)))


CIFAR_ARCHS = {"cifar_cnn": CifarCNN, "cifar_resnet": CifarResNet}


def build_cifar_model(arch: str = "cifar_cnn") -> nn.Module:
    """Fresh (untrained) CIFAR model wrapped with its normalization layer."""
    return nn.Sequential(Normalize(CIFAR_MEAN, CIFAR_STD), CIFAR_ARCHS[arch]())


def load_cifar_checkpoint(path: str | Path, arch: str | None = None, map_location="cpu") -> nn.Module:
    """Load a checkpoint saved by ``studio.defenses.adv_train.save_checkpoint``."""
    obj = torch.load(path, map_location=map_location, weights_only=False)
    if isinstance(obj, nn.Module):
        return obj.eval()
    if isinstance(obj, dict) and "state_dict" in obj:
        arch = arch or obj.get("arch", "cifar_cnn")
        model = build_cifar_model(arch)
        model.load_state_dict(obj["state_dict"])
        return model.eval()
    if isinstance(obj, dict):  # raw state dict
        model = build_cifar_model(arch or "cifar_cnn")
        model.load_state_dict(obj)
        return model.eval()
    raise ValueError(f"Unrecognised checkpoint format: {path}")


def load_torchvision(name: str = "resnet18", pretrained: bool = True) -> nn.Module:
    """ImageNet model from torchvision, wrapped with ImageNet normalization.

    If ``pretrained`` weights cannot be downloaded (offline box), falls back to
    random init and prints a warning -- the transfer demo then becomes a
    plumbing demo only, which is labelled in the report.
    """
    import torchvision.models as tvm

    ctor = {"resnet18": (tvm.resnet18, tvm.ResNet18_Weights),
            "efficientnet_b0": (tvm.efficientnet_b0, tvm.EfficientNet_B0_Weights),
            "mobilenet_v3_small": (tvm.mobilenet_v3_small, tvm.MobileNet_V3_Small_Weights)}[name]
    fn, weights = ctor
    try:
        net = fn(weights=weights.DEFAULT if pretrained else None)
    except Exception as e:  # pragma: no cover - network dependent
        print(f"[models] could not fetch {name} weights ({e}); using random init")
        net = fn(weights=None)
    return nn.Sequential(Normalize(IMAGENET_MEAN, IMAGENET_STD), net).eval()


def imagenet_labels() -> list[str]:
    import torchvision.models as tvm

    return list(tvm.ResNet18_Weights.DEFAULT.meta["categories"])


def load_model(spec: str) -> nn.Module:
    """Generic entry point: a torchvision name or a path to a .pt checkpoint."""
    if spec in ("resnet18", "efficientnet_b0", "mobilenet_v3_small"):
        return load_torchvision(spec)
    if spec in CIFAR_ARCHS:
        return build_cifar_model(spec).eval()
    return load_cifar_checkpoint(spec)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
