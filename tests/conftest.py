import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from studio.models import build_cifar_model  # noqa: E402


@pytest.fixture(scope="session")
def model():
    """Small CIFAR model with a few quick SGD steps on random data (no dataset needed)."""
    torch.manual_seed(0)
    m = build_cifar_model("cifar_cnn")
    opt = torch.optim.SGD(m.parameters(), lr=0.05)
    x, y = torch.rand(64, 3, 32, 32), torch.randint(0, 10, (64,))
    for _ in range(5):
        loss = torch.nn.functional.cross_entropy(m(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
    return m.eval()


@pytest.fixture
def batch():
    torch.manual_seed(1)
    return torch.rand(16, 3, 32, 32), torch.randint(0, 10, (16,))
