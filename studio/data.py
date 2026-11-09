"""CIFAR-10 loading.

Primary source is torchvision's downloader. Some networks block the original
Toronto host, so we fall back to the Hugging Face parquet mirror
(``uoft-cs/cifar10``). Either way the data is cached once as uint8 tensors in
``data/cifar10_{split}.pt`` for fast reloads.
"""
from __future__ import annotations

import io
import os
import urllib.request
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

HF_URL = "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/{split}-00000-of-00001.parquet"


def _from_torchvision(root: Path, train: bool):
    import torchvision

    ds = torchvision.datasets.CIFAR10(str(root), train=train, download=True)
    x = torch.from_numpy(ds.data).permute(0, 3, 1, 2).contiguous()  # N,3,32,32 uint8
    y = torch.tensor(ds.targets, dtype=torch.long)
    return x, y


def _from_hf_parquet(root: Path, split: str):
    import pyarrow.parquet as pq
    from PIL import Image

    pq_dir = root / "cifar10_hf"
    pq_dir.mkdir(parents=True, exist_ok=True)
    path = pq_dir / f"{split}.parquet"
    if not path.exists():
        urllib.request.urlretrieve(HF_URL.format(split=split), path)
    table = pq.read_table(path).to_pydict()
    imgs = [np.array(Image.open(io.BytesIO(d["bytes"])).convert("RGB")) for d in table["img"]]
    x = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).contiguous()
    y = torch.tensor(table["label"], dtype=torch.long)
    return x, y


def load_cifar10(root: str | os.PathLike = "data", train: bool = False, limit: int | None = None):
    """Return (x uint8 [N,3,32,32], y long [N]) for the requested split."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    split = "train" if train else "test"
    cache = root / f"cifar10_{split}.pt"
    if cache.exists():
        x, y = torch.load(cache)
    else:
        try:
            if (root / "cifar10_hf" / f"{split}.parquet").exists():
                raise RuntimeError("parquet mirror already present")
            x, y = _from_torchvision(root, train)
        except Exception:
            x, y = _from_hf_parquet(root, split)
        torch.save((x, y), cache)
    if limit is not None:
        x, y = x[:limit], y[:limit]
    return x, y


def to_float(x_uint8: torch.Tensor) -> torch.Tensor:
    return x_uint8.float().div_(255.0)


def cifar10_tensor_dataset(root="data", train=False, limit=None) -> TensorDataset:
    x, y = load_cifar10(root, train, limit)
    return TensorDataset(to_float(x), y)
