"""Train the CIFAR-10 checkpoints used by the benchmark and UI.

    python experiments/train_models.py --config config/default.yaml

Produces (by default):
    checkpoints/cifar_cnn_vanilla.pt   - standard training
    checkpoints/cifar_cnn_pgdat.pt     - PGD-AT, eps=8/255
    checkpoints/cifar_resnet_vanilla.pt - second architecture (transfer victim)
Existing checkpoints are skipped unless --force.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from studio.data import load_cifar10  # noqa: E402
from studio.defenses.adv_train import save_checkpoint, train  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", default=None, help="train only this checkpoint name")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    torch.set_num_threads(cfg.get("threads", 8))
    x, y = load_cifar10(cfg["data_dir"], train=True)
    for name, spec in cfg["training"]["checkpoints"].items():
        if args.only and name != args.only:
            continue
        path = Path(cfg["checkpoint_dir"]) / f"{name}.pt"
        if path.exists() and not args.force:
            print(f"skip {path} (exists)")
            continue
        t0 = time.time()
        model, hist = train(x, y, arch=spec["arch"], mode=spec["mode"], epochs=spec["epochs"],
                            batch_size=cfg["training"]["batch_size"], lr=cfg["training"]["lr"],
                            eps=eval(str(cfg["training"]["eps"])), train_steps=cfg["training"]["pgd_steps"],
                            seed=cfg["seed"], log=lambda *a: print(*a, flush=True))
        meta = {"mode": spec["mode"], "epochs": spec["epochs"], "history": hist,
                "train_seconds": round(time.time() - t0, 1)}
        save_checkpoint(model, path, spec["arch"], meta)
        print(f"saved {path} in {meta['train_seconds']}s")
        print(json.dumps(hist[-1]))


if __name__ == "__main__":
    main()
