"""Plots: original | adversarial | amplified noise grids, confidence bars and
robust-accuracy-vs-epsilon curves. Matplotlib only (Agg backend), so it works
headless and inside Streamlit.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402


def to_img(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()


def amplify_noise(delta: torch.Tensor) -> np.ndarray:
    """Map a perturbation to [0,1] for display: 0.5 + delta / (2 * max|delta|)."""
    d = delta.detach().cpu()
    m = d.abs().max().clamp_min(1e-12)
    return (0.5 + d / (2 * m)).permute(1, 2, 0).numpy()


def perturbation_heatmap(delta: torch.Tensor) -> np.ndarray:
    return delta.detach().cpu().abs().max(0).values.numpy()


def triplet_figure(x, x_adv, clean_label: str, adv_label: str, clean_conf: float, adv_conf: float,
                   title: str | None = None):
    """Single example: original | adversarial | amplified noise | |delta| heatmap."""
    fig, ax = plt.subplots(1, 4, figsize=(11, 3))
    delta = x_adv - x
    ax[0].imshow(to_img(x)); ax[0].set_title(f"original\n{clean_label} ({clean_conf:.0%})")
    ax[1].imshow(to_img(x_adv)); ax[1].set_title(f"adversarial\n{adv_label} ({adv_conf:.0%})", color="crimson")
    ax[2].imshow(amplify_noise(delta)); ax[2].set_title(f"noise (amplified)\nL∞={delta.abs().max():.4f}")
    im = ax[3].imshow(perturbation_heatmap(delta), cmap="magma"); ax[3].set_title("|δ| heatmap")
    fig.colorbar(im, ax=ax[3], fraction=0.046)
    for a in ax:
        a.axis("off")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def grid_figure(x, x_adv, clean_pred, adv_pred, labels, y=None, n: int = 8, title: str | None = None):
    """Hero figure: rows = original / adversarial / amplified noise, columns = examples."""
    n = min(n, len(x))
    fig, ax = plt.subplots(3, n, figsize=(1.6 * n, 5.4))
    for i in range(n):
        ok = y is None or clean_pred[i] == y[i]
        ax[0, i].imshow(to_img(x[i])); ax[0, i].set_title(labels[clean_pred[i]], fontsize=9,
                                                          color="black" if ok else "gray")
        fooled = adv_pred[i] != clean_pred[i]
        ax[1, i].imshow(to_img(x_adv[i])); ax[1, i].set_title(labels[adv_pred[i]], fontsize=9,
                                                              color="crimson" if fooled else "green")
        ax[2, i].imshow(amplify_noise(x_adv[i] - x[i]))
        for r in range(3):
            ax[r, i].set_xticks([]); ax[r, i].set_yticks([])
    ax[0, 0].set_ylabel("clean", fontsize=10)
    ax[1, 0].set_ylabel("adversarial", fontsize=10)
    ax[2, 0].set_ylabel("noise ×amp", fontsize=10)
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig


def confidence_bars(p_clean: torch.Tensor, p_adv: torch.Tensor, labels, top_k: int = 5):
    """Side-by-side top-k softmax bars before / after the attack."""
    fig, ax = plt.subplots(1, 2, figsize=(10, 3))
    for a, p, name, col in ((ax[0], p_clean, "clean", "tab:blue"), (ax[1], p_adv, "adversarial", "tab:red")):
        v, i = p.topk(min(top_k, p.numel()))
        a.barh([labels[j] for j in i.tolist()][::-1], v.tolist()[::-1], color=col)
        a.set_xlim(0, 1); a.set_title(f"{name}: top-{len(v)} confidence")
    fig.tight_layout()
    return fig


def eps_curve_figure(curves: dict[str, list[dict]], title="Robust accuracy vs ε (PGD-L∞)"):
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, rows in curves.items():
        eps = [r["eps"] * 255 for r in rows]
        ax.plot(eps, [r["robust_acc"] * 100 for r in rows], marker="o", label=name)
    ax.set_xlabel("ε (×1/255, L∞)"); ax.set_ylabel("accuracy (%)"); ax.set_title(title)
    ax.grid(alpha=0.3); ax.legend(); ax.set_ylim(0, 100)
    fig.tight_layout()
    return fig


def save(fig, path: str | Path, dpi: int = 130):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path
