"""AdvRobust Studio — Streamlit UI.

    streamlit run app/ui.py

Upload an image (or pick a CIFAR-10 test image) → choose model, attack and ε
with a slider → toggle a defense → see original | adversarial | noise and the
confidence bars before/after.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from studio import visualize as viz  # noqa: E402
from studio.attacks import ATTACKS  # noqa: E402
from studio.data import CIFAR10_CLASSES, load_cifar10, to_float  # noqa: E402
from studio.defenses import wrap_with_defense  # noqa: E402
from studio.models import imagenet_labels, load_cifar_checkpoint, load_torchvision  # noqa: E402

st.set_page_config(page_title="AdvRobust Studio", layout="wide")
CK = ROOT / "checkpoints"


@st.cache_resource
def get_model(name: str):
    if name in ("resnet18", "efficientnet_b0"):
        return load_torchvision(name), imagenet_labels(), 224
    return load_cifar_checkpoint(CK / f"{name}.pt"), CIFAR10_CLASSES, 32


@st.cache_data
def cifar_samples(n=50):
    x, y = load_cifar10(ROOT / "data", train=False, limit=n)
    return to_float(x), y


def to_tensor(img: Image.Image, size: int) -> torch.Tensor:
    img = img.convert("RGB")
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2)).resize((size, size), Image.BICUBIC)
    return torch.from_numpy(np.asarray(img, dtype=np.float32) / 255).permute(2, 0, 1)


st.title("🛡️ AdvRobust Studio")
st.caption("White-box adversarial attacks vs. defenses on image classifiers you own. Educational / research use.")

models = [p.stem for p in sorted(CK.glob("*.pt"))] + ["resnet18", "efficientnet_b0"]
with st.sidebar:
    model_name = st.selectbox("Model", models, index=0)
    attack = st.selectbox("Attack", list(ATTACKS), index=1)
    is_l2 = attack in ("pgd_l2", "cw_l2")
    if is_l2:
        eps = st.slider("ε (L2)", 0.0, 3.0, 0.5, 0.05)
    else:
        eps = st.slider("ε (L∞, ×1/255)", 0, 32, 8) / 255
    steps = st.slider("steps (PGD / C&W)", 1, 100, 20)
    targeted = st.checkbox("Targeted")
    defense = st.radio("Defense (inference-time)", ["none", "jpeg", "bit_depth", "smoothing"], horizontal=False)
    adaptive = st.checkbox("Adaptive attacker (differentiate through defense)", value=True)

model, labels, size = get_model(model_name)
col_in, _ = st.columns([1, 2])
with col_in:
    up = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if up is not None:
        x = to_tensor(Image.open(up), size)
    elif size == 32:
        xs, ys = cifar_samples()
        idx = st.number_input("…or CIFAR-10 test image #", 0, len(xs) - 1, 0)
        x = xs[int(idx)]
        st.write(f"ground truth: **{CIFAR10_CLASSES[ys[int(idx)]]}**")
    else:
        demo = sorted((ROOT / "assets/imagenet_demo").glob("*.jpg"))
        pick = st.selectbox("…or demo photo", [p.name for p in demo])
        x = to_tensor(Image.open(ROOT / "assets/imagenet_demo" / pick), size)

x = x.unsqueeze(0)
defended = wrap_with_defense(model, defense)
with torch.no_grad():
    p_clean = defended(x).softmax(1)[0]
y_clean = p_clean.argmax().view(1)
if targeted:
    target = st.selectbox("Target class", labels, index=(int(y_clean) + 1) % len(labels))
    y_att = torch.tensor([labels.index(target)])
else:
    y_att = y_clean

if True:  # re-runs automatically whenever a widget changes
    attack_model = defended if adaptive else model
    kw = {"eps": eps, "targeted": targeted, "steps": steps}
    with st.spinner("crafting…"):
        x_adv = x if eps == 0 else ATTACKS[attack](attack_model, x, y_att, **kw)
    with torch.no_grad():
        p_adv = defended(x_adv).softmax(1)[0]
    c, a = int(p_clean.argmax()), int(p_adv.argmax())
    fooled = (a == int(y_att)) if targeted else (a != c)
    delta = (x_adv - x)[0]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("clean prediction", labels[c], f"{p_clean[c]:.1%}")
    m2.metric("adversarial prediction", labels[a], f"{p_adv[a]:.1%}", delta_color="inverse")
    m3.metric("L∞ / L2", f"{delta.abs().max() * 255:.1f}/255", f"L2 {delta.norm():.3f}", delta_color="off")
    m4.metric("result", "FOOLED ❌" if fooled else "robust ✅")
    # the pair of images (CIFAR thumbnails upscaled 8x with nearest-neighbour for viewing)
    pair = [viz.to_img(t[0]) for t in (x, x_adv)]
    if size == 32:
        pair = [np.kron(p, np.ones((8, 8, 1))) for p in pair]
    st.image(pair, caption=[f"original: {labels[c]}", f"adversarial: {labels[a]}"], width=256, clamp=True)
    st.pyplot(viz.triplet_figure(x[0], x_adv[0], labels[c], labels[a], p_clean[c].item(), p_adv[a].item()))
    st.pyplot(viz.confidence_bars(p_clean, p_adv, labels))
    st.caption("Amplified noise is rescaled to the full display range — the actual perturbation is ≤ ε.")
