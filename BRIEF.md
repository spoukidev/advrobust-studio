# AdvRobust Studio: An Adversarial Attack and Defense Workbench for Image Classifiers

**Slug:** `advrobust-studio`  
**Level:** Intermediate  
**Estimated effort:** 2–3 weeks part time  
**Why it ships well on GitHub:** Visual before/after grids, clean PyTorch code, and a one-command benchmark make this instantly readable to recruiters.

## 1. Goal

Build an interactive research prototype that lets a user load a pretrained image classifier (ResNet, EfficientNet, or a custom checkpoint), generate adversarial examples with standard white-box attacks, compare two or three defenses, and export a robustness report.

The app should answer, visually and with numbers: *How easily can this model be fooled, and which defense actually helps?*

## 2. Threat model

| Item | Description |
|---|---|
| **Asset** | Integrity of image-classifier predictions (e.g. content moderation, medical or document classifiers in a demo setting) |
| **Attacker** | White-box: knows architecture and weights; can craft perturbations under an L∞ or L2 budget |
| **Attacker goals** | Untargeted misclassification, or targeted (force class *t*) while keeping the image looking clean |
| **Constraints** | Perturbation budget ε (e.g. 8/255 L∞); optional query limit for a black-box transfer suite |
| **Out of scope** | Physical-world attacks (printed stickers), poisoning the training set, and attacking hosted APIs you do not own |

## 3. Tech stack

- Python 3.11+, **PyTorch**, torchvision
- Attacks: `torchattacks` and/or a small hand-written FGSM / PGD / C&W module so the math is visible in-repo
- Datasets: CIFAR-10 and a ImageNet-pretrained demo on a handful of local images (no huge downloads required)
- App: Streamlit or Gradio
- Metrics and plots: numpy, matplotlib / plotly
- Tests: pytest (attack budget respected, defense does not crash, report schema)

## 4. Key components / structure

```
advrobust-studio/
├── BRIEF.md
├── README.md                 # hero: clean vs adversarial grid + robust accuracy table
├── pyproject.toml
├── Makefile
├── config/default.yaml
├── studio/
│   ├── models.py             # load torchvision or a local .pt
│   ├── attacks/
│   │   ├── fgsm.py
│   │   ├── pgd.py
│   │   ├── cw.py             # optional
│   │   └── transfer.py       # black-box: attack a surrogate, evaluate on victim
│   ├── defenses/
│   │   ├── jpeg_denoise.py   # input preprocessing
│   │   ├── bit_depth.py
│   │   └── adv_train.py      # short PGD adversarial training on CIFAR-10
│   ├── eval.py               # clean acc, robust acc @ ε, ASR, mean L∞/L2
│   ├── visualize.py          # side-by-side + perturbation heatmaps
│   └── report.py             # Markdown / HTML export
├── app/
│   └── ui.py                 # upload image → attack slider → defense toggle
├── experiments/
│   └── run_benchmark.py      # `make bench` → results/report.md
├── tests/
└── results/sample/           # committed sample report + images
```

## 5. Key features

1. **White-box attacks:** FGSM, multi-step PGD (L∞ and L2), optional C&W.
2. **Targeted and untargeted** modes with a live perturbation budget slider.
3. **Three defenses:** JPEG / bit-depth squeezing (preprocessing), randomized smoothing (lightweight), and a short adversarially trained CIFAR-10 checkpoint compared to a vanilla one.
4. **Transfer / black-box demo:** craft on ResNet-18, evaluate on EfficientNet-B0 to show transferability.
5. **Visualizations:** original | adversarial | amplified noise; confidence bars before and after.
6. **Benchmark command** that writes robust accuracy vs ε curves.

## 6. Success criteria

- `make bench` produces a report with clean vs robust accuracy for at least two models and two attacks.
- UI can fool a pretrained CIFAR-10 model at ε = 8/255 and display the pair of images.
- Adversarially trained model shows a clearly higher robust accuracy than the vanilla baseline in the sample report.
- pytest covers perturbation-budget checks and report generation.
- README leads with a figure and the robustness table; LICENSE and sample results included.

## 7. Why it's good for a skills profile

Shows real PyTorch (custom attack loops, training, eval), not just API wrapping. The visual demo is screenshot-friendly. Recruiters in ML *and* security immediately understand “I attacked this model and measured whether the defense worked.”
