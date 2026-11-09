# BUILT — AdvRobust Studio

**Status:** complete, ready to publish (suitable for a public skills-profile repo).
**Built:** 2026-11-09 (Africa/Algiers, UTC+1) · CPU only · Python 3.12, torch 2.15.0+cpu, torchvision 0.29.2, streamlit 1.67.0
**Location:** `/workspace/projects/advrobust-studio/` · local git repo, branch `main`, not pushed anywhere.

## Measured results (`make bench`, 500 CIFAR-10 test images, seed 0)

Robust accuracy (clean acc in first column):

| Model | Clean | FGSM 8/255 | PGD-L∞ 8/255 (10 steps) | PGD-L2 0.5 | C&W-L2 0.5 (n=200) |
|---|---|---|---|---|---|
| cifar_cnn_vanilla | 86.8% | 2.2% | 0.0% | 0.4% | 1.5% |
| cifar_resnet_vanilla | 81.0% | 2.2% | 0.0% | 0.0% | 1.0% |
| cifar_cnn_pgdat | 60.4% | 39.2% | 37.4% | 48.2% | 49.0% |

- **AT vs vanilla gap (PGD-L∞ 8/255): 0.0% → 37.4% = +37.4 pp.** Clean accuracy cost 86.8% → 60.4%.
- ε-curve (PGD-L∞, n=250): PGD-AT keeps 48.0% at 4/255 and 13.6% at 16/255; both vanilla models hit 0% by 4/255.
- Defenses (adaptive BPDA/EOT, n=200), vanilla CNN: JPEG 1.0%, bit-depth 0.0%, smoothing 0.0% (smoothing also drops clean acc to 48%). Unaware-attacker JPEG/smoothing: 6.5%.
- Transfer (PGD-L∞ 8/255): CNN→ResNet 81.0%→9.6%; ResNet→CNN 86.8%→14.0%; vanilla CNN→PGD-AT CNN 60.4%→59.8%. ImageNet ResNet-18 → EfficientNet-B0 (ε 4/255): prediction flipped on 3/5 demo photos.
- Benchmark runtime ≈ 2 min on 8 CPU cores.

Full report: `results/sample/report.md` (+ `report.html`, `results.json`, `figures/`).

## Tests

`make test` → **37 passed** (≈5 s). Covers L∞/L2 budget checks for every attack, box constraints, targeted mode, defense wrappers not crashing, report schema/validation, model loading, and a headless Streamlit test that fools the vanilla CNN at ε = 8/255. `streamlit run app/ui.py` also boots and answers `/_stcore/health`.

## How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu && pip install -e .
make test && make bench && make ui
```

## Checkpoints

`checkpoints/cifar_cnn_vanilla.pt` (1.2 MB, 10 ep, train acc 0.887), `cifar_resnet_vanilla.pt` (0.3 MB, 10 ep, 0.794), `cifar_cnn_pgdat.pt` (1.2 MB, PGD-AT ε=8/255, 3-step, 10 ep, adversarial train acc 0.357, 2233 s). All committed.

## Deviations from the brief

- Hand-written attacks only; `torchattacks` is an optional extra, not used.
- Small custom CIFAR models (VGG-style CNN ~289K params, ResNet-8-style ~78K) instead of ResNet-18 on CIFAR, so everything trains on CPU in under an hour total.
- Streamlit chosen (brief allowed Streamlit or Gradio). No plotly; matplotlib only.
- C&W is a simplified fixed-c variant (no binary search), projected into the L2 ε-ball.
- Randomized smoothing is prediction-only (no certified radius).
- Added beyond brief: adaptive (BPDA/EOT) defense evaluation, budget audit in the report, HTML + JSON reports.

## Limitations

- Low absolute accuracies (short CPU training); robust numbers use 10-step PGD with one random start, not AutoAttack, so PGD-AT robustness is likely somewhat overestimated.
- Single seed and test subsets (500 / 250 / 200 images) → a few points of sampling noise.
- ImageNet demo has 5 photos without ground truth (agreement with each model's own clean prediction).
- Provenance/licence of the 5 demo photos in `assets/imagenet_demo/` (astronaut, grace_hopper, three dogs; they look like the standard public-domain ML sample images) was not independently verified — worth a check before publishing.
- CIFAR-10 and torchvision ImageNet weights are downloaded at runtime (not redistributed).
