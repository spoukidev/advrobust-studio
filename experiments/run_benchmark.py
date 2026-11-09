"""One-command robustness benchmark (``make bench``).

Steps
  1. load the trained CIFAR-10 checkpoints (train them first with ``make train``)
  2. white-box FGSM / PGD-L∞ / PGD-L2 / C&W-L2 on every model
  3. robust-accuracy-vs-ε curves (PGD-L∞)
  4. preprocessing + randomized-smoothing defenses, defense-unaware and adaptive
  5. black-box transfer (CIFAR CNN <-> ResNet, ImageNet ResNet-18 -> EfficientNet-B0)
  6. figures + results/report.md, report.html, results.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from studio import visualize as viz  # noqa: E402
from studio.attacks import ATTACKS, pgd_linf, transfer_attack  # noqa: E402
from studio.data import CIFAR10_CLASSES, load_cifar10, to_float  # noqa: E402
from studio.defenses import wrap_with_defense  # noqa: E402
from studio.eval import evaluate_attack, eps_curve, predict  # noqa: E402
from studio.models import count_params, imagenet_labels, load_cifar_checkpoint, load_torchvision  # noqa: E402
from studio.report import write_report  # noqa: E402


def frac(v) -> float:
    """Parse '8/255'-style strings from YAML."""
    if isinstance(v, str) and "/" in v:
        a, b = v.split("/")
        return float(a) / float(b)
    return float(v)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def imagenet_demo(cfg, out_dir: Path, figures: dict):
    """Transfer demo on a handful of local photos (pseudo-labels = surrogate's clean top-1)."""
    from PIL import Image
    import torchvision.transforms as T

    paths = sorted(Path(ROOT / cfg["images"]).glob("*.jpg"))
    if not paths:
        return None
    tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
    x = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths])
    sur, vic = load_torchvision(cfg["surrogate"]), load_torchvision(cfg["victim"])
    labels = imagenet_labels()
    y_sur = predict(sur, x, 8)
    y_vic = predict(vic, x, 8)
    eps = frac(cfg["eps"])
    x_adv = pgd_linf(sur, x, y_sur, eps=eps, steps=cfg["steps"])
    s_adv, v_adv = predict(sur, x_adv, 8), predict(vic, x_adv, 8)
    fig = viz.grid_figure(x, x_adv, y_vic.tolist(), v_adv.tolist(), labels, n=len(x),
                          title=f"ImageNet transfer: PGD-L∞ ε={cfg['eps']} crafted on {cfg['surrogate']}, "
                                f"shown = {cfg['victim']} predictions")
    figures["imagenet_transfer"] = str(viz.save(fig, out_dir / "figures" / "imagenet_transfer.png").relative_to(out_dir))
    return {
        "surrogate": cfg["surrogate"], "victim": cfg["victim"], "attack": f"pgd_linf ({cfg['steps']} steps)",
        "eps_str": str(cfg["eps"]), "n": len(x),
        # accuracies here are agreement with each model's own clean top-1 (no ground truth for local photos)
        "surrogate_adv_acc": (s_adv == y_sur).float().mean().item(),
        "victim_clean_acc": 1.0,
        "victim_adv_acc": (v_adv == y_vic).float().mean().item(),
        "transfer_rate": (v_adv != y_vic).float().mean().item(),
        "per_image": [{"file": p.name, "victim_clean": labels[a], "victim_adv": labels[b],
                       "surrogate_clean": labels[c], "surrogate_adv": labels[d]}
                      for p, a, b, c, d in zip(paths, y_vic.tolist(), v_adv.tolist(), y_sur.tolist(), s_adv.tolist())],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config/default.yaml"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--quick", action="store_true", help="tiny n for smoke tests")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    b = cfg["benchmark"]
    torch.manual_seed(cfg["seed"])
    torch.set_num_threads(cfg.get("threads", 8))
    out_dir = Path(args.out or ROOT / cfg["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 64 if args.quick else b["n_eval"]
    n_def = 32 if args.quick else b["defense_n_eval"]
    n_cw = 32 if args.quick else b["cw"]["n_eval"]
    eps_inf, eps_l2 = frac(b["eps_linf"]), frac(b["eps_l2"])
    steps = 5 if args.quick else b["pgd_steps"]

    xt, yt = load_cifar10(ROOT / cfg["data_dir"], train=False, limit=n)
    x, y = to_float(xt), yt

    ckdir = ROOT / cfg["checkpoint_dir"]
    models, info = {}, {}
    for name in cfg["training"]["checkpoints"]:
        p = ckdir / f"{name}.pt"
        if not p.exists():
            sys.exit(f"missing checkpoint {p}; run `make train` first")
        ck = torch.load(p, map_location="cpu", weights_only=False)
        models[name] = load_cifar_checkpoint(p)
        meta = ck.get("meta", {})
        info[name] = {"arch": ck["arch"], "mode": meta.get("mode"), "epochs": meta.get("epochs"),
                      "params": count_params(models[name]), "train_seconds": meta.get("train_seconds")}

    results = {"meta": {}, "models": info, "attacks": [], "defenses": [], "eps_curves": {}, "transfer": [],
               "figures": {}}

    # 2. white-box attacks
    attack_specs = [("fgsm", eps_inf, {}), ("pgd_linf", eps_inf, {"steps": steps}),
                    ("pgd_l2", eps_l2, {"steps": steps}),
                    ("cw_l2", eps_l2, {"steps": 10 if args.quick else b["cw"]["steps"], "c": b["cw"]["c"],
                                       "lr": b["cw"]["lr"]})]
    for name, model in models.items():
        for atk, eps, kw in attack_specs:
            xs, ys = (x[:n_cw], y[:n_cw]) if atk == "cw_l2" else (x, y)
            r = evaluate_attack(model, xs, ys, atk, eps, model_name=name, batch_size=b["batch_size"], **kw)
            results["attacks"].append(r.to_dict())
            log(f"{name:22s} {atk:9s} clean={r.clean_acc:.3f} robust={r.robust_acc:.3f} asr={r.asr:.3f} "
                f"maxLinf={r.max_linf:.4f} maxL2={r.max_l2:.3f} ({r.seconds}s)")

    # AT vs vanilla gap
    def row(model, atk):
        return next(r for r in results["attacks"] if r["model"] == model and r["attack"] == atk)
    v, a = row("cifar_cnn_vanilla", "pgd_linf"), row("cifar_cnn_pgdat", "pgd_linf")
    results["at_gap"] = {"attack": f"PGD-L∞ ({steps} steps)", "eps_str": b["eps_linf"], "vanilla": v["robust_acc"],
                         "pgdat": a["robust_acc"], "gap": a["robust_acc"] - v["robust_acc"],
                         "vanilla_clean": v["clean_acc"], "pgdat_clean": a["clean_acc"]}

    # 3. eps curves
    eps_list = [frac(e) for e in b["eps_curve"]]
    for name, model in models.items():
        nc = 32 if args.quick else b.get("curve_n_eval", n)
        results["eps_curves"][name] = eps_curve(model, x[:nc], y[:nc], eps_list, "pgd_linf", model_name=name,
                                                steps=5 if args.quick else b["curve_pgd_steps"])
        log(f"curve {name}: " + " ".join(f"{r['robust_acc']:.2f}" for r in results["eps_curves"][name]))

    # 4. defenses (PGD-L∞ @ eps) on vanilla and AT CNN
    xd, yd = x[:n_def], y[:n_def]
    for base_name in ("cifar_cnn_vanilla", "cifar_cnn_pgdat"):
        base = models[base_name]
        for dname, dkw in b["defenses"].items():
            defended = wrap_with_defense(base, dname, **dkw)
            for setting, attack_model in (("unaware", base), ("adaptive", defended)):
                torch.manual_seed(cfg["seed"])
                r = evaluate_attack(attack_model, xd, yd, "pgd_linf", eps_inf, model_name=base_name,
                                    eval_model=defended, defense=dname, batch_size=b["batch_size"], steps=steps)
                d = r.to_dict(); d["setting"] = setting
                results["defenses"].append(d)
                log(f"defense {base_name} {dname:9s} {setting:8s} clean={r.clean_acc:.3f} robust={r.robust_acc:.3f}")
        r = evaluate_attack(base, xd, yd, "pgd_linf", eps_inf, model_name=base_name, steps=steps)
        d = r.to_dict(); d["setting"] = "white-box"; d["defense"] = "none"
        results["defenses"].insert(len(results["defenses"]) - 2 * len(b["defenses"]), d)

    # 5. transfer
    pairs = [("cifar_cnn_vanilla", "cifar_resnet_vanilla"), ("cifar_resnet_vanilla", "cifar_cnn_vanilla"),
             ("cifar_cnn_vanilla", "cifar_cnn_pgdat")]
    for s, vname in pairs:
        t = transfer_attack(models[s], models[vname], x, y, eps=eps_inf, steps=steps, norm="linf")
        t.pop("x_adv")
        results["transfer"].append({"surrogate": s, "victim": vname, "attack": "pgd_linf", "eps_str": b["eps_linf"],
                                    "n": n, **t})
        log(f"transfer {s} -> {vname}: victim {t['victim_clean_acc']:.3f} -> {t['victim_adv_acc']:.3f}")
    if not args.quick and b.get("imagenet_demo"):
        try:
            demo = imagenet_demo(b["imagenet_demo"], out_dir, results["figures"])
            if demo:
                results["transfer"].append(demo)
                results["imagenet_demo"] = demo["per_image"]
                log(f"imagenet transfer: rate={demo['transfer_rate']:.2f}")
        except Exception as e:  # network / weights unavailable
            log(f"imagenet demo skipped: {e}")

    # 6. figures
    vm = models["cifar_cnn_vanilla"]
    ok = (predict(vm, x[:100]) == y[:100]).nonzero().flatten()[:10]  # hero: correctly-classified images
    xh, yh = x[ok], y[ok]
    xh_adv = pgd_linf(vm, xh, yh, eps=eps_inf, steps=steps)
    cp, ap_ = predict(vm, xh).tolist(), predict(vm, xh_adv).tolist()
    results["figures"]["hero"] = str(viz.save(viz.grid_figure(
        xh, xh_adv, cp, ap_, CIFAR10_CLASSES, y=yh.tolist(), n=10,
        title="PGD-L∞ ε=8/255 vs vanilla CIFAR-10 CNN (red = fooled)"), out_dir / "figures/hero_grid.png").relative_to(out_dir))
    with torch.no_grad():
        pc, pa = vm(xh[:1]).softmax(1)[0], vm(xh_adv[:1]).softmax(1)[0]
    results["figures"]["triplet"] = str(viz.save(viz.triplet_figure(
        xh[0], xh_adv[0], CIFAR10_CLASSES[pc.argmax()], CIFAR10_CLASSES[pa.argmax()], pc.max().item(), pa.max().item()),
        out_dir / "figures/triplet.png").relative_to(out_dir))
    results["figures"]["confidence"] = str(viz.save(viz.confidence_bars(pc, pa, CIFAR10_CLASSES),
                                                    out_dir / "figures/confidence.png").relative_to(out_dir))
    results["figures"]["eps_curve"] = str(viz.save(viz.eps_curve_figure(results["eps_curves"]),
                                                   out_dir / "figures/eps_curve.png").relative_to(out_dir))

    results["meta"] = {
        "generated_at": dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z"),
        "torch": torch.__version__, "device": "cpu", "n_eval": n, "n_defense": n_def, "n_cw": n_cw,
        "seed": cfg["seed"], "pgd_steps": steps, "n_curve": b.get("curve_n_eval", n), "quick": args.quick,
        "notes": [
            "C&W is a simplified fixed-c variant (no binary search), projected into the same L2 ball as PGD-L2.",
            "ImageNet transfer row uses local photos without ground truth: accuracies are agreement with each "
            "model's own clean top-1 prediction.",
            "Randomized smoothing here is prediction-only (Monte-Carlo vote), no certified radius.",
        ],
    }
    paths = write_report(results, out_dir)
    log(f"wrote {paths['md']}")


if __name__ == "__main__":
    main()
