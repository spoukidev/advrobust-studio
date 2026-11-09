import json

import pytest

from studio.eval import evaluate_attack, eps_curve, perturbation_stats
from studio.report import REPORT_SCHEMA_KEYS, render_markdown, validate, write_report
from studio import visualize as viz


def _results(model, batch):
    x, y = batch
    rows = [evaluate_attack(model, x, y, a, e, model_name="m", steps=2).to_dict()
            for a, e in (("fgsm", 8 / 255), ("pgd_linf", 8 / 255), ("pgd_l2", 0.5))]
    curve = eps_curve(model, x, y, [0, 4 / 255, 8 / 255], model_name="m", steps=2)
    return {"meta": {"n_eval": len(y), "seed": 0}, "models": {"m": {"arch": "cifar_cnn", "mode": "vanilla",
            "epochs": 0, "params": 1}}, "attacks": rows, "defenses": [], "eps_curves": {"m": curve},
            "transfer": [], "figures": {}}


def test_eval_metrics_in_range(model, batch):
    r = evaluate_attack(model, *batch, "pgd_linf", 8 / 255, steps=2)
    assert 0 <= r.clean_acc <= 1 and 0 <= r.robust_acc <= 1 and 0 <= r.asr <= 1
    assert r.max_linf <= 8 / 255 + 1e-6


def test_eps_zero_is_identity(model, batch):
    r = evaluate_attack(model, *batch, "pgd_linf", 0.0)
    assert r.robust_acc == r.clean_acc and r.max_linf == 0


def test_robust_never_above_clean_for_whitebox_pgd(model, batch):
    curve = eps_curve(model, *batch, [0, 8 / 255, 32 / 255], steps=5)
    assert curve[-1]["robust_acc"] <= curve[0]["robust_acc"]


def test_perturbation_stats():
    import torch
    x = torch.zeros(2, 3, 4, 4)
    s = perturbation_stats(x, x + 0.1)
    assert abs(s["max_linf"] - 0.1) < 1e-6


def test_report_generation(tmp_path, model, batch):
    res = _results(model, batch)
    paths = write_report(res, tmp_path)
    data = json.loads(paths["json"].read_text())
    assert all(k in data for k in REPORT_SCHEMA_KEYS)
    md = paths["md"].read_text()
    assert "Robustness Report" in md and "| m | pgd_linf | 8/255 |" in md
    assert "Robust accuracy vs ε" in md
    assert paths["html"].read_text().startswith("<!doctype html>")


def test_report_schema_rejects_bad_rows(model, batch):
    res = _results(model, batch)
    res["attacks"][0]["robust_acc"] = 1.7
    with pytest.raises(ValueError):
        validate(res)
    del res["eps_curves"]
    with pytest.raises(ValueError):
        render_markdown(res)


def test_figures_render(tmp_path, model, batch):
    x, y = batch
    fig = viz.grid_figure(x, x, [0] * 8, [1] * 8, [str(i) for i in range(10)], n=8)
    assert viz.save(fig, tmp_path / "g.png").stat().st_size > 1000
    fig = viz.triplet_figure(x[0], (x[0] + 0.01).clamp(0, 1), "a", "b", 0.9, 0.8)
    assert viz.save(fig, tmp_path / "t.png").exists()
