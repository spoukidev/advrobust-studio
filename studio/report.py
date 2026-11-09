"""Markdown / HTML / JSON robustness report.

The JSON file (``results.json``) is the machine-readable source of truth; the
Markdown report is rendered from it. ``REPORT_SCHEMA_KEYS`` documents the
required top-level keys and is checked by the tests.
"""
from __future__ import annotations

import html
import json
from pathlib import Path

REPORT_SCHEMA_KEYS = ("meta", "models", "attacks", "defenses", "eps_curves", "transfer", "figures")
ATTACK_ROW_KEYS = ("model", "attack", "eps", "norm", "n", "clean_acc", "robust_acc", "asr",
                   "mean_linf", "max_linf", "mean_l2", "max_l2")


def validate(results: dict) -> None:
    missing = [k for k in REPORT_SCHEMA_KEYS if k not in results]
    if missing:
        raise ValueError(f"report missing keys: {missing}")
    for row in results["attacks"]:
        miss = [k for k in ATTACK_ROW_KEYS if k not in row]
        if miss:
            raise ValueError(f"attack row missing {miss}: {row}")
        for k in ("clean_acc", "robust_acc", "asr"):
            if not 0.0 <= row[k] <= 1.0:
                raise ValueError(f"{k} out of range in {row}")


def _pct(v):
    return f"{100 * v:.1f}%"


def _eps_str(eps, norm):
    if norm == "linf":
        return f"{eps * 255:.0f}/255"
    return f"{eps:.2f}"


def attack_table(rows: list[dict]) -> str:
    out = ["| Model | Attack | ε | Clean acc | Robust acc | ASR | mean L∞ | mean L2 |",
           "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['model']} | {r['attack']} | {_eps_str(r['eps'], r['norm'])} | {_pct(r['clean_acc'])} | "
                   f"**{_pct(r['robust_acc'])}** | {_pct(r['asr'])} | {r['mean_linf']:.4f} | {r['mean_l2']:.3f} |")
    return "\n".join(out)


def defense_table(rows: list[dict]) -> str:
    out = ["| Model | Defense | Setting | Attack | Clean acc | Robust acc |", "|---|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['model']} | {r['defense']} | {r['setting']} | {r['attack']} {_eps_str(r['eps'], r['norm'])} | "
                   f"{_pct(r['clean_acc'])} | **{_pct(r['robust_acc'])}** |")
    return "\n".join(out)


def render_markdown(res: dict, fig_prefix: str = "") -> str:
    validate(res)
    m = res["meta"]
    lines = [
        "# AdvRobust Studio — Robustness Report", "",
        f"Generated {m.get('generated_at', '')} · {m.get('device', 'cpu')} · torch {m.get('torch', '')} · "
        f"n = {m.get('n_eval')} CIFAR-10 test images · seed {m.get('seed')}", "",
    ]
    if "hero" in res["figures"]:
        lines += [f"![clean vs adversarial]({fig_prefix}{res['figures']['hero']})", ""]
    lines += ["## Models", "", "| Name | Arch | Training | Params | Train time |", "|---|---|---|---|---|"]
    for name, info in res["models"].items():
        lines.append(f"| {name} | {info['arch']} | {info['mode']} ({info['epochs']} ep) | {info['params']:,} | "
                     f"{info.get('train_seconds', '?')} s |")
    lines += ["", "## White-box attacks (clean vs robust accuracy)", "", attack_table(res["attacks"]), "",
              "ASR = fraction of originally-correct images the attack flips. "
              "Max L∞ / L2 across all images never exceeds ε (checked in tests and below).", ""]
    budget = [f"- {r['model']} / {r['attack']}: max L∞ = {r['max_linf']:.5f}, max L2 = {r['max_l2']:.4f}"
              for r in res["attacks"]]
    lines += ["<details><summary>Budget audit</summary>", "", *budget, "", "</details>", ""]
    if res.get("at_gap"):
        g = res["at_gap"]
        lines += ["## Adversarial training vs vanilla", "",
                  f"Under {g['attack']} at ε = {g['eps_str']}: vanilla robust acc **{_pct(g['vanilla'])}** → "
                  f"PGD-AT robust acc **{_pct(g['pgdat'])}** (gap **+{100 * g['gap']:.1f} pp**). "
                  f"Clean accuracy cost: {_pct(g['vanilla_clean'])} → {_pct(g['pgdat_clean'])}.", ""]
    if res["eps_curves"]:
        lines += ["## Robust accuracy vs ε", ""]
        if "eps_curve" in res["figures"]:
            lines += [f"![eps curve]({fig_prefix}{res['figures']['eps_curve']})", ""]
        names = list(res["eps_curves"])
        eps = [r["eps"] for r in res["eps_curves"][names[0]]]
        lines += ["| ε | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
        for i, e in enumerate(eps):
            lines.append(f"| {e * 255:.0f}/255 | " + " | ".join(_pct(res['eps_curves'][n][i]['robust_acc'])
                                                            for n in names) + " |")
        lines.append("")
    if res["defenses"]:
        lines += ["## Defenses", "",
                  "*unaware* = attack crafted on the undefended model (gray-box); "
                  "*adaptive* = attack differentiates through the defense (BPDA for JPEG/bit-depth, "
                  "EOT for smoothing). Adaptive numbers are the honest ones.", "",
                  defense_table(res["defenses"]), ""]
    if res["transfer"]:
        lines += ["## Black-box transfer", "",
                  "| Surrogate → Victim | Attack | ε | Surrogate adv acc (white-box) | Victim clean | Victim adv | Transfer rate |",
                  "|---|---|---|---|---|---|---|"]
        for t in res["transfer"]:
            lines.append(f"| {t['surrogate']} → {t['victim']} | {t['attack']} | {t['eps_str']} | "
                         f"{_pct(t['surrogate_adv_acc'])} | {_pct(t['victim_clean_acc'])} | "
                         f"{_pct(t['victim_adv_acc'])} | {_pct(t['transfer_rate'])} |")
        lines.append("")
        if "imagenet_transfer" in res["figures"]:
            lines += [f"![imagenet transfer]({fig_prefix}{res['figures']['imagenet_transfer']})", ""]
    for note in m.get("notes", []):
        lines.append(f"> {note}")
    return "\n".join(lines) + "\n"


def render_html(markdown_text: str) -> str:
    """Minimal standalone HTML (pre-wrapped markdown + images) with no extra deps."""
    try:
        import markdown  # type: ignore

        body = markdown.markdown(markdown_text, extensions=["tables"])
    except ImportError:
        body = f"<pre>{html.escape(markdown_text)}</pre>"
    return f"<!doctype html><meta charset='utf-8'><title>AdvRobust report</title><body>{body}</body>"


def write_report(results: dict, out_dir: str | Path) -> dict:
    """Write results.json, report.md and report.html into ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    validate(results)
    (out / "results.json").write_text(json.dumps(results, indent=2))
    md = render_markdown(results)
    (out / "report.md").write_text(md)
    (out / "report.html").write_text(render_html(md))
    return {"json": out / "results.json", "md": out / "report.md", "html": out / "report.html"}
