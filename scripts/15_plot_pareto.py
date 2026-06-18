#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PANELS = [
    ("nmse", "eval NMSE", "Reconstruction vs sparsity", "#378ADD"),
    ("top1", "replacement top-1 agreement", "Top-1 fidelity vs sparsity", "#1D9E75"),
    ("kl", "replacement KL divergence", "KL fidelity vs sparsity", "#D85A30"),
    ("recovered", "recovered fraction (KL, vs mean)", "Recovered fraction vs sparsity", "#8C6BB1"),
]


def parse_run(value):
    if ":" not in value:
        raise argparse.ArgumentTypeError(f"--run wants LABEL:DIR, got {value!r}")
    label, path = value.split(":", 1)
    return label, path


def last_eval(metrics_path):
    l0 = nmse = None
    with open(metrics_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("kind") == "eval":
                l0 = d.get("eval_l0_mean", l0)
                nmse = d.get("eval_nmse_mean", nmse)
    return l0, nmse


def read_replacement(eval_path):
    if not Path(eval_path).exists():
        return {}
    data = json.loads(Path(eval_path).read_text())
    metrics = data.get("metrics", {})
    recovered = (
        data.get("baseline_comparison", {})
        .get("recovered_fraction", {})
        .get("mean", {})
        .get("recovered_fraction_kl_div")
    )
    return {
        "top1": metrics.get("top1_agreement"),
        "kl": metrics.get("kl_div"),
        "recovered": recovered,
    }


def main():
    ap = argparse.ArgumentParser(description="Plot sparsity-fidelity Pareto from sweep runs.")
    ap.add_argument("--run", action="append", type=parse_run, required=True)
    ap.add_argument("-o", "--output", default="pareto.svg")
    args = ap.parse_args()

    points = []
    for label, d in args.run:
        d = Path(d)
        m = d / "metrics.jsonl"
        if not m.exists():
            print(f"skip {label}: no {m}")
            continue
        l0, nmse = last_eval(m)
        if l0 is None or nmse is None:
            print(f"skip {label}: no eval rows in {m}")
            continue
        rec = read_replacement(d / "replacement_eval_metrics.json")
        row = {"label": label, "l0": float(l0), "nmse": float(nmse), **rec}
        points.append(row)
        print(f"{label}: L0={l0:.0f} NMSE={nmse:.3f} top1={rec.get('top1')} "
              f"kl={rec.get('kl')} recovered={rec.get('recovered')}")

    if not points:
        ap.error("no usable runs")

    points.sort(key=lambda p: p["l0"])
    active = [pan for pan in PANELS if any(p.get(pan[0]) is not None for p in points)]
    ncols = 2 if len(active) > 1 else 1
    nrows = math.ceil(len(active) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.4 * ncols, 4.4 * nrows), squeeze=False)
    flat = [ax for row in axes for ax in row]

    for ax, (key, ylabel, title, color) in zip(flat, active):
        pts = [p for p in points if p.get(key) is not None]
        xs = [p["l0"] for p in pts]
        ys = [p[key] for p in pts]
        ax.plot(xs, ys, "-o", color=color, lw=1.5)
        for p in pts:
            ax.annotate(p["label"], (p["l0"], p[key]), fontsize=8,
                        xytext=(4, 4), textcoords="offset points")
        ax.set_xlabel("L0 (active features / token)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.25)

    for ax in flat[len(active):]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(args.output, bbox_inches="tight")
    print("wrote", args.output)


if __name__ == "__main__":
    main()
