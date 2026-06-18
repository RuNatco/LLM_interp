#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROWS = [
    ("top1_agreement", "top1", "up"),
    ("last_token_top1_agreement", "last-tok top1", "up"),
    ("kl_div", "KL", "down"),
    ("last_token_kl_div", "last-tok KL", "down"),
    ("logit_mse", "logit MSE", "down"),
    ("target_logit_diff_mae", "tgt logit-diff MAE", "down"),
]
RECOVERED = [
    ("mean", "recovered vs mean"),
    ("zero", "recovered vs zero"),
    ("random_clt", "recovered vs random"),
]


def load_run(path):
    data = json.loads(Path(path).read_text())
    name = data.get("project") or Path(path).stem
    metrics = data.get("metrics", {})
    recovered = (
        data.get("baseline_comparison", {}).get("recovered_fraction", {})
    )
    rec = {
        k: recovered.get(k, {}).get("recovered_fraction_kl_div")
        for k in ("mean", "zero", "random_clt")
    }
    return name, metrics, rec


def fmt(v):
    return "—" if v is None else f"{v:.4f}"


def main():
    ap = argparse.ArgumentParser(
        description="Compare replacement_eval_metrics.json runs side by side."
    )
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    runs = [load_run(p) for p in args.runs]
    names = [n for n, _, _ in runs]

    label_w = max(20, max(len(lbl) for _, lbl, _ in ROWS + [(0, r, 0) for _, r in RECOVERED]) + 2)
    col_w = max(12, max(len(n) for n in names) + 2)

    lines = []
    header = "metric".ljust(label_w) + "dir".ljust(6) + "".join(n.ljust(col_w) for n in names)
    lines.append(header)
    lines.append("-" * len(header))

    for key, lbl, direction in ROWS:
        arrow = "\u2191" if direction == "up" else "\u2193"
        cells = "".join(fmt(m.get(key)).ljust(col_w) for _, m, _ in runs)
        lines.append(lbl.ljust(label_w) + arrow.ljust(6) + cells)

    lines.append("")
    for key, lbl in RECOVERED:
        cells = "".join(fmt(r.get(key)).ljust(col_w) for _, _, r in runs)
        lines.append(lbl.ljust(label_w) + "\u2191".ljust(6) + cells)

    if len(runs) >= 2:
        lines.append("")
        lines.append(f"delta ({names[-1]} - {names[0]}):")
        base_m = runs[0][1]
        last_m = runs[-1][1]
        for key, lbl, direction in ROWS:
            a, b = base_m.get(key), last_m.get(key)
            if a is None or b is None:
                continue
            d = b - a
            better = (d > 0) if direction == "up" else (d < 0)
            mark = "better" if better else "worse"
            lines.append(f"  {lbl.ljust(label_w)} {d:+.4f}  ({mark})")

    text = "\n".join(lines)
    print(text)

    if args.output:
        out = Path(args.output)
        if out.suffix == ".md":
            md = ["| metric | dir | " + " | ".join(names) + " |"]
            md.append("|" + "---|" * (len(names) + 2))
            for key, lbl, direction in ROWS:
                arrow = "up" if direction == "up" else "down"
                md.append("| " + lbl + " | " + arrow + " | "
                          + " | ".join(fmt(m.get(key)) for _, m, _ in runs) + " |")
            for key, lbl in RECOVERED:
                md.append("| " + lbl + " | up | "
                          + " | ".join(fmt(r.get(key)) for _, _, r in runs) + " |")
            out.write_text("\n".join(md) + "\n")
        else:
            out.write_text(text + "\n")
        print("wrote", out)


if __name__ == "__main__":
    main()
