#!/usr/bin/env python
# SUPERSEDED by scripts/17_feature_faithfulness.py
# Input-position perturbation; degenerate for single-position deep-trace graphs (thesis 3.4.8). Kept for record.
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import _path_setup  # noqa: F401

from qwen_clt.utils.config import load_config
from qwen_clt.models.qwen_hooks import load_qwen_model_and_tokenizer

FEAT_POS_RE = re.compile(r"feature:L\d+:P(\d+):F\d+")


def trapz(y, x):
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    return float(np.sum((y[1:] + y[:-1]) * (x[1:] - x[:-1]) / 2.0))


def position_attribution(edges):
    agg = {}
    for e in edges:
        if not str(e.get("target", "")).startswith("target"):
            continue
        m = FEAT_POS_RE.match(str(e.get("source", "")))
        if not m:
            continue
        p = int(m.group(1))
        agg[p] = agg.get(p, 0.0) + abs(float(e.get("score", 0.0)))
    return agg


def load_edges(result, summary_dir):
    gp = result.get("graph_path")
    candidates = []
    if gp:
        candidates = [Path(gp), summary_dir / Path(gp).name, Path(Path(gp).name)]
    for cand in candidates:
        try:
            if cand.exists():
                return json.loads(cand.read_text()).get("edges", [])
        except Exception:
            pass
    return result.get("top_causal_edges", [])


def first_token(tokenizer, text):
    ids = tokenizer.encode(text, add_special_tokens=False)
    return ids[0]


@torch.no_grad()
def metric_at(model, ids, attn, target_pos, pos_tok, neg_tok):
    logits = model(input_ids=ids, attention_mask=attn).logits[0, target_pos]
    return float(logits[pos_tok] - logits[neg_tok])


@torch.no_grad()
def curve(model, ids, attn, order, target_pos, pos_tok, neg_tok, base_tok, fracs):
    out = []
    n = len(order)
    for fr in fracs:
        k = int(round(fr * n))
        ab = ids.clone()
        for p in order[:k]:
            ab[0, p] = base_tok
        out.append(metric_at(model, ab, attn, target_pos, pos_tok, neg_tok))
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Input-perturbation faithfulness of deep-trace attribution graphs."
    )
    ap.add_argument("--config", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--output", default="faithfulness.json")
    ap.add_argument("--plot", default=None)
    ap.add_argument("--max-prompts", type=int, default=0)
    ap.add_argument("--random-seeds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=11)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = cfg["model"].get("device", "cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_qwen_model_and_tokenizer(cfg)

    summary = json.loads(Path(args.summary).read_text())
    summary_dir = Path(args.summary).parent
    pos_tok = first_token(tokenizer, summary["positive"])
    neg_tok = first_token(tokenizer, summary["negative"])
    target_pos = int(summary.get("target_pos", -1))
    base_tok = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    fracs = np.linspace(0.0, 1.0, args.steps)
    results = summary["results"]
    if args.max_prompts:
        results = results[: args.max_prompts]

    rng = np.random.default_rng(0)
    per_prompt = []
    graph_curves = []
    rand_curves = []

    for r in results:
        prompt = r["prompt"]
        enc = tokenizer(prompt, return_tensors="pt")
        ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)
        seq_len = ids.shape[1]
        last = target_pos % seq_len
        ablatable = [p for p in range(seq_len) if p != last]

        attribution = position_attribution(load_edges(r, summary_dir))
        graph_order = sorted(ablatable, key=lambda p: attribution.get(p, 0.0), reverse=True)

        base = metric_at(model, ids, attn, target_pos, pos_tok, neg_tok)
        denom = base if abs(base) > 1e-6 else 1.0

        g = curve(model, ids, attn, graph_order, target_pos, pos_tok, neg_tok, base_tok, fracs)
        g = [v / denom for v in g]

        rand_runs = []
        for _ in range(args.random_seeds):
            ro = list(ablatable)
            rng.shuffle(ro)
            rc = curve(model, ids, attn, ro, target_pos, pos_tok, neg_tok, base_tok, fracs)
            rand_runs.append([v / denom for v in rc])
        rmean = np.mean(rand_runs, axis=0).tolist()

        auc_g = trapz(g, fracs)
        auc_r = trapz(rmean, fracs)
        graph_curves.append(g)
        rand_curves.append(rmean)
        per_prompt.append({
            "idx": r.get("idx"),
            "prompt": prompt,
            "base_metric": round(base, 4),
            "auc_graph": round(auc_g, 4),
            "auc_random": round(auc_r, 4),
            "faithfulness_gap": round(auc_r - auc_g, 4),
        })
        print(f"prompt {r.get('idx')}: AUC graph {auc_g:.3f} | random {auc_r:.3f} | gap {auc_r - auc_g:+.3f}")

    mean_g = np.mean(graph_curves, axis=0)
    mean_r = np.mean(rand_curves, axis=0)
    summary_out = {
        "n_prompts": len(per_prompt),
        "mean_auc_graph": round(float(np.mean([p["auc_graph"] for p in per_prompt])), 4),
        "mean_auc_random": round(float(np.mean([p["auc_random"] for p in per_prompt])), 4),
        "mean_faithfulness_gap": round(float(np.mean([p["faithfulness_gap"] for p in per_prompt])), 4),
        "fractions": fracs.tolist(),
        "mean_curve_graph": mean_g.tolist(),
        "mean_curve_random": mean_r.tolist(),
        "per_prompt": per_prompt,
    }
    Path(args.output).write_text(json.dumps(summary_out, ensure_ascii=False, indent=2))
    print(f"\nmean AUC graph {summary_out['mean_auc_graph']:.3f} vs random "
          f"{summary_out['mean_auc_random']:.3f} | gap {summary_out['mean_faithfulness_gap']:+.3f}")
    print("wrote", args.output)

    if args.plot:
        fig, ax = plt.subplots(figsize=(6.4, 4.6))
        ax.plot(fracs, mean_g, "-o", color="#378ADD", label="graph-ranked ablation")
        ax.plot(fracs, mean_r, "-o", color="#888780", label="random ablation")
        ax.set_xlabel("fraction of positions ablated")
        ax.set_ylabel("target logit-diff (normalized)")
        ax.set_title("Attribution-graph faithfulness")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(args.plot, bbox_inches="tight")
        print("wrote", args.plot)


if __name__ == "__main__":
    main()
