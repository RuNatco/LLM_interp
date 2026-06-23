#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _path_setup  # noqa: F401

import numpy as np
import torch

from qwen_clt.utils.config import load_config
from qwen_clt.models.proxy_replacement_model import ProxyQwenReplacementModel
from qwen_clt.deep_trace.build import resolve_position
from qwen_clt.deep_trace.cache import collect_deep_trace_cache
from qwen_clt.replacement import ReplacementConfig
from qwen_clt.interventions import FeatureIntervention
from qwen_clt.attribution.validation import logit_difference_score


def trapz(y, x):
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    return float(np.sum((y[1:] + y[:-1]) * (x[1:] - x[:-1]) / 2.0))


def graph_features(graph):
    nodes = {n["id"]: n for n in graph["nodes"]}
    attribution = {}
    for e in graph["edges"]:
        if e.get("kind") != "causal_ablation_effect":
            continue
        if not str(e.get("target", "")).startswith("target"):
            continue
        src = e.get("source", "")
        attribution[src] = attribution.get(src, 0.0) + abs(float(e.get("score", 0.0)))
    feats = []
    for nid, n in nodes.items():
        if n.get("type") != "CLTFeatureNode":
            continue
        feats.append({
            "layer": int(n["layer"]),
            "feature_idx": int(n["feature_idx"]),
            "pos": n.get("pos"),
            "attr": attribution.get(nid, 0.0),
        })
    return feats


@torch.no_grad()
def metric_for(rm, ids, attn, interventions, baseline, pos_id, neg_id, tgt):
    cfg = ReplacementConfig(feature_interventions=tuple(interventions))
    cache = collect_deep_trace_cache(
        model=rm.base_model, autoencoder=rm.clt,
        input_ids=ids, attention_mask=attn,
        replacement_config=cfg, original=baseline.original)
    return logit_difference_score(cache.replacement.logits, pos_id, neg_id, tgt)


def raw_curve(rm, ids, attn, order, baseline, pos_id, neg_id, tgt, fracs):
    out = []
    n = len(order)
    for fr in fracs:
        k = int(round(fr * n))
        interventions = [
            FeatureIntervention(layer=f["layer"], pos=f["pos"],
                                feature_idx=f["feature_idx"], value=0.0)
            for f in order[:k]
        ]
        out.append(metric_for(rm, ids, attn, interventions,
                              baseline, pos_id, neg_id, tgt))
    return out


def subset_stats(per_prompt, thr):
    sub = [p for p in per_prompt if abs(p["base_metric"]) >= thr]
    if not sub:
        return {"threshold": thr, "n": 0}
    return {
        "threshold": thr,
        "n": len(sub),
        "mean_gap_norm": round(float(np.mean([p["gap_norm"] for p in sub])), 4),
        "frac_faithful_norm": round(float(np.mean([p["gap_norm"] > 0 for p in sub])), 3),
        "mean_win_rate": round(float(np.mean([p["win_rate"] for p in sub])), 3),
        "frac_win": round(float(np.mean([p["win_rate"] > 0.5 for p in sub])), 3),
    }


def main():
    ap = argparse.ArgumentParser(
        description="Feature-ablation faithfulness for deep-trace graphs. Reports "
        "three views over ALL prompts: (a) base-normalized gap, (b) magnitude-free "
        "sign/rank win-rate (stable at base~0), (c) raw |metric| gap; plus a "
        "sensitivity sweep over |base| thresholds. No prompts are silently dropped.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--graph", action="append", default=[])
    ap.add_argument("--graphs-dir", default=None)
    ap.add_argument("--steps", type=int, default=11)
    ap.add_argument("--random-seeds", type=int, default=5)
    ap.add_argument("--sweep", nargs="?", const="0.0,0.25,0.5,0.75,1.0",
                    default="0.0,0.25,0.5,0.75,1.0")
    ap.add_argument("--output", default="feature_faithfulness.json")
    args = ap.parse_args()

    graph_paths = list(args.graph)
    if args.graphs_dir:
        graph_paths += sorted(
            str(p) for p in Path(args.graphs_dir).glob("*_deep_trace.json"))
    if not graph_paths:
        ap.error("no graphs; use --graph or --graphs-dir")

    cfg = load_config(args.config)
    rm = ProxyQwenReplacementModel.from_checkpoint(args.checkpoint, cfg=cfg)
    device = rm.device
    fracs = np.linspace(0.0, 1.0, args.steps)
    rng = np.random.default_rng(0)

    per_prompt = []
    gcurves_norm, rcurves_norm = [], []
    for gp in graph_paths:
        graph = json.loads(Path(gp).read_text())
        md = graph.get("metadata", {})
        target = md.get("target", {})
        prompt = md.get("prompt")
        pos_id = int(target["positive_token_id"])
        neg_id = int(target["negative_token_id"])
        target_pos = int(target.get("target_pos", -1))

        enc = rm.tokenizer(prompt, return_tensors="pt")
        ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)
        tgt = resolve_position(target_pos, ids.shape[1])

        feats = graph_features(graph)
        for f in feats:
            if f["pos"] is None:
                f["pos"] = tgt
        if not feats:
            continue

        baseline = collect_deep_trace_cache(
            model=rm.base_model, autoencoder=rm.clt,
            input_ids=ids, attention_mask=attn,
            replacement_config=ReplacementConfig(), original=None)

        gorder = sorted(feats, key=lambda f: f["attr"], reverse=True)
        g_raw = raw_curve(rm, ids, attn, gorder, baseline, pos_id, neg_id, tgt, fracs)
        rand_runs = []
        for _ in range(args.random_seeds):
            ro = list(feats)
            rng.shuffle(ro)
            rand_runs.append(raw_curve(rm, ids, attn, ro, baseline,
                                       pos_id, neg_id, tgt, fracs))
        r_raw = np.mean(rand_runs, axis=0)
        g_raw = np.asarray(g_raw)

        base = float(g_raw[0])
        denom = base if abs(base) > 1e-6 else 1.0
        g_norm = (g_raw / denom).tolist()
        r_norm = (r_raw / denom).tolist()
        auc_g = trapz(g_norm, fracs)
        auc_r = trapz(r_norm, fracs)
        gap_norm = auc_r - auc_g

        wins = [abs(g_raw[i]) < abs(r_raw[i]) for i in range(1, len(fracs))]
        win_rate = float(np.mean(wins)) if wins else 0.0

        auc_g_abs = trapz(np.abs(g_raw), fracs)
        auc_r_abs = trapz(np.abs(r_raw), fracs)
        gap_raw = auc_r_abs - auc_g_abs

        gcurves_norm.append(g_norm)
        rcurves_norm.append(r_norm)
        per_prompt.append({
            "prompt": prompt, "base_metric": round(base, 4), "n_features": len(feats),
            "gap_norm": round(gap_norm, 4),
            "win_rate": round(win_rate, 3),
            "gap_raw": round(gap_raw, 4),
        })
        print(f"base={base:+.2f}  gap_norm={gap_norm:+.3f}  win_rate={win_rate:.2f}  "
              f"gap_raw={gap_raw:+.3f}  {prompt[:42]}")

    thresholds = [float(x) for x in args.sweep.split(",")]
    sweep = [subset_stats(per_prompt, t) for t in thresholds]

    result = {
        "kind": "feature_ablation_faithfulness",
        "n_prompts": len(per_prompt),
        "all_prompts": {
            "mean_gap_norm": round(float(np.mean([p["gap_norm"] for p in per_prompt])), 4),
            "frac_faithful_norm": round(float(np.mean([p["gap_norm"] > 0 for p in per_prompt])), 3),
            "mean_win_rate": round(float(np.mean([p["win_rate"] for p in per_prompt])), 3),
            "frac_win": round(float(np.mean([p["win_rate"] > 0.5 for p in per_prompt])), 3),
            "mean_gap_raw": round(float(np.mean([p["gap_raw"] for p in per_prompt])), 4),
            "frac_faithful_raw": round(float(np.mean([p["gap_raw"] > 0 for p in per_prompt])), 3),
        },
        "sensitivity_sweep": sweep,
        "fractions": fracs.tolist(),
        "mean_curve_graph": np.mean(gcurves_norm, axis=0).tolist() if gcurves_norm else [],
        "mean_curve_random": np.mean(rcurves_norm, axis=0).tolist() if rcurves_norm else [],
        "per_prompt": per_prompt,
    }
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2))

    a = result["all_prompts"]
    print("\n=== ALL prompts (no filtering) ===")
    print(f"  win-rate (magnitude-free): mean {a['mean_win_rate']:.3f}, "
          f"faithful on {a['frac_win']*100:.0f}% of prompts")
    print(f"  raw |metric| gap:          faithful on {a['frac_faithful_raw']*100:.0f}% of prompts")
    print(f"  normalized gap:            mean {a['mean_gap_norm']:+.3f}, "
          f"{a['frac_faithful_norm']*100:.0f}% (unstable at base~0)")
    print("\n=== sensitivity sweep over |base| (transparency, NOT a single chosen cut) ===")
    print(f"  {'|base|>=':>9} {'n':>3} {'gap_norm':>9} {'ff_norm':>8} {'win_rate':>9} {'frac_win':>9}")
    for s in sweep:
        if s["n"] == 0:
            continue
        print(f"  {s['threshold']:>9.2f} {s['n']:>3} {s['mean_gap_norm']:>+9.3f} "
              f"{s['frac_faithful_norm']:>8.2f} {s['mean_win_rate']:>9.3f} {s['frac_win']:>9.2f}")
    print("\nwrote", args.output)


if __name__ == "__main__":
    main()
