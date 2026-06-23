#!/usr/bin/env python
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import _path_setup  # noqa: F401

import numpy as np
import torch

from qwen_clt.utils.config import load_config
from qwen_clt.models.proxy_replacement_model import ProxyQwenReplacementModel
from qwen_clt.deep_trace.cache import collect_deep_trace_cache
from qwen_clt.replacement import ReplacementConfig
from qwen_clt.interventions import FeatureIntervention
from qwen_clt.attribution.validation import logit_difference_score


def feature_nodes(graph):
    out = []
    for n in graph["nodes"]:
        if n.get("type") == "CLTFeatureNode":
            out.append({
                "id": n["id"],
                "layer": int(n["layer"]),
                "feature_idx": int(n["feature_idx"]),
                "value": float(n.get("value") or 0.0),
            })
    return out


def target_std(clt, layer):
    if not bool(getattr(clt, "normalize_targets", False)):
        return None
    return (clt.target_var[layer].detach().float().cpu()
            + clt.normalization_eps).sqrt()


def residual_write(clt, s_layer, s_idx, t_layer):
    acc = torch.zeros(clt.encoders[t_layer].shape[0])
    for tp in range(s_layer, t_layer):
        key = f"{s_layer}->{tp}"
        if key not in clt.decoders:
            continue
        w = clt.decoders[key][s_idx].detach().float().cpu()
        ts = target_std(clt, tp)
        if ts is not None:
            w = w * ts
        acc = acc + w
    return acc


def target_factor(clt, cache, layer, pos):
    if bool(getattr(clt, "normalize_inputs", False)):
        inv = (clt.input_var[layer].detach().float().cpu()
               + clt.normalization_eps).rsqrt()
    else:
        inv = torch.ones(clt.encoders[layer].shape[0])
    mi = cache.replacement.mlp_inputs[layer]
    ri = cache.replacement.residual_inputs[layer]
    if mi is None or ri is None:
        return inv
    mi = mi[0, pos].float().cpu()
    ri = ri[0, pos].float().cpu()
    phi = torch.where(ri.abs() > 1e-8, mi / ri, torch.zeros_like(ri))
    return phi * inv


def frozen_de_weight(clt, t_layer, t_idx, residual_delta, factor):
    enc = clt.encoders[t_layer][:, t_idx].detach().float().cpu()
    return float(torch.dot(residual_delta * factor, enc).item())


def spearman(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 2:
        return float("nan")
    ra = a.argsort().argsort().astype(float)
    rb = b.argsort().argsort().astype(float)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def sign_match(pred, obs, eps):
    pred = np.asarray(pred, dtype=float)
    obs = np.asarray(obs, dtype=float)
    mask = (np.abs(pred) > eps) & (np.abs(obs) > eps)
    if not mask.any():
        return float("nan"), 0
    agree = np.sign(pred[mask]) == np.sign(obs[mask])
    return float(agree.mean()), int(mask.sum())


def run_baseline(model, input_ids, attention_mask):
    return collect_deep_trace_cache(
        model=model.base_model,
        autoencoder=model.clt,
        input_ids=input_ids,
        attention_mask=attention_mask,
        replacement_config=ReplacementConfig(),
    )


def run_intervened(model, input_ids, attention_mask, original, layer, pos, idx, value):
    return collect_deep_trace_cache(
        model=model.base_model,
        autoencoder=model.clt,
        input_ids=input_ids,
        attention_mask=attention_mask,
        replacement_config=ReplacementConfig(
            feature_interventions=(
                FeatureIntervention(layer=layer, pos=pos, feature_idx=idx, value=value),
            )
        ),
        original=original,
    )


def feat_act(cache, layer, pos, idx):
    feats = cache.replacement_features[layer]
    if feats is None:
        return None
    return float(feats[0, pos, idx].item())


def evaluate_graph(model, graph, perturb, top_targets, min_activation, eps):
    meta = graph.get("metadata", {})
    tgt = meta.get("target", {})
    prompt = meta.get("prompt")
    pos = int(tgt.get("resolved_target_pos"))
    raw_pos = tgt.get("target_pos", -1)
    pos_id = int(tgt["positive_token_id"])
    neg_id = int(tgt["negative_token_id"])

    input_ids, attention_mask = model.tokenize(prompt)
    base = run_baseline(model, input_ids, attention_mask)
    base_out = logit_difference_score(
        base.replacement.logits, positive_token_id=pos_id,
        negative_token_id=neg_id, target_pos=raw_pos)

    feats = feature_nodes(graph)
    by_layer = {}
    for f in feats:
        by_layer.setdefault(f["layer"], []).append(f)
    factor_by_layer = {L: target_factor(model.clt, base, L, pos) for L in by_layer}

    sources = []
    path_edges = []
    k_edge = max(perturb, key=lambda kk: abs(kk - 1.0))
    out_dir_consistent = 0
    out_dir_total = 0
    for s in feats:
        a_s = s["value"]
        if abs(a_s) < min_activation:
            continue
        downstream = [t for t in feats if t["layer"] > s["layer"]]
        if not downstream:
            continue
        dlayers = sorted({t["layer"] for t in downstream})
        rw_by_layer = {L: residual_write(model.clt, s["layer"], s["feature_idx"], L)
                       for L in dlayers}
        weights = {t["id"]: frozen_de_weight(model.clt, t["layer"], t["feature_idx"],
                                             rw_by_layer[t["layer"]],
                                             factor_by_layer[t["layer"]])
                   for t in downstream}

        per_k = {}
        out_by_k = {}
        for k in perturb:
            interv = run_intervened(model, input_ids, attention_mask, base.original,
                                    s["layer"], pos, s["feature_idx"], k * a_s)
            out_by_k[k] = logit_difference_score(
                interv.replacement.logits, positive_token_id=pos_id,
                negative_token_id=neg_id, target_pos=raw_pos) - base_out
            delta_a_s = (k - 1.0) * a_s
            pred, obs = [], []
            for t in downstream:
                bt = feat_act(base, t["layer"], pos, t["feature_idx"])
                it = feat_act(interv, t["layer"], pos, t["feature_idx"])
                if bt is None or it is None:
                    continue
                p_t = delta_a_s * weights[t["id"]]
                o_t = it - bt
                pred.append(p_t)
                obs.append(o_t)
                if k == k_edge:
                    agree = (abs(p_t) > eps and abs(o_t) > eps
                             and np.sign(p_t) == np.sign(o_t))
                    path_edges.append({
                        "source": s["id"], "target": t["id"],
                        "predicted": float(p_t), "observed": float(o_t),
                        "agree": bool(agree),
                    })
            if top_targets and len(pred) > top_targets:
                order = np.argsort([-abs(p) for p in pred])[:top_targets]
                pred = [pred[i] for i in order]
                obs = [obs[i] for i in order]
            sm, n = sign_match(pred, obs, eps)
            per_k[k] = {
                "sign_match": sm,
                "n_targets": n,
                "spearman": spearman(pred, obs),
                "output_delta": out_by_k[k],
            }

        lo = min(perturb)
        hi = max(perturb)
        if lo < 1.0 < hi or (lo < 1.0 and hi > lo):
            d_lo = out_by_k.get(lo)
            d_hi = out_by_k.get(hi)
            if d_lo is not None and d_hi is not None and abs(d_lo) > eps and abs(d_hi) > eps:
                out_dir_total += 1
                if np.sign(d_lo) != np.sign(d_hi):
                    out_dir_consistent += 1

        sources.append({"id": s["id"], "layer": s["layer"],
                        "feature_idx": s["feature_idx"], "activation": a_s,
                        "n_downstream": len(downstream), "per_k": per_k})

    sm_vals, sp_vals = [], []
    for src in sources:
        for k, r in src["per_k"].items():
            if not np.isnan(r["sign_match"]):
                sm_vals.append(r["sign_match"])
            if not np.isnan(r["spearman"]):
                sp_vals.append(r["spearman"])
    return {
        "prompt": prompt,
        "n_sources": len(sources),
        "mean_sign_match": float(np.mean(sm_vals)) if sm_vals else float("nan"),
        "mean_spearman": float(np.mean(sp_vals)) if sp_vals else float("nan"),
        "output_direction_consistency": (
            out_dir_consistent / out_dir_total if out_dir_total else float("nan")),
        "perturb_for_edges": k_edge,
        "path_edges": path_edges,
        "sources": sources,
    }


def main():
    ap = argparse.ArgumentParser(
        description="Path-level faithfulness: perturb each graph feature and check "
        "whether downstream feature changes match the graph's linear edge predictions "
        "(feature->feature), plus interventional output-direction consistency.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--graph", action="append", default=[])
    ap.add_argument("--graphs-dir", default=None)
    ap.add_argument("--perturb", default="0.0,2.0")
    ap.add_argument("--top-targets", type=int, default=0)
    ap.add_argument("--min-activation", type=float, default=1e-6)
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--output", default="path_faithfulness.json")
    args = ap.parse_args()

    paths = list(args.graph)
    if args.graphs_dir:
        paths.extend(sorted(glob.glob(str(Path(args.graphs_dir) / "*_deep_trace.json"))))
    if not paths:
        raise SystemExit("no graphs given (use --graph or --graphs-dir)")
    perturb = [float(x) for x in args.perturb.split(",") if x.strip() != ""]

    cfg = load_config(args.config)
    model = ProxyQwenReplacementModel.from_checkpoint(args.checkpoint, cfg=cfg)

    results = []
    for p in paths:
        graph = json.loads(Path(p).read_text())
        res = evaluate_graph(model, graph, perturb, args.top_targets,
                             args.min_activation, args.eps)
        res["graph"] = Path(p).name
        results.append(res)
        print(f"{Path(p).name[:48]:48}  sign_match={res['mean_sign_match']:.3f}  "
              f"spearman={res['mean_spearman']:.3f}  "
              f"out_dir={res['output_direction_consistency']:.3f}  "
              f"(sources={res['n_sources']})")

    sm = [r["mean_sign_match"] for r in results if not np.isnan(r["mean_sign_match"])]
    sp = [r["mean_spearman"] for r in results if not np.isnan(r["mean_spearman"])]
    od = [r["output_direction_consistency"] for r in results
          if not np.isnan(r["output_direction_consistency"])]
    summary = {
        "n_graphs": len(results),
        "perturb": perturb,
        "mean_sign_match": float(np.mean(sm)) if sm else float("nan"),
        "mean_spearman": float(np.mean(sp)) if sp else float("nan"),
        "mean_output_direction_consistency": float(np.mean(od)) if od else float("nan"),
        "per_graph": results,
    }
    Path(args.output).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print("=" * 60)
    print(f"graphs: {summary['n_graphs']}  "
          f"sign_match={summary['mean_sign_match']:.3f}  "
          f"spearman={summary['mean_spearman']:.3f}  "
          f"output_direction_consistency={summary['mean_output_direction_consistency']:.3f}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
