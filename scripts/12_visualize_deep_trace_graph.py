#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import matplotlib.cm as cm

LATIN = re.compile(r"[A-Za-z][A-Za-z\-]+")

TYPE_STYLE = {
    "CLTFeatureNode": ("#378ADD", "o", "feature"),
    "MLPErrorNode": ("#8C84A0", "s", "mlp error"),
    "LogitTargetNode": ("#1D9E75", "D", "target"),
    "ResidualStreamNode": ("#c4c2ba", ".", "residual"),
    "AttentionLayerNode": ("#cdd2d6", ".", "attention"),
    "AttentionHeadNode": ("#cdd2d6", ".", "attention"),
    "LayerNormNode": ("#d4d2c9", ".", "layernorm"),
}
DEFAULT_STYLE = ("#d4d2c9", ".", "other")
CONTEXT_TYPES = {"ResidualStreamNode", "AttentionLayerNode",
                 "AttentionHeadNode", "LayerNormNode"}


def clean_concept(top_promoted):
    for raw in sorted(top_promoted, key=lambda t: (not t.startswith(" "),)):
        t = raw.strip()
        if LATIN.fullmatch(t) and len(t) >= 2:
            return t.lower()
    return "?"


def feature_label(key, layer, idx, labels):
    lab = labels.get(key)
    if isinstance(lab, dict) and "target_effect" in lab:
        eff = float(lab["target_effect"])
        arrow = "\u2191" if eff > 0 else "\u2193"
        concept = "?" if float(lab.get("effect_norm", 1.0)) < 0.9 \
            else clean_concept(lab.get("top_promoted", []))
        return f"L{layer}:F{idx}\n{arrow}{concept} ({eff:+.2f})", eff
    if isinstance(lab, str) and lab:
        return (lab if len(lab) <= 24 else lab[:23] + "\u2026"), None
    return f"L{layer} F{idx}", None


def short_label(node, labels):
    t = node.get("type")
    layer = node.get("layer")
    if t == "CLTFeatureNode":
        text, _ = feature_label(f"L{layer}:F{node.get('feature_idx')}",
                                layer, node.get("feature_idx"), labels)
        return text
    if t == "MLPErrorNode":
        return f"err L{layer}"
    if t == "LogitTargetNode":
        return node.get("label") or "target"
    return ""


def is_causal_edge(edge):
    if "causal" in str(edge.get("kind", "")).lower():
        return True
    return bool(edge.get("metadata", {}).get("is_causal"))


def is_error_edge(edge):
    return "error" in str(edge.get("kind", "")).lower()


def edge_influence(edge):
    score = float(edge.get("score", 0.0))
    return -score if "ablation" in str(edge.get("kind", "")).lower() else score


def select_edges(edges, top_edges):
    priority = [e for e in edges if is_causal_edge(e) or is_error_edge(e)]
    rest = sorted((e for e in edges if not (is_causal_edge(e) or is_error_edge(e))),
                  key=lambda e: abs(float(e.get("score", 0.0))), reverse=True)
    keep = priority + rest[: max(0, top_edges - len(priority))]
    seen, out = set(), []
    for e in keep:
        k = (e["source"], e["target"], e.get("kind"))
        if k not in seen:
            seen.add(k)
            out.append(e)
    return out


def render(graph_path, out_path, top_edges, figsize, labels, model_output, faith_map):
    payload = json.loads(Path(graph_path).read_text())
    prompt = (payload.get("metadata", {}) or {}).get("prompt")
    tgt_meta = (payload.get("metadata", {}) or {}).get("target", {}) or {}
    pos_name = (tgt_meta.get("positive") or " increase").strip()
    neg_name = (tgt_meta.get("negative") or " decrease").strip()
    if model_output is None and faith_map and prompt:
        model_output = faith_map.get(prompt)
        if model_output is None:
            for kk, vv in faith_map.items():
                if prompt in kk or kk in prompt:
                    model_output = vv
                    break
    nodes = {n["id"]: n for n in payload["nodes"]}
    edges = select_edges(payload["edges"], top_edges)

    kept = set()
    for e in edges:
        kept.add(e["source"])
        kept.add(e["target"])
    for nid, n in nodes.items():
        if n.get("type") == "LogitTargetNode":
            kept.add(nid)

    present = [(nodes[nid].get("layer") or 0) for nid in kept]
    target_y = (max(present) if present else 1) + 1.6

    by_layer = {}
    for nid in kept:
        n = nodes[nid]
        y = target_y if n.get("type") == "LogitTargetNode" else (n.get("layer") or 0)
        by_layer.setdefault(y, []).append(nid)

    pos = {}
    for y, ids in by_layer.items():
        ids = sorted(ids, key=lambda i: (nodes[i].get("pos") or 0, i))
        for j, nid in enumerate(ids):
            pos[nid] = ((j + 1) / (len(ids) + 1), y)

    w, h = (float(v) for v in figsize.split(","))
    fig, ax = plt.subplots(figsize=(w, h))

    for y in sorted({p[1] for p in pos.values()}):
        ax.axhline(y, color="#eceae3", lw=6, zorder=0)

    causal_map = {}
    for e in payload["edges"]:
        if e.get("kind") == "causal_ablation_effect":
            causal_map[e["source"]] = float(e.get("score", 0.0))
    maxc = max((abs(v) for v in causal_map.values()), default=1.0) or 1.0

    causal_edges = [e for e in edges if is_causal_edge(e)]
    other_edges = [e for e in edges if not is_causal_edge(e)]
    for e in other_edges:
        s, t = e["source"], e["target"]
        if s not in pos or t not in pos:
            continue
        score = float(e.get("score", 0.0))
        color = "#73c2a4" if edge_influence(e) >= 0 else "#e8a98f"
        ax.add_patch(FancyArrowPatch(
            pos[s], pos[t], arrowstyle="-", mutation_scale=7,
            color=color, lw=0.7, alpha=0.22, shrinkA=5, shrinkB=6,
            connectionstyle="arc3,rad=0.08", zorder=1))
    for e in causal_edges:
        s, t = e["source"], e["target"]
        if s not in pos or t not in pos:
            continue
        score = float(e.get("score", 0.0))
        color = "#1D9E75" if edge_influence(e) >= 0 else "#D85A30"
        lw = 1.0 + 3.4 * min(1.0, abs(score) / 0.4)
        ax.add_patch(FancyArrowPatch(
            pos[s], pos[t], arrowstyle="-|>", mutation_scale=11,
            color=color, lw=lw, alpha=0.85, shrinkA=6, shrinkB=9,
            connectionstyle="arc3,rad=0.06", zorder=2))

    offsets = [(0, 13), (0, -22), (0, 20), (0, -15)]
    for nid in kept:
        n = nodes[nid]
        typ = n.get("type")
        color, marker, _ = TYPE_STYLE.get(typ, DEFAULT_STYLE)
        x, y = pos[nid]
        if typ in CONTEXT_TYPES:
            ax.scatter([x], [y], s=26, c=color, marker="o",
                       edgecolors="none", alpha=0.55, zorder=3)
            continue
        if typ == "CLTFeatureNode":
            frac = min(1.0, abs(causal_map.get(nid, 0.0)) / maxc)
            size = 70 + 520 * frac
            fill = cm.get_cmap("Blues")(0.28 + 0.64 * frac)
            ax.scatter([x], [y], s=size, color=[fill], marker="o",
                       edgecolors="#1f4e79", linewidths=0.8, zorder=4)
        else:
            val = abs(float(n.get("value") or 0.0))
            size = 380 if typ == "LogitTargetNode" else 150
            if typ == "LogitTargetNode" and model_output is not None:
                color = "#1D9E75" if model_output >= 0 else "#D85A30"
            ax.scatter([x], [y], s=size, c=color, marker=marker,
                       edgecolors="white", linewidths=0.9, zorder=4)
        label = short_label(n, labels)
        if label:
            j = sorted(by_layer[y]).index(nid) if y in by_layer else 0
            if typ == "MLPErrorNode":
                ax.annotate(label, (x, y), fontsize=6.3, ha="center", va="top",
                            xytext=(0, -11), textcoords="offset points",
                            color="#5E5670", zorder=5)
            else:
                feat_off = [(0, 15), (0, 27), (0, 18), (0, 33)]
                dx, dy = feat_off[j % len(feat_off)]
                ax.annotate(label, (x, y), fontsize=7.4, ha="center", va="center",
                            xytext=(dx, dy), textcoords="offset points",
                            zorder=6, linespacing=1.05, fontweight="medium",
                            bbox=dict(boxstyle="round,pad=0.2", fc="white",
                                      ec="#dcdcdc", lw=0.5, alpha=0.85))

    if model_output is not None and target_y:
        tnode = next((nid for nid in kept
                      if nodes[nid].get("type") == "LogitTargetNode"), None)
        if tnode:
            tx, ty = pos[tnode]
            d = pos_name if model_output > 0 else neg_name
            ax.annotate(f"\u0432\u044b\u0445\u043e\u0434 \u2248 {model_output:+.2f} \u2192 {d}",
                        (tx, ty), fontsize=8, ha="center", va="bottom",
                        xytext=(0, 30), textcoords="offset points",
                        color="#9b2c2c", zorder=6, fontweight="bold")

    handles = []
    for typ in ["CLTFeatureNode", "MLPErrorNode", "LogitTargetNode"]:
        c, m, lab = TYPE_STYLE[typ]
        if typ == "LogitTargetNode":
            c, lab = "#5f6368", "target (\u0446\u0432\u0435\u0442 = \u0432\u044b\u0445\u043e\u0434)"
        handles.append(plt.Line2D([], [], marker=m, color="w", markerfacecolor=c,
                                  markersize=9, label=lab))
    handles.append(plt.Line2D([], [], color="#9aa0a6", marker="o", lw=0,
                              markersize=5, label="attn / residual / ln"))
    handles.append(plt.Line2D([], [], color="#1D9E75", lw=3,
                              label=f"causal score > 0 (\u2192 {pos_name})"))
    handles.append(plt.Line2D([], [], color="#D85A30", lw=3,
                              label=f"causal score < 0 (\u2192 {neg_name})"))
    ax.legend(handles=handles, loc="upper left", fontsize=8, frameon=False,
              ncol=1, labelspacing=0.35, handletextpad=0.5, borderaxespad=0.4)
    ax.annotate("\u0440\u0430\u0437\u043c\u0435\u0440/\u043d\u0430\u0441\u044b\u0449\u0435\u043d\u043d\u043e\u0441\u0442\u044c \u0444\u0438\u0447 \u221d |\u043a\u0430\u0443\u0437\u0430\u043b\u044c\u043d\u044b\u0439 \u0431\u0430\u043b\u043b| (\u0437\u0430\u0434\u0435\u0439\u0441\u0442\u0432\u043e\u0432\u0430\u043d\u043d\u043e\u0441\u0442\u044c)",
                xy=(0.0, 0.02), xycoords="axes fraction", fontsize=7.5,
                color="#555", va="bottom")

    ax.set_yticks(sorted({round(p[1]) for p in pos.values()}))
    ax.set_ylabel("layer  (target at top)")
    ax.set_xticks([])
    mx = [pos[nid][0] for nid in kept
          if nodes[nid].get("type") in ("CLTFeatureNode", "MLPErrorNode", "LogitTargetNode")]
    if mx:
        ax.set_xlim(min(mx) - 0.10, max(mx) + 0.10)
    else:
        ax.set_xlim(-0.04, 1.04)
    info = (f"Deep Trace circuit \u2014 {len(causal_edges)} causal + "
            f"{len(other_edges)} context edges")
    if prompt:
        ax.set_title(info, fontsize=9, color="#666", pad=20)
        fig.suptitle(f"\u00ab{prompt} ___\u00bb", fontsize=13,
                     fontweight="bold", y=0.995)
    else:
        ax.set_title(info, fontsize=11)
    for sp in ["top", "right", "bottom"]:
        ax.spines[sp].set_visible(False)
    ax.margins(0.06)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("wrote", out_path)


def main():
    ap = argparse.ArgumentParser(
        description="Render Deep Trace attribution graph(s): full layered "
        "structure (features, mlp-error, target, context nodes) with causal "
        "edges emphasised and output-effect feature labels.")
    ap.add_argument("graphs", nargs="+")
    ap.add_argument("-o", "--output", default=None)
    ap.add_argument("--top-edges", type=int, default=50)
    ap.add_argument("--figsize", default="12.5,9")
    ap.add_argument("--labels", default=None)
    ap.add_argument("--model-output", type=float, default=None)
    ap.add_argument("--faithfulness", default=None)
    args = ap.parse_args()

    if args.output and len(args.graphs) > 1:
        ap.error("--output cannot be used with multiple input graphs")

    faith_map = {}
    if args.faithfulness:
        fj = json.loads(Path(args.faithfulness).read_text())
        for p in fj.get("per_prompt", []):
            if p.get("prompt") is not None and p.get("base_metric") is not None:
                faith_map[p["prompt"]] = float(p["base_metric"])

    labels = {}
    if args.labels:
        raw = json.loads(Path(args.labels).read_text())
        labels = raw if all(isinstance(v, dict) for v in raw.values()) else {
            k: (v if isinstance(v, str) else v.get("label", "")) for k, v in raw.items()}

    for graph_path in args.graphs:
        out_path = args.output or str(Path(graph_path).with_suffix(".svg"))
        render(graph_path, out_path, args.top_edges, args.figsize, labels,
               args.model_output, faith_map)


if __name__ == "__main__":
    main()
