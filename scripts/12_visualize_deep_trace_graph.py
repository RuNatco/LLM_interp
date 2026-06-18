#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

TYPE_STYLE = {
    "CLTFeatureNode": ("#378ADD", "o", "feature"),
    "MLPErrorNode": ("#D85A30", "s", "mlp error"),
    "LogitTargetNode": ("#1D9E75", "D", "target"),
    "ResidualStreamNode": ("#888780", ".", "residual"),
    "AttentionLayerNode": ("#9aa0a6", ".", "attention"),
    "AttentionHeadNode": ("#9aa0a6", ".", "attention"),
    "LayerNormNode": ("#b4b2a9", ".", "layernorm"),
}
DEFAULT_STYLE = ("#b4b2a9", ".", "other")


def short_label(node, labels):
    t = node.get("type")
    layer = node.get("layer")
    if t == "CLTFeatureNode":
        key = f"L{layer}:F{node.get('feature_idx')}"
        named = labels.get(key)
        if named:
            return named if len(named) <= 24 else named[:23] + "\u2026"
        return f"L{layer} F{node.get('feature_idx')}"
    if t == "MLPErrorNode":
        return f"err L{layer}"
    if t == "LogitTargetNode":
        return node.get("label") or "target"
    return ""


def is_causal_edge(edge):
    if "causal" in str(edge.get("kind", "")).lower():
        return True
    return bool(edge.get("metadata", {}).get("is_causal"))


def select_edges(edges, top_edges):
    causal = [e for e in edges if is_causal_edge(e)]
    rest = sorted(
        (e for e in edges if not is_causal_edge(e)),
        key=lambda e: abs(float(e.get("score", 0.0))),
        reverse=True,
    )
    keep = causal + rest[: max(0, top_edges - len(causal))]
    seen = set()
    out = []
    for e in keep:
        key = (e["source"], e["target"], e.get("kind"))
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
    return out


def render(graph_path, out_path, top_edges, figsize, labels):
    payload = json.loads(Path(graph_path).read_text())
    nodes = {n["id"]: n for n in payload["nodes"]}
    edges = select_edges(payload["edges"], top_edges)

    kept = set()
    for e in edges:
        kept.add(e["source"])
        kept.add(e["target"])
    for nid, n in nodes.items():
        if n.get("type") == "LogitTargetNode":
            kept.add(nid)

    present_layers = [
        (nodes[nid].get("layer") if nodes[nid].get("layer") is not None else 0)
        for nid in kept
    ]
    target_y = (max(present_layers) if present_layers else 1) + 1.5

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

    for e in edges:
        s, t = e["source"], e["target"]
        if s not in pos or t not in pos:
            continue
        score = float(e.get("score", 0.0))
        color = "#1D9E75" if score >= 0 else "#D85A30"
        lw = 0.6 + 3.2 * min(1.0, abs(score) / 0.4)
        ax.add_patch(FancyArrowPatch(
            pos[s], pos[t], arrowstyle="-|>", mutation_scale=9,
            color=color, lw=lw, alpha=0.55, shrinkA=6, shrinkB=8,
            connectionstyle="arc3,rad=0.06", zorder=1,
        ))

    for nid in kept:
        n = nodes[nid]
        color, marker, _ = TYPE_STYLE.get(n.get("type"), DEFAULT_STYLE)
        x, y = pos[nid]
        val = abs(float(n.get("value") or 0.0))
        size = 360 if n.get("type") == "LogitTargetNode" else 60 + 240 * min(1.0, val / 10.0)
        ax.scatter([x], [y], s=size, c=color, marker=marker,
                   edgecolors="white", linewidths=0.8, zorder=3)
        label = short_label(n, labels)
        if label:
            ax.annotate(label, (x, y), fontsize=7.5, ha="center", va="center",
                        xytext=(0, 11), textcoords="offset points", zorder=4)

    handles = []
    for typ in ["CLTFeatureNode", "MLPErrorNode", "LogitTargetNode"]:
        c, m, lab = TYPE_STYLE[typ]
        handles.append(plt.Line2D([], [], marker=m, color="w", markerfacecolor=c,
                                  markersize=9, label=lab))
    handles.append(plt.Line2D([], [], color="#1D9E75", lw=3, label="edge score > 0"))
    handles.append(plt.Line2D([], [], color="#D85A30", lw=3, label="edge score < 0"))
    ax.legend(handles=handles, loc="upper left", fontsize=8, frameon=False)

    ax.set_yticks(sorted({round(p[1]) for p in pos.values()}))
    ax.set_ylabel("layer  (target at top)")
    ax.set_xticks([])
    ax.set_title(f"Deep Trace circuit — top {len(edges)} edges", fontsize=11)
    for sp in ["top", "right", "bottom"]:
        ax.spines[sp].set_visible(False)
    ax.margins(0.05)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out_path)


def main():
    parser = argparse.ArgumentParser(
        description="Render Deep Trace attribution graph(s) to image files."
    )
    parser.add_argument("graphs", nargs="+")
    parser.add_argument("-o", "--output", default=None)
    parser.add_argument("--top-edges", type=int, default=40)
    parser.add_argument("--figsize", default="13,9")
    parser.add_argument("--labels", default=None)
    args = parser.parse_args()

    if args.output and len(args.graphs) > 1:
        parser.error("--output cannot be used with multiple input graphs")

    labels = {}
    if args.labels:
        raw = json.loads(Path(args.labels).read_text())
        labels = {
            k: (v.get("label") if isinstance(v, dict) else str(v))
            for k, v in raw.items()
            if (v.get("label") if isinstance(v, dict) else v)
        }

    for graph_path in args.graphs:
        out_path = args.output or str(Path(graph_path).with_suffix(".svg"))
        render(graph_path, out_path, args.top_edges, args.figsize, labels)


if __name__ == "__main__":
    main()
