#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import _path_setup  # noqa: F401

from qwen_clt.deep_trace import (
    load_deep_trace_payload,
    summarize_deep_trace_payload,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize a Deep Trace stage 1 graph JSON."
    )
    parser.add_argument("--graph", required=True)
    parser.add_argument("--top-n", type=int, default=8)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def print_edge(edge: dict[str, Any]) -> None:
    print(
        "  "
        f"{edge.get('kind')} "
        f"{edge.get('source')} -> {edge.get('target')} "
        f"score={_fmt(float(edge.get('score', 0.0)))}"
    )


def print_mlp_error_node(node: dict[str, Any]) -> None:
    metadata = node.get("metadata", {})
    print(
        "  "
        f"{node.get('id')} "
        f"layer={node.get('layer')} "
        f"pos={node.get('pos')} "
        f"error_norm={_fmt(float(node.get('value', 0.0)))} "
        "target_projection="
        f"{_fmt(float(metadata.get('direct_target_projection', 0.0)))}"
    )


def main() -> None:
    args = parse_args()

    payload = load_deep_trace_payload(args.graph)
    summary = summarize_deep_trace_payload(payload, top_n=args.top_n)

    if args.output is not None:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"Saved Deep Trace summary to: {output_path}")

    print("Deep Trace summary")
    print("=" * 80)
    print("Graph:", args.graph)
    print("Trace kind:", summary.get("trace_kind"))
    print("Checkpoint:", summary.get("checkpoint"))
    print("Prompt:", summary.get("prompt"))
    print("Nodes:", summary.get("num_nodes"))
    print("Edges:", summary.get("num_edges"))
    print("Node type counts:", summary.get("node_type_counts"))
    print("Edge kind counts:", summary.get("edge_kind_counts"))

    causal = summary.get("causal_pruning", {})
    print("Causal sign match rate:", causal.get("sign_match_rate"))

    cache_summary = summary.get("cache_summary", {})
    print("Mean MLP error norm:", cache_summary.get("mean_mlp_error_norm"))

    fidelity = summary.get("replacement_fidelity", {})
    if fidelity:
        print("\nReplacement fidelity:")
        for key, value in fidelity.get("metrics", {}).items():
            print(f"  {key}: {value}")
        for warning in fidelity.get("warnings", []):
            print(f"  warning: {warning}")

    print("\nTop MLP error nodes:")
    for node in summary.get("top_mlp_error_nodes", []):
        print_mlp_error_node(node)

    print("\nTop causal feature edges:")
    for edge in summary.get("top_causal_edges", []):
        print_edge(edge)

    print("\nTop direct feature edges:")
    for edge in summary.get("top_direct_feature_edges", []):
        print_edge(edge)

    print("\nTop replacement error edges:")
    for edge in summary.get("top_replacement_error_edges", []):
        print_edge(edge)


if __name__ == "__main__":
    main()
