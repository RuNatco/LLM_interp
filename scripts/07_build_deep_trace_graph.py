#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import _path_setup  # noqa: F401

from qwen_clt.attribution.fidelity import (
    print_fidelity_report,
    replacement_fidelity_report,
)
from qwen_clt.deep_trace import (
    build_deep_trace_graph,
    save_deep_trace_cache,
)
from qwen_clt.models.proxy_replacement_model import ProxyQwenReplacementModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build Deep Trace stage 1 typed graph from a Qwen CLT checkpoint."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--prompt",
        default="Demand is greater than generation, so the price will",
    )
    parser.add_argument("--positive", default=" increase")
    parser.add_argument("--negative", default=" decrease")
    parser.add_argument("--target-pos", type=int, default=-1)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-output", default=None)
    parser.add_argument("--max-feature-nodes", type=int, default=64)
    parser.add_argument("--top-error-nodes", type=int, default=8)
    parser.add_argument("--causal-top-k", type=int, default=8)
    parser.add_argument("--min-activation", type=float, default=0.0)
    parser.add_argument("--feature-residual-edges-per-node", type=int, default=3)
    parser.add_argument("--replacement-metrics", default=None)
    parser.add_argument("--min-last-token-top1", type=float, default=0.3)
    parser.add_argument("--max-target-logit-diff-mae", type=float, default=2.0)
    parser.add_argument("--strict-fidelity", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    fidelity = replacement_fidelity_report(
        checkpoint_path=args.checkpoint,
        metrics_path=args.replacement_metrics,
        min_last_token_top1=args.min_last_token_top1,
        max_target_logit_diff_mae=args.max_target_logit_diff_mae,
    )
    print_fidelity_report(fidelity, strict=args.strict_fidelity)

    model = ProxyQwenReplacementModel.from_checkpoint(args.checkpoint)
    graph, cache = build_deep_trace_graph(
        replacement_model=model,
        prompt=args.prompt,
        positive=args.positive,
        negative=args.negative,
        target_pos=args.target_pos,
        max_feature_nodes=args.max_feature_nodes,
        top_error_nodes=args.top_error_nodes,
        causal_top_k=args.causal_top_k,
        min_activation=args.min_activation,
        feature_residual_edges_per_node=args.feature_residual_edges_per_node,
    )
    graph.metadata["replacement_fidelity"] = fidelity
    graph.metadata["checkpoint"] = str(Path(args.checkpoint))

    output_path = Path(args.output)
    graph.to_json(output_path)

    if args.cache_output is not None:
        cache_path = Path(args.cache_output)
        save_deep_trace_cache(cache, cache_path)
        graph.metadata["cache_path"] = str(cache_path)
        graph.to_json(output_path)
        print(f"Saved Deep Trace cache to: {cache_path}")

    print(f"Saved Deep Trace graph to: {output_path}")
    print("Trace kind:", graph.metadata.get("trace_kind"))
    print("Nodes:", len(graph.nodes))
    print("Edges:", len(graph.edges))
    print("Node types:", ", ".join(graph.metadata.get("node_types", [])))
    print(
        "Causal sign match rate:",
        graph.metadata.get("causal_pruning", {}).get("sign_match_rate"),
    )
    print(
        "Important: Deep Trace stage 1 is deeper than proxy attribution, "
        "but still not full path attribution."
    )


if __name__ == "__main__":
    main()
