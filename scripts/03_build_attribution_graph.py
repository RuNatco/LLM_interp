#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import _path_setup  # noqa: F401

from qwen_clt.models.proxy_replacement_model import ProxyQwenReplacementModel
from qwen_clt.attribution.fidelity import (
    print_fidelity_report,
    replacement_fidelity_report,
)
from qwen_clt.attribution.targets import logit_difference_target
from qwen_clt.attribution.attribute import build_feature_to_target_graph
from qwen_clt.attribution.prune import prune_graph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", default="Demand is greater than generation, so the price will")
    parser.add_argument("--positive", default=" increase")
    parser.add_argument("--negative", default=" decrease")
    parser.add_argument("--output", default="outputs/graph.json")
    parser.add_argument("--max-feature-nodes", type=int, default=2048)
    parser.add_argument("--node-threshold", type=float, default=0.8)
    parser.add_argument("--replacement-metrics", default=None)
    parser.add_argument("--min-last-token-top1", type=float, default=0.3)
    parser.add_argument("--max-target-logit-diff-mae", type=float, default=2.0)
    parser.add_argument("--strict-fidelity", action="store_true")
    args = parser.parse_args()

    fidelity = replacement_fidelity_report(
        checkpoint_path=args.checkpoint,
        metrics_path=args.replacement_metrics,
        min_last_token_top1=args.min_last_token_top1,
        max_target_logit_diff_mae=args.max_target_logit_diff_mae,
    )
    print_fidelity_report(fidelity, strict=args.strict_fidelity)

    model = ProxyQwenReplacementModel.from_checkpoint(args.checkpoint)
    target = logit_difference_target(model.tokenizer, model.base_model, args.positive, args.negative)
    graph = build_feature_to_target_graph(
        replacement_model=model,
        prompt=args.prompt,
        targets=[target],
        max_feature_nodes=args.max_feature_nodes,
    )
    graph = prune_graph(graph, node_threshold=args.node_threshold)
    graph.metadata = graph.metadata or {}
    graph.metadata["replacement_fidelity"] = fidelity
    out = Path(args.output)
    graph.to_json(out)
    print(f"Saved proxy attribution graph to: {out}")
    print("Method:", graph.metadata.get("method") if graph.metadata else "unknown")
    print("Important: graph edges are hypotheses; validate with feature interventions.")


if __name__ == "__main__":
    main()
