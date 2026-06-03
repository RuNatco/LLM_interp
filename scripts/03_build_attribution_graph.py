#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from qwen_clt.models.proxy_replacement_model import ProxyQwenReplacementModel
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
    args = parser.parse_args()

    model = ProxyQwenReplacementModel.from_checkpoint(args.checkpoint)
    target = logit_difference_target(model.tokenizer, model.base_model, args.positive, args.negative)
    graph = build_feature_to_target_graph(
        replacement_model=model,
        prompt=args.prompt,
        targets=[target],
        max_feature_nodes=args.max_feature_nodes,
    )
    graph = prune_graph(graph, node_threshold=args.node_threshold)
    out = Path(args.output)
    graph.to_json(out)
    print(f"Saved proxy attribution graph to: {out}")
    print("Method:", graph.metadata.get("method") if graph.metadata else "unknown")
    print("Important: graph edges are hypotheses; validate with feature interventions.")


if __name__ == "__main__":
    main()
