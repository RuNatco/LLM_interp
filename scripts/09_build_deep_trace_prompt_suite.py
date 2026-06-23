#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import _path_setup  # noqa: F401

from tqdm import tqdm

from qwen_clt.attribution.fidelity import (
    print_fidelity_report,
    replacement_fidelity_report,
)
from qwen_clt.deep_trace import (
    build_deep_trace_graph,
    save_deep_trace_cache,
    summarize_deep_trace_payload,
)
from qwen_clt.models.proxy_replacement_model import ProxyQwenReplacementModel


DEFAULT_PROMPTS = [
    "Demand is greater than generation, so the price will",
    "Demand is lower than supply, so the price will",
    "Fuel supply falls while demand stays high, so the price will",
    "Generation exceeds demand, so the price will",
    "A product becomes scarce while buyers still need it, so the price will",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Deep Trace stage 2 graphs for a prompt suite."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", action="append", default=None)
    parser.add_argument(
        "--prompts-file",
        default=None,
        help="JSON list of prompt strings. If omitted, uses a small default suite.",
    )
    parser.add_argument("--positive", default=" increase")
    parser.add_argument("--negative", default=" decrease")
    parser.add_argument("--target-pos", type=int, default=-1)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--summary-output", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--max-feature-nodes", type=int, default=64)
    parser.add_argument("--top-error-nodes", type=int, default=8)
    parser.add_argument("--causal-top-k", type=int, default=8)
    parser.add_argument("--min-activation", type=float, default=0.0)
    parser.add_argument("--feature-residual-edges-per-node", type=int, default=3)
    parser.add_argument("--node-threshold", type=float, default=1.0)
    parser.add_argument("--replacement-metrics", default=None)
    parser.add_argument("--min-last-token-top1", type=float, default=0.3)
    parser.add_argument("--max-target-logit-diff-mae", type=float, default=2.0)
    parser.add_argument("--strict-fidelity", action="store_true")
    return parser.parse_args()


def load_prompts(args: argparse.Namespace) -> list[str]:
    prompts: list[str] = []

    if args.prompts_file is not None:
        path = Path(args.prompts_file)
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, list) or not all(
            isinstance(item, str) for item in payload
        ):
            raise TypeError(
                "--prompts-file must contain a JSON list of prompt strings."
            )
        prompts.extend(payload)

    if args.prompt is not None:
        prompts.extend(args.prompt)

    if not prompts:
        prompts = list(DEFAULT_PROMPTS)

    return prompts


def prompt_slug(prompt: str, idx: int) -> str:
    slug = prompt.lower()
    slug = re.sub(r"[^a-z0-9]+", "_", slug).strip("_")
    slug = slug[:64].strip("_")
    if not slug:
        slug = "prompt"
    return f"{idx:02d}_{slug}"


def main() -> None:
    args = parse_args()
    prompts = load_prompts(args)

    fidelity = replacement_fidelity_report(
        checkpoint_path=args.checkpoint,
        metrics_path=args.replacement_metrics,
        min_last_token_top1=args.min_last_token_top1,
        max_target_logit_diff_mae=args.max_target_logit_diff_mae,
    )
    print_fidelity_report(fidelity, strict=args.strict_fidelity)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir is not None else None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    model = ProxyQwenReplacementModel.from_checkpoint(args.checkpoint)

    results: list[dict[str, Any]] = []
    for idx, prompt in tqdm(
        list(enumerate(prompts, start=1)),
        desc="Building Deep Trace prompt suite",
    ):
        slug = prompt_slug(prompt, idx)
        graph_path = output_dir / f"{slug}_deep_trace.json"
        cache_path = None if cache_dir is None else cache_dir / f"{slug}_cache.pt"

        graph, cache = build_deep_trace_graph(
            replacement_model=model,
            prompt=prompt,
            positive=args.positive,
            negative=args.negative,
            target_pos=args.target_pos,
            max_feature_nodes=args.max_feature_nodes,
            top_error_nodes=args.top_error_nodes,
            causal_top_k=args.causal_top_k,
            min_activation=args.min_activation,
            feature_residual_edges_per_node=args.feature_residual_edges_per_node,
            node_threshold=args.node_threshold,
        )
        graph.metadata["replacement_fidelity"] = fidelity
        graph.metadata["checkpoint"] = str(Path(args.checkpoint))
        if cache_path is not None:
            save_deep_trace_cache(cache, cache_path)
            graph.metadata["cache_path"] = str(cache_path)

        graph.to_json(graph_path)
        summary = summarize_deep_trace_payload(graph.to_payload(), top_n=3)
        results.append(
            {
                "idx": idx,
                "prompt": prompt,
                "graph_path": str(graph_path),
                "cache_path": None if cache_path is None else str(cache_path),
                "num_nodes": summary["num_nodes"],
                "num_edges": summary["num_edges"],
                "causal_sign_match_rate": summary.get(
                    "causal_pruning",
                    {},
                ).get("sign_match_rate"),
                "mean_mlp_error_norm": summary.get(
                    "cache_summary",
                    {},
                ).get("mean_mlp_error_norm"),
                "top_mlp_error_nodes": summary["top_mlp_error_nodes"],
                "top_causal_edges": summary["top_causal_edges"],
                "top_causal_residual_delta_edges": summary[
                    "top_causal_residual_delta_edges"
                ],
            }
        )

    suite_summary = {
        "kind": "deep_trace_stage2_prompt_suite",
        "checkpoint": str(Path(args.checkpoint)),
        "positive": args.positive,
        "negative": args.negative,
        "target_pos": args.target_pos,
        "replacement_fidelity": fidelity,
        "num_prompts": len(prompts),
        "results": results,
    }

    summary_output = (
        Path(args.summary_output)
        if args.summary_output is not None
        else output_dir / "deep_trace_suite_summary.json"
    )
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    with summary_output.open("w", encoding="utf-8") as f:
        json.dump(suite_summary, f, ensure_ascii=False, indent=2)

    print(f"Saved Deep Trace prompt suite summary to: {summary_output}")
    for item in results:
        print(
            f"{item['idx']}. sign_match={item['causal_sign_match_rate']} "
            f"mean_error_norm={item['mean_mlp_error_norm']} "
            f"graph={item['graph_path']}"
        )


if __name__ == "__main__":
    main()
