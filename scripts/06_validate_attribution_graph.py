#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _path_setup  # noqa: F401

from tqdm import tqdm

from qwen_clt.attribution.validation import (
    ablation_matches_proxy_sign,
    feature_value,
    load_graph_payload,
    logit_difference_score,
    pearson_correlation,
    proxy_effect,
    select_top_node_indices,
    sign_match_rate,
    single_token_id,
)
from qwen_clt.interventions import FeatureIntervention
from qwen_clt.models.proxy_replacement_model import ProxyQwenReplacementModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate proxy attribution graph nodes with causal feature interventions."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--positive", default=" increase")
    parser.add_argument("--negative", default=" decrease")
    parser.add_argument("--target-index", type=int, default=0)
    parser.add_argument("--target-pos", type=int, default=-1)
    parser.add_argument("--top-k", type=int, default=24)
    parser.add_argument("--min-activation", type=float, default=1e-8)
    parser.add_argument("--value", type=float, default=0.0)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def default_output_path(graph_path: str | Path) -> Path:
    path = Path(graph_path)
    return path.with_name(f"{path.stem}_validated.json")


def main() -> None:
    args = parse_args()

    graph_payload = load_graph_payload(args.graph)
    node_indices = select_top_node_indices(
        graph_payload,
        target_index=args.target_index,
        top_k=args.top_k,
        min_activation=args.min_activation,
    )

    model = ProxyQwenReplacementModel.from_checkpoint(args.checkpoint)
    positive_id = single_token_id(model.tokenizer, args.positive)
    negative_id = single_token_id(model.tokenizer, args.negative)

    results = []
    for rank, node_index in tqdm(
        enumerate(node_indices, start=1),
        total=len(node_indices),
        desc="Validating graph nodes",
    ):
        node = graph_payload["nodes"][node_index]
        intervention = FeatureIntervention(
            layer=int(node["layer"]),
            pos=int(node["pos"]),
            feature_idx=int(node["feature_idx"]),
            value=float(args.value),
        )
        run = model.feature_intervention(args.prompt, [intervention])

        original_score = logit_difference_score(
            run["logits"],
            positive_token_id=positive_id,
            negative_token_id=negative_id,
            target_pos=args.target_pos,
        )
        replacement_score = logit_difference_score(
            run["replacement_logits"],
            positive_token_id=positive_id,
            negative_token_id=negative_id,
            target_pos=args.target_pos,
        )
        intervened_score = logit_difference_score(
            run["intervened_logits"],
            positive_token_id=positive_id,
            negative_token_id=negative_id,
            target_pos=args.target_pos,
        )

        causal_effect = intervened_score - replacement_score
        node_proxy_effect = proxy_effect(
            graph_payload,
            node_index=node_index,
            target_index=args.target_index,
        )
        replacement_feature_value = feature_value(
            run["replacement_features"],
            layer=int(node["layer"]),
            pos=int(node["pos"]),
            feature_idx=int(node["feature_idx"]),
        )
        intervened_feature_value = feature_value(
            run["intervened_features"],
            layer=int(node["layer"]),
            pos=int(node["pos"]),
            feature_idx=int(node["feature_idx"]),
        )
        actual_feature_delta = (
            None
            if replacement_feature_value is None or intervened_feature_value is None
            else intervened_feature_value - replacement_feature_value
        )
        graph_activation = float(node.get("activation", 0.0))
        proxy_write_coefficient = (
            None
            if abs(graph_activation) <= 1e-12
            else node_proxy_effect / graph_activation
        )
        proxy_causal_effect_estimate = (
            None
            if proxy_write_coefficient is None or actual_feature_delta is None
            else proxy_write_coefficient * actual_feature_delta
        )
        proxy_causal_abs_error = (
            None
            if proxy_causal_effect_estimate is None
            else abs(causal_effect - proxy_causal_effect_estimate)
        )
        sign_match = (
            ablation_matches_proxy_sign(node_proxy_effect, causal_effect)
            if float(args.value) == 0.0
            else None
        )

        results.append(
            {
                "rank": rank,
                "node_index": node_index,
                "node": node,
                "proxy_effect": node_proxy_effect,
                "abs_proxy_effect": abs(node_proxy_effect),
                "intervention": {
                    "kind": "set_feature_value",
                    "value": float(args.value),
                },
                "feature_value_before": replacement_feature_value,
                "feature_value_after": intervened_feature_value,
                "actual_feature_delta": actual_feature_delta,
                "proxy_write_coefficient": proxy_write_coefficient,
                "proxy_causal_effect_estimate": proxy_causal_effect_estimate,
                "original_target_score": original_score,
                "replacement_target_score": replacement_score,
                "intervened_target_score": intervened_score,
                "causal_effect": causal_effect,
                "abs_causal_effect": abs(causal_effect),
                "proxy_causal_abs_error": proxy_causal_abs_error,
                "ablation_matches_proxy_sign": sign_match,
            }
        )

    output_path = Path(args.output) if args.output else default_output_path(args.graph)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    comparable = [
        item
        for item in results
        if item["proxy_causal_effect_estimate"] is not None
    ]
    proxy_estimates = [
        float(item["proxy_causal_effect_estimate"])
        for item in comparable
    ]
    causal_effects = [float(item["causal_effect"]) for item in comparable]
    abs_errors = [
        float(item["proxy_causal_abs_error"])
        for item in comparable
        if item["proxy_causal_abs_error"] is not None
    ]
    sign_matches = [
        item["ablation_matches_proxy_sign"]
        for item in results
    ]

    payload = {
        "source_graph": str(Path(args.graph)),
        "checkpoint": str(Path(args.checkpoint)),
        "prompt": args.prompt,
        "target": {
            "positive": args.positive,
            "negative": args.negative,
            "positive_token_id": positive_id,
            "negative_token_id": negative_id,
            "target_index": args.target_index,
            "target_pos": args.target_pos,
            "score": "logit(positive)-logit(negative)",
        },
        "validation": {
            "kind": "causal_feature_intervention",
            "baseline_logits": "replacement_logits",
            "effect_definition": (
                "logit_diff(intervened_logits) - logit_diff(replacement_logits)"
            ),
            "top_k": args.top_k,
            "selected_nodes": len(results),
            "min_activation": float(args.min_activation),
            "intervention_value": float(args.value),
        },
        "summary": {
            "comparable_nodes": len(comparable),
            "proxy_causal_pearson": pearson_correlation(
                proxy_estimates,
                causal_effects,
            ),
            "ablation_sign_match_rate": sign_match_rate(sign_matches),
            "mean_abs_proxy_causal_error": (
                None
                if not abs_errors
                else sum(abs_errors) / len(abs_errors)
            ),
            "mean_abs_causal_effect": (
                None
                if not causal_effects
                else sum(abs(value) for value in causal_effects) / len(causal_effects)
            ),
        },
        "graph_metadata": graph_payload.get("metadata"),
        "results": results,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Saved causal validation report to: {output_path}")
    print(f"Validated nodes: {len(results)}")
    print("Comparable nodes:", payload["summary"]["comparable_nodes"])
    print("Proxy/causal Pearson:", payload["summary"]["proxy_causal_pearson"])
    print("Ablation sign match rate:", payload["summary"]["ablation_sign_match_rate"])


if __name__ == "__main__":
    main()
