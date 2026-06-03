#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import _path_setup  # noqa: F401

import torch
import torch.nn.functional as F

from qwen_clt.interventions import FeatureIntervention
from qwen_clt.attribution.validation import (
    feature_value,
    load_graph_payload,
    logit_difference_score,
    proxy_effect,
    select_top_node_indices,
    single_token_id,
)
from qwen_clt.models.proxy_replacement_model import (
    ProxyQwenReplacementModel,
)


def top_next_tokens(tokenizer, logits: torch.Tensor, k: int = 5) -> list[tuple[str, float]]:
    probs = F.softmax(logits[0, -1].float(), dim=-1)
    values, ids = torch.topk(probs, k=k)
    return [(tokenizer.decode([int(i)]), float(v)) for i, v in zip(ids, values)]


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object in {path}, got {type(payload)}")
    return payload


def select_node_from_graph(args) -> tuple[dict[str, Any], dict[str, Any]]:
    graph = load_graph_payload(args.graph)

    if args.node_index is not None:
        node_index = int(args.node_index)
    else:
        ranked = select_top_node_indices(
            graph,
            target_index=args.target_index,
            top_k=args.rank,
            min_activation=args.min_activation,
        )
        if len(ranked) < args.rank:
            raise ValueError(
                f"Graph has only {len(ranked)} selectable nodes after filtering; "
                f"cannot select rank={args.rank}."
            )
        node_index = ranked[args.rank - 1]

    node = graph["nodes"][node_index]
    source = {
        "kind": "graph",
        "path": args.graph,
        "node_index": node_index,
        "rank": args.rank,
        "proxy_effect": proxy_effect(
            graph,
            node_index=node_index,
            target_index=args.target_index,
        ),
        "target_index": args.target_index,
    }
    return node, source


def select_node_from_validation_report(args) -> tuple[dict[str, Any], dict[str, Any]]:
    report = load_json(args.validation_report)
    results = report.get("results", [])
    if not isinstance(results, list) or not results:
        raise ValueError(f"Validation report has no results: {args.validation_report}")

    select_by = args.select_by or "abs_causal_effect"
    if select_by == "rank":
        ordered = sorted(results, key=lambda item: int(item.get("rank", 10**9)))
    elif select_by == "abs_proxy_effect":
        ordered = sorted(
            results,
            key=lambda item: abs(float(item.get("proxy_effect", 0.0))),
            reverse=True,
        )
    elif select_by == "abs_causal_effect":
        ordered = sorted(
            results,
            key=lambda item: abs(float(item.get("causal_effect", 0.0))),
            reverse=True,
        )
    else:
        raise ValueError(f"Unsupported --select-by value: {select_by}")

    if len(ordered) < args.rank:
        raise ValueError(
            f"Validation report has only {len(ordered)} results; "
            f"cannot select rank={args.rank}."
        )

    selected = ordered[args.rank - 1]
    node = selected["node"]
    source = {
        "kind": "validation_report",
        "path": args.validation_report,
        "rank": args.rank,
        "select_by": select_by,
        "node_index": selected.get("node_index"),
        "proxy_effect": selected.get("proxy_effect"),
        "causal_effect": selected.get("causal_effect"),
        "feature_value_before": selected.get("feature_value_before"),
    }

    target = report.get("target", {})
    if args.positive is None and target.get("positive") is not None:
        args.positive = target["positive"]
    if args.negative is None and target.get("negative") is not None:
        args.negative = target["negative"]
    if args.target_pos is None and target.get("target_pos") is not None:
        args.target_pos = int(target["target_pos"])

    return node, source


def resolve_selected_feature(args) -> tuple[FeatureIntervention, dict[str, Any]]:
    if args.rank < 1:
        raise ValueError(f"--rank must be >= 1. Got {args.rank}.")
    if args.graph is not None and args.validation_report is not None:
        raise ValueError("Use either --graph or --validation-report, not both.")

    manual_values = [args.layer, args.pos, args.feature]
    if any(value is not None for value in manual_values):
        if not all(value is not None for value in manual_values):
            raise ValueError(
                "Manual selection requires all of --layer, --pos and --feature."
            )
        return (
            FeatureIntervention(
                layer=int(args.layer),
                pos=int(args.pos),
                feature_idx=int(args.feature),
                value=float(args.value),
            ),
            {
                "kind": "manual",
                "layer": int(args.layer),
                "pos": int(args.pos),
                "feature_idx": int(args.feature),
            },
        )

    if args.validation_report is not None:
        node, source = select_node_from_validation_report(args)
    elif args.graph is not None:
        node, source = select_node_from_graph(args)
    else:
        raise ValueError(
            "Provide either manual --layer/--pos/--feature, --graph, "
            "or --validation-report."
        )

    return (
        FeatureIntervention(
            layer=int(node["layer"]),
            pos=int(node["pos"]),
            feature_idx=int(node["feature_idx"]),
            value=float(args.value),
        ),
        {
            **source,
            "layer": int(node["layer"]),
            "pos": int(node["pos"]),
            "feature_idx": int(node["feature_idx"]),
            "graph_activation": float(node.get("activation", 0.0)),
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--pos", type=int, default=None)
    parser.add_argument("--feature", type=int, default=None)
    parser.add_argument("--graph", default=None)
    parser.add_argument("--validation-report", default=None)
    parser.add_argument("--node-index", type=int, default=None)
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--target-index", type=int, default=0)
    parser.add_argument("--min-activation", type=float, default=1e-8)
    parser.add_argument(
        "--select-by",
        choices=["rank", "abs_proxy_effect", "abs_causal_effect"],
        default=None,
    )
    parser.add_argument("--value", type=float, default=0.0)
    parser.add_argument("--positive", default=None)
    parser.add_argument("--negative", default=None)
    parser.add_argument("--target-pos", type=int, default=-1)
    args = parser.parse_args()

    intervention, selection = resolve_selected_feature(args)
    model = ProxyQwenReplacementModel.from_checkpoint(args.checkpoint)
    result = model.feature_intervention(
        args.prompt,
        [intervention],
    )

    delta = result["intervention_logit_delta"][0, -1].float()
    feature_before = feature_value(
        result["replacement_features"],
        layer=intervention.layer,
        pos=intervention.pos,
        feature_idx=intervention.feature_idx,
    )
    feature_after = feature_value(
        result["intervened_features"],
        layer=intervention.layer,
        pos=intervention.pos,
        feature_idx=intervention.feature_idx,
    )

    print("Intervention applied inside CLT replacement forward pass.")
    print("Selection source:", selection)
    print("Original logits shape:", tuple(result["logits"].shape))
    print("Replacement logits shape:", tuple(result["replacement_logits"].shape))
    print("Intervened logits shape:", tuple(result["intervened_logits"].shape))
    print("Number of CLT feature layers:", len(result["intervened_features"]))
    print("Selected feature value before:", feature_before)
    print("Selected feature value after:", feature_after)
    print(
        "Selected feature actual delta:",
        None if feature_before is None or feature_after is None else feature_after - feature_before,
    )
    print("Mean abs next-token logit delta:", float(delta.abs().mean().item()))
    print("Max abs next-token logit delta:", float(delta.abs().max().item()))

    if args.positive is not None and args.negative is not None:
        positive_id = single_token_id(model.tokenizer, args.positive)
        negative_id = single_token_id(model.tokenizer, args.negative)
        original_score = logit_difference_score(
            result["logits"],
            positive_token_id=positive_id,
            negative_token_id=negative_id,
            target_pos=args.target_pos,
        )
        replacement_score = logit_difference_score(
            result["replacement_logits"],
            positive_token_id=positive_id,
            negative_token_id=negative_id,
            target_pos=args.target_pos,
        )
        intervened_score = logit_difference_score(
            result["intervened_logits"],
            positive_token_id=positive_id,
            negative_token_id=negative_id,
            target_pos=args.target_pos,
        )
        print(
            f"Target logit diff original ({args.positive!r}-{args.negative!r}):",
            original_score,
        )
        print("Target logit diff replacement:", replacement_score)
        print("Target logit diff intervened:", intervened_score)
        print("Target causal effect:", intervened_score - replacement_score)

    print("Top original tokens:", top_next_tokens(model.tokenizer, result["logits"]))
    print("Top replacement tokens:", top_next_tokens(model.tokenizer, result["replacement_logits"]))
    print("Top intervened tokens:", top_next_tokens(model.tokenizer, result["intervened_logits"]))


if __name__ == "__main__":
    main()
