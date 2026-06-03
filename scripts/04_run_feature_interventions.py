#!/usr/bin/env python
from __future__ import annotations

import argparse

import _path_setup  # noqa: F401

import torch
import torch.nn.functional as F

from qwen_clt.interventions import FeatureIntervention
from qwen_clt.attribution.validation import (
    feature_value,
    logit_difference_score,
    single_token_id,
)
from qwen_clt.models.proxy_replacement_model import (
    ProxyQwenReplacementModel,
)


def top_next_tokens(tokenizer, logits: torch.Tensor, k: int = 5) -> list[tuple[str, float]]:
    probs = F.softmax(logits[0, -1].float(), dim=-1)
    values, ids = torch.topk(probs, k=k)
    return [(tokenizer.decode([int(i)]), float(v)) for i, v in zip(ids, values)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--pos", type=int, required=True)
    parser.add_argument("--feature", type=int, required=True)
    parser.add_argument("--value", type=float, default=0.0)
    parser.add_argument("--positive", default=None)
    parser.add_argument("--negative", default=None)
    parser.add_argument("--target-pos", type=int, default=-1)
    args = parser.parse_args()

    model = ProxyQwenReplacementModel.from_checkpoint(args.checkpoint)
    result = model.feature_intervention(
        args.prompt,
        [FeatureIntervention(layer=args.layer, pos=args.pos, feature_idx=args.feature, value=args.value)],
    )

    delta = result["intervention_logit_delta"][0, -1].float()
    feature_before = feature_value(
        result["replacement_features"],
        layer=args.layer,
        pos=args.pos,
        feature_idx=args.feature,
    )
    feature_after = feature_value(
        result["intervened_features"],
        layer=args.layer,
        pos=args.pos,
        feature_idx=args.feature,
    )

    print("Intervention applied inside CLT replacement forward pass.")
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
