#!/usr/bin/env python
from __future__ import annotations

import argparse
import torch
import torch.nn.functional as F

from qwen_clt.interventions import FeatureIntervention
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
    args = parser.parse_args()

    model = ProxyQwenReplacementModel.from_checkpoint(args.checkpoint)
    result = model.feature_intervention(
        args.prompt,
        [FeatureIntervention(layer=args.layer, pos=args.pos, feature_idx=args.feature, value=args.value)],
    )

    delta = result["intervention_logit_delta"][0, -1].float()

    print("Intervention applied inside CLT replacement forward pass.")
    print("Original logits shape:", tuple(result["logits"].shape))
    print("Replacement logits shape:", tuple(result["replacement_logits"].shape))
    print("Intervened logits shape:", tuple(result["intervened_logits"].shape))
    print("Number of CLT feature layers:", len(result["intervened_features"]))
    print("Mean abs next-token logit delta:", float(delta.abs().mean().item()))
    print("Max abs next-token logit delta:", float(delta.abs().max().item()))
    print("Top original tokens:", top_next_tokens(model.tokenizer, result["logits"]))
    print("Top replacement tokens:", top_next_tokens(model.tokenizer, result["replacement_logits"]))
    print("Top intervened tokens:", top_next_tokens(model.tokenizer, result["intervened_logits"]))


if __name__ == "__main__":
    main()
