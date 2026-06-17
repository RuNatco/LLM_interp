from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass
class CustomTarget:
    name: str
    vector: torch.Tensor


def logit_difference_target(tokenizer, model, positive: str, negative: str) -> CustomTarget:
    pos_ids = tokenizer.encode(positive, add_special_tokens=False)
    neg_ids = tokenizer.encode(negative, add_special_tokens=False)
    if len(pos_ids) != 1 or len(neg_ids) != 1:
        raise ValueError("Use single-token strings for v0 logit difference targets.")
    unembed = model.lm_head.weight.detach()
    vec = unembed[pos_ids[0]] - unembed[neg_ids[0]]
    return CustomTarget(name=f"logit({positive})-logit({negative})", vector=vec)
