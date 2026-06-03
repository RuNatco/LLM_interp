from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class ReplacementMetrics:
    kl_div: float
    logit_mse: float
    top1_agreement: float
    mean_abs_logit_diff: float


@torch.no_grad()
def compute_replacement_metrics(
    original_logits: torch.Tensor,
    replacement_logits: torch.Tensor,
) -> ReplacementMetrics:
    if original_logits.shape != replacement_logits.shape:
        raise ValueError(
            f"Shape mismatch: original={original_logits.shape}, "
            f"replacement={replacement_logits.shape}"
        )

    original_log_probs = F.log_softmax(original_logits, dim=-1)
    replacement_log_probs = F.log_softmax(replacement_logits, dim=-1)
    original_probs = original_log_probs.exp()

    kl_div = F.kl_div(
        replacement_log_probs,
        original_probs,
        reduction="batchmean",
        log_target=False,
    )

    logit_mse = F.mse_loss(
        replacement_logits,
        original_logits,
    )

    original_top1 = original_logits.argmax(dim=-1)
    replacement_top1 = replacement_logits.argmax(dim=-1)

    top1_agreement = (original_top1 == replacement_top1).float().mean()

    mean_abs_logit_diff = (
        original_logits - replacement_logits
    ).abs().mean()

    return ReplacementMetrics(
        kl_div=float(kl_div.item()),
        logit_mse=float(logit_mse.item()),
        top1_agreement=float(top1_agreement.item()),
        mean_abs_logit_diff=float(mean_abs_logit_diff.item()),
    )


def metrics_to_dict(metrics: ReplacementMetrics) -> dict[str, float]:
    return {
        "kl_div": metrics.kl_div,
        "logit_mse": metrics.logit_mse,
        "top1_agreement": metrics.top1_agreement,
        "mean_abs_logit_diff": metrics.mean_abs_logit_diff,
    }
