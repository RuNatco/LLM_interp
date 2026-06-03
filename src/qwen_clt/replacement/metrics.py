from __future__ import annotations

from dataclasses import dataclass
from dataclasses import fields

import torch


@dataclass
class ReplacementMetrics:
    num_tokens: int
    num_sequences: int
    kl_div: float
    logit_mse: float
    top1_agreement: float
    mean_abs_logit_diff: float
    last_token_kl_div: float
    last_token_logit_mse: float
    last_token_top1_agreement: float
    last_token_mean_abs_logit_diff: float
    target_logit_diff_original_mean: float | None = None
    target_logit_diff_replacement_mean: float | None = None
    target_logit_diff_mae: float | None = None
    target_logit_diff_mse: float | None = None


def _validate_logits(
    original_logits: torch.Tensor,
    replacement_logits: torch.Tensor,
) -> None:
    if original_logits.shape != replacement_logits.shape:
        raise ValueError(
            f"Shape mismatch: original={original_logits.shape}, "
            f"replacement={replacement_logits.shape}"
        )
    if original_logits.ndim != 3:
        raise ValueError(
            "Expected logits with shape [batch, seq, vocab], "
            f"got {original_logits.shape}."
        )


def _attention_mask(
    attention_mask: torch.Tensor | None,
    logits: torch.Tensor,
) -> torch.Tensor:
    if attention_mask is None:
        return torch.ones(
            logits.shape[:2],
            dtype=torch.bool,
            device=logits.device,
        )

    if attention_mask.shape != logits.shape[:2]:
        raise ValueError(
            f"attention_mask shape {attention_mask.shape} does not match "
            f"logits batch/seq shape {logits.shape[:2]}."
        )

    mask = attention_mask.to(device=logits.device).bool()
    if not bool(mask.any()):
        raise ValueError("attention_mask has no active tokens.")
    return mask


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    return float(values[mask].float().mean().item())


def _last_token_positions(mask: torch.Tensor) -> torch.Tensor:
    if bool((mask.long().sum(dim=1) <= 0).any()):
        raise ValueError("Every sequence must have at least one active token.")

    positions = torch.arange(mask.shape[1], device=mask.device)
    positions = positions.view(1, -1).expand_as(mask)
    return positions.masked_fill(~mask, -1).max(dim=1).values


def _resolve_target_positions(mask: torch.Tensor, target_pos: int) -> torch.Tensor:
    if target_pos < 0:
        resolved = []
        for row in mask:
            active_positions = torch.nonzero(row, as_tuple=False).flatten()
            if abs(target_pos) > active_positions.numel():
                raise IndexError(
                    f"target_pos={target_pos} is outside at least one sequence."
                )
            resolved.append(active_positions[int(target_pos)])

        return torch.stack(resolved)

    positions = torch.full(
        (mask.shape[0],),
        int(target_pos),
        dtype=torch.long,
        device=mask.device,
    )

    if bool((positions >= mask.shape[1]).any()):
        raise IndexError(
            f"target_pos={target_pos} is outside at least one sequence."
        )

    return positions


def _indexed_logits(
    logits: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    batch_idx = torch.arange(logits.shape[0], device=logits.device)
    return logits[batch_idx, positions].float()


def _logit_difference(
    logits: torch.Tensor,
    positions: torch.Tensor,
    positive_token_id: int,
    negative_token_id: int,
) -> torch.Tensor:
    selected = _indexed_logits(logits, positions)
    return selected[:, positive_token_id] - selected[:, negative_token_id]


@torch.no_grad()
def compute_replacement_metrics(
    original_logits: torch.Tensor,
    replacement_logits: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    target_token_ids: tuple[int, int] | None = None,
    target_pos: int = -1,
) -> ReplacementMetrics:
    _validate_logits(original_logits, replacement_logits)

    original_logits = original_logits.float()
    replacement_logits = replacement_logits.float()
    mask = _attention_mask(attention_mask, original_logits)

    original_log_probs = torch.log_softmax(original_logits, dim=-1)
    replacement_log_probs = torch.log_softmax(replacement_logits, dim=-1)
    original_probs = original_log_probs.exp()

    kl_per_token = (
        original_probs * (original_log_probs - replacement_log_probs)
    ).sum(dim=-1)
    logit_mse_per_token = (
        (replacement_logits - original_logits).pow(2).mean(dim=-1)
    )
    mean_abs_logit_diff_per_token = (
        (original_logits - replacement_logits).abs().mean(dim=-1)
    )

    original_top1 = original_logits.argmax(dim=-1)
    replacement_top1 = replacement_logits.argmax(dim=-1)
    top1_per_token = (original_top1 == replacement_top1).float()

    last_positions = _last_token_positions(mask)
    original_last = _indexed_logits(original_logits, last_positions)
    replacement_last = _indexed_logits(replacement_logits, last_positions)

    original_last_log_probs = torch.log_softmax(original_last, dim=-1)
    replacement_last_log_probs = torch.log_softmax(replacement_last, dim=-1)
    original_last_probs = original_last_log_probs.exp()

    last_kl = (
        original_last_probs
        * (original_last_log_probs - replacement_last_log_probs)
    ).sum(dim=-1)
    last_logit_mse = (replacement_last - original_last).pow(2).mean(dim=-1)
    last_mean_abs_logit_diff = (replacement_last - original_last).abs().mean(dim=-1)
    last_top1 = (
        original_last.argmax(dim=-1) == replacement_last.argmax(dim=-1)
    ).float()

    target_original_mean = None
    target_replacement_mean = None
    target_mae = None
    target_mse = None

    if target_token_ids is not None:
        positive_token_id, negative_token_id = target_token_ids
        target_positions = _resolve_target_positions(mask, target_pos)
        original_target = _logit_difference(
            original_logits,
            target_positions,
            positive_token_id=positive_token_id,
            negative_token_id=negative_token_id,
        )
        replacement_target = _logit_difference(
            replacement_logits,
            target_positions,
            positive_token_id=positive_token_id,
            negative_token_id=negative_token_id,
        )
        target_delta = replacement_target - original_target

        target_original_mean = float(original_target.mean().item())
        target_replacement_mean = float(replacement_target.mean().item())
        target_mae = float(target_delta.abs().mean().item())
        target_mse = float(target_delta.pow(2).mean().item())

    return ReplacementMetrics(
        num_tokens=int(mask.sum().item()),
        num_sequences=int(mask.shape[0]),
        kl_div=_masked_mean(kl_per_token, mask),
        logit_mse=_masked_mean(logit_mse_per_token, mask),
        top1_agreement=_masked_mean(top1_per_token, mask),
        mean_abs_logit_diff=_masked_mean(mean_abs_logit_diff_per_token, mask),
        last_token_kl_div=float(last_kl.mean().item()),
        last_token_logit_mse=float(last_logit_mse.mean().item()),
        last_token_top1_agreement=float(last_top1.mean().item()),
        last_token_mean_abs_logit_diff=float(last_mean_abs_logit_diff.mean().item()),
        target_logit_diff_original_mean=target_original_mean,
        target_logit_diff_replacement_mean=target_replacement_mean,
        target_logit_diff_mae=target_mae,
        target_logit_diff_mse=target_mse,
    )


def metrics_to_dict(metrics: ReplacementMetrics) -> dict[str, float | int]:
    result = {}
    for field in fields(metrics):
        value = getattr(metrics, field.name)
        if value is not None:
            result[field.name] = value
    return result
