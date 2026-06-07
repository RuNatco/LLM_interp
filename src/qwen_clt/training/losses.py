from __future__ import annotations

import torch
import torch.nn.functional as F
from qwen_clt.models.cross_layer_transcoder import CrossLayerTranscoder


def normalized_mse(y_hat: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mse = F.mse_loss(y_hat.float(), y.float())
    centered = y.float() - y.float().mean(dim=(0, 1), keepdim=True)
    denom = centered.pow(2).mean().clamp_min(eps)
    return mse / denom


def reconstruction_loss(
    mlp_recons: list[torch.Tensor],
    mlp_targets: list[torch.Tensor],
    layer_weights: list[float] | torch.Tensor | None = None,
) -> torch.Tensor:
    losses = [normalized_mse(y_hat, y) for y_hat, y in zip(mlp_recons, mlp_targets)]
    loss_tensor = torch.stack(losses)

    if layer_weights is None:
        return loss_tensor.mean()

    weights = torch.as_tensor(
        layer_weights,
        dtype=loss_tensor.dtype,
        device=loss_tensor.device,
    )
    if weights.shape != loss_tensor.shape:
        raise ValueError(
            f"layer_weights shape {weights.shape} must match layer losses "
            f"shape {loss_tensor.shape}."
        )
    if bool((weights < 0).any()):
        raise ValueError("layer_weights must be non-negative.")
    if float(weights.sum().item()) <= 0.0:
        raise ValueError("At least one layer weight must be positive.")

    return (loss_tensor * weights).sum() / weights.sum()


def _last_token_positions(
    attention_mask: torch.Tensor | None,
    logits: torch.Tensor,
) -> torch.Tensor:
    if attention_mask is None:
        return torch.full(
            (logits.shape[0],),
            logits.shape[1] - 1,
            dtype=torch.long,
            device=logits.device,
        )

    mask = attention_mask.to(device=logits.device).bool()
    if mask.shape != logits.shape[:2]:
        raise ValueError(
            f"attention_mask shape {mask.shape} does not match logits shape "
            f"{logits.shape[:2]}."
        )
    if bool((mask.long().sum(dim=1) <= 0).any()):
        raise ValueError("Every sequence must have at least one active token.")

    positions = torch.arange(mask.shape[1], device=mask.device)
    positions = positions.view(1, -1).expand_as(mask)
    return positions.masked_fill(~mask, -1).max(dim=1).values


def _last_token_logits(
    logits: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    positions = _last_token_positions(attention_mask, logits)
    batch_idx = torch.arange(logits.shape[0], device=logits.device)
    return logits[batch_idx, positions]


def _center_scale_logits(logits: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    logits = logits.float()
    centered = logits - logits.mean(dim=-1, keepdim=True)
    scale = centered.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(eps)
    return centered / scale


def last_token_logit_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    top_k: int | None = 256,
    temperature: float = 1.0,
    normalize_logits: bool = True,
    loss_type: str = "centered_mse",
) -> torch.Tensor:
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            f"Shape mismatch: student={student_logits.shape}, "
            f"teacher={teacher_logits.shape}"
        )
    if student_logits.ndim != 3:
        raise ValueError(
            "Expected logits with shape [batch, seq, vocab], "
            f"got {student_logits.shape}."
        )

    if temperature <= 0:
        raise ValueError(f"temperature must be positive. Got {temperature}.")

    student_last = _last_token_logits(student_logits, attention_mask)
    teacher_last = _last_token_logits(
        teacher_logits.detach().to(student_logits.device),
        attention_mask,
    )

    if top_k is not None and int(top_k) > 0 and int(top_k) < teacher_last.shape[-1]:
        token_ids = teacher_last.float().topk(int(top_k), dim=-1).indices
        student_last = student_last.gather(dim=-1, index=token_ids)
        teacher_last = teacher_last.gather(dim=-1, index=token_ids)

    loss_type = loss_type.lower()
    if loss_type == "centered_mse":
        normalize_logits = True
        loss_type = "mse"

    student_scores = student_last.float() / float(temperature)
    teacher_scores = teacher_last.float() / float(temperature)

    if normalize_logits:
        student_scores = _center_scale_logits(student_scores)
        teacher_scores = _center_scale_logits(teacher_scores)

    if loss_type == "mse":
        return F.mse_loss(student_scores, teacher_scores)

    if loss_type == "kl":
        student_log_probs = F.log_softmax(student_scores, dim=-1)
        teacher_probs = F.softmax(teacher_scores, dim=-1)
        return (
            F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
            * float(temperature) ** 2
        )

    raise ValueError(
        "Unsupported logit distillation loss_type. "
        "Expected 'centered_mse', 'mse', or 'kl'. "
        f"Got {loss_type!r}."
    )


def tanh_sparsity_loss(
    features_by_layer: list[torch.Tensor],
    clt: CrossLayerTranscoder,
    c: float = 1.0,
) -> torch.Tensor:
    penalties = []
    for layer_idx, a in enumerate(features_by_layer):
        dec_norm = clt.decoder_feature_norms(layer_idx).to(a.device, dtype=a.dtype)
        penalty = torch.tanh(float(c) * a.abs() * dec_norm.view(1, 1, -1))
        penalties.append(penalty.mean())
    return torch.stack(penalties).mean()
