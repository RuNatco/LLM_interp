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
) -> torch.Tensor:
    losses = [normalized_mse(y_hat, y) for y_hat, y in zip(mlp_recons, mlp_targets)]
    return torch.stack(losses).mean()


def build_layer_sparsity_weights(
    n_layers: int,
    *,
    mode: str = "linear",
    min_weight: float = 1.0,
    max_weight: float = 4.0,
) -> list[float]:
    if n_layers <= 1:
        return [1.0] * n_layers
    if mode == "uniform":
        return [1.0] * n_layers

    weights = []
    for i in range(n_layers):
        t = i / (n_layers - 1)
        if mode == "quadratic":
            t = t ** 2
        weights.append(min_weight + (max_weight - min_weight) * t)
    return weights


def tanh_sparsity_loss(
    features_by_layer: list[torch.Tensor],
    clt: CrossLayerTranscoder,
    c: float = 1.0,
    layer_weights: list[float] | None = None,
) -> torch.Tensor:
    n_layers = len(features_by_layer)
    if layer_weights is None:
        layer_weights = [1.0] * n_layers
    if len(layer_weights) != n_layers:
        raise ValueError(
            f"layer_weights length {len(layer_weights)} != n_layers {n_layers}."
        )

    penalties = []
    for layer_idx, a in enumerate(features_by_layer):
        dec_norm = clt.decoder_feature_norms(layer_idx).to(a.device, dtype=a.dtype)
        penalty = torch.tanh(float(c) * a.abs() * dec_norm.view(1, 1, -1))
        penalties.append(penalty.mean() * float(layer_weights[layer_idx]))
    return torch.stack(penalties).mean()
