from __future__ import annotations

import torch
from qwen_clt.training.losses import normalized_mse


@torch.no_grad()
def l0_by_layer(features_by_layer: list[torch.Tensor]) -> list[float]:
    out = []
    for a in features_by_layer:
        out.append(float((a > 0).float().sum(dim=-1).mean().item()))
    return out


@torch.no_grad()
def nmse_by_layer(mlp_recons: list[torch.Tensor], mlp_targets: list[torch.Tensor]) -> list[float]:
    return [float(normalized_mse(y_hat, y).item()) for y_hat, y in zip(mlp_recons, mlp_targets)]


@torch.no_grad()
def summarize_metrics(features_by_layer, mlp_recons, mlp_targets) -> dict:
    l0s = l0_by_layer(features_by_layer)
    nmses = nmse_by_layer(mlp_recons, mlp_targets)
    return {
        "l0_mean": sum(l0s) / len(l0s),
        "nmse_mean": sum(nmses) / len(nmses),
        "l0_by_layer": l0s,
        "nmse_by_layer": nmses,
    }
