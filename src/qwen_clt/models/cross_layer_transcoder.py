from __future__ import annotations

import math
import torch
import torch.nn as nn


def clt_normalization_kwargs(clt_cfg: dict) -> dict:
    normalization_cfg = clt_cfg.get("normalization", {}) or {}
    enabled = bool(normalization_cfg.get("enabled", False))

    return {
        "normalize_inputs": bool(
            normalization_cfg.get("normalize_inputs", enabled)
        ),
        "normalize_targets": bool(
            normalization_cfg.get("normalize_targets", enabled)
        ),
        "normalization_eps": float(normalization_cfg.get("eps", 1e-5)),
        "normalization_momentum": float(
            normalization_cfg.get("momentum", 0.01)
        ),
    }


def clt_jumprelu_kwargs(clt_cfg: dict) -> dict:
    """Read JumpReLU hyperparameters (init threshold + STE bandwidth) from config.

    Supports either a nested `clt.jumprelu` block or a top-level `init_threshold`
    for backward compatibility.
    """
    jumprelu_cfg = clt_cfg.get("jumprelu", {}) or {}
    init_threshold = float(
        jumprelu_cfg.get("init_threshold", clt_cfg.get("init_threshold", 0.05))
    )
    return {
        "init_threshold": init_threshold,
        "jumprelu_bandwidth": float(jumprelu_cfg.get("bandwidth", 0.05)),
    }


class _JumpReLU(torch.autograd.Function):
    """JumpReLU activation with a straight-through estimator for the threshold.

    Forward:  out = z * H(z - theta)          (hard gate, theta > 0)

    Backward (STE, rectangular kernel K of width = bandwidth, centered at z=theta):
      d out / d z      = H(z - theta)                       (ReLU-style gate)
      d out / d theta  = -(theta / bandwidth) * K((z - theta)/bandwidth)

    The threshold thus receives gradient from any loss that flows through the
    activation magnitude: reconstruction pulls theta down (admit features),
    the sparsity penalty pulls it up (drop near-boundary features). Without this
    the threshold is a dead parameter and JumpReLU degenerates into ReLU.
    Reference: Rajamanoharan et al., 2024 (JumpReLU SAEs).
    """

    @staticmethod
    def forward(ctx, z: torch.Tensor, threshold: torch.Tensor, bandwidth: float):
        gate = z > threshold
        ctx.save_for_backward(z, threshold, gate)
        ctx.bandwidth = float(bandwidth)
        return z * gate.to(z.dtype)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        z, threshold, gate = ctx.saved_tensors
        eps = ctx.bandwidth

        grad_z = grad_out * gate.to(grad_out.dtype)

        within = ((z - threshold).abs() <= (0.5 * eps)).to(grad_out.dtype)
        grad_threshold = grad_out * (-(threshold) / eps) * within

        # Reduce over broadcast (non-feature) dims back to `threshold`'s shape.
        while grad_threshold.dim() > threshold.dim():
            grad_threshold = grad_threshold.sum(dim=0)
        for i, (g_dim, t_dim) in enumerate(zip(grad_threshold.shape, threshold.shape)):
            if t_dim == 1 and g_dim != 1:
                grad_threshold = grad_threshold.sum(dim=i, keepdim=True)

        return grad_z, grad_threshold, None


class CrossLayerTranscoder(nn.Module):
    """Small-scale dense Cross-Layer Transcoder.

    Feature layer src reads MLP input at src and writes to MLP outputs at tgt >= src.
    This v0 implementation is intentionally simple and dense. For larger runs,
    replace triangular decoder loops with chunked/sparse kernels.
    """

    def __init__(
        self,
        n_layers: int,
        d_model: int,
        features_per_layer: int,
        init_threshold: float = 0.05,
        jumprelu_bandwidth: float = 0.05,
        decoder_init_scale: float = 0.02,
        normalize_inputs: bool = False,
        normalize_targets: bool = False,
        normalization_eps: float = 1e-5,
        normalization_momentum: float = 0.01,
    ):
        super().__init__()
        self.n_layers = int(n_layers)
        self.d_model = int(d_model)
        self.features_per_layer = int(features_per_layer)
        self.normalize_inputs = bool(normalize_inputs)
        self.normalize_targets = bool(normalize_targets)
        self.normalization_eps = float(normalization_eps)
        self.normalization_momentum = float(normalization_momentum)

        if self.normalization_eps <= 0.0:
            raise ValueError(
                f"normalization_eps must be positive. Got {self.normalization_eps}."
            )
        if not 0.0 < self.normalization_momentum <= 1.0:
            raise ValueError(
                "normalization_momentum must be in (0, 1]. "
                f"Got {self.normalization_momentum}."
            )

        self.encoders = nn.ParameterList([
            nn.Parameter(torch.empty(self.d_model, self.features_per_layer))
            for _ in range(self.n_layers)
        ])
        self.encoder_bias = nn.ParameterList([
            nn.Parameter(torch.zeros(self.features_per_layer))
            for _ in range(self.n_layers)
        ])

        self.decoders = nn.ParameterDict()
        for src in range(self.n_layers):
            for tgt in range(src, self.n_layers):
                self.decoders[f"{src}->{tgt}"] = nn.Parameter(
                    torch.empty(self.features_per_layer, self.d_model)
                )

        self.decoder_bias = nn.ParameterList([
            nn.Parameter(torch.zeros(self.d_model))
            for _ in range(self.n_layers)
        ])

        if float(init_threshold) <= 0.0:
            raise ValueError(
                "init_threshold must be positive for the JumpReLU log-parametrization. "
                f"Got {init_threshold}."
            )
        if float(jumprelu_bandwidth) <= 0.0:
            raise ValueError(
                f"jumprelu_bandwidth must be positive. Got {jumprelu_bandwidth}."
            )
        self.jumprelu_init_threshold = float(init_threshold)
        self.jumprelu_bandwidth = float(jumprelu_bandwidth)
        # theta = exp(log_threshold) keeps the threshold strictly positive; it is
        # learned via the straight-through estimator in _JumpReLU.
        self.log_threshold = nn.Parameter(
            torch.full(
                (self.n_layers, self.features_per_layer),
                math.log(float(init_threshold)),
            )
        )
        self.decoder_init_scale = float(decoder_init_scale)

        if self.normalize_inputs:
            self.register_buffer(
                "input_mean",
                torch.zeros(self.n_layers, self.d_model),
                persistent=True,
            )
            self.register_buffer(
                "input_var",
                torch.ones(self.n_layers, self.d_model),
                persistent=True,
            )

        if self.normalize_targets:
            self.register_buffer(
                "target_mean",
                torch.zeros(self.n_layers, self.d_model),
                persistent=True,
            )
            self.register_buffer(
                "target_var",
                torch.ones(self.n_layers, self.d_model),
                persistent=True,
            )

        if self.normalize_inputs or self.normalize_targets:
            self.register_buffer(
                "normalization_updates",
                torch.zeros(self.n_layers, dtype=torch.long),
                persistent=True,
            )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for W in self.encoders:
            nn.init.kaiming_uniform_(W, a=math.sqrt(5))
        for W in self.decoders.values():
            nn.init.normal_(W, mean=0.0, std=self.decoder_init_scale)

    def normalization_enabled(self) -> bool:
        return self.normalize_inputs or self.normalize_targets

    def _layer_mean_var(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.detach().float()
        mean = x.mean(dim=(0, 1))
        var = (x - mean.view(1, 1, -1)).pow(2).mean(dim=(0, 1))
        return mean, var.clamp_min(self.normalization_eps)

    @torch.no_grad()
    def update_normalization_stats(
        self,
        mlp_inputs: list[torch.Tensor],
        mlp_targets: list[torch.Tensor] | None = None,
    ) -> None:
        if not self.normalization_enabled():
            return

        if self.normalize_targets and mlp_targets is None:
            raise ValueError(
                "normalize_targets=True requires mlp_targets when updating "
                "CLT normalization stats."
            )

        for layer_idx in range(self.n_layers):
            updates = int(self.normalization_updates[layer_idx].item())
            momentum = 1.0 if updates == 0 else self.normalization_momentum

            if self.normalize_inputs:
                mean, var = self._layer_mean_var(mlp_inputs[layer_idx])
                self.input_mean[layer_idx].lerp_(
                    mean.to(self.input_mean.device),
                    momentum,
                )
                self.input_var[layer_idx].lerp_(
                    var.to(self.input_var.device),
                    momentum,
                )

            if self.normalize_targets:
                assert mlp_targets is not None
                mean, var = self._layer_mean_var(mlp_targets[layer_idx])
                self.target_mean[layer_idx].lerp_(
                    mean.to(self.target_mean.device),
                    momentum,
                )
                self.target_var[layer_idx].lerp_(
                    var.to(self.target_var.device),
                    momentum,
                )

            self.normalization_updates[layer_idx] += 1

    def normalize_input(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        if not self.normalize_inputs:
            return x

        mean = self.input_mean[layer_idx].to(device=x.device, dtype=x.dtype)
        std = (
            self.input_var[layer_idx]
            .to(device=x.device, dtype=x.dtype)
            .add(self.normalization_eps)
            .sqrt()
        )
        return (x - mean.view(1, 1, -1)) / std.view(1, 1, -1)

    def normalize_target(self, y: torch.Tensor, layer_idx: int) -> torch.Tensor:
        if not self.normalize_targets:
            return y

        mean = self.target_mean[layer_idx].to(device=y.device, dtype=y.dtype)
        std = (
            self.target_var[layer_idx]
            .to(device=y.device, dtype=y.dtype)
            .add(self.normalization_eps)
            .sqrt()
        )
        return (y - mean.view(1, 1, -1)) / std.view(1, 1, -1)

    def denormalize_output(self, y: torch.Tensor, layer_idx: int) -> torch.Tensor:
        if not self.normalize_targets:
            return y

        mean = self.target_mean[layer_idx].to(device=y.device, dtype=y.dtype)
        std = (
            self.target_var[layer_idx]
            .to(device=y.device, dtype=y.dtype)
            .add(self.normalization_eps)
            .sqrt()
        )
        return y * std.view(1, 1, -1) + mean.view(1, 1, -1)

    @property
    def effective_thresholds(self) -> torch.Tensor:
        """Positive per-(layer, feature) JumpReLU thresholds: theta = exp(log_threshold)."""
        return self.log_threshold.exp()

    def effective_threshold(self, layer_idx: int) -> torch.Tensor:
        return self.log_threshold[layer_idx].exp()

    def jump_relu(self, z: torch.Tensor, layer_idx: int) -> torch.Tensor:
        threshold = self.effective_threshold(layer_idx).to(z.dtype)
        return _JumpReLU.apply(z, threshold, self.jumprelu_bandwidth)

    def encode_layer(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        x = self.normalize_input(x, layer_idx)
        z = x @ self.encoders[layer_idx].to(x.dtype) + self.encoder_bias[layer_idx].to(x.dtype)
        return self.jump_relu(z, layer_idx)

    def forward(
        self,
        mlp_inputs: list[torch.Tensor],
        mlp_targets: list[torch.Tensor] | None = None,
        update_normalization_stats: bool = False,
    ):
        if update_normalization_stats:
            self.update_normalization_stats(mlp_inputs, mlp_targets)

        features_by_layer: list[torch.Tensor] = []
        for layer_idx, x in enumerate(mlp_inputs):
            features_by_layer.append(self.encode_layer(x, layer_idx))

        normalized_recons = []
        for tgt in range(self.n_layers):
            bias = self.decoder_bias[tgt].to(
                device=mlp_inputs[tgt].device,
                dtype=mlp_inputs[tgt].dtype,
            )
            normalized_recons.append(
                bias.view(1, 1, -1).expand_as(mlp_inputs[tgt]).clone()
            )

        for src, a in enumerate(features_by_layer):
            for tgt in range(src, self.n_layers):
                W = self.decoders[f"{src}->{tgt}"].to(
                    device=a.device,
                    dtype=a.dtype,
                )
                normalized_recons[tgt] = normalized_recons[tgt] + a @ W

        mlp_recons = [
            self.denormalize_output(recon, tgt)
            for tgt, recon in enumerate(normalized_recons)
        ]
        return features_by_layer, mlp_recons

    def decoder_matrix_for_raw_output(self, src: int, tgt: int) -> torch.Tensor:
        matrix = self.decoders[f"{src}->{tgt}"]
        if not self.normalize_targets:
            return matrix

        std = (
            self.target_var[tgt]
            .to(device=matrix.device, dtype=matrix.dtype)
            .add(self.normalization_eps)
            .sqrt()
        )
        return matrix * std.view(1, -1)

    def decoder_row_for_raw_output(
        self,
        src: int,
        tgt: int,
        feature_idx: int,
    ) -> torch.Tensor:
        return self.decoder_matrix_for_raw_output(src, tgt)[feature_idx]

    def decoder_feature_norms(self, src: int) -> torch.Tensor:
        norms = []
        for tgt in range(src, self.n_layers):
            norms.append(self.decoder_matrix_for_raw_output(src, tgt).norm(dim=1))
        return torch.stack(norms, dim=0).sum(dim=0)
