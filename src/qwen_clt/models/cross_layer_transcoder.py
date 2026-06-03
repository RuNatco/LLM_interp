from __future__ import annotations

import math
import torch
import torch.nn as nn


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
        init_threshold: float = 0.0,
        decoder_init_scale: float = 0.02,
    ):
        super().__init__()
        self.n_layers = int(n_layers)
        self.d_model = int(d_model)
        self.features_per_layer = int(features_per_layer)

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

        self.thresholds = nn.Parameter(
            torch.full((self.n_layers, self.features_per_layer), float(init_threshold))
        )
        self.decoder_init_scale = float(decoder_init_scale)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for W in self.encoders:
            nn.init.kaiming_uniform_(W, a=math.sqrt(5))
        for W in self.decoders.values():
            nn.init.normal_(W, mean=0.0, std=self.decoder_init_scale)

    def jump_relu(self, z: torch.Tensor, layer_idx: int) -> torch.Tensor:
        threshold = self.thresholds[layer_idx].to(z.dtype)
        return torch.where(z > threshold, z, torch.zeros_like(z))

    def encode_layer(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        z = x @ self.encoders[layer_idx].to(x.dtype) + self.encoder_bias[layer_idx].to(x.dtype)
        return self.jump_relu(z, layer_idx)

    def forward(self, mlp_inputs: list[torch.Tensor]):
        features_by_layer: list[torch.Tensor] = []
        for layer_idx, x in enumerate(mlp_inputs):
            features_by_layer.append(self.encode_layer(x, layer_idx))

        mlp_recons = [torch.zeros_like(mlp_inputs[tgt]) for tgt in range(self.n_layers)]
        for src, a in enumerate(features_by_layer):
            for tgt in range(src, self.n_layers):
                W = self.decoders[f"{src}->{tgt}"].to(a.dtype)
                mlp_recons[tgt] = mlp_recons[tgt] + a @ W
        return features_by_layer, mlp_recons

    def decoder_feature_norms(self, src: int) -> torch.Tensor:
        norms = []
        for tgt in range(src, self.n_layers):
            norms.append(self.decoders[f"{src}->{tgt}"].norm(dim=1))
        return torch.stack(norms, dim=0).sum(dim=0)
