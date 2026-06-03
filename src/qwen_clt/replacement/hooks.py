from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from qwen_clt.interventions import FeatureIntervention


@dataclass
class ReplacementConfig:
    layer_idx: int | None = None
    read_from: str = "mlp_normed_input"
    replace_at: str = "mlp_output"
    replace_mode: str = "full"
    detach_reconstruction: bool = False
    feature_interventions: tuple[FeatureIntervention, ...] = field(default_factory=tuple)


class LayerReplacementHook:
    """
    MLP-output replacement hook для Qwen + CrossLayerTranscoder.

    CLT обучается на паре:

        layer.mlp input  = mlp_normed_input
        layer.mlp output = mlp_output

    Поэтому replacement должен вешаться на `layer.mlp`, читать inputs[0] и
    заменять output этого MLP-модуля. Хук на весь transformer layer был бы
    методологически неверным: там output уже включает residual path и attention.

    CrossLayerTranscoder.forward(...) ожидает входы сразу по всем слоям.
    Поэтому внутри MLP forward-hook нельзя напрямую вызывать:

        autoencoder(mlp_input)

    Вместо этого мы вручную делаем online-реконструкцию текущего слоя tgt:

        recon[tgt] = sum_{src <= tgt} features[src] @ decoder[src->tgt]

    где features[src] получаются из сохранённых MLP inputs предыдущих слоёв.
    """

    def __init__(
        self,
        model: nn.Module,
        autoencoder: nn.Module,
        config: ReplacementConfig,
    ):
        self.model = model
        self.autoencoder = autoencoder
        self.config = config
        self.handles: list[Any] = []

        # cache: layer_idx -> mlp_normed_input
        self.cached_inputs: dict[int, torch.Tensor] = {}
        self.features_by_layer: dict[int, torch.Tensor] = {}
        self.reconstructions_by_layer: dict[int, torch.Tensor] = {}
        self.interventions_by_layer = self._group_interventions(
            self.config.feature_interventions
        )

        self._validate_config()

    def _validate_config(self) -> None:
        if self.config.replace_mode != "full":
            raise ValueError(
                "LayerReplacementHook currently supports only "
                "replace_mode='full'. "
                f"Got replace_mode={self.config.replace_mode!r}."
            )

        if self.config.read_from != "mlp_normed_input":
            raise ValueError(
                "LayerReplacementHook currently supports only "
                "read_from='mlp_normed_input'. "
                f"Got read_from={self.config.read_from!r}."
            )

        if self.config.replace_at != "mlp_output":
            raise ValueError(
                "LayerReplacementHook replaces layer.mlp outputs and currently "
                "supports only replace_at='mlp_output'. "
                f"Got replace_at={self.config.replace_at!r}."
            )

        for item in self.config.feature_interventions:
            if item.layer < 0:
                raise IndexError(
                    f"Feature intervention layer must be non-negative. Got {item.layer}."
                )
            if item.layer >= self.autoencoder.n_layers:
                raise IndexError(
                    f"Feature intervention layer={item.layer} is outside CLT layers "
                    f"0..{self.autoencoder.n_layers - 1}."
                )
            if item.feature_idx < 0:
                raise IndexError(
                    "Feature intervention feature_idx must be non-negative. "
                    f"Got {item.feature_idx}."
                )
            if item.feature_idx >= self.autoencoder.features_per_layer:
                raise IndexError(
                    f"Feature intervention feature_idx={item.feature_idx} is outside "
                    f"CLT features 0..{self.autoencoder.features_per_layer - 1}."
                )

    def _group_interventions(
        self,
        interventions: tuple[FeatureIntervention, ...],
    ) -> dict[int, list[FeatureIntervention]]:
        grouped: dict[int, list[FeatureIntervention]] = {}
        for item in interventions:
            grouped.setdefault(int(item.layer), []).append(item)
        return grouped

    def _get_layers(self):
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers

        if hasattr(self.model, "base_model") and hasattr(self.model.base_model, "model"):
            base = self.model.base_model
            if hasattr(base.model, "layers"):
                return base.model.layers

        raise AttributeError(
            "Cannot find transformer layers. "
            "Expected model.model.layers or model.base_model.model.layers."
        )

    def _get_mlp_modules(self):
        layers = self._get_layers()
        mlp_modules = []

        for idx, layer in enumerate(layers):
            if not hasattr(layer, "mlp"):
                raise AttributeError(
                    f"Cannot find layer.mlp for transformer layer {idx}. "
                    "Expected Qwen-like decoder layers with an MLP submodule."
                )
            mlp_modules.append(layer.mlp)

        return mlp_modules

    def _extract_mlp_output(self, output):
        if isinstance(output, tuple):
            return output[0], output[1:]

        return output, None

    def _pack_output(self, reconstructed: torch.Tensor, rest):
        if rest is not None:
            return (reconstructed, *rest)

        return reconstructed

    def _encode_layer(
        self,
        mlp_input: torch.Tensor,
        src: int,
    ) -> torch.Tensor:
        encoder = self.autoencoder.encoders[src].to(
            device=mlp_input.device,
            dtype=mlp_input.dtype,
        )
        bias = self.autoencoder.encoder_bias[src].to(
            device=mlp_input.device,
            dtype=mlp_input.dtype,
        )
        threshold = self.autoencoder.thresholds[src].to(
            device=mlp_input.device,
            dtype=mlp_input.dtype,
        )

        pre = mlp_input @ encoder + bias
        features = pre * (pre > threshold)

        return features

    def _apply_feature_interventions(
        self,
        features: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        interventions = self.interventions_by_layer.get(layer_idx)
        if not interventions:
            return features

        features = features.clone()

        for item in interventions:
            if isinstance(item.pos, int):
                pos = item.pos
                if pos < 0:
                    pos = features.shape[1] + pos
                if pos < 0 or pos >= features.shape[1]:
                    raise IndexError(
                        f"Feature intervention pos={item.pos} is outside sequence "
                        f"length {features.shape[1]} for layer {layer_idx}."
                    )
            else:
                pos = item.pos

            features[:, pos, item.feature_idx] = item.value

        return features

    def _features_for_layer(self, src: int) -> torch.Tensor:
        if src in self.features_by_layer:
            return self.features_by_layer[src]

        if src not in self.cached_inputs:
            raise KeyError(
                f"No cached MLP input for source layer {src}."
            )

        features = self._encode_layer(
            mlp_input=self.cached_inputs[src],
            src=src,
        )
        features = self._apply_feature_interventions(features, src)
        self.features_by_layer[src] = features

        return features

    def _decode_to_target(
        self,
        features: torch.Tensor,
        src: int,
        tgt: int,
    ) -> torch.Tensor:
        decoder_key = f"{src}->{tgt}"

        if decoder_key not in self.autoencoder.decoders:
            raise KeyError(
                f"Decoder {decoder_key} not found in autoencoder.decoders."
            )

        decoder = self.autoencoder.decoders[decoder_key].to(
            device=features.device,
            dtype=features.dtype,
        )

        return features @ decoder

    def _reconstruct_target_layer(
        self,
        tgt: int,
        target_shape_like: torch.Tensor,
    ) -> torch.Tensor:
        if tgt not in self.cached_inputs:
            raise KeyError(
                f"No cached input for target layer {tgt}. "
                "The MLP hook must cache current layer input before reconstruction."
            )

        reconstruction = torch.zeros_like(target_shape_like)

        max_src = min(tgt, self.autoencoder.n_layers - 1)

        for src in range(max_src + 1):
            if src not in self.cached_inputs:
                continue

            features = self._features_for_layer(src)

            contribution = self._decode_to_target(
                features=features,
                src=src,
                tgt=tgt,
            )

            reconstruction = reconstruction + contribution

        return reconstruction

    def _make_hook_fn(self, layer_idx: int):
        def hook_fn(module, inputs, output):
            if not inputs:
                raise RuntimeError(
                    f"MLP hook for layer {layer_idx} received no inputs."
                )

            mlp_input = inputs[0]
            mlp_output, rest = self._extract_mlp_output(output)

            # Cache the exact activation point used by CLT training:
            # layer.mlp input, i.e. post-attention-normalized hidden states.
            self.cached_inputs[layer_idx] = mlp_input

            should_replace = (
                self.config.layer_idx is None
                or int(self.config.layer_idx) == int(layer_idx)
            )

            if not should_replace:
                return output

            # Если checkpoint содержит меньше CLT-слоёв, чем Qwen,
            # не заменяем слои за пределами CLT.
            if layer_idx >= self.autoencoder.n_layers:
                return output

            reconstructed = self._reconstruct_target_layer(
                tgt=layer_idx,
                target_shape_like=mlp_output,
            )
            self.reconstructions_by_layer[layer_idx] = reconstructed

            if self.config.detach_reconstruction:
                reconstructed = reconstructed.detach()

            return self._pack_output(reconstructed, rest)

        return hook_fn

    def install(self):
        if self.handles:
            raise RuntimeError("Replacement hooks are already installed.")

        self.cached_inputs = {}
        self.features_by_layer = {}
        self.reconstructions_by_layer = {}

        mlp_modules = self._get_mlp_modules()

        if self.config.layer_idx is not None:
            target_layer = int(self.config.layer_idx)
            if target_layer < 0:
                raise IndexError(
                    f"layer_idx must be non-negative. Got {target_layer}."
                )
            if target_layer >= len(mlp_modules):
                raise IndexError(
                    f"layer_idx={target_layer} is outside model layers "
                    f"0..{len(mlp_modules) - 1}."
                )
            if target_layer >= self.autoencoder.n_layers:
                raise IndexError(
                    f"layer_idx={target_layer} is outside CLT layers "
                    f"0..{self.autoencoder.n_layers - 1}."
                )
            max_layer = min(
                target_layer + 1,
                len(mlp_modules),
                self.autoencoder.n_layers,
            )
            layer_indices = list(range(max_layer))
        else:
            layer_indices = list(range(min(len(mlp_modules), self.autoencoder.n_layers)))

        for layer_idx in layer_indices:
            handle = mlp_modules[layer_idx].register_forward_hook(
                self._make_hook_fn(layer_idx)
            )
            self.handles.append(handle)

        return self

    def remove(self):
        for handle in self.handles:
            handle.remove()

        self.handles = []
        self.cached_inputs = {}

    def __enter__(self):
        self.install()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.remove()
