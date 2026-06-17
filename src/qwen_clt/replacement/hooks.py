from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from qwen_clt.interventions import FeatureIntervention


@dataclass
class ReplacementConfig:
    layer_idx: int | None = None
    replacement_layers: tuple[int, ...] | None = None
    read_from: str = "mlp_normed_input"
    replace_at: str = "mlp_output"
    replace_mode: str = "full"
    detach_reconstruction: bool = False
    feature_interventions: tuple[FeatureIntervention, ...] = field(default_factory=tuple)


class LayerReplacementHook:

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

        if self.config.layer_idx is not None and self.config.replacement_layers is not None:
            raise ValueError(
                "Use either layer_idx for a single replacement layer or "
                "replacement_layers for an explicit set, not both."
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
        if hasattr(self.autoencoder, "normalize_input"):
            mlp_input = self.autoencoder.normalize_input(mlp_input, src)

        encoder = self.autoencoder.encoders[src].to(
            device=mlp_input.device,
            dtype=mlp_input.dtype,
        )
        bias = self.autoencoder.encoder_bias[src].to(
            device=mlp_input.device,
            dtype=mlp_input.dtype,
        )

        pre = mlp_input @ encoder + bias
        features = self.autoencoder.jump_relu(pre, src)

        return features

    def _denormalize_target(
        self,
        reconstruction: torch.Tensor,
        tgt: int,
    ) -> torch.Tensor:
        if hasattr(self.autoencoder, "denormalize_output"):
            return self.autoencoder.denormalize_output(reconstruction, tgt)

        return reconstruction

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

    def _target_bias(
        self,
        tgt: int,
        target_shape_like: torch.Tensor,
    ) -> torch.Tensor:
        if not hasattr(self.autoencoder, "decoder_bias"):
            return torch.zeros_like(target_shape_like)

        bias = self.autoencoder.decoder_bias[tgt].to(
            device=target_shape_like.device,
            dtype=target_shape_like.dtype,
        )

        return bias.view(1, 1, -1).expand_as(target_shape_like).clone()

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

        reconstruction = self._target_bias(
            tgt=tgt,
            target_shape_like=target_shape_like,
        )

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

        return self._denormalize_target(reconstruction, tgt)

    def _make_hook_fn(self, layer_idx: int):
        def hook_fn(module, inputs, output):
            if not inputs:
                raise RuntimeError(
                    f"MLP hook for layer {layer_idx} received no inputs."
                )

            mlp_input = inputs[0]
            mlp_output, rest = self._extract_mlp_output(output)

            self.cached_inputs[layer_idx] = mlp_input

            should_replace = self._should_replace_layer(layer_idx)

            if not should_replace:
                return output

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

    def _replacement_layer_set(self) -> set[int] | None:
        if self.config.layer_idx is not None:
            return {int(self.config.layer_idx)}
        if self.config.replacement_layers is not None:
            return {int(layer_idx) for layer_idx in self.config.replacement_layers}
        return None

    def _should_replace_layer(self, layer_idx: int) -> bool:
        replacement_layers = self._replacement_layer_set()
        return replacement_layers is None or int(layer_idx) in replacement_layers

    def _validate_replacement_layers(
        self,
        mlp_modules: list[nn.Module],
    ) -> list[int]:
        replacement_layers = self._replacement_layer_set()

        if replacement_layers is None:
            return list(range(min(len(mlp_modules), self.autoencoder.n_layers)))

        if not replacement_layers:
            raise ValueError("replacement_layers cannot be empty.")

        for layer_idx in replacement_layers:
            if layer_idx < 0:
                raise IndexError(
                    f"replacement layer must be non-negative. Got {layer_idx}."
                )
            if layer_idx >= len(mlp_modules):
                raise IndexError(
                    f"replacement layer={layer_idx} is outside model layers "
                    f"0..{len(mlp_modules) - 1}."
                )
            if layer_idx >= self.autoencoder.n_layers:
                raise IndexError(
                    f"replacement layer={layer_idx} is outside CLT layers "
                    f"0..{self.autoencoder.n_layers - 1}."
                )

        max_layer = max(replacement_layers)
        return list(range(max_layer + 1))

    def install(self):
        if self.handles:
            raise RuntimeError("Replacement hooks are already installed.")

        self.cached_inputs = {}
        self.features_by_layer = {}
        self.reconstructions_by_layer = {}

        mlp_modules = self._get_mlp_modules()

        layer_indices = self._validate_replacement_layers(mlp_modules)

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
