from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from qwen_clt.replacement import LayerReplacementHook, ReplacementConfig
from qwen_clt.utils.io import save_checkpoint


@dataclass
class DeepTraceActivations:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor | None
    logits: torch.Tensor
    residual_inputs: list[torch.Tensor]
    input_layernorm_outputs: list[torch.Tensor | None]
    attention_outputs: list[torch.Tensor | None]
    attention_head_outputs: list[torch.Tensor | None]
    post_attention_layernorm_outputs: list[torch.Tensor | None]
    mlp_inputs: list[torch.Tensor]
    mlp_outputs: list[torch.Tensor]


@dataclass
class DeepTraceCache:
    original: DeepTraceActivations
    replacement: DeepTraceActivations
    replacement_features: list[torch.Tensor | None]
    replacement_mlp_recons: list[torch.Tensor | None]
    replacement_errors: list[torch.Tensor | None]


def _detach_tensor(value: torch.Tensor) -> torch.Tensor:
    return value.detach().cpu()


def _first_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, tuple):
        output = output[0]
    if not isinstance(output, torch.Tensor):
        raise TypeError(f"Expected tensor output, got {type(output)}")
    return output


def _get_layers(model: nn.Module):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        base = model.base_model
        if hasattr(base.model, "layers"):
            return base.model.layers
    raise AttributeError(
        "Cannot find transformer layers. Expected model.model.layers."
    )


class QwenDeepTraceCollector:
    """Collects a compact typed trace for Qwen-like decoder layers."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.handles: list[Any] = []
        self.n_layers = len(_get_layers(model))
        self.reset()

    def reset(self) -> None:
        self.residual_inputs: list[torch.Tensor | None] = [None] * self.n_layers
        self.input_layernorm_outputs: list[torch.Tensor | None] = [None] * self.n_layers
        self.attention_outputs: list[torch.Tensor | None] = [None] * self.n_layers
        self.attention_head_outputs: list[torch.Tensor | None] = [None] * self.n_layers
        self.post_attention_layernorm_outputs: list[torch.Tensor | None] = [
            None
        ] * self.n_layers
        self.mlp_inputs: list[torch.Tensor | None] = [None] * self.n_layers
        self.mlp_outputs: list[torch.Tensor | None] = [None] * self.n_layers

    def _make_layer_pre_hook(self, layer_idx: int):
        def hook(module, inputs):
            if not inputs:
                raise RuntimeError(f"Layer {layer_idx} pre-hook received no inputs.")
            self.residual_inputs[layer_idx] = inputs[0]

        return hook

    def _make_output_hook(self, store: list, layer_idx: int):
        def hook(module, inputs, output):
            store[layer_idx] = _first_tensor(output)

        return hook

    def _infer_attention_heads(self, attention_module, o_proj: nn.Module) -> tuple[int, int] | None:
        n_heads = getattr(attention_module, "num_heads", None)
        if n_heads is None:
            n_heads = getattr(attention_module, "num_attention_heads", None)
        if n_heads is None:
            config = getattr(self.model, "config", None)
            n_heads = getattr(config, "num_attention_heads", None)
        if n_heads is None:
            return None

        n_heads = int(n_heads)
        if n_heads <= 0:
            return None

        head_dim = getattr(attention_module, "head_dim", None)
        in_features = int(o_proj.weight.shape[1])
        if head_dim is None:
            if in_features % n_heads != 0:
                return None
            head_dim = in_features // n_heads
        head_dim = int(head_dim)

        if n_heads * head_dim != in_features:
            return None

        return n_heads, head_dim

    def _make_o_proj_hook(self, layer_idx: int, attention_module):
        def hook(module, inputs, output):
            if not inputs:
                return

            projected_input = inputs[0]
            if projected_input.ndim != 3:
                return

            head_shape = self._infer_attention_heads(attention_module, module)
            if head_shape is None:
                return

            n_heads, head_dim = head_shape
            batch, seq_len, _ = projected_input.shape
            head_values = projected_input.reshape(batch, seq_len, n_heads, head_dim)
            head_weights = module.weight.reshape(
                module.out_features,
                n_heads,
                head_dim,
            )
            head_outputs = torch.einsum(
                "bshd,ohd->bsho",
                head_values,
                head_weights,
            )
            self.attention_head_outputs[layer_idx] = head_outputs

        return hook

    def _make_mlp_hook(self, layer_idx: int):
        def hook(module, inputs, output):
            if not inputs:
                raise RuntimeError(f"MLP hook for layer {layer_idx} received no inputs.")
            self.mlp_inputs[layer_idx] = inputs[0]
            self.mlp_outputs[layer_idx] = _first_tensor(output)

        return hook

    def __enter__(self):
        self.reset()
        layers = _get_layers(self.model)
        for layer_idx, layer in enumerate(layers):
            self.handles.append(
                layer.register_forward_pre_hook(
                    self._make_layer_pre_hook(layer_idx)
                )
            )

            if hasattr(layer, "input_layernorm"):
                self.handles.append(
                    layer.input_layernorm.register_forward_hook(
                        self._make_output_hook(
                            self.input_layernorm_outputs,
                            layer_idx,
                        )
                    )
                )

            if hasattr(layer, "self_attn"):
                self.handles.append(
                    layer.self_attn.register_forward_hook(
                        self._make_output_hook(self.attention_outputs, layer_idx)
                    )
                )
                if hasattr(layer.self_attn, "o_proj"):
                    self.handles.append(
                        layer.self_attn.o_proj.register_forward_hook(
                            self._make_o_proj_hook(layer_idx, layer.self_attn)
                        )
                    )

            if hasattr(layer, "post_attention_layernorm"):
                self.handles.append(
                    layer.post_attention_layernorm.register_forward_hook(
                        self._make_output_hook(
                            self.post_attention_layernorm_outputs,
                            layer_idx,
                        )
                    )
                )

            if not hasattr(layer, "mlp"):
                raise AttributeError(f"Layer {layer_idx} has no mlp module.")
            self.handles.append(
                layer.mlp.register_forward_hook(self._make_mlp_hook(layer_idx))
            )

        return self

    def __exit__(self, exc_type, exc, tb):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _require_all(self, values: list[torch.Tensor | None], name: str) -> list[torch.Tensor]:
        if any(value is None for value in values):
            missing = [idx for idx, value in enumerate(values) if value is None]
            raise RuntimeError(f"Missing {name} activations for layers: {missing}")
        return [_detach_tensor(value) for value in values if value is not None]

    @torch.no_grad()
    def run(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> DeepTraceActivations:
        with self:
            out = self.model(input_ids=input_ids, attention_mask=attention_mask)

        return DeepTraceActivations(
            input_ids=_detach_tensor(input_ids),
            attention_mask=(
                None if attention_mask is None else _detach_tensor(attention_mask)
            ),
            logits=_detach_tensor(out.logits),
            residual_inputs=self._require_all(
                self.residual_inputs,
                "residual_inputs",
            ),
            input_layernorm_outputs=[
                None if value is None else _detach_tensor(value)
                for value in self.input_layernorm_outputs
            ],
            attention_outputs=[
                None if value is None else _detach_tensor(value)
                for value in self.attention_outputs
            ],
            attention_head_outputs=[
                None if value is None else _detach_tensor(value)
                for value in self.attention_head_outputs
            ],
            post_attention_layernorm_outputs=[
                None if value is None else _detach_tensor(value)
                for value in self.post_attention_layernorm_outputs
            ],
            mlp_inputs=self._require_all(self.mlp_inputs, "mlp_inputs"),
            mlp_outputs=self._require_all(self.mlp_outputs, "mlp_outputs"),
        )


def _layer_dict_to_list(
    values: dict[int, torch.Tensor],
    n_layers: int,
) -> list[torch.Tensor | None]:
    return [
        None if values.get(layer_idx) is None else _detach_tensor(values[layer_idx])
        for layer_idx in range(n_layers)
    ]


@torch.no_grad()
def collect_deep_trace_cache(
    *,
    model,
    autoencoder,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    replacement_config: ReplacementConfig | None = None,
    original: DeepTraceActivations | None = None,
) -> DeepTraceCache:
    collector = QwenDeepTraceCollector(model)
    if original is None:
        original = collector.run(input_ids=input_ids, attention_mask=attention_mask)

    config = replacement_config or ReplacementConfig()
    with LayerReplacementHook(
        model=model,
        autoencoder=autoencoder,
        config=config,
    ) as replacement_hook:
        replacement = collector.run(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    recons = _layer_dict_to_list(
        replacement_hook.reconstructions_by_layer,
        autoencoder.n_layers,
    )
    features = _layer_dict_to_list(
        replacement_hook.features_by_layer,
        autoencoder.n_layers,
    )
    errors: list[torch.Tensor | None] = []
    for layer_idx in range(autoencoder.n_layers):
        recon = recons[layer_idx]
        if recon is None:
            errors.append(None)
            continue
        errors.append(original.mlp_outputs[layer_idx] - recon)

    return DeepTraceCache(
        original=original,
        replacement=replacement,
        replacement_features=features,
        replacement_mlp_recons=recons,
        replacement_errors=errors,
    )


def _activation_payload(acts: DeepTraceActivations) -> dict[str, Any]:
    return {
        "input_ids": acts.input_ids,
        "attention_mask": acts.attention_mask,
        "logits": acts.logits,
        "residual_inputs": acts.residual_inputs,
        "input_layernorm_outputs": acts.input_layernorm_outputs,
        "attention_outputs": acts.attention_outputs,
        "attention_head_outputs": acts.attention_head_outputs,
        "post_attention_layernorm_outputs": acts.post_attention_layernorm_outputs,
        "mlp_inputs": acts.mlp_inputs,
        "mlp_outputs": acts.mlp_outputs,
    }


def save_deep_trace_cache(cache: DeepTraceCache, path: str | Path) -> None:
    save_checkpoint(
        {
            "kind": "deep_trace_stage2_cache",
            "original": _activation_payload(cache.original),
            "replacement": _activation_payload(cache.replacement),
            "replacement_features": cache.replacement_features,
            "replacement_mlp_recons": cache.replacement_mlp_recons,
            "replacement_errors": cache.replacement_errors,
        },
        path,
    )
