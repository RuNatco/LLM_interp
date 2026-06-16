"""Control baselines for the MLP-output replacement model.

The trained-CLT replacement only *means* something relative to controls. We
replace the exact same activation point (`layer.mlp` output, "full" mode, the
same layer set the CLT covers) with deliberately uninformative substitutes and
measure the same logit-fidelity metrics. The gap between the trained CLT and
each control is the actual evidence the method recovered real computation:

    zero        -> delete the MLP entirely (how much the MLPs matter at all)
    mean        -> per-layer dataset mean of the MLP output (drops all
                   token-specific information, keeps the average contribution)
    random_clt  -> same CLT architecture, untrained weights (isolates the
                   contribution of *training* from the contribution of shape)

All hooks here mirror LayerReplacementHook: they hook `layer.mlp`, handle the
tuple/tensor output packing, and replace exactly the CLT-covered layers so the
comparison against the trained CLT is apples-to-apples.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from qwen_clt.models.cross_layer_transcoder import CrossLayerTranscoder


# ---------------------------------------------------------------------------
# Module discovery (kept consistent with LayerReplacementHook)
# ---------------------------------------------------------------------------

def get_decoder_layers(model: nn.Module):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers

    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        base = model.base_model
        if hasattr(base.model, "layers"):
            return base.model.layers

    raise AttributeError(
        "Cannot find transformer layers. "
        "Expected model.model.layers or model.base_model.model.layers."
    )


def get_mlp_modules(model: nn.Module) -> list[nn.Module]:
    layers = get_decoder_layers(model)
    mlp_modules: list[nn.Module] = []
    for idx, layer in enumerate(layers):
        if not hasattr(layer, "mlp"):
            raise AttributeError(
                f"Cannot find layer.mlp for transformer layer {idx}. "
                "Expected Qwen-like decoder layers with an MLP submodule."
            )
        mlp_modules.append(layer.mlp)
    return mlp_modules


def _split_mlp_output(output):
    if isinstance(output, tuple):
        return output[0], output[1:]
    return output, None


def _pack_mlp_output(tensor: torch.Tensor, rest):
    if rest is not None:
        return (tensor, *rest)
    return tensor


def _covered_layers(model: nn.Module, n_clt_layers: int) -> list[int]:
    """The layer set the trained CLT replaces, so baselines match it exactly."""
    n_model_layers = len(get_mlp_modules(model))
    return list(range(min(n_model_layers, int(n_clt_layers))))


# ---------------------------------------------------------------------------
# Mean-MLP-output collection (for the mean-ablation baseline)
# ---------------------------------------------------------------------------

class _MlpOutputTap:
    """Read-only forward hooks that stash the latest MLP output per layer."""

    def __init__(self, model: nn.Module, layers: list[int]):
        self.model = model
        self.layers = list(layers)
        self.latest: dict[int, torch.Tensor] = {}
        self.handles: list[Any] = []

    def _make_hook(self, layer_idx: int):
        def hook(module, inputs, output):
            tensor, _ = _split_mlp_output(output)
            self.latest[layer_idx] = tensor.detach()
        return hook

    def __enter__(self):
        mlp_modules = get_mlp_modules(self.model)
        for layer_idx in self.layers:
            self.handles.append(
                mlp_modules[layer_idx].register_forward_hook(
                    self._make_hook(layer_idx)
                )
            )
        return self

    def __exit__(self, exc_type, exc, tb):
        for handle in self.handles:
            handle.remove()
        self.handles = []


class MeanMlpOutputAccumulator:
    """Running, attention-masked mean of each layer's MLP output.

    Call `observe(...)` once per batch *after* a forward pass that ran inside a
    `_MlpOutputTap`. Padded positions are excluded via the attention mask so the
    mean reflects real tokens only.
    """

    def __init__(self, layers: list[int], d_model: int, device: str):
        self.layers = list(layers)
        self.d_model = int(d_model)
        self.device = device
        self.sums: dict[int, torch.Tensor] = {
            layer_idx: torch.zeros(self.d_model, dtype=torch.float32, device=device)
            for layer_idx in self.layers
        }
        self.token_count: float = 0.0

    def observe(
        self,
        latest: dict[int, torch.Tensor],
        attention_mask: torch.Tensor | None,
    ) -> None:
        ref = next(iter(latest.values()))
        if attention_mask is None:
            mask = torch.ones(ref.shape[:2], dtype=torch.float32, device=ref.device)
        else:
            mask = attention_mask.to(device=ref.device, dtype=torch.float32)

        weight = mask.unsqueeze(-1)  # [B, S, 1]
        batch_tokens = float(mask.sum().item())
        if batch_tokens <= 0.0:
            return

        for layer_idx, tensor in latest.items():
            if layer_idx not in self.sums:
                continue
            contribution = (tensor.float() * weight).sum(dim=(0, 1))
            self.sums[layer_idx] += contribution.to(self.device)

        self.token_count += batch_tokens

    def finalize(self) -> dict[int, torch.Tensor]:
        if self.token_count <= 0.0:
            raise ValueError("No tokens observed for mean computation.")
        return {
            layer_idx: (self.sums[layer_idx] / self.token_count)
            for layer_idx in self.layers
        }


def compute_mean_mlp_outputs(
    *,
    model: nn.Module,
    encoded_batches,
    n_clt_layers: int,
    d_model: int,
    device: str,
    run_logits,
    progress=None,
) -> dict[int, torch.Tensor]:
    """One pass over the eval set to get per-layer mean MLP outputs.

    `encoded_batches` is an iterable of already-encoded dicts (with
    attention_mask). `run_logits(model, encoded)` runs the forward pass; its
    return value is ignored here (we only need the tapped activations).
    """
    layers = _covered_layers(model, n_clt_layers)
    accumulator = MeanMlpOutputAccumulator(layers, d_model, device)

    iterator = encoded_batches if progress is None else progress(encoded_batches)
    with _MlpOutputTap(model, layers) as tap:
        for encoded in iterator:
            tap.latest = {}
            run_logits(model, encoded)
            accumulator.observe(tap.latest, encoded.get("attention_mask"))

    return accumulator.finalize()


# ---------------------------------------------------------------------------
# Constant-substitution baselines: zero and mean
# ---------------------------------------------------------------------------

class ConstantReplacementHook:
    """Replace `layer.mlp` outputs with a constant: zeros or per-layer mean.

    mode="zero": output -> zeros_like(output)
    mode="mean": output -> per_layer_mean[layer] broadcast over [batch, seq]
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        mode: str,
        n_clt_layers: int,
        per_layer_value: dict[int, torch.Tensor] | None = None,
    ):
        if mode not in {"zero", "mean"}:
            raise ValueError(f"Unsupported constant baseline mode={mode!r}.")
        if mode == "mean" and not per_layer_value:
            raise ValueError("mode='mean' requires per_layer_value.")

        self.model = model
        self.mode = mode
        self.layers = _covered_layers(model, n_clt_layers)
        self.per_layer_value = per_layer_value or {}
        self.handles: list[Any] = []

    def _make_hook(self, layer_idx: int):
        def hook(module, inputs, output):
            tensor, rest = _split_mlp_output(output)

            if self.mode == "zero":
                replaced = torch.zeros_like(tensor)
            else:
                value = self.per_layer_value[layer_idx].to(
                    device=tensor.device,
                    dtype=tensor.dtype,
                )
                replaced = value.view(1, 1, -1).expand_as(tensor).clone()

            return _pack_mlp_output(replaced, rest)

        return hook

    def install(self):
        if self.handles:
            raise RuntimeError("Constant replacement hooks already installed.")
        mlp_modules = get_mlp_modules(self.model)
        for layer_idx in self.layers:
            self.handles.append(
                mlp_modules[layer_idx].register_forward_hook(
                    self._make_hook(layer_idx)
                )
            )
        return self

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def __enter__(self):
        return self.install()

    def __exit__(self, exc_type, exc, tb):
        self.remove()


# ---------------------------------------------------------------------------
# Random untrained CLT baseline
# ---------------------------------------------------------------------------

def build_random_clt_like(
    autoencoder: CrossLayerTranscoder,
    *,
    device: str,
    seed: int = 0,
) -> CrossLayerTranscoder:
    """A fresh, untrained CLT with the same shape/normalization as `autoencoder`.

    Reuses the trained CLT's normalization buffers (if any) so the only
    difference is the *learned* encoder/decoder/threshold weights. The global
    RNG is snapshotted and restored so seeding here has no side effects on the
    rest of the eval.
    """
    rng_state = torch.get_rng_state()
    try:
        torch.manual_seed(int(seed))
        random_clt = CrossLayerTranscoder(
            n_layers=autoencoder.n_layers,
            d_model=autoencoder.d_model,
            features_per_layer=autoencoder.features_per_layer,
            decoder_init_scale=autoencoder.decoder_init_scale,
            normalize_inputs=autoencoder.normalize_inputs,
            normalize_targets=autoencoder.normalize_targets,
            normalization_eps=autoencoder.normalization_eps,
            normalization_momentum=autoencoder.normalization_momentum,
        )
    finally:
        torch.set_rng_state(rng_state)

    # Match learned normalization statistics so the control differs only in the
    # trained read/write weights, not in the input/output scaling.
    random_state = random_clt.state_dict()
    trained_state = autoencoder.state_dict()
    for key in random_state:
        if any(
            key.startswith(prefix)
            for prefix in ("input_mean", "input_var", "target_mean", "target_var",
                           "normalization_updates")
        ):
            if key in trained_state:
                random_state[key] = trained_state[key].clone()
    random_clt.load_state_dict(random_state, strict=True)

    random_clt.to(device)
    random_clt.eval()
    return random_clt
