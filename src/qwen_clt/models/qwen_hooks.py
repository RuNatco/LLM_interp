from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class QwenActivations:
    mlp_inputs: list[torch.Tensor]
    mlp_outputs: list[torch.Tensor]
    logits: torch.Tensor


def dtype_from_name(name: str) -> torch.dtype:
    table = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    return table.get(name.lower(), torch.float32)


def load_qwen_model_and_tokenizer(cfg: dict[str, Any]):
    model_cfg = cfg["model"]
    dtype = dtype_from_name(model_cfg.get("dtype", "bfloat16"))
    device = model_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["name"],
        trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name"],
        torch_dtype=dtype,
        trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
        device_map=None,
    ).to(device)
    model.config.use_cache = bool(model_cfg.get("use_cache", False))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tokenizer


class QwenMLPHookCollector:

    def __init__(self, model):
        self.model = model
        self.handles = []
        self.mlp_inputs: list[torch.Tensor] = []
        self.mlp_outputs: list[torch.Tensor] = []

    def _make_hook(self, layer_idx: int):
        def hook(module, inputs, output):
            self.mlp_inputs[layer_idx] = inputs[0]
            self.mlp_outputs[layer_idx] = output
        return hook

    def __enter__(self):
        layers = self.model.model.layers
        self.mlp_inputs = [None for _ in range(len(layers))]
        self.mlp_outputs = [None for _ in range(len(layers))]
        for idx, layer in enumerate(layers):
            self.handles.append(layer.mlp.register_forward_hook(self._make_hook(idx)))
        return self

    def __exit__(self, exc_type, exc, tb):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    @torch.no_grad()
    def run(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> QwenActivations:
        with self:
            out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        if any(x is None for x in self.mlp_inputs) or any(y is None for y in self.mlp_outputs):
            raise RuntimeError("Failed to collect all MLP activations. Check model architecture/hook names.")
        return QwenActivations(
            mlp_inputs=[x.detach() for x in self.mlp_inputs],
            mlp_outputs=[y.detach() for y in self.mlp_outputs],
            logits=out.logits.detach(),
        )
