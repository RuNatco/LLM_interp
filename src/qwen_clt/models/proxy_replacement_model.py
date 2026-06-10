from __future__ import annotations

import torch

from qwen_clt.models.cross_layer_transcoder import (
    CrossLayerTranscoder,
    clt_normalization_kwargs,
)
from qwen_clt.models.qwen_hooks import load_qwen_model_and_tokenizer, QwenMLPHookCollector
from qwen_clt.interventions import FeatureIntervention
from qwen_clt.replacement import LayerReplacementHook, ReplacementConfig


class ProxyQwenReplacementModel:
    """Proxy wrapper for CLT feature analysis.

    It exposes the CLT feature space needed by v0 attribution graphs and uses
    LayerReplacementHook for causal feature interventions inside a real forward
    pass:

    - collect original Qwen MLP inputs/outputs;
    - encode them with the trained CLT;
    - run feature interventions through an in-forward CLT replacement hook.

    Use qwen_clt.replacement.LayerReplacementHook when the experiment needs
    replacement logits and metrics from a real forward pass.
    """

    def __init__(self, base_model, tokenizer, clt: CrossLayerTranscoder, cfg: dict):
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.clt = clt
        self.cfg = cfg
        self.collector = QwenMLPHookCollector(base_model)
        self.device = next(base_model.parameters()).device

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, cfg: dict | None = None):
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if not isinstance(ckpt, dict):
            raise TypeError(
                f"Expected checkpoint dict, got {type(ckpt)} "
                f"in {checkpoint_path}."
            )
        if "model_state_dict" not in ckpt:
            raise KeyError(
                f"Checkpoint {checkpoint_path} has no 'model_state_dict'. "
                f"Available keys: {list(ckpt.keys())}"
            )
        cfg = cfg or ckpt.get("cfg")
        if cfg is None:
            raise KeyError(
                f"Checkpoint {checkpoint_path} has no 'cfg' and none was provided."
            )
        base_model, tokenizer = load_qwen_model_and_tokenizer(cfg)
        clt_cfg = cfg["clt"]
        device = next(base_model.parameters()).device
        clt = CrossLayerTranscoder(
            n_layers=int(clt_cfg["n_layers"]),
            d_model=int(clt_cfg["d_model"]),
            features_per_layer=int(clt_cfg["features_per_layer"]),
            init_threshold=float(clt_cfg.get("init_threshold", 0.0)),
            decoder_init_scale=float(clt_cfg.get("decoder_init_scale", 0.02)),
            **clt_normalization_kwargs(clt_cfg),
        ).to(device)
        # strict=True: any key mismatch indicates a real architecture divergence
        # and should surface immediately rather than silently produce wrong results.
        try:
            clt.load_state_dict(ckpt["model_state_dict"], strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                f"CLT state dict mismatch loading {checkpoint_path}. "
                "If you changed the CLT architecture (n_layers, d_model, "
                "features_per_layer), create a fresh checkpoint rather than "
                f"loading an incompatible one.\nOriginal error: {exc}"
            ) from exc
        clt.eval()
        return cls(base_model, tokenizer, clt, cfg)

    def format_prompt(self, prompt: str) -> str:
        model_cfg = self.cfg.get("model", {})
        if not bool(model_cfg.get("chat_template", False)):
            return prompt

        wrapper = self.cfg.get("data", {}).get("instruct_wrapper", {}) or {}
        system = wrapper.get("system", "You are a helpful assistant.")
        user_prefix = wrapper.get(
            "user_prefix",
            "Continue or answer the following text:",
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"{user_prefix}\n\n{prompt}"},
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def tokenize(self, prompt: str):
        formatted_prompt = self.format_prompt(prompt)
        toks = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        return toks["input_ids"], toks.get("attention_mask")

    @torch.no_grad()
    def get_activations(self, prompt: str):
        input_ids, attention_mask = self.tokenize(prompt)
        acts = self.collector.run(input_ids, attention_mask)
        features, recons = self.clt(acts.mlp_inputs)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "logits": acts.logits,
            "mlp_inputs": acts.mlp_inputs,
            "mlp_outputs": acts.mlp_outputs,
            "features": features,
            "mlp_recons": recons,
        }

    @torch.no_grad()
    def _forward_logits(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None):
        out = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        return out.logits.detach()

    def _layer_dict_to_list(self, values: dict[int, torch.Tensor]) -> list[torch.Tensor | None]:
        return [values.get(i) for i in range(self.clt.n_layers)]

    @torch.no_grad()
    def feature_intervention(self, prompt: str, interventions: list[FeatureIntervention]):
        """Run feature interventions causally inside the CLT replacement model.

        Runs exactly 2 forward passes (baseline replacement + intervened),
        reusing the tokenized input.

        Returned logits:
        - `logits`: original Qwen logits (from baseline pass, no replacement);
        - `replacement_logits`: logits with CLT reconstruction replacing MLP outputs;
        - `intervened_logits`: logits with the same replacement plus feature edits.

        The causal feature effect should be measured against `replacement_logits`,
        because CLT reconstruction itself can move logits relative to the original
        model.
        """
        input_ids, attention_mask = self.tokenize(prompt)

        # Pass 1: baseline replacement (also captures original logits via hook)
        replacement_config = ReplacementConfig()
        with LayerReplacementHook(
            model=self.base_model,
            autoencoder=self.clt,
            config=replacement_config,
        ) as replacement_hook:
            out = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            replacement_logits = out.logits.detach()

        # Recompute original logits + activations without replacement in one pass
        acts = self.collector.run(input_ids, attention_mask)
        original_logits = acts.logits
        features, recons = self.clt(acts.mlp_inputs)

        # Pass 2: intervened replacement
        intervention_config = ReplacementConfig(
            feature_interventions=tuple(interventions),
        )
        with LayerReplacementHook(
            model=self.base_model,
            autoencoder=self.clt,
            config=intervention_config,
        ) as intervention_hook:
            out = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            intervened_logits = out.logits.detach()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "logits": original_logits,
            "mlp_inputs": acts.mlp_inputs,
            "mlp_outputs": acts.mlp_outputs,
            "features": features,
            "mlp_recons": recons,
            "replacement_logits": replacement_logits,
            "replacement_features": self._layer_dict_to_list(
                replacement_hook.features_by_layer
            ),
            "replacement_mlp_recons": self._layer_dict_to_list(
                replacement_hook.reconstructions_by_layer
            ),
            "intervened_logits": intervened_logits,
            "intervened_features": self._layer_dict_to_list(
                intervention_hook.features_by_layer
            ),
            "intervened_mlp_recons": self._layer_dict_to_list(
                intervention_hook.reconstructions_by_layer
            ),
            "intervention_logit_delta": intervened_logits - replacement_logits,
        }
