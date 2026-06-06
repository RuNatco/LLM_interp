from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import _path_setup  # noqa: F401

import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from qwen_clt.utils.config import load_config
from qwen_clt.utils.seed import set_seed
from qwen_clt.replacement import (
    LayerReplacementHook,
    ReplacementConfig,
    compute_replacement_metrics,
    load_autoencoder_from_checkpoint,
    metrics_to_dict,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen + CLT replacement model."
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment YAML config.",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional override for CLT checkpoint path.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional override for output metrics path.",
    )

    parser.add_argument(
        "--max-eval-tokens",
        type=int,
        default=None,
        help="Optional override for replacement_eval.max_eval_tokens.",
    )

    parser.add_argument(
        "--batch-size-sequences",
        type=int,
        default=None,
        help="Optional override for replacement_eval.batch_size_sequences.",
    )

    return parser.parse_args()


def ensure_parent_dir(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def get_torch_dtype(dtype_name: str):
    dtype_name = str(dtype_name).lower()

    if dtype_name in {"float16", "fp16"}:
        return torch.float16

    if dtype_name in {"bfloat16", "bf16"}:
        return torch.bfloat16

    if dtype_name in {"float32", "fp32"}:
        return torch.float32

    raise ValueError(f"Unsupported dtype: {dtype_name}")


def batch_iter(items: list[str], batch_size: int) -> Iterable[list[str]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def format_text_for_model(
    text: str,
    tokenizer,
    cfg: dict[str, Any],
) -> str:
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]

    use_chat_template = bool(model_cfg.get("chat_template", False))

    if not use_chat_template:
        return text

    wrapper = data_cfg.get("instruct_wrapper", {})
    system = wrapper.get("system", "You are a helpful assistant.")
    user_prefix = wrapper.get("user_prefix", "Continue or answer the following text:")

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"{user_prefix}\n\n{text}"},
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def load_eval_texts(
    cfg: dict[str, Any],
    tokenizer,
    max_eval_tokens: int,
) -> list[str]:
    data_cfg = cfg["data"]

    dataset_name = data_cfg["dataset_name"]
    dataset_config = data_cfg.get("dataset_config")
    split = data_cfg.get("split", "train")
    text_field = data_cfg.get("text_field", "text")

    if dataset_config:
        dataset = load_dataset(dataset_name, dataset_config, split=split)
    else:
        dataset = load_dataset(dataset_name, split=split)

    texts: list[str] = []
    total_tokens = 0

    for row in dataset:
        raw_text = row.get(text_field, "")

        if raw_text is None:
            continue

        raw_text = str(raw_text).strip()

        if not raw_text:
            continue

        formatted_text = format_text_for_model(raw_text, tokenizer, cfg)

        token_count = len(
            tokenizer(
                formatted_text,
                add_special_tokens=True,
                truncation=False,
            )["input_ids"]
        )

        if token_count == 0:
            continue

        texts.append(formatted_text)
        total_tokens += token_count

        if total_tokens >= max_eval_tokens:
            break

    if len(texts) == 0:
        raise ValueError(
            "No evaluation texts loaded. "
            f"Check dataset={dataset_name}, split={split}, text_field={text_field}."
        )

    return texts


@torch.no_grad()
def encode_texts(
    tokenizer,
    texts: list[str],
    device: str,
    seq_len: int,
) -> dict[str, torch.Tensor]:
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=seq_len,
        return_tensors="pt",
    )

    return {k: v.to(device) for k, v in encoded.items()}


@torch.no_grad()
def run_model_logits(
    model,
    encoded: dict[str, torch.Tensor],
) -> torch.Tensor:
    outputs = model(**encoded)

    return outputs.logits.detach().cpu()


def single_token_id(tokenizer, text: str) -> int:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) != 1:
        raise ValueError(
            f"Expected a single-token string, got {text!r} -> {token_ids}"
        )
    return int(token_ids[0])


def target_token_ids_from_config(
    tokenizer,
    eval_cfg: dict[str, Any],
) -> tuple[tuple[int, int] | None, dict[str, Any] | None]:
    target_cfg = eval_cfg.get("target_logit_diff")
    if not target_cfg:
        return None, None

    positive = str(target_cfg["positive"])
    negative = str(target_cfg["negative"])
    positive_id = single_token_id(tokenizer, positive)
    negative_id = single_token_id(tokenizer, negative)

    metadata = {
        "positive": positive,
        "negative": negative,
        "positive_token_id": positive_id,
        "negative_token_id": negative_id,
        "target_pos": int(target_cfg.get("target_pos", -1)),
        "score": "logit(positive)-logit(negative)",
    }

    return (positive_id, negative_id), metadata


def _metric_weight_key(metric_name: str) -> str:
    if metric_name.startswith("last_token_"):
        return "num_sequences"
    if metric_name.startswith("target_logit_diff_"):
        return "num_sequences"
    return "num_tokens"


def aggregate_batch_metrics(
    batch_metrics: list[dict[str, float | int]],
) -> dict[str, float | int]:
    if len(batch_metrics) == 0:
        raise ValueError("No batch metrics to aggregate.")

    result: dict[str, float | int] = {}
    metric_names = sorted({key for metrics in batch_metrics for key in metrics})

    for key in metric_names:
        if key in {"num_tokens", "num_sequences"}:
            result[key] = int(sum(int(m.get(key, 0)) for m in batch_metrics))
            continue

        weight_key = _metric_weight_key(key)
        weighted_sum = 0.0
        total_weight = 0.0

        for metrics in batch_metrics:
            if key not in metrics:
                continue
            weight = float(metrics.get(weight_key, 1.0))
            weighted_sum += float(metrics[key]) * weight
            total_weight += weight

        if total_weight > 0.0:
            result[key] = weighted_sum / total_weight

    return result


def make_replacement_config(
    replacement_cfg: dict[str, Any],
    clt_cfg: dict[str, Any],
    *,
    layer_idx: int | None = None,
    replacement_layers: tuple[int, ...] | None = None,
) -> ReplacementConfig:
    return ReplacementConfig(
        layer_idx=layer_idx,
        replacement_layers=replacement_layers,
        read_from=replacement_cfg.get("read_from", clt_cfg.get("input_kind")),
        replace_at=replacement_cfg.get("replace_at", "mlp_output"),
        replace_mode=replacement_cfg.get("replace_mode", "full"),
        detach_reconstruction=bool(replacement_cfg.get("detach_reconstruction", False)),
    )


@torch.no_grad()
def evaluate_replacement(
    *,
    model,
    tokenizer,
    autoencoder,
    eval_texts: list[str],
    replacement_config: ReplacementConfig,
    device: str,
    seq_len: int,
    batch_size_sequences: int,
    desc: str,
    target_token_ids: tuple[int, int] | None,
    target_pos: int,
) -> tuple[dict[str, float | int], list[dict[str, float | int]]]:
    batch_metrics: list[dict[str, float | int]] = []
    num_batches = math.ceil(len(eval_texts) / batch_size_sequences)

    for texts in tqdm(
        batch_iter(eval_texts, batch_size_sequences),
        total=num_batches,
        desc=desc,
    ):
        encoded = encode_texts(
            tokenizer=tokenizer,
            texts=texts,
            device=device,
            seq_len=seq_len,
        )
        attention_mask = encoded.get("attention_mask")
        attention_mask_cpu = (
            attention_mask.detach().cpu() if attention_mask is not None else None
        )

        original_logits = run_model_logits(
            model=model,
            encoded=encoded,
        )

        with LayerReplacementHook(
            model=model,
            autoencoder=autoencoder,
            config=replacement_config,
        ):
            replacement_logits = run_model_logits(
                model=model,
                encoded=encoded,
            )

        metrics = compute_replacement_metrics(
            original_logits=original_logits,
            replacement_logits=replacement_logits,
            attention_mask=attention_mask_cpu,
            target_token_ids=target_token_ids,
            target_pos=target_pos,
        )

        batch_metrics.append(metrics_to_dict(metrics))

    return aggregate_batch_metrics(batch_metrics), batch_metrics


def diagnostic_modes(
    diagnostics_cfg: dict[str, Any],
    n_layers: int,
) -> list[tuple[str, dict[str, Any]]]:
    if not bool(diagnostics_cfg.get("enabled", False)):
        return []

    modes: list[tuple[str, dict[str, Any]]] = []

    if bool(diagnostics_cfg.get("layerwise", True)):
        stride = int(diagnostics_cfg.get("layer_stride", 1))
        for layer_idx in range(0, n_layers, stride):
            modes.append(
                (
                    f"layer_{layer_idx}",
                    {
                        "kind": "single_layer",
                        "layer_idx": layer_idx,
                    },
                )
            )

    if bool(diagnostics_cfg.get("prefix", True)):
        stride = int(diagnostics_cfg.get("prefix_stride", 1))
        for layer_idx in range(0, n_layers, stride):
            modes.append(
                (
                    f"prefix_0_to_{layer_idx}",
                    {
                        "kind": "prefix",
                        "replacement_layers": tuple(range(layer_idx + 1)),
                    },
                )
            )

    return modes


def validate_config(cfg: dict[str, Any]) -> None:
    for section in ["project", "model", "data", "clt", "replacement", "replacement_eval"]:
        if section not in cfg:
            raise KeyError(f"Missing config section: {section}")

    if not bool(cfg["replacement"].get("enabled", False)):
        raise ValueError("replacement.enabled must be true for this script.")

    if not bool(cfg["replacement_eval"].get("enabled", False)):
        raise ValueError("replacement_eval.enabled must be true for this script.")

    replace_mode = cfg["replacement"].get("replace_mode", "full")
    if replace_mode != "full":
        raise ValueError(
            f"Unsupported replacement.replace_mode={replace_mode}. "
            "Current baseline supports only replace_mode='full'."
        )

    read_from = cfg["replacement"].get("read_from", cfg["clt"].get("input_kind"))
    replace_at = cfg["replacement"].get("replace_at")
    input_kind = cfg["clt"].get("input_kind")

    if read_from != input_kind:
        raise ValueError(
            f"replacement.read_from={read_from} must match clt.input_kind={input_kind}. "
            "The replacement hook should encode the same activation point used for CLT training."
        )

    if replace_at != "mlp_output":
        raise ValueError(
            f"replacement.replace_at={replace_at} is unsupported. "
            "CLT reconstructs MLP outputs, so replacement.replace_at must be 'mlp_output'."
        )


def main() -> None:
    args = parse_args()

    cfg = load_config(args.config)
    validate_config(cfg)

    seed = int(cfg.get("training", {}).get("seed", 42))
    set_seed(seed)

    project_cfg = cfg["project"]
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    clt_cfg = cfg["clt"]
    replacement_cfg = cfg["replacement"]
    eval_cfg = cfg["replacement_eval"]

    project_name = project_cfg["name"]

    model_name = model_cfg["name"]
    device = model_cfg.get("device", "cuda")
    dtype = get_torch_dtype(model_cfg.get("dtype", "bfloat16"))
    trust_remote_code = bool(model_cfg.get("trust_remote_code", True))
    use_cache = bool(model_cfg.get("use_cache", False))

    seq_len = int(data_cfg["seq_len"])

    max_eval_tokens = (
        int(args.max_eval_tokens)
        if args.max_eval_tokens is not None
        else int(eval_cfg.get("max_eval_tokens", data_cfg.get("max_eval_tokens", 50000)))
    )

    batch_size_sequences = (
        int(args.batch_size_sequences)
        if args.batch_size_sequences is not None
        else int(eval_cfg.get("batch_size_sequences", 2))
    )

    checkpoint_path = Path(
        args.checkpoint
        if args.checkpoint is not None
        else replacement_cfg["checkpoint_path"]
    )

    output_path = Path(
        args.output
        if args.output is not None
        else eval_cfg["output_path"]
    )

    ensure_parent_dir(output_path)

    print("=" * 80)
    print("Replacement model evaluation")
    print("=" * 80)
    print(f"Project: {project_name}")
    print(f"Config: {args.config}")
    print(f"Model: {model_name}")
    print(f"Device: {device}")
    print(f"Dtype: {dtype}")
    print(f"Seq len: {seq_len}")
    print(f"CLT checkpoint: {checkpoint_path}")
    print(f"Replacement read from: {replacement_cfg.get('read_from', clt_cfg.get('input_kind'))}")
    print(f"Replacement point: {replacement_cfg.get('replace_at')}")
    print(f"Max eval tokens: {max_eval_tokens}")
    print(f"Batch size sequences: {batch_size_sequences}")
    print(f"Output: {output_path}")
    print("=" * 80)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"CLT checkpoint not found: {checkpoint_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
        device_map=None,
    )

    model.to(device)
    model.eval()
    model.config.use_cache = use_cache

    autoencoder = load_autoencoder_from_checkpoint(
        checkpoint_path=checkpoint_path,
        device=device,
    )

    autoencoder.eval()

    eval_texts = load_eval_texts(
        cfg=cfg,
        tokenizer=tokenizer,
        max_eval_tokens=max_eval_tokens,
    )

    print(f"Loaded eval sequences: {len(eval_texts)}")

    target_token_ids, target_metadata = target_token_ids_from_config(
        tokenizer=tokenizer,
        eval_cfg=eval_cfg,
    )
    target_pos = (
        int(target_metadata["target_pos"])
        if target_metadata is not None
        else -1
    )

    replacement_config = make_replacement_config(
        replacement_cfg=replacement_cfg,
        clt_cfg=clt_cfg,
    )

    aggregated_metrics, batch_metrics = evaluate_replacement(
        model=model,
        tokenizer=tokenizer,
        autoencoder=autoencoder,
        eval_texts=eval_texts,
        replacement_config=replacement_config,
        device=device,
        seq_len=seq_len,
        batch_size_sequences=batch_size_sequences,
        desc="Evaluating full replacement",
        target_token_ids=target_token_ids,
        target_pos=target_pos,
    )

    diagnostics_cfg = eval_cfg.get("diagnostics", {})
    diagnostics: dict[str, Any] = {}
    diagnostics_texts: list[str] | None = None
    diagnostics_max_eval_tokens = int(
        diagnostics_cfg.get("max_eval_tokens", min(max_eval_tokens, 5000))
    )

    for mode_name, mode_cfg in diagnostic_modes(
        diagnostics_cfg=diagnostics_cfg,
        n_layers=autoencoder.n_layers,
    ):
        if diagnostics_texts is None:
            diagnostics_texts = load_eval_texts(
                cfg=cfg,
                tokenizer=tokenizer,
                max_eval_tokens=diagnostics_max_eval_tokens,
            )
            print(f"Loaded diagnostic eval sequences: {len(diagnostics_texts)}")

        if mode_cfg["kind"] == "single_layer":
            diagnostic_config = make_replacement_config(
                replacement_cfg=replacement_cfg,
                clt_cfg=clt_cfg,
                layer_idx=int(mode_cfg["layer_idx"]),
            )
        elif mode_cfg["kind"] == "prefix":
            diagnostic_config = make_replacement_config(
                replacement_cfg=replacement_cfg,
                clt_cfg=clt_cfg,
                replacement_layers=mode_cfg["replacement_layers"],
            )
        else:
            raise ValueError(f"Unknown diagnostic kind: {mode_cfg['kind']}")

        diagnostic_metrics, _ = evaluate_replacement(
            model=model,
            tokenizer=tokenizer,
            autoencoder=autoencoder,
            eval_texts=diagnostics_texts,
            replacement_config=diagnostic_config,
            device=device,
            seq_len=seq_len,
            batch_size_sequences=batch_size_sequences,
            desc=f"Diagnostic {mode_name}",
            target_token_ids=target_token_ids,
            target_pos=target_pos,
        )

        diagnostics[mode_name] = {
            "config": {
                key: value
                for key, value in mode_cfg.items()
                if key != "replacement_layers"
            },
            "replacement_layers": (
                list(mode_cfg["replacement_layers"])
                if "replacement_layers" in mode_cfg
                else None
            ),
            "max_eval_tokens": diagnostics_max_eval_tokens,
            "metrics": diagnostic_metrics,
        }

    result: dict[str, Any] = {
        "project": project_name,
        "config_path": args.config,
        "model_name": model_name,
        "checkpoint_path": str(checkpoint_path),
        "replacement": {
            "read_from": replacement_cfg.get("read_from", clt_cfg.get("input_kind")),
            "replace_at": replacement_cfg.get("replace_at"),
            "replace_mode": replacement_cfg.get("replace_mode", "full"),
            "detach_reconstruction": bool(
                replacement_cfg.get("detach_reconstruction", False)
            ),
        },
        "eval": {
            "seq_len": seq_len,
            "max_eval_tokens": max_eval_tokens,
            "batch_size_sequences": batch_size_sequences,
            "num_eval_sequences": len(eval_texts),
        },
        "target_logit_diff": target_metadata,
        "clt": {
            "n_layers": clt_cfg.get("n_layers"),
            "d_model": clt_cfg.get("d_model"),
            "features_per_layer": clt_cfg.get("features_per_layer"),
            "input_kind": clt_cfg.get("input_kind"),
            "nonlinearity": clt_cfg.get("nonlinearity"),
            "normalization": clt_cfg.get("normalization", {}),
        },
        "metrics": aggregated_metrics,
    }

    if bool(eval_cfg.get("save_batch_metrics", True)):
        result["batch_metrics"] = batch_metrics

    if diagnostics:
        result["diagnostics"] = diagnostics

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("\nReplacement metrics:")
    for key, value in aggregated_metrics.items():
        if isinstance(value, int):
            print(f"{key}: {value}")
        else:
            print(f"{key}: {value:.6f}")

    if diagnostics:
        print("\nDiagnostics:")
        for mode_name, payload in diagnostics.items():
            mode_metrics = payload["metrics"]
            print(
                f"{mode_name}: "
                f"last_top1={mode_metrics.get('last_token_top1_agreement', 0.0):.4f} "
                f"last_kl={mode_metrics.get('last_token_kl_div', 0.0):.4f} "
                f"top1={mode_metrics.get('top1_agreement', 0.0):.4f}"
            )

    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
