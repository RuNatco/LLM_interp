from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

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


def resolve_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


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
def run_model_logits(
    model,
    tokenizer,
    texts: list[str],
    device: str,
    seq_len: int,
) -> torch.Tensor:
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=seq_len,
        return_tensors="pt",
    )

    encoded = {k: v.to(device) for k, v in encoded.items()}

    outputs = model(**encoded)

    return outputs.logits.detach().cpu()


def aggregate_batch_metrics(batch_metrics: list[dict[str, float]]) -> dict[str, float]:
    if len(batch_metrics) == 0:
        raise ValueError("No batch metrics to aggregate.")

    result: dict[str, float] = {}

    for key in batch_metrics[0].keys():
        values = [float(m[key]) for m in batch_metrics]
        result[key] = float(sum(values) / len(values))

    return result


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

    checkpoint_path = (
        resolve_path(args.checkpoint)
        if args.checkpoint is not None
        else resolve_path(replacement_cfg["checkpoint_path"])
    )

    output_path = (
        resolve_path(args.output)
        if args.output is not None
        else resolve_path(eval_cfg["output_path"])
    )

    ensure_parent_dir(output_path)

    print("=" * 80)
    print("Replacement model evaluation")
    print("=" * 80)
    print(f"Project: {project_name}")
    print(f"Config: {resolve_path(args.config)}")
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

    replacement_config = ReplacementConfig(
        layer_idx=None,
        read_from=replacement_cfg.get("read_from", clt_cfg.get("input_kind")),
        replace_at=replacement_cfg.get("replace_at", "mlp_output"),
        replace_mode=replacement_cfg.get("replace_mode", "full"),
        detach_reconstruction=bool(replacement_cfg.get("detach_reconstruction", False)),
    )

    batch_metrics: list[dict[str, float]] = []

    num_batches = math.ceil(len(eval_texts) / batch_size_sequences)

    for texts in tqdm(
        batch_iter(eval_texts, batch_size_sequences),
        total=num_batches,
        desc="Evaluating replacement",
    ):
        original_logits = run_model_logits(
            model=model,
            tokenizer=tokenizer,
            texts=texts,
            device=device,
            seq_len=seq_len,
        )

        with LayerReplacementHook(
            model=model,
            autoencoder=autoencoder,
            config=replacement_config,
        ):
            replacement_logits = run_model_logits(
                model=model,
                tokenizer=tokenizer,
                texts=texts,
                device=device,
                seq_len=seq_len,
            )

        metrics = compute_replacement_metrics(
            original_logits=original_logits,
            replacement_logits=replacement_logits,
        )

        batch_metrics.append(metrics_to_dict(metrics))

    aggregated_metrics = aggregate_batch_metrics(batch_metrics)

    result: dict[str, Any] = {
        "project": project_name,
        "config_path": str(resolve_path(args.config)),
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
        "clt": {
            "n_layers": clt_cfg.get("n_layers"),
            "d_model": clt_cfg.get("d_model"),
            "features_per_layer": clt_cfg.get("features_per_layer"),
            "input_kind": clt_cfg.get("input_kind"),
            "nonlinearity": clt_cfg.get("nonlinearity"),
        },
        "metrics": aggregated_metrics,
    }

    if bool(eval_cfg.get("save_batch_metrics", True)):
        result["batch_metrics"] = batch_metrics

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("\nReplacement metrics:")
    for key, value in aggregated_metrics.items():
        print(f"{key}: {value:.6f}")

    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
