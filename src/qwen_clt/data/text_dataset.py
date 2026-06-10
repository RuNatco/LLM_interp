from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator
import torch
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase


@dataclass
class TokenBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor


def _wrap_instruct(text: str, cfg: dict, tokenizer: PreTrainedTokenizerBase) -> str:
    wrapper = cfg.get("data", {}).get("instruct_wrapper") or {}
    system = wrapper.get("system", "You are a helpful assistant.")
    user_prefix = wrapper.get("user_prefix", "Continue the following text:")
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"{user_prefix}\n{text}"},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"System: {system}\nUser: {user_prefix}\n{text}\nAssistant:"


def _iter_batches_from_split(
    cfg: dict,
    tokenizer: PreTrainedTokenizerBase,
    device: str,
    *,
    split: str,
    max_tokens: int,
    batch_size: int,
    rank: int = 0,
    world_size: int = 1,
) -> Iterator[TokenBatch]:
    """Shared tokenization + batching logic for any HuggingFace split.

    Multi-GPU: each rank takes every world_size-th batch starting at its rank
    (interleave). This ensures non-overlapping data across ranks without
    requiring the dataset to be pre-split.

    max_tokens counts tokens emitted by *this rank only*.
    """
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    ds = load_dataset(
        data_cfg["dataset_name"],
        data_cfg.get("dataset_config"),
        split=split,
    )
    text_field = data_cfg.get("text_field", "text")
    seq_len = int(data_cfg["seq_len"])
    use_chat_template = bool(model_cfg.get("chat_template", False))

    buffer: list[torch.Tensor] = []
    emitted_tokens = 0
    batch_index = 0  # global batch counter used for rank assignment

    for row in ds:
        text = (row.get(text_field) or "").strip()
        if not text:
            continue
        if use_chat_template:
            text = _wrap_instruct(text, cfg, tokenizer)
        ids = tokenizer(text, add_special_tokens=True, return_tensors="pt")["input_ids"][0]
        if ids.numel() < 2:
            continue
        for start in range(0, ids.numel() - 1, seq_len):
            chunk = ids[start : start + seq_len]
            if chunk.numel() < seq_len:
                continue
            buffer.append(chunk)
            if len(buffer) == batch_size:
                # Interleave: rank r processes batches 0, world_size, 2*world_size, ...
                # offset by r, i.e. batch_index % world_size == rank
                if batch_index % world_size == rank:
                    input_ids = torch.stack(buffer).to(device)
                    attention_mask = torch.ones_like(input_ids, device=device)
                    emitted_tokens += int(input_ids.numel())
                    yield TokenBatch(input_ids=input_ids, attention_mask=attention_mask)
                    if emitted_tokens >= max_tokens:
                        return
                batch_index += 1
                buffer = []


def iter_token_batches(
    cfg: dict,
    tokenizer: PreTrainedTokenizerBase,
    device: str,
    rank: int = 0,
    world_size: int = 1,
) -> Iterator[TokenBatch]:
    data_cfg = cfg["data"]
    yield from _iter_batches_from_split(
        cfg,
        tokenizer,
        device,
        split=data_cfg.get("split", "train"),
        max_tokens=int(data_cfg.get("max_train_tokens", 1_000_000)),
        batch_size=int(cfg["training"]["batch_size_sequences"]),
        rank=rank,
        world_size=world_size,
    )


def iter_eval_batches(
    cfg: dict,
    tokenizer: PreTrainedTokenizerBase,
    device: str,
) -> Iterator[TokenBatch]:
    """Iterate over the validation/test split for reconstruction eval.

    Uses data.eval_split (default: "test") and data.max_eval_tokens.
    Falls back gracefully to "validation" if "test" is unavailable.
    Batch size is taken from training.eval_batch_size_sequences (default: same
    as training.batch_size_sequences).
    """
    data_cfg = cfg["data"]
    training_cfg = cfg["training"]

    eval_split = data_cfg.get("eval_split", "test")
    max_eval_tokens = int(data_cfg.get("max_eval_tokens", 200_000))
    batch_size = int(
        training_cfg.get(
            "eval_batch_size_sequences",
            training_cfg["batch_size_sequences"],
        )
    )

    try:
        yield from _iter_batches_from_split(
            cfg,
            tokenizer,
            device,
            split=eval_split,
            max_tokens=max_eval_tokens,
            batch_size=batch_size,
        )
    except Exception as exc:
        # Some datasets don't have a "test" split — try "validation"
        if eval_split == "test":
            import warnings
            warnings.warn(
                f"[iter_eval_batches] split={eval_split!r} failed "
                f"({exc}), retrying with 'validation'."
            )
            yield from _iter_batches_from_split(
                cfg,
                tokenizer,
                device,
                split="validation",
                max_tokens=max_eval_tokens,
                batch_size=batch_size,
            )
        else:
            raise
