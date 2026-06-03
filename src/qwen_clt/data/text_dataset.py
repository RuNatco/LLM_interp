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


def iter_token_batches(
    cfg: dict,
    tokenizer: PreTrainedTokenizerBase,
    device: str,
) -> Iterator[TokenBatch]:
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    ds = load_dataset(
        data_cfg["dataset_name"],
        data_cfg.get("dataset_config"),
        split=data_cfg.get("split", "train"),
    )
    text_field = data_cfg.get("text_field", "text")
    seq_len = int(data_cfg["seq_len"])
    batch_size = int(cfg["training"]["batch_size_sequences"])
    max_tokens = int(data_cfg.get("max_train_tokens", 1_000_000))
    use_chat_template = bool(model_cfg.get("chat_template", False))

    buffer: list[torch.Tensor] = []
    emitted_tokens = 0

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
                input_ids = torch.stack(buffer).to(device)
                attention_mask = torch.ones_like(input_ids, device=device)
                emitted_tokens += int(input_ids.numel())
                yield TokenBatch(input_ids=input_ids, attention_mask=attention_mask)
                buffer = []
                if emitted_tokens >= max_tokens:
                    return
