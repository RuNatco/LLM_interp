from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
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


# ---------------------------------------------------------------------------
# Pre-tokenized dataset
# ---------------------------------------------------------------------------

def _tokenize_and_chunk(
    dataset,
    tokenizer: PreTrainedTokenizerBase,
    *,
    text_field: str,
    seq_len: int,
    use_chat_template: bool,
    cfg: dict,
    num_proc: int = 8,
) -> Dataset:
    """Tokenize the whole HF dataset in parallel, then chunk into seq_len windows.

    Uses HF datasets .map() with batched=True for multi-process CPU tokenization.
    Result is cached automatically by HF datasets (keyed on tokenizer + seq_len).
    """

    def tokenize_batch(batch):
        texts = batch[text_field]
        if use_chat_template:
            texts = [_wrap_instruct(t, cfg, tokenizer) for t in texts]
        # Tokenize without truncation — we'll chunk manually
        enc = tokenizer(
            texts,
            add_special_tokens=True,
            truncation=False,
            return_attention_mask=False,
        )
        return {"input_ids": enc["input_ids"]}

    def chunk_batch(batch):
        all_ids = []
        for ids in batch["input_ids"]:
            for start in range(0, len(ids) - 1, seq_len):
                chunk = ids[start : start + seq_len]
                if len(chunk) == seq_len:
                    all_ids.append(chunk)
        return {"input_ids": all_ids}

    tokenized = dataset.map(
        tokenize_batch,
        batched=True,
        batch_size=1000,
        num_proc=num_proc,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )
    chunked = tokenized.map(
        chunk_batch,
        batched=True,
        batch_size=1000,
        num_proc=num_proc,
        desc="Chunking",
    )
    chunked.set_format(type="torch", columns=["input_ids"])
    return chunked


class ChunkedTokenDataset(Dataset):
    """Thin wrapper around a pre-tokenized HF dataset."""

    def __init__(self, hf_dataset):
        self._ds = hf_dataset

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int) -> dict:
        return {"input_ids": self._ds[idx]["input_ids"]}


def _collate(batch: list[dict]) -> TokenBatch:
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.ones_like(input_ids)
    return TokenBatch(input_ids=input_ids, attention_mask=attention_mask)


def _build_dataloader(
    cfg: dict,
    tokenizer: PreTrainedTokenizerBase,
    *,
    split: str,
    batch_size: int,
    rank: int = 0,
    world_size: int = 1,
    shuffle: bool = True,
    num_workers: int | None = None,
) -> DataLoader:
    """Build a DataLoader over a pre-tokenized + chunked HF dataset.

    Tokenization result is cached by HF datasets, so the second run is instant.
    Uses DistributedSampler for multi-GPU, which splits the dataset evenly
    across ranks without data overlap.
    """
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    seq_len = int(data_cfg["seq_len"])
    use_chat_template = bool(model_cfg.get("chat_template", False))
    text_field = data_cfg.get("text_field", "text")

    # Optional override for the HF datasets cache location. The raw dataset
    # download AND the derived tokenize/chunk .map() caches both live here,
    # so re-runs skip tokenization entirely.
    cache_dir = data_cfg.get("tokenized_cache_dir") or None

    raw_ds = load_dataset(
        data_cfg["dataset_name"],
        data_cfg.get("dataset_config"),
        split=split,
        cache_dir=cache_dir,
    )

    n_workers = num_workers if num_workers is not None else int(data_cfg.get("num_workers", 4))

    chunked = _tokenize_and_chunk(
        raw_ds,
        tokenizer,
        text_field=text_field,
        seq_len=seq_len,
        use_chat_template=use_chat_template,
        cfg=cfg,
        num_proc=max(1, n_workers),
    )
    dataset = ChunkedTokenDataset(chunked)

    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            drop_last=True,
        )
        shuffle = False  # DistributedSampler handles shuffle

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=n_workers,
        collate_fn=_collate,
        pin_memory=True,
        drop_last=True,
        persistent_workers=n_workers > 0,
    )


# ---------------------------------------------------------------------------
# Public iterators (unchanged signature — drop-in replacement)
# ---------------------------------------------------------------------------

def iter_token_batches(
    cfg: dict,
    tokenizer: PreTrainedTokenizerBase,
    device: str,
    rank: int = 0,
    world_size: int = 1,
) -> Iterator[TokenBatch]:
    """Iterate over training batches using a pre-tokenized cached dataset.

    On first call tokenizes the whole dataset (parallel, cached to disk).
    Subsequent calls are instant. Uses DistributedSampler for multi-GPU.
    """
    data_cfg = cfg["data"]
    max_tokens = int(data_cfg.get("max_train_tokens", 1_000_000))
    batch_size = int(cfg["training"]["batch_size_sequences"])

    loader = _build_dataloader(
        cfg,
        tokenizer,
        split=data_cfg.get("split", "train"),
        batch_size=batch_size,
        rank=rank,
        world_size=world_size,
        shuffle=True,
    )

    emitted_tokens = 0
    epoch = 0
    # Loop over the dataloader repeatedly until max_tokens is reached.
    # One pass through the loader may not be enough for large max_train_tokens.
    while True:
        if hasattr(loader.sampler, "set_epoch"):
            # Advance epoch so DistributedSampler reshuffles each pass
            loader.sampler.set_epoch(epoch)
        epoch += 1

        for batch in loader:
            batch = TokenBatch(
                input_ids=batch.input_ids.to(device),
                attention_mask=batch.attention_mask.to(device),
            )
            yield batch
            emitted_tokens += int(batch.input_ids.numel())
            if emitted_tokens >= max_tokens:
                return


def iter_eval_batches(
    cfg: dict,
    tokenizer: PreTrainedTokenizerBase,
    device: str,
) -> Iterator[TokenBatch]:
    """Iterate over the test/validation split for reconstruction eval.

    Uses data.eval_split (default: "test"). Falls back to "validation".
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

    def _iter(split: str) -> Iterator[TokenBatch]:
        loader = _build_dataloader(
            cfg,
            tokenizer,
            split=split,
            batch_size=batch_size,
            rank=0,
            world_size=1,   # eval runs on rank 0 only
            shuffle=False,
        )
        emitted = 0
        for batch in loader:
            batch = TokenBatch(
                input_ids=batch.input_ids.to(device),
                attention_mask=batch.attention_mask.to(device),
            )
            yield batch
            emitted += int(batch.input_ids.numel())
            if emitted >= max_eval_tokens:
                return

    try:
        yield from _iter(eval_split)
    except Exception as exc:
        if eval_split == "test":
            import warnings
            warnings.warn(
                f"[iter_eval_batches] split={eval_split!r} failed "
                f"({exc}), retrying with 'validation'."
            )
            yield from _iter("validation")
        else:
            raise
