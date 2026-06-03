from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import torch


def load_graph_payload(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    required = {"nodes", "targets", "adjacency_matrix"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"Graph JSON is missing required keys: {sorted(missing)}")

    return payload


def proxy_effect(
    graph_payload: dict[str, Any],
    node_index: int,
    target_index: int = 0,
) -> float:
    adjacency = graph_payload["adjacency_matrix"]
    if target_index < 0 or target_index >= len(adjacency):
        raise IndexError(f"target_index={target_index} is outside graph targets")

    row = adjacency[target_index]
    if node_index < 0 or node_index >= len(row):
        raise IndexError(f"node_index={node_index} is outside graph nodes")

    return float(row[node_index])


def select_top_node_indices(
    graph_payload: dict[str, Any],
    target_index: int = 0,
    top_k: int = 24,
    min_activation: float = 0.0,
) -> list[int]:
    if top_k <= 0:
        return []

    nodes = graph_payload["nodes"]
    scored = []
    for idx, node in enumerate(nodes):
        activation = abs(float(node.get("activation", 0.0)))
        if activation < min_activation:
            continue
        scored.append((idx, abs(proxy_effect(graph_payload, idx, target_index))))

    scored.sort(key=lambda item: item[1], reverse=True)
    return [idx for idx, _ in scored[:top_k]]


def single_token_id(tokenizer, text: str) -> int:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) != 1:
        raise ValueError(
            f"Expected a single-token string, got {text!r} -> {token_ids}"
        )
    return int(token_ids[0])


def resolve_position(pos: int, seq_len: int) -> int:
    if pos < 0:
        pos = seq_len + pos
    if pos < 0 or pos >= seq_len:
        raise IndexError(f"pos={pos} is outside sequence length {seq_len}")
    return pos


def logit_difference_score(
    logits: torch.Tensor,
    positive_token_id: int,
    negative_token_id: int,
    target_pos: int = -1,
) -> float:
    if logits.ndim != 3:
        raise ValueError(
            f"Expected logits with shape [batch, seq, vocab], got {logits.shape}"
        )
    if logits.shape[0] != 1:
        raise ValueError("Validation currently expects batch size 1.")

    pos = resolve_position(target_pos, logits.shape[1])
    next_token_logits = logits[0, pos].float()
    return float(
        next_token_logits[positive_token_id].item()
        - next_token_logits[negative_token_id].item()
    )


def ablation_matches_proxy_sign(
    proxy_score: float,
    causal_effect: float,
    eps: float = 1e-8,
) -> bool | None:
    if abs(proxy_score) <= eps or abs(causal_effect) <= eps:
        return None
    return (proxy_score > 0 and causal_effect < 0) or (
        proxy_score < 0 and causal_effect > 0
    )


def feature_value(
    features_by_layer: list[torch.Tensor | None],
    layer: int,
    pos: int,
    feature_idx: int,
) -> float | None:
    if layer < 0 or layer >= len(features_by_layer):
        return None

    features = features_by_layer[layer]
    if features is None:
        return None

    resolved_pos = resolve_position(pos, features.shape[1])
    return float(features[0, resolved_pos, feature_idx].detach().float().item())


def pearson_correlation(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys):
        raise ValueError("xs and ys must have the same length.")
    if len(xs) < 2:
        return None

    x = torch.tensor(xs, dtype=torch.float32)
    y = torch.tensor(ys, dtype=torch.float32)
    x = x - x.mean()
    y = y - y.mean()
    denom = x.norm() * y.norm()

    if float(denom.item()) == 0.0:
        return None

    return float((x @ y / denom).item())


def sign_match_rate(values: list[bool | None]) -> float | None:
    resolved = [item for item in values if item is not None]
    if not resolved:
        return None
    return float(sum(1 for item in resolved if item) / len(resolved))
