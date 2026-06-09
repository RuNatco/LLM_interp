from __future__ import annotations

from pathlib import Path
from typing import Any
import json


def load_deep_trace_payload(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        raise TypeError(f"Expected Deep Trace JSON object, got {type(payload)}")

    required = {"nodes", "edges", "metadata"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"Deep Trace JSON is missing keys: {sorted(missing)}")

    return payload


def _abs_float(value: Any) -> float:
    if value is None:
        return 0.0
    return abs(float(value))


def nodes_by_type(
    payload: dict[str, Any],
    node_type: str,
) -> list[dict[str, Any]]:
    return [
        node
        for node in payload.get("nodes", [])
        if node.get("type") == node_type
    ]


def edges_by_kind(
    payload: dict[str, Any],
    edge_kind: str,
) -> list[dict[str, Any]]:
    return [
        edge
        for edge in payload.get("edges", [])
        if edge.get("kind") == edge_kind
    ]


def top_mlp_error_nodes(
    payload: dict[str, Any],
    top_n: int = 8,
    sort_by: str = "abs_target_projection",
) -> list[dict[str, Any]]:
    nodes = nodes_by_type(payload, "MLPErrorNode")

    if sort_by == "error_norm":
        key = lambda node: _abs_float(node.get("value"))
    elif sort_by == "abs_target_projection":
        key = lambda node: _abs_float(
            node.get("metadata", {}).get("direct_target_projection")
        )
    else:
        raise ValueError(
            "sort_by must be 'error_norm' or 'abs_target_projection'. "
            f"Got {sort_by!r}."
        )

    return sorted(nodes, key=key, reverse=True)[:top_n]


def top_edges_by_kind(
    payload: dict[str, Any],
    edge_kind: str,
    top_n: int = 8,
) -> list[dict[str, Any]]:
    edges = edges_by_kind(payload, edge_kind)
    return sorted(
        edges,
        key=lambda edge: _abs_float(edge.get("score")),
        reverse=True,
    )[:top_n]


def summarize_deep_trace_payload(
    payload: dict[str, Any],
    top_n: int = 8,
) -> dict[str, Any]:
    metadata = payload.get("metadata", {})
    nodes = payload.get("nodes", [])
    edges = payload.get("edges", [])

    node_type_counts: dict[str, int] = {}
    for node in nodes:
        node_type = str(node.get("type", "unknown"))
        node_type_counts[node_type] = node_type_counts.get(node_type, 0) + 1

    edge_kind_counts: dict[str, int] = {}
    for edge in edges:
        edge_kind = str(edge.get("kind", "unknown"))
        edge_kind_counts[edge_kind] = edge_kind_counts.get(edge_kind, 0) + 1

    return {
        "trace_kind": metadata.get("trace_kind"),
        "checkpoint": metadata.get("checkpoint"),
        "prompt": metadata.get("prompt"),
        "target": metadata.get("target"),
        "num_nodes": len(nodes),
        "num_edges": len(edges),
        "node_type_counts": node_type_counts,
        "edge_kind_counts": edge_kind_counts,
        "causal_pruning": metadata.get("causal_pruning", {}),
        "cache_summary": metadata.get("cache_summary", {}),
        "replacement_fidelity": metadata.get("replacement_fidelity", {}),
        "top_mlp_error_nodes": top_mlp_error_nodes(payload, top_n=top_n),
        "top_causal_edges": top_edges_by_kind(
            payload,
            "causal_ablation_effect",
            top_n=top_n,
        ),
        "top_causal_residual_delta_edges": top_edges_by_kind(
            payload,
            "causal_feature_to_residual_delta",
            top_n=top_n,
        ),
        "top_direct_feature_edges": top_edges_by_kind(
            payload,
            "direct_decoder_write_to_logit_direction",
            top_n=top_n,
        ),
        "top_replacement_error_edges": top_edges_by_kind(
            payload,
            "replacement_error_projection",
            top_n=top_n,
        ),
    }
