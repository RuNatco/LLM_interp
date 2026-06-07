from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from qwen_clt.attribution.targets import logit_difference_target
from qwen_clt.attribution.validation import (
    ablation_matches_proxy_sign,
    logit_difference_score,
    single_token_id,
)
from qwen_clt.deep_trace.cache import DeepTraceCache, collect_deep_trace_cache
from qwen_clt.deep_trace.graph import DeepTraceEdge, DeepTraceGraph, DeepTraceNode
from qwen_clt.interventions import FeatureIntervention
from qwen_clt.replacement import ReplacementConfig


DEEP_TRACE_LIMITATIONS = [
    "Deep Trace stage 1 is not full path attribution.",
    "Attention is represented as layer-level attention output, not per-head output.",
    "Feature-to-residual edges are decoder-write proxies.",
    "Residual, layernorm and attention edges are diagnostic projections, not causal proofs.",
    "Important nodes and edges should still be validated with interventions.",
]


@dataclass(frozen=True)
class FeatureCandidate:
    layer: int
    pos: int
    feature_idx: int
    activation: float
    direct_score: float

    @property
    def node_id(self) -> str:
        return f"feature:L{self.layer}:P{self.pos}:F{self.feature_idx}"


def resolve_position(pos: int, seq_len: int) -> int:
    if pos < 0:
        pos = seq_len + pos
    if pos < 0 or pos >= seq_len:
        raise IndexError(f"pos={pos} is outside sequence length {seq_len}")
    return pos


def _tensor_at_pos(tensor: torch.Tensor, pos: int) -> torch.Tensor:
    if tensor.ndim != 3:
        raise ValueError(f"Expected [batch, seq, d_model], got {tensor.shape}")
    if tensor.shape[0] != 1:
        raise ValueError("Deep Trace stage 1 currently expects batch size 1.")
    return tensor[0, pos].float()


def _projection_score(
    tensor: torch.Tensor,
    target_vector: torch.Tensor,
    pos: int,
) -> float:
    value = _tensor_at_pos(tensor, pos)
    target_vector = target_vector.to(value.device).float()
    return float(torch.dot(value, target_vector).item())


def _norm_at_pos(tensor: torch.Tensor, pos: int) -> float:
    return float(_tensor_at_pos(tensor, pos).norm().item())


def _decoder_row_for_raw_output(clt, src: int, tgt: int, feature_idx: int) -> torch.Tensor:
    if hasattr(clt, "decoder_row_for_raw_output"):
        return clt.decoder_row_for_raw_output(src, tgt, feature_idx).detach().float()

    return clt.decoders[f"{src}->{tgt}"][feature_idx].detach().float()


def feature_direct_score(
    *,
    clt,
    candidate: FeatureCandidate,
    target_vector: torch.Tensor,
) -> float:
    write = torch.zeros_like(target_vector, dtype=torch.float32)
    for tgt in range(candidate.layer, clt.n_layers):
        row = _decoder_row_for_raw_output(
            clt,
            src=candidate.layer,
            tgt=tgt,
            feature_idx=candidate.feature_idx,
        )
        write = write + row.to(write.device)
    return float(candidate.activation) * float(torch.dot(write, target_vector).item())


def select_feature_candidates(
    *,
    clt,
    features_by_layer: list[torch.Tensor | None],
    target_vector: torch.Tensor,
    pos: int,
    max_feature_nodes: int,
    min_activation: float,
) -> list[FeatureCandidate]:
    candidates: list[FeatureCandidate] = []
    for layer, features in enumerate(features_by_layer):
        if features is None:
            continue
        active = torch.nonzero(features[0, pos].float() > min_activation, as_tuple=False)
        for item in active.flatten().tolist():
            activation = float(features[0, pos, item].float().item())
            candidate = FeatureCandidate(
                layer=layer,
                pos=pos,
                feature_idx=int(item),
                activation=activation,
                direct_score=0.0,
            )
            candidate = FeatureCandidate(
                layer=candidate.layer,
                pos=candidate.pos,
                feature_idx=candidate.feature_idx,
                activation=candidate.activation,
                direct_score=feature_direct_score(
                    clt=clt,
                    candidate=candidate,
                    target_vector=target_vector,
                ),
            )
            candidates.append(candidate)

    candidates.sort(key=lambda item: abs(item.direct_score), reverse=True)
    return candidates[:max_feature_nodes]


def _target_node(target_name: str, target_pos: int) -> DeepTraceNode:
    return DeepTraceNode(
        id="target:0",
        type="LogitTargetNode",
        label=target_name,
        pos=target_pos,
        metadata={"score": "logit(positive)-logit(negative)"},
    )


def _feature_node(
    candidate: FeatureCandidate,
    causal_payload: dict[str, Any] | None,
) -> DeepTraceNode:
    metadata: dict[str, Any] = {
        "activation": candidate.activation,
        "direct_decoder_score": candidate.direct_score,
    }
    if causal_payload is not None:
        metadata["causal_validation"] = causal_payload

    return DeepTraceNode(
        id=candidate.node_id,
        type="CLTFeatureNode",
        label=f"L{candidate.layer} P{candidate.pos} F{candidate.feature_idx}",
        layer=candidate.layer,
        pos=candidate.pos,
        feature_idx=candidate.feature_idx,
        value=candidate.activation,
        metadata=metadata,
    )


def _residual_node(layer: int, pos: int, norm: float) -> DeepTraceNode:
    return DeepTraceNode(
        id=f"residual:L{layer}:P{pos}",
        type="ResidualStreamNode",
        label=f"Residual L{layer} P{pos}",
        layer=layer,
        pos=pos,
        value=norm,
        metadata={"value": "norm(replacement_residual_input)"},
    )


def _attention_node(layer: int, pos: int, norm: float) -> DeepTraceNode:
    return DeepTraceNode(
        id=f"attention:L{layer}:P{pos}",
        type="AttentionHeadNode",
        label=f"Attention L{layer} P{pos}",
        layer=layer,
        pos=pos,
        value=norm,
        metadata={
            "granularity": "layer_output",
            "head_idx": None,
            "value": "norm(replacement_attention_output)",
        },
    )


def _layernorm_node(
    *,
    layer: int,
    pos: int,
    kind: str,
    norm: float,
) -> DeepTraceNode:
    return DeepTraceNode(
        id=f"layernorm:{kind}:L{layer}:P{pos}",
        type="LayerNormNode",
        label=f"{kind} layernorm L{layer} P{pos}",
        layer=layer,
        pos=pos,
        value=norm,
        metadata={"kind": kind, "value": "norm(layernorm_output)"},
    )


def _error_node(
    *,
    layer: int,
    pos: int,
    error_norm: float,
    direct_score: float,
) -> DeepTraceNode:
    return DeepTraceNode(
        id=f"mlp_error:L{layer}:P{pos}",
        type="MLPErrorNode",
        label=f"MLP error L{layer} P{pos}",
        layer=layer,
        pos=pos,
        value=error_norm,
        metadata={
            "definition": "original_mlp_output - clt_reconstruction",
            "direct_target_projection": direct_score,
        },
    )


def _feature_decoder_residual_edges(
    *,
    clt,
    candidate: FeatureCandidate,
    per_feature_limit: int,
) -> list[DeepTraceEdge]:
    scored_edges: list[DeepTraceEdge] = []
    for tgt in range(candidate.layer, clt.n_layers):
        row = _decoder_row_for_raw_output(
            clt,
            src=candidate.layer,
            tgt=tgt,
            feature_idx=candidate.feature_idx,
        )
        next_residual_layer = tgt + 1
        if next_residual_layer >= clt.n_layers:
            continue
        score = float(candidate.activation) * float(row.norm().item())
        scored_edges.append(
            DeepTraceEdge(
                source=candidate.node_id,
                target=f"residual:L{next_residual_layer}:P{candidate.pos}",
                score=score,
                kind="decoder_write_to_next_residual_proxy",
                metadata={
                    "decoder": f"{candidate.layer}->{tgt}",
                    "is_causal": False,
                },
            )
        )

    scored_edges.sort(key=lambda edge: abs(edge.score), reverse=True)
    return scored_edges[:per_feature_limit]


def _causal_validation_for_candidate(
    *,
    replacement_model,
    prompt: str,
    candidate: FeatureCandidate,
    positive_token_id: int,
    negative_token_id: int,
    target_pos: int,
) -> dict[str, Any]:
    run = replacement_model.feature_intervention(
        prompt,
        [
            FeatureIntervention(
                layer=candidate.layer,
                pos=candidate.pos,
                feature_idx=candidate.feature_idx,
                value=0.0,
            )
        ],
    )
    replacement_score = logit_difference_score(
        run["replacement_logits"],
        positive_token_id=positive_token_id,
        negative_token_id=negative_token_id,
        target_pos=target_pos,
    )
    intervened_score = logit_difference_score(
        run["intervened_logits"],
        positive_token_id=positive_token_id,
        negative_token_id=negative_token_id,
        target_pos=target_pos,
    )
    causal_effect = intervened_score - replacement_score
    return {
        "kind": "feature_ablation",
        "baseline_logits": "replacement_logits",
        "intervention_value": 0.0,
        "replacement_target_score": replacement_score,
        "intervened_target_score": intervened_score,
        "causal_effect": causal_effect,
        "ablation_matches_direct_score_sign": ablation_matches_proxy_sign(
            candidate.direct_score,
            causal_effect,
        ),
    }


def _cache_summary(
    cache: DeepTraceCache,
    target_pos: int,
) -> dict[str, Any]:
    return {
        "n_layers": len(cache.replacement.mlp_outputs),
        "target_pos": target_pos,
        "replacement_logits_shape": list(cache.replacement.logits.shape),
        "original_logits_shape": list(cache.original.logits.shape),
        "mean_mlp_error_norm": float(
            torch.tensor(
                [
                    _norm_at_pos(error, target_pos)
                    for error in cache.replacement_errors
                    if error is not None
                ],
                dtype=torch.float32,
            ).mean().item()
        ),
    }


@torch.no_grad()
def build_deep_trace_graph(
    *,
    replacement_model,
    prompt: str,
    positive: str,
    negative: str,
    target_pos: int = -1,
    max_feature_nodes: int = 64,
    top_error_nodes: int = 8,
    causal_top_k: int = 8,
    min_activation: float = 0.0,
    feature_residual_edges_per_node: int = 3,
) -> tuple[DeepTraceGraph, DeepTraceCache]:
    input_ids, attention_mask = replacement_model.tokenize(prompt)
    replacement_config = ReplacementConfig()
    cache = collect_deep_trace_cache(
        model=replacement_model.base_model,
        autoencoder=replacement_model.clt,
        input_ids=input_ids,
        attention_mask=attention_mask,
        replacement_config=replacement_config,
    )

    resolved_target_pos = resolve_position(target_pos, cache.replacement.logits.shape[1])
    positive_token_id = single_token_id(replacement_model.tokenizer, positive)
    negative_token_id = single_token_id(replacement_model.tokenizer, negative)
    target = logit_difference_target(
        replacement_model.tokenizer,
        replacement_model.base_model,
        positive,
        negative,
    )
    target_vector = target.vector.detach().float().cpu()

    candidates = select_feature_candidates(
        clt=replacement_model.clt,
        features_by_layer=cache.replacement_features,
        target_vector=target_vector,
        pos=resolved_target_pos,
        max_feature_nodes=max_feature_nodes,
        min_activation=min_activation,
    )

    causal_payloads: dict[str, dict[str, Any]] = {}
    for candidate in candidates[:causal_top_k]:
        causal_payloads[candidate.node_id] = _causal_validation_for_candidate(
            replacement_model=replacement_model,
            prompt=prompt,
            candidate=candidate,
            positive_token_id=positive_token_id,
            negative_token_id=negative_token_id,
            target_pos=target_pos,
        )

    nodes: list[DeepTraceNode] = [_target_node(target.name, resolved_target_pos)]
    edges: list[DeepTraceEdge] = []

    for layer in range(replacement_model.clt.n_layers):
        residual = cache.replacement.residual_inputs[layer]
        nodes.append(_residual_node(layer, resolved_target_pos, _norm_at_pos(residual, resolved_target_pos)))
        edges.append(
            DeepTraceEdge(
                source=f"residual:L{layer}:P{resolved_target_pos}",
                target="target:0",
                score=_projection_score(residual, target_vector, resolved_target_pos),
                kind="activation_projection_to_target_direction",
                metadata={"is_causal": False},
            )
        )

        attention = cache.replacement.attention_outputs[layer]
        if attention is not None:
            nodes.append(_attention_node(layer, resolved_target_pos, _norm_at_pos(attention, resolved_target_pos)))
            edges.append(
                DeepTraceEdge(
                    source=f"attention:L{layer}:P{resolved_target_pos}",
                    target=f"residual:L{layer}:P{resolved_target_pos}",
                    score=_projection_score(attention, target_vector, resolved_target_pos),
                    kind="attention_output_projection_proxy",
                    metadata={"is_causal": False, "granularity": "layer_output"},
                )
            )

        for kind, layernorms in [
            ("input", cache.replacement.input_layernorm_outputs),
            ("post_attention", cache.replacement.post_attention_layernorm_outputs),
        ]:
            value = layernorms[layer]
            if value is None:
                continue
            nodes.append(
                _layernorm_node(
                    layer=layer,
                    pos=resolved_target_pos,
                    kind=kind,
                    norm=_norm_at_pos(value, resolved_target_pos),
                )
            )

    error_scores: list[tuple[int, float, float]] = []
    for layer, error in enumerate(cache.replacement_errors):
        if error is None:
            continue
        error_norm = _norm_at_pos(error, resolved_target_pos)
        direct_score = _projection_score(error, target_vector, resolved_target_pos)
        error_scores.append((layer, error_norm, direct_score))

    error_scores.sort(key=lambda item: item[1], reverse=True)
    for layer, error_norm, direct_score in error_scores[:top_error_nodes]:
        nodes.append(
            _error_node(
                layer=layer,
                pos=resolved_target_pos,
                error_norm=error_norm,
                direct_score=direct_score,
            )
        )
        edges.append(
            DeepTraceEdge(
                source=f"mlp_error:L{layer}:P{resolved_target_pos}",
                target="target:0",
                score=direct_score,
                kind="replacement_error_projection",
                metadata={"is_causal": False},
            )
        )

    for candidate in candidates:
        causal_payload = causal_payloads.get(candidate.node_id)
        nodes.append(_feature_node(candidate, causal_payload))
        edges.append(
            DeepTraceEdge(
                source=candidate.node_id,
                target="target:0",
                score=candidate.direct_score,
                kind="direct_decoder_write_to_logit_direction",
                metadata={"is_causal": False},
            )
        )
        if causal_payload is not None:
            edges.append(
                DeepTraceEdge(
                    source=candidate.node_id,
                    target="target:0",
                    score=float(causal_payload["causal_effect"]),
                    kind="causal_ablation_effect",
                    metadata={
                        "is_causal": True,
                        "intervention": "set_feature_value_to_zero",
                    },
                )
            )
        edges.extend(
            _feature_decoder_residual_edges(
                clt=replacement_model.clt,
                candidate=candidate,
                per_feature_limit=feature_residual_edges_per_node,
            )
        )

    sign_matches = [
        payload["ablation_matches_direct_score_sign"]
        for payload in causal_payloads.values()
        if payload["ablation_matches_direct_score_sign"] is not None
    ]

    metadata = {
        "trace_kind": "deep_trace_stage1",
        "is_full_circuit_tracing": False,
        "is_deeper_than_proxy_attribution": True,
        "method": "replacement_conditioned_typed_trace_with_error_nodes",
        "prompt": prompt,
        "target": {
            "positive": positive,
            "negative": negative,
            "positive_token_id": positive_token_id,
            "negative_token_id": negative_token_id,
            "target_pos": target_pos,
            "resolved_target_pos": resolved_target_pos,
            "score": "logit(positive)-logit(negative)",
        },
        "node_types": sorted({node.type for node in nodes}),
        "edge_kinds": sorted({edge.kind for edge in edges}),
        "selection": {
            "max_feature_nodes": max_feature_nodes,
            "selected_feature_nodes": len(candidates),
            "top_error_nodes": top_error_nodes,
            "causal_top_k": causal_top_k,
            "min_activation": min_activation,
        },
        "causal_pruning": {
            "kind": "feature_ablation_sign_check",
            "validated_feature_nodes": len(causal_payloads),
            "sign_match_rate": (
                None
                if not sign_matches
                else sum(1 for item in sign_matches if item) / len(sign_matches)
            ),
        },
        "cache_summary": _cache_summary(cache, resolved_target_pos),
        "limitations": DEEP_TRACE_LIMITATIONS,
    }

    return DeepTraceGraph(nodes=nodes, edges=edges, metadata=metadata), cache
