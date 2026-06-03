from __future__ import annotations

import torch

from qwen_clt.attribution.graph import AttributionGraph, FeatureNode, LogitTargetNode
from qwen_clt.attribution.targets import CustomTarget


PROXY_ATTRIBUTION_LIMITATIONS = [
    "Not full circuit tracing.",
    "Scores only direct CLT decoder writes to a target logit direction.",
    "Ignores attention-mediated feature-feature paths.",
    "Ignores residual stream dynamics outside reconstructed MLP outputs.",
    "Ignores layernorm/error nodes and replacement-model error terms.",
    "Use causal feature interventions to validate any interpreted feature.",
]


def _resolve_target_pos(target_pos: int | None, seq_len: int) -> int | None:
    """Resolve Python-style negative positions.

    target_pos=None means "use all positions".
    target_pos=-1 means "use the last prompt position", which is the correct
    default for next-token attribution in a causal LM.
    """
    if target_pos is None:
        return None
    if target_pos < 0:
        target_pos = seq_len + target_pos
    if target_pos < 0 or target_pos >= seq_len:
        raise ValueError(f"target_pos={target_pos} is outside sequence length {seq_len}")
    return target_pos


def _proxy_metadata(
    *,
    target_pos: int | None,
    resolved_target_pos: int | None,
    max_feature_nodes: int,
    candidate_pool_size: int | None,
    min_activation: float,
) -> dict:
    return {
        "attribution_kind": "proxy",
        "method": "active_clt_feature_decoder_write_to_logit_direction",
        "is_full_circuit_tracing": False,
        "score_definition": (
            "feature_activation * dot(sum_outgoing_decoder_rows, target_logit_direction)"
        ),
        "target_pos": target_pos,
        "resolved_target_pos": resolved_target_pos,
        "candidate_pool_size": candidate_pool_size,
        "max_feature_nodes": max_feature_nodes,
        "min_activation": min_activation,
        "limitations": PROXY_ATTRIBUTION_LIMITATIONS,
        "validation_note": (
            "Treat graph edges as hypotheses; validate important nodes with "
            "feature_intervention and compare intervened_logits against replacement_logits."
        ),
    }


def build_feature_to_target_graph(
    replacement_model,
    prompt: str,
    targets: list[CustomTarget],
    max_feature_nodes: int = 2048,
    target_pos: int | None = -1,
    candidate_pool_size: int | None = None,
    min_activation: float = 0.0,
) -> AttributionGraph:
    """v0 attribution graph: active CLT features -> custom residual/logit directions.

    This is still a cheap proxy, not the full local replacement-model attribution
    from the Circuit Tracing paper.

    Important behavior:
    - By default, only features at the last prompt position are considered
      (`target_pos=-1`). This is the correct default for next-token causal LM
      attribution: logits for the next token are read from the last prompt
      position.
    - Nodes are ranked by absolute target effect, not by raw activation.
      The previous version ranked by activation first, which could over-select
      highly active features on early positions such as P0.
    - Set `target_pos=None` to consider all positions for debugging.
    """
    data = replacement_model.get_activations(prompt)
    features = data["features"]

    seq_len = features[0].shape[1]
    resolved_target_pos = _resolve_target_pos(target_pos, seq_len)
    metadata = _proxy_metadata(
        target_pos=target_pos,
        resolved_target_pos=resolved_target_pos,
        max_feature_nodes=max_feature_nodes,
        candidate_pool_size=candidate_pool_size,
        min_activation=min_activation,
    )

    # Collect active features. For next-token attribution, default to the last
    # position. This avoids the misleading "all top features at P0" behavior
    # caused by ranking high activations across every prompt position.
    candidates: list[FeatureNode] = []
    for layer, a in enumerate(features):
        # batch=1 assumed for graph building
        active = torch.nonzero(a[0] > min_activation, as_tuple=False)
        for pos, feat_idx in active.tolist():
            if resolved_target_pos is not None and pos != resolved_target_pos:
                continue
            act = float(a[0, pos, feat_idx].detach().float().item())
            candidates.append(FeatureNode(layer, pos, feat_idx, act))

    # Optional safety cap before expensive scoring. Use activation as a cheap
    # prefilter only when requested; final ranking is still by effect.
    if candidate_pool_size is not None and len(candidates) > candidate_pool_size:
        candidates.sort(key=lambda n: abs(n.activation), reverse=True)
        candidates = candidates[:candidate_pool_size]

    if not candidates:
        adjacency = torch.zeros((len(targets), 0), dtype=torch.float32)
        return AttributionGraph(
            nodes=[],
            targets=[LogitTargetNode(t.name) for t in targets],
            adjacency_matrix=adjacency,
            metadata=metadata,
        )

    # Compute effects target-by-target for every candidate.
    effect_rows: list[list[float]] = []
    for target in targets:
        vec = target.vector.to(replacement_model.device).float()
        effects = []

        for node in candidates:
            # Approximate downstream write by summing all outgoing decoder rows.
            # This ignores attention-mediated feature-feature paths and error
            # nodes, so it is a proxy rather than full circuit-tracer attribution.
            write = torch.zeros_like(vec)
            for tgt in range(node.layer, replacement_model.clt.n_layers):
                W = replacement_model.clt.decoders[f"{node.layer}->{tgt}"]
                write = write + W[node.feature_idx].detach().float().to(vec.device)

            effects.append(float(node.activation) * float(torch.dot(write, vec).item()))

        effect_rows.append(effects)

    effects_tensor = torch.tensor(effect_rows, dtype=torch.float32)

    # Rank nodes by maximum absolute effect across targets.
    node_scores = effects_tensor.abs().max(dim=0).values
    order = torch.argsort(node_scores, descending=True)
    order = order[:max_feature_nodes]

    nodes = [candidates[int(i)] for i in order.tolist()]
    adjacency = effects_tensor[:, order].contiguous()

    return AttributionGraph(
        nodes=nodes,
        targets=[LogitTargetNode(t.name) for t in targets],
        adjacency_matrix=adjacency,
        metadata=metadata,
    )
