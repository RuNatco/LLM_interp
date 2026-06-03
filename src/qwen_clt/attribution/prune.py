from __future__ import annotations

import torch
from qwen_clt.attribution.graph import AttributionGraph


def prune_graph(graph: AttributionGraph, node_threshold: float = 0.8) -> AttributionGraph:
    scores = graph.adjacency_matrix.abs().sum(dim=0)
    total = scores.sum().clamp_min(1e-12)
    order = torch.argsort(scores, descending=True)
    cumulative = torch.cumsum(scores[order], dim=0) / total
    keep_n = int((cumulative <= node_threshold).sum().item()) + 1
    keep = order[:keep_n].tolist()
    nodes = [graph.nodes[i] for i in keep]
    adjacency = graph.adjacency_matrix[:, keep]
    metadata = dict(graph.metadata or {})
    metadata["pruning"] = {
        "node_threshold": node_threshold,
        "nodes_before": len(graph.nodes),
        "nodes_after": len(nodes),
    }
    return AttributionGraph(
        nodes=nodes,
        targets=graph.targets,
        adjacency_matrix=adjacency,
        metadata=metadata,
    )
