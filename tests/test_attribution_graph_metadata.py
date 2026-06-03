import json

import torch

from qwen_clt.attribution.graph import AttributionGraph, FeatureNode, LogitTargetNode
from qwen_clt.attribution.prune import prune_graph


def test_attribution_graph_serializes_proxy_metadata(tmp_path):
    graph = AttributionGraph(
        nodes=[FeatureNode(layer=0, pos=1, feature_idx=2, activation=3.0)],
        targets=[LogitTargetNode(name="target")],
        adjacency_matrix=torch.tensor([[1.0]]),
        metadata={
            "attribution_kind": "proxy",
            "is_full_circuit_tracing": False,
        },
    )

    path = tmp_path / "graph.json"
    graph.to_json(path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["metadata"]["attribution_kind"] == "proxy"
    assert payload["metadata"]["is_full_circuit_tracing"] is False


def test_prune_graph_preserves_metadata_and_adds_pruning_info():
    graph = AttributionGraph(
        nodes=[
            FeatureNode(layer=0, pos=1, feature_idx=2, activation=3.0),
            FeatureNode(layer=1, pos=1, feature_idx=4, activation=2.0),
        ],
        targets=[LogitTargetNode(name="target")],
        adjacency_matrix=torch.tensor([[2.0, 1.0]]),
        metadata={"attribution_kind": "proxy"},
    )

    pruned = prune_graph(graph, node_threshold=0.5)

    assert pruned.metadata["attribution_kind"] == "proxy"
    assert pruned.metadata["pruning"]["nodes_before"] == 2
    assert pruned.metadata["pruning"]["nodes_after"] == len(pruned.nodes)
