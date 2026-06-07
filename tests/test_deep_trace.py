from types import SimpleNamespace
import json

import torch
import torch.nn as nn

from qwen_clt.deep_trace.build import (
    resolve_position,
    select_feature_candidates,
)
from qwen_clt.deep_trace.cache import QwenDeepTraceCollector
from qwen_clt.deep_trace.graph import DeepTraceEdge, DeepTraceGraph, DeepTraceNode
from qwen_clt.deep_trace.summary import summarize_deep_trace_payload
from qwen_clt.models.cross_layer_transcoder import CrossLayerTranscoder


class DummyTraceLayer(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(d_model)
        self.self_attn = nn.Linear(d_model, d_model, bias=False)
        self.post_attention_layernorm = nn.LayerNorm(d_model)
        self.mlp = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        x = x + self.self_attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class DummyTraceBackbone(nn.Module):
    def __init__(self, d_model: int, n_layers: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [DummyTraceLayer(d_model) for _ in range(n_layers)]
        )


class DummyTraceModel(nn.Module):
    def __init__(self, d_model: int = 4, n_layers: int = 2, vocab: int = 16):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.model = DummyTraceBackbone(d_model=d_model, n_layers=n_layers)
        self.lm_head = nn.Linear(d_model, vocab, bias=False)

    def forward(self, input_ids, attention_mask=None):
        x = self.embed(input_ids)
        for layer in self.model.layers:
            x = layer(x)
        return SimpleNamespace(logits=self.lm_head(x))


def test_deep_trace_graph_serializes_typed_nodes_and_edges(tmp_path):
    graph = DeepTraceGraph(
        nodes=[
            DeepTraceNode(
                id="feature:L0:P1:F2",
                type="CLTFeatureNode",
                label="L0 P1 F2",
                layer=0,
                pos=1,
                feature_idx=2,
                value=3.0,
            )
        ],
        edges=[
            DeepTraceEdge(
                source="feature:L0:P1:F2",
                target="target:0",
                score=1.5,
                kind="causal_ablation_effect",
            )
        ],
        metadata={"trace_kind": "deep_trace_stage1"},
    )

    path = tmp_path / "deep_trace.json"
    graph.to_json(path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["metadata"]["trace_kind"] == "deep_trace_stage1"
    assert payload["nodes"][0]["type"] == "CLTFeatureNode"
    assert payload["edges"][0]["kind"] == "causal_ablation_effect"


def test_select_feature_candidates_ranks_by_abs_direct_score():
    clt = CrossLayerTranscoder(n_layers=1, d_model=2, features_per_layer=2)
    with torch.no_grad():
        clt.decoders["0->0"].zero_()
        clt.decoders["0->0"][0].copy_(torch.tensor([1.0, 0.0]))
        clt.decoders["0->0"][1].copy_(torch.tensor([0.0, -3.0]))

    features = [torch.zeros(1, 2, 2)]
    features[0][0, 1, 0] = 2.0
    features[0][0, 1, 1] = 1.0
    target_vector = torch.tensor([1.0, 1.0])

    candidates = select_feature_candidates(
        clt=clt,
        features_by_layer=features,
        target_vector=target_vector,
        pos=1,
        max_feature_nodes=2,
        min_activation=0.0,
    )

    assert candidates[0].feature_idx == 1
    assert candidates[1].feature_idx == 0


def test_qwen_deep_trace_collector_on_dummy_qwen_like_model():
    model = DummyTraceModel()
    collector = QwenDeepTraceCollector(model)
    input_ids = torch.tensor([[1, 2, 3]])

    trace = collector.run(input_ids=input_ids)

    assert trace.logits.shape == (1, 3, 16)
    assert len(trace.residual_inputs) == 2
    assert len(trace.attention_outputs) == 2
    assert trace.mlp_outputs[0].shape == (1, 3, 4)


def test_resolve_position_handles_negative_positions():
    assert resolve_position(-1, 5) == 4


def test_deep_trace_summary_ranks_error_nodes_and_causal_edges():
    graph = DeepTraceGraph(
        nodes=[
            DeepTraceNode(
                id="mlp_error:L0:P1",
                type="MLPErrorNode",
                label="error 0",
                layer=0,
                pos=1,
                value=10.0,
                metadata={"direct_target_projection": 0.5},
            ),
            DeepTraceNode(
                id="mlp_error:L1:P1",
                type="MLPErrorNode",
                label="error 1",
                layer=1,
                pos=1,
                value=1.0,
                metadata={"direct_target_projection": -3.0},
            ),
        ],
        edges=[
            DeepTraceEdge(
                source="feature:L0:P1:F0",
                target="target:0",
                score=0.2,
                kind="causal_ablation_effect",
            ),
            DeepTraceEdge(
                source="feature:L1:P1:F0",
                target="target:0",
                score=-4.0,
                kind="causal_ablation_effect",
            ),
        ],
        metadata={"trace_kind": "deep_trace_stage1"},
    )

    summary = summarize_deep_trace_payload(graph.to_payload(), top_n=1)

    assert summary["top_mlp_error_nodes"][0]["id"] == "mlp_error:L1:P1"
    assert summary["top_causal_edges"][0]["source"] == "feature:L1:P1:F0"
