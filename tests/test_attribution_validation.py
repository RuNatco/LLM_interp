import torch

from qwen_clt.attribution.validation import (
    ablation_matches_proxy_sign,
    logit_difference_score,
    proxy_effect,
    select_top_node_indices,
)


def test_select_top_node_indices_uses_abs_proxy_effect():
    graph_payload = {
        "nodes": [{}, {}, {}],
        "targets": [{"name": "target"}],
        "adjacency_matrix": [[0.1, -0.7, 0.4]],
    }

    assert select_top_node_indices(graph_payload, top_k=2) == [1, 2]


def test_proxy_effect_reads_target_row():
    graph_payload = {
        "nodes": [{}, {}],
        "targets": [{"name": "a"}, {"name": "b"}],
        "adjacency_matrix": [[1.0, 2.0], [3.0, 4.0]],
    }

    assert proxy_effect(graph_payload, node_index=1, target_index=1) == 4.0


def test_logit_difference_score_uses_target_position():
    logits = torch.zeros(1, 2, 5)
    logits[0, 1, 3] = 4.0
    logits[0, 1, 1] = 1.5

    score = logit_difference_score(
        logits,
        positive_token_id=3,
        negative_token_id=1,
        target_pos=-1,
    )

    assert score == 2.5


def test_ablation_sign_match_is_opposite_to_proxy_sign():
    assert ablation_matches_proxy_sign(proxy_score=2.0, causal_effect=-0.5) is True
    assert ablation_matches_proxy_sign(proxy_score=2.0, causal_effect=0.5) is False
    assert ablation_matches_proxy_sign(proxy_score=0.0, causal_effect=0.5) is None
