import torch

from qwen_clt.replacement.metrics import compute_replacement_metrics, metrics_to_dict


def test_replacement_metrics_mask_padding_tokens():
    original = torch.zeros(1, 2, 4)
    replacement = original.clone()
    replacement[0, 1, 3] = 100.0
    attention_mask = torch.tensor([[1, 0]])

    metrics = compute_replacement_metrics(
        original_logits=original,
        replacement_logits=replacement,
        attention_mask=attention_mask,
    )

    assert metrics.num_tokens == 1
    assert metrics.top1_agreement == 1.0
    assert metrics.mean_abs_logit_diff == 0.0


def test_replacement_metrics_report_last_token_and_target_diff():
    original = torch.zeros(1, 2, 4)
    replacement = original.clone()
    original[0, 1, 2] = 3.0
    original[0, 1, 1] = 1.0
    replacement[0, 1, 2] = 2.0
    replacement[0, 1, 1] = 2.0
    attention_mask = torch.tensor([[1, 1]])

    metrics = compute_replacement_metrics(
        original_logits=original,
        replacement_logits=replacement,
        attention_mask=attention_mask,
        target_token_ids=(2, 1),
        target_pos=-1,
    )
    payload = metrics_to_dict(metrics)

    assert metrics.last_token_top1_agreement == 0.0
    assert metrics.target_logit_diff_original_mean == 2.0
    assert metrics.target_logit_diff_replacement_mean == 0.0
    assert metrics.target_logit_diff_mae == 2.0
    assert "target_logit_diff_mae" in payload
