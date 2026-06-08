import pytest
import torch

from qwen_clt.training.losses import reconstruction_loss
from qwen_clt.training.losses import target_logit_difference_loss
from qwen_clt.training.train_clt import layer_loss_weights_from_config


def test_reconstruction_loss_accepts_layer_weights():
    targets = [
        torch.zeros(1, 2, 3),
        torch.zeros(1, 2, 3),
    ]
    recons = [
        torch.zeros(1, 2, 3),
        torch.ones(1, 2, 3),
    ]

    unweighted = reconstruction_loss(recons, targets)
    weighted = reconstruction_loss(recons, targets, layer_weights=[0.0, 1.0])

    assert weighted > unweighted


def test_layer_loss_weights_from_config_applies_late_layer_overrides():
    weights = layer_loss_weights_from_config(
        {
            "layer_loss_weights": {
                "default": 1.0,
                "layers": {
                    18: 2.0,
                    20: 3.0,
                },
            }
        },
        n_layers=24,
    )

    assert weights is not None
    assert weights[0] == 1.0
    assert weights[18] == 2.0
    assert weights[20] == 3.0


def test_layer_loss_weights_from_config_rejects_invalid_layer():
    with pytest.raises(IndexError):
        layer_loss_weights_from_config(
            {
                "layer_loss_weights": {
                    "default": 1.0,
                    "layers": {24: 2.0},
                }
            },
            n_layers=24,
        )


def test_target_logit_difference_loss_is_zero_for_matching_target_diff():
    teacher = torch.zeros(2, 3, 8)
    student = torch.zeros(2, 3, 8)
    attention_mask = torch.tensor(
        [
            [1, 1, 1],
            [1, 1, 0],
        ]
    )

    teacher[:, :, 4] = 3.0
    teacher[:, :, 5] = 1.0
    student[:, :, 4] = 5.0
    student[:, :, 5] = 3.0

    loss = target_logit_difference_loss(
        student_logits=student,
        teacher_logits=teacher,
        positive_token_id=4,
        negative_token_id=5,
        attention_mask=attention_mask,
        target_pos=-1,
    )

    assert torch.allclose(loss, torch.tensor(0.0))


def test_target_logit_difference_loss_uses_last_active_position():
    teacher = torch.zeros(1, 3, 8)
    student = torch.zeros(1, 3, 8)
    attention_mask = torch.tensor([[1, 1, 0]])

    teacher[0, 1, 4] = 2.0
    teacher[0, 1, 5] = 0.0
    student[0, 1, 4] = 0.0
    student[0, 1, 5] = 2.0
    student[0, 2, 4] = 2.0
    student[0, 2, 5] = 0.0

    loss = target_logit_difference_loss(
        student_logits=student,
        teacher_logits=teacher,
        positive_token_id=4,
        negative_token_id=5,
        attention_mask=attention_mask,
        target_pos=-1,
    )

    assert float(loss.item()) > 0.0
