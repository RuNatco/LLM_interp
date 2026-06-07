import pytest
import torch

from qwen_clt.training.losses import reconstruction_loss
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
