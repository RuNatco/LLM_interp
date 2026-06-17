from __future__ import annotations

import torch

from qwen_clt.models.cross_layer_transcoder import (
    CrossLayerTranscoder,
    _JumpReLU,
)


def test_forward_equals_hard_gate():
    z = torch.tensor([-1.0, 0.0, 0.4, 0.6, 2.0])
    theta = torch.tensor(0.5)
    out = _JumpReLU.apply(z, theta, 0.1)
    expected = z * (z > theta)
    assert torch.allclose(out, expected)


def test_grad_wrt_z_is_the_gate():
    z = torch.tensor([0.0, 0.9, 1.1, 2.0], requires_grad=True)
    theta = torch.tensor(1.0)
    _JumpReLU.apply(z, theta, 0.5).sum().backward()
    assert torch.allclose(z.grad, (z.detach() > theta).float())


def test_grad_wrt_threshold_kernel():
    z = torch.tensor([0.5, 0.9, 1.1, 2.0])
    theta = torch.tensor(1.0, requires_grad=True)
    _JumpReLU.apply(z, theta, 0.5).sum().backward()
    expected = -(1.0 / 0.5) * 2.0
    assert torch.allclose(theta.grad, torch.tensor(expected))


def test_grad_wrt_threshold_zero_outside_band():
    z = torch.tensor([0.0, 0.1, 5.0, 6.0])
    theta = torch.tensor(1.0, requires_grad=True)
    _JumpReLU.apply(z, theta, 0.2).sum().backward()
    assert torch.allclose(theta.grad, torch.tensor(0.0))


def test_per_feature_threshold_grad_reduces_to_feature_shape():
    torch.manual_seed(0)
    z = torch.randn(4, 5, 8)
    theta = torch.full((8,), 0.0, requires_grad=True)
    _JumpReLU.apply(z, theta, 1.0).sum().backward()
    assert theta.grad.shape == (8,)


def test_effective_thresholds_are_positive():
    clt = CrossLayerTranscoder(
        n_layers=2, d_model=4, features_per_layer=6,
        init_threshold=0.05, jumprelu_bandwidth=0.1,
    )
    assert (clt.effective_thresholds > 0).all()
    assert torch.allclose(
        clt.effective_thresholds,
        torch.full_like(clt.effective_thresholds, 0.05),
    )


def test_threshold_is_no_longer_dead():
    torch.manual_seed(0)
    clt = CrossLayerTranscoder(
        n_layers=1, d_model=4, features_per_layer=32,
        init_threshold=0.5, jumprelu_bandwidth=1.0,
    )
    x = torch.randn(4, 4, 4)
    features = clt.encode_layer(x, 0)
    loss = (features ** 2).sum()
    loss.backward()

    assert clt.log_threshold.grad is not None
    assert clt.log_threshold.grad.abs().sum() > 0


def test_optimizer_step_moves_threshold():
    torch.manual_seed(0)
    clt = CrossLayerTranscoder(
        n_layers=1, d_model=4, features_per_layer=32,
        init_threshold=0.5, jumprelu_bandwidth=1.0,
    )
    opt = torch.optim.SGD(clt.parameters(), lr=1.0)
    before = clt.log_threshold.detach().clone()

    x = torch.randn(8, 4, 4)
    features = clt.encode_layer(x, 0)
    (features.abs().sum()).backward()
    opt.step()

    after = clt.log_threshold.detach()
    assert not torch.allclose(before, after)
    assert (after > before).any()
