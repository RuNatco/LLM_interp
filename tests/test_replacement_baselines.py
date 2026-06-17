from __future__ import annotations

import torch
import torch.nn as nn

from qwen_clt.models.cross_layer_transcoder import CrossLayerTranscoder
from qwen_clt.replacement.baselines import (
    ConstantReplacementHook,
    build_random_clt_like,
    compute_mean_mlp_outputs,
    get_mlp_modules,
)


class _FakeMLP(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.lin = nn.Linear(d_model, d_model)

    def forward(self, x):
        return self.lin(x)


class _FakeLayer(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.mlp = _FakeMLP(d_model)


class _FakeInner(nn.Module):
    def __init__(self, n_layers: int, d_model: int):
        super().__init__()
        self.layers = nn.ModuleList([_FakeLayer(d_model) for _ in range(n_layers)])


class _FakeModel(nn.Module):

    def __init__(self, n_layers: int, d_model: int):
        super().__init__()
        self.model = _FakeInner(n_layers, d_model)

    def forward(self, hidden):
        return [layer.mlp(hidden) for layer in self.model.layers]


def _run(model, encoded):
    return model(encoded["hidden"])


def test_get_mlp_modules_finds_all_layers():
    model = _FakeModel(n_layers=4, d_model=3)
    assert len(get_mlp_modules(model)) == 4


def test_zero_baseline_zeroes_mlp_outputs():
    torch.manual_seed(0)
    model = _FakeModel(n_layers=3, d_model=4)
    hidden = torch.randn(2, 5, 4)

    with ConstantReplacementHook(model, mode="zero", n_clt_layers=3):
        outputs = model(hidden)

    for out in outputs:
        assert torch.allclose(out, torch.zeros_like(out))


def test_mean_baseline_broadcasts_per_layer_value():
    torch.manual_seed(0)
    model = _FakeModel(n_layers=2, d_model=4)
    hidden = torch.randn(2, 5, 4)

    per_layer = {0: torch.arange(4).float(), 1: torch.ones(4) * 7.0}

    with ConstantReplacementHook(
        model, mode="mean", n_clt_layers=2, per_layer_value=per_layer
    ):
        outputs = model(hidden)

    assert torch.allclose(outputs[0], per_layer[0].view(1, 1, -1).expand(2, 5, 4))
    assert torch.allclose(outputs[1], per_layer[1].view(1, 1, -1).expand(2, 5, 4))


def test_baseline_respects_clt_layer_coverage():
    torch.manual_seed(0)
    model = _FakeModel(n_layers=4, d_model=3)
    hidden = torch.randn(1, 2, 3)

    clean = model(hidden)
    with ConstantReplacementHook(model, mode="zero", n_clt_layers=2):
        replaced = model(hidden)

    assert torch.allclose(replaced[0], torch.zeros_like(replaced[0]))
    assert torch.allclose(replaced[1], torch.zeros_like(replaced[1]))
    assert torch.allclose(replaced[2], clean[2])
    assert torch.allclose(replaced[3], clean[3])


def test_compute_mean_masks_padded_tokens():
    d_model = 3
    model = _FakeModel(n_layers=1, d_model=d_model)

    with torch.no_grad():
        model.model.layers[0].mlp.lin.weight.copy_(torch.eye(d_model))
        model.model.layers[0].mlp.lin.bias.zero_()

    hidden = torch.tensor([[[1.0, 1.0, 1.0],
                            [3.0, 3.0, 3.0],
                            [99.0, 99.0, 99.0]]])
    mask = torch.tensor([[1, 1, 0]])
    encoded = {"hidden": hidden, "attention_mask": mask}

    means = compute_mean_mlp_outputs(
        model=model,
        encoded_batches=[encoded],
        n_clt_layers=1,
        d_model=d_model,
        device="cpu",
        run_logits=_run,
    )

    assert torch.allclose(means[0], torch.full((d_model,), 2.0))


def _tiny_clt() -> CrossLayerTranscoder:
    return CrossLayerTranscoder(n_layers=3, d_model=4, features_per_layer=5)


def test_random_clt_matches_shape_and_is_deterministic():
    trained = _tiny_clt()
    a = build_random_clt_like(trained, device="cpu", seed=123)
    b = build_random_clt_like(trained, device="cpu", seed=123)

    assert a.n_layers == trained.n_layers
    assert a.d_model == trained.d_model
    assert a.features_per_layer == trained.features_per_layer

    for key in a.state_dict():
        assert torch.allclose(a.state_dict()[key], b.state_dict()[key])


def test_random_clt_differs_from_trained_weights():
    trained = _tiny_clt()
    with torch.no_grad():
        for p in trained.parameters():
            p.add_(1.0)

    random_clt = build_random_clt_like(trained, device="cpu", seed=7)
    enc_key = "encoders.0"
    assert not torch.allclose(
        random_clt.state_dict()[enc_key], trained.state_dict()[enc_key]
    )


def test_random_clt_does_not_disturb_global_rng():
    trained = _tiny_clt()
    torch.manual_seed(2024)
    before = torch.randn(3)

    torch.manual_seed(2024)
    build_random_clt_like(trained, device="cpu", seed=999)
    after = torch.randn(3)

    assert torch.allclose(before, after)
