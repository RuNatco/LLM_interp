import torch

from qwen_clt.models.cross_layer_transcoder import (
    CrossLayerTranscoder,
    clt_nonlinearity,
    clt_topk_kwargs,
)


def _mk(nonlinearity, k=4, F=16, L=3, d=8):
    return CrossLayerTranscoder(
        n_layers=L, d_model=d, features_per_layer=F,
        nonlinearity=nonlinearity, topk_k=k,
    )


def test_topk_exact_l0():
    torch.manual_seed(0)
    clt = _mk("topk", k=4, F=16, L=3, d=8)
    x = [torch.randn(2, 5, 8) for _ in range(3)]
    feats, recons = clt(x)
    assert len(feats) == 3 and len(recons) == 3
    for f in feats:
        l0 = (f > 0).sum(dim=-1)
        assert int(l0.max()) <= 4
        assert f.shape == (2, 5, 16)


def test_topk_gradients_flow():
    clt = _mk("topk", k=3)
    x = [torch.randn(2, 4, 8, requires_grad=False) for _ in range(3)]
    feats, recons = clt(x)
    loss = sum((r ** 2).mean() for r in recons)
    loss.backward()
    assert clt.encoders[0].grad is not None
    assert torch.isfinite(clt.encoders[0].grad).all()


def test_batchtopk_average_l0():
    torch.manual_seed(0)
    clt = _mk("batchtopk", k=4, F=16, L=2, d=8)
    clt.train()
    x = [torch.randn(8, 8, 8) for _ in range(2)]
    feats, _ = clt(x)
    avg_l0 = (feats[0] > 0).float().sum(dim=-1).mean()
    assert 1.0 <= float(avg_l0) <= 9.0


def test_jumprelu_default_unchanged():
    clt = _mk("jumprelu", k=4)
    assert clt.nonlinearity == "jumprelu"
    x = [torch.randn(2, 4, 8) for _ in range(3)]
    feats, recons = clt(x)
    assert feats[0].shape == (2, 4, 16)


def test_config_helpers():
    assert clt_nonlinearity({"nonlinearity": "TopK"}) == "topk"
    assert clt_topk_kwargs({"topk": {"k": 64}})["topk_k"] == 64
    assert clt_nonlinearity({}) == "jumprelu"


def test_invalid_k_raises():
    try:
        _mk("topk", k=999, F=16)
        assert False
    except ValueError:
        pass
