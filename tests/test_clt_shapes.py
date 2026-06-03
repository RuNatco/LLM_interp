import torch
from qwen_clt.models.cross_layer_transcoder import CrossLayerTranscoder


def test_clt_shapes():
    clt = CrossLayerTranscoder(n_layers=3, d_model=8, features_per_layer=4)
    xs = [torch.randn(2, 5, 8) for _ in range(3)]
    features, recons = clt(xs)
    assert len(features) == 3
    assert len(recons) == 3
    assert features[0].shape == (2, 5, 4)
    assert recons[0].shape == (2, 5, 8)
