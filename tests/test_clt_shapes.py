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


def test_decoder_bias_is_added_to_reconstruction():
    clt = CrossLayerTranscoder(n_layers=1, d_model=3, features_per_layer=2)

    with torch.no_grad():
        clt.encoders[0].zero_()
        clt.encoder_bias[0].zero_()
        clt.decoders["0->0"].zero_()
        clt.decoder_bias[0].copy_(torch.tensor([1.0, -2.0, 0.5]))

    _, recons = clt([torch.randn(2, 4, 3)])

    expected = torch.tensor([1.0, -2.0, 0.5]).view(1, 1, 3).expand(2, 4, 3)
    assert torch.allclose(recons[0], expected)
