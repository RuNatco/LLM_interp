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


def test_target_normalization_denormalizes_reconstruction_to_raw_space():
    clt = CrossLayerTranscoder(
        n_layers=1,
        d_model=2,
        features_per_layer=1,
        normalize_targets=True,
    )

    with torch.no_grad():
        clt.target_mean[0].copy_(torch.tensor([10.0, -4.0]))
        clt.target_var[0].copy_(torch.tensor([4.0, 9.0]))
        clt.encoders[0].zero_()
        clt.encoder_bias[0].zero_()
        clt.decoders["0->0"].zero_()
        clt.decoder_bias[0].copy_(torch.tensor([1.0, -1.0]))

    _, recons = clt([torch.randn(2, 3, 2)])

    expected = torch.tensor([12.0, -7.0]).view(1, 1, 2).expand(2, 3, 2)
    assert torch.allclose(recons[0], expected, atol=1e-4)


def test_decoder_row_for_raw_output_scales_target_normalized_decoder():
    clt = CrossLayerTranscoder(
        n_layers=1,
        d_model=2,
        features_per_layer=1,
        normalize_targets=True,
    )

    with torch.no_grad():
        clt.target_var[0].copy_(torch.tensor([4.0, 9.0]))
        clt.decoders["0->0"][0].copy_(torch.tensor([0.5, -2.0]))

    row = clt.decoder_row_for_raw_output(src=0, tgt=0, feature_idx=0)

    assert torch.allclose(row, torch.tensor([1.0, -6.0]), atol=1e-4)
