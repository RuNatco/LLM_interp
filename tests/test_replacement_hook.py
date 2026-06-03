import torch
import torch.nn as nn

from qwen_clt.interventions import FeatureIntervention
from qwen_clt.models.cross_layer_transcoder import CrossLayerTranscoder
from qwen_clt.replacement.hooks import LayerReplacementHook, ReplacementConfig


class DummyDecoderLayer(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.mlp = nn.Linear(d_model, d_model, bias=False)
        nn.init.eye_(self.mlp.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        return residual + self.mlp(x)


class DummyBackbone(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.layers = nn.ModuleList([DummyDecoderLayer(d_model)])


class DummyQwenLikeModel(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.model = DummyBackbone(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.model.layers:
            x = layer(x)
        return x


def test_replacement_hook_replaces_mlp_output_not_decoder_layer_output():
    d_model = 2
    model = DummyQwenLikeModel(d_model)
    autoencoder = CrossLayerTranscoder(
        n_layers=1,
        d_model=d_model,
        features_per_layer=2,
    )

    with torch.no_grad():
        autoencoder.decoders["0->0"].zero_()

    x = torch.ones(1, 3, d_model)
    assert torch.allclose(model(x), 2 * x)

    with LayerReplacementHook(
        model=model,
        autoencoder=autoencoder,
        config=ReplacementConfig(),
    ):
        out = model(x)

    assert torch.allclose(out, x)


def test_replacement_hook_applies_feature_interventions_in_forward_pass():
    d_model = 2
    model = DummyQwenLikeModel(d_model)
    autoencoder = CrossLayerTranscoder(
        n_layers=1,
        d_model=d_model,
        features_per_layer=2,
    )

    with torch.no_grad():
        autoencoder.encoders[0].copy_(torch.eye(d_model))
        autoencoder.encoder_bias[0].zero_()
        autoencoder.thresholds.zero_()
        autoencoder.decoders["0->0"].zero_()
        autoencoder.decoders["0->0"][0, 0] = 2.0

    x = torch.ones(1, 3, d_model)

    with LayerReplacementHook(
        model=model,
        autoencoder=autoencoder,
        config=ReplacementConfig(),
    ) as replacement_hook:
        replacement_out = model(x)

    with LayerReplacementHook(
        model=model,
        autoencoder=autoencoder,
        config=ReplacementConfig(
            feature_interventions=(
                FeatureIntervention(
                    layer=0,
                    pos=slice(None),
                    feature_idx=0,
                    value=0.0,
                ),
            ),
        ),
    ):
        intervened_out = model(x)

    assert torch.allclose(replacement_out[..., 0], torch.full((1, 3), 3.0))
    assert 0 in replacement_hook.features_by_layer
    assert 0 in replacement_hook.reconstructions_by_layer
    assert torch.allclose(intervened_out, x)
