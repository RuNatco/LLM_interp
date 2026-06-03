from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from qwen_clt.models.cross_layer_transcoder import CrossLayerTranscoder


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> dict[str, Any]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=map_location)

    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"Expected checkpoint to be dict, got {type(checkpoint)}"
        )

    required_keys = ["cfg", "model_state_dict", "step"]

    for key in required_keys:
        if key not in checkpoint:
            raise KeyError(
                f"Missing key '{key}' in checkpoint. "
                f"Available keys: {list(checkpoint.keys())}"
            )

    return checkpoint


def infer_n_layers_from_state_dict(state_dict: dict[str, torch.Tensor]) -> int:
    """
    В checkpoint encoders лежат как:
        encoders.0
        encoders.1
        ...
        encoders.N

    Поэтому фактическое число слоёв можно восстановить по ключам.
    """

    encoder_indices = []

    for key in state_dict.keys():
        if key.startswith("encoders."):
            parts = key.split(".")
            if len(parts) >= 2 and parts[1].isdigit():
                encoder_indices.append(int(parts[1]))

    if len(encoder_indices) == 0:
        raise ValueError(
            "Cannot infer n_layers from state_dict: no keys like 'encoders.0'."
        )

    return max(encoder_indices) + 1


def build_clt_from_checkpoint_cfg(
    checkpoint: dict[str, Any],
) -> CrossLayerTranscoder:
    cfg = checkpoint["cfg"]
    state_dict = checkpoint["model_state_dict"]

    if "clt" not in cfg:
        raise KeyError(
            "Checkpoint cfg has no 'clt' section. "
            f"Available cfg keys: {list(cfg.keys())}"
        )

    clt_cfg = cfg["clt"]

    n_layers_from_cfg = int(clt_cfg["n_layers"])
    n_layers_from_state = infer_n_layers_from_state_dict(state_dict)

    if n_layers_from_cfg != n_layers_from_state:
        print(
            "[loader warning] "
            f"cfg['clt']['n_layers']={n_layers_from_cfg}, "
            f"but state_dict has encoders for {n_layers_from_state} layers. "
            "Using state_dict value."
        )

    model = CrossLayerTranscoder(
        n_layers=n_layers_from_state,
        d_model=int(clt_cfg["d_model"]),
        features_per_layer=int(clt_cfg["features_per_layer"]),
        init_threshold=float(clt_cfg.get("init_threshold", 0.0)),
        decoder_init_scale=float(clt_cfg.get("decoder_init_scale", 0.02)),
    )

    return model


def load_autoencoder_from_checkpoint(
    checkpoint_path: str | Path,
    device: str = "cuda",
) -> CrossLayerTranscoder:
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")

    model = build_clt_from_checkpoint_cfg(checkpoint)

    state_dict = checkpoint["model_state_dict"]
    missing_keys, unexpected_keys = model.load_state_dict(
        state_dict,
        strict=False,
    )

    if missing_keys:
        print("[loader warning] Missing keys:")
        for key in missing_keys:
            print(f"  - {key}")

    if unexpected_keys:
        print("[loader warning] Unexpected keys:")
        for key in unexpected_keys:
            print(f"  - {key}")

    model.to(device)
    model.eval()

    print(
        f"Loaded CLT checkpoint from {checkpoint_path} "
        f"at step={checkpoint.get('step')}"
    )

    return model
