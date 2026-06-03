from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureIntervention:
    layer: int
    pos: int | slice
    feature_idx: int
    value: float
