from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
import json
import torch


@dataclass(frozen=True)
class FeatureNode:
    layer: int
    pos: int
    feature_idx: int
    activation: float


@dataclass(frozen=True)
class LogitTargetNode:
    name: str


@dataclass
class AttributionGraph:
    nodes: list[FeatureNode]
    targets: list[LogitTargetNode]
    adjacency_matrix: torch.Tensor  # [n_targets, n_nodes] for v0
    metadata: dict[str, Any] | None = None

    def to_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "nodes": [asdict(n) for n in self.nodes],
            "targets": [asdict(t) for t in self.targets],
            "adjacency_matrix": self.adjacency_matrix.detach().cpu().tolist(),
        }
        if self.metadata is not None:
            payload["metadata"] = self.metadata
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
