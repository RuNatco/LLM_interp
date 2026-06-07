from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import json


@dataclass(frozen=True)
class DeepTraceNode:
    id: str
    type: str
    label: str
    layer: int | None = None
    pos: int | None = None
    feature_idx: int | None = None
    value: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DeepTraceEdge:
    source: str
    target: str
    score: float
    kind: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DeepTraceGraph:
    nodes: list[DeepTraceNode]
    edges: list[DeepTraceEdge]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {
            "nodes": [asdict(node) for node in self.nodes],
            "edges": [asdict(edge) for edge in self.edges],
            "metadata": self.metadata,
        }

    def to_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_payload(), f, ensure_ascii=False, indent=2)
