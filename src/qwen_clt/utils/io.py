from __future__ import annotations

from pathlib import Path
import torch


def save_checkpoint(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = path.with_name(f".{path.name}.tmp")
    try:
        torch.save(obj, tmp_path)
        tmp_path.replace(path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise
