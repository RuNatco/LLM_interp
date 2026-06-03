from __future__ import annotations

import sys
from pathlib import Path


LOCAL_SRC = Path("src")

if LOCAL_SRC.is_dir():
    local_src = str(LOCAL_SRC)
    if local_src not in sys.path:
        sys.path.insert(0, local_src)
