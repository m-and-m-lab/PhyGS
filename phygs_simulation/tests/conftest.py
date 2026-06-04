from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
SERVICE_ROOT = REPO_ROOT / "docker" / "aograsp" / "services"

for path in (SCRIPTS_ROOT, SERVICE_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
