from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]  # The PhyGS checkout: phygs/ and phygs_robot_tests/ live here.

for path in (REPO_ROOT / "phygs_robot_tests" / "episodes", REPO_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
