from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]  # The PhyGS checkout: phygs/ lives here.
# The robot tests are the package phygs_robot_tests (tests/robot_tests, mapped by pyproject); the editable
# install provides it. Episode plans are importable by file name for the unit tests.
for path in (REPO_ROOT / "tests" / "robot_tests" / "episodes", REPO_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
