# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The package list in ``pyproject.toml`` names every package in the tree, so an install never misses
one. The list stays explicit because the robot tests are installed under another name
(``tests/robot_tests`` is ``phygs_robot_tests``), which setuptools' discovery cannot map. A package is a
folder of Python modules (some have no ``__init__.py``); inside the ``spot_description`` submodule only
folders with an ``__init__.py`` count (its ROS launch folder is not a package)."""

from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROOTS = {"phygs": ROOT / "phygs", "phygs_robot_tests": ROOT / "tests" / "robot_tests"}
SUBMODULE = ROOT / "phygs" / "robots" / "spot" / "assets" / "spot_description"


def packages_in_tree() -> set[str]:
    found = set()
    for name, folder in ROOTS.items():
        for module in folder.rglob("*.py"):
            parts = module.parent.relative_to(folder).parts
            inside = module.parent.is_relative_to(SUBMODULE)
            if "__pycache__" in parts or (inside and not (module.parent / "__init__.py").exists()):
                continue
            found.add(".".join((name, *parts)))
    return found


def test_pyproject_lists_every_package_in_the_tree():
    listed = set(tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["setuptools"]["packages"])
    tree = packages_in_tree()
    assert listed == tree, f"missing from pyproject: {sorted(tree - listed)}; not in the tree: {sorted(listed - tree)}"
