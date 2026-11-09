# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The package list in ``pyproject.toml`` names every package in the tree, so an install never misses
one. The list stays explicit because the robot tests are installed under another name
(``tests/robot_tests`` is ``phygs_robot_tests``), which setuptools' discovery cannot map. A package is a
folder of Python modules (some have no ``__init__.py``); inside the ``spot_description`` submodule only
folders with an ``__init__.py`` count (its ROS launch folder is not a package).

The version: ``pyproject.toml`` and ``phygs.__version__`` agree, a release version sits only on its tag,
and a ``.dev`` version is not released yet (AGENTS.md, Releasing)."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

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


def test_pyproject_and_the_package_say_the_same_version():
    from phygs import __version__

    assert tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"] == __version__


def test_a_release_version_is_only_on_its_tag_and_a_dev_version_is_not_yet_released():
    """A release version X.Y.Z is on the commit tagged vX.Y.Z and nowhere else: right after the tag the next
    commit starts X.Y+1.0.dev0 (AGENTS.md, Releasing). A dev version X.Y.Z.devN is not tagged vX.Y.Z yet.
    Local tags only (no network); skipped, with git's reason, where git cannot tell (no git, a checkout it
    refuses, a release commit before its tag, a clone without tags)."""
    import subprocess

    from phygs import __version__

    def git(*args: str) -> subprocess.CompletedProcess:
        try:
            return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, timeout=30)
        except OSError as error:
            pytest.skip(f"git cannot run here: {error}")

    def tagged(tag: str) -> str | None:
        """The commit ``tag`` names, None when it is not tagged here."""
        found = git("rev-parse", "--verify", "-q", f"refs/tags/{tag}^{{commit}}")
        if found.returncode == 1:
            return None
        if found.returncode != 0:
            pytest.skip(f"git cannot read {ROOT}: {found.stderr.strip()}")
        return found.stdout.strip()

    if ".dev" in __version__:
        release = __version__.split(".dev")[0]
        assert tagged(f"v{release}") is None, (
            f"phygs says {__version__}, but v{release} is tagged: start the next .dev0."
        )
        return
    commit = tagged(f"v{__version__}")
    if commit is None:
        pytest.skip(f"v{__version__} is not tagged here (the release commit before its tag, or a clone without tags).")
    head = git("rev-parse", "HEAD")
    assert head.returncode == 0, head.stderr
    here = head.stdout.strip()
    assert here == commit, f"phygs says {__version__} on {here[:7]}, not its tag {commit[:7]}: start the next .dev0."
