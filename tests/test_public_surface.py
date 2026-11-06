# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The public surfaces (ARCHITECTURE.md, Boundaries) and the package's independence.

``phygs.session`` is what agents use, ``phygs.skills.composite`` the composite skills over it;
``phygs.world`` is what benchmarks use. A name added to or removed from any is an API change: edit
the list here and bump the version (minor) with a CHANGELOG line. The package itself never imports
the robot tests or reads the repository around it.
"""

import ast
from pathlib import Path

import pytest

import phygs.session
import phygs.skills.composite
import phygs.world

PACKAGE = Path(__file__).resolve().parents[1] / "phygs"

SESSION = [
    "AgentFailed",
    "AgentSession",
    "ArticulatedPart",
    "Command",
    "EndEffectorTrajectory",
    "EpisodeEnded",
    "HeldObject",
    "JointTrajectory",
    "ObjectTarget",
    "PixelTarget",
    "PointTarget",
    "PublicTask",
    "RobotInfo",
    "SkillFailure",
    "TargetHint",
    "TrajectoryRefused",
    "entry_depth",
    "motion_failure",
    "unit_approach",
]
COMPOSITE = ["TUNING", "CompositeSkills", "Outcome", "Tuning"]
WORLD = [
    "REPORT_SCHEMA",
    "SCHEMA",
    "SETTINGS",
    "EpisodeEnd",
    "EpisodeSpec",
    "EpisodeView",
    "NoTask",
    "Task",
    "Truth",
    "Viewer",
    "World",
    "code_version",
    "file_sha256",
    "load_agent",
    "load_settings",
    "load_task",
    "run_episode",
    "summary",
]


def test_the_agent_surface_is_the_listed_names():
    assert sorted(phygs.session.__all__) == sorted(SESSION)


def test_the_composite_surface_is_the_listed_names():
    assert sorted(phygs.skills.composite.__all__) == sorted(COMPOSITE)


def test_the_old_composite_path_is_a_deprecated_alias_of_the_moved_package():
    import importlib
    import sys

    sys.modules.pop("phygs.composite", None)
    with pytest.warns(DeprecationWarning, match="phygs.skills.composite"):
        old = importlib.import_module("phygs.composite")
    assert old.CompositeSkills is phygs.skills.composite.CompositeSkills
    assert importlib.import_module("phygs.composite.tuning") is importlib.import_module("phygs.skills.composite.tuning")


def test_the_benchmark_surface_is_the_listed_names_and_each_is_exported():
    assert sorted(phygs.world.__all__) == sorted(WORLD)
    assert set(phygs.world.EXPORTS) == set(WORLD)


def test_the_package_never_imports_the_robot_tests_or_reads_the_checkout():
    """Only the ``phygs`` command (``__main__.py``) knows a source checkout's ``examples/`` and ``tests/``."""
    offenders = []
    for path in sorted(PACKAGE.rglob("*.py")):
        if path.name == "__main__.py" or "spot_description" in path.parts:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            if any(name.split(".")[0] == "phygs_robot_tests" for name in names):
                offenders.append(f"{path.relative_to(PACKAGE)}: imports the robot tests")
            if isinstance(node, ast.Constant) and node.value in ("tests", "examples", "robot_tests"):
                offenders.append(f"{path.relative_to(PACKAGE)}: names the checkout folder {node.value!r}")
            if isinstance(node, ast.Constant) and str(node.value).startswith("phygs_robot_tests"):
                offenders.append(f"{path.relative_to(PACKAGE)}: names the robot tests ({node.value!r})")
    assert offenders == []
