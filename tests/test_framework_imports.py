# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The framework knows no task: no module outside ``phygs_robot_tests.tasks`` (and the agents,
which import only the session) imports the tasks package, and no episode spec holds a file path.
A task reaches the Environment as an object (episodes/task.py); scenes are named by id."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import yaml

PROJECT = Path(__file__).resolve().parents[1]
PACKAGE = PROJECT / "phygs"
FRAMEWORK = sorted(path for path in PACKAGE.rglob("*.py") if path.relative_to(PACKAGE).parts[0] != "agents")
SKILLS = sorted((PACKAGE / "skills").rglob("*.py"))
SPECS = sorted((PROJECT / "phygs_robot_tests" / "configs" / "episodes").glob("*.yaml")) + sorted(
    (PACKAGE / "configs" / "ui").glob("*.yaml")
)


def imported_modules(path: Path) -> set[str]:
    names = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


@pytest.mark.parametrize("path", FRAMEWORK + SKILLS, ids=lambda path: str(path.relative_to(PROJECT)))
def test_no_framework_module_imports_the_tasks_package(path):
    tasks = {name for name in imported_modules(path) if name.split(".")[:2] == ["phygs_robot_tests", "tasks"]}
    assert not tasks, f"{path.relative_to(PROJECT)} imports {sorted(tasks)}"


@pytest.mark.parametrize("path", SPECS, ids=lambda path: path.name)
def test_no_spec_holds_a_path_that_climbs_out_of_its_folder(path):
    text = path.read_text()
    assert ".." not in text.replace("...", ""), f"{path.name} holds a '..' path"
    data = yaml.safe_load(text)
    if "scene" in data:
        assert "usd" not in data["scene"], f"{path.name} names a scene file; name the scene by id"
