# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Agents act through the session only: the example agents (examples/agents) import nothing else
of the project (the barrier is a convention until agents run in their own process). The composite
skills (``phygs.skills.composite``) are agent-side code and keep the same rule, and the robot tests'
episode plans use only the session and the composite skills, so they prove those are enough."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]
AGENTS = PROJECT / "examples" / "agents"
ROBOT_TESTS = PROJECT / "tests" / "robot_tests"
COMPOSITE = sorted((PROJECT / "phygs" / "skills" / "composite").glob("*.py"))
ALLOWED = {"phygs.session", "numpy", "yaml"}
# An episode test also imports its harness helper (episode.py).
PLANS = sorted(ROBOT_TESTS.glob("episodes/check_*.py"))


def imports(path: Path) -> set[str]:
    names = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                raise AssertionError(f"{path.name}: relative import")
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
    return names


def allowed(name: str, extra: tuple[str, ...] = ()) -> bool:
    """``name`` is an imported module, or ``module.symbol`` for a from-import."""
    top = name.split(".")[0]
    heads = ALLOWED | set(extra)
    return top in sys.stdlib_module_names or any(name == head or name.startswith(head + ".") for head in heads)


@pytest.mark.parametrize("path", sorted(AGENTS.glob("*.py")), ids=lambda path: path.name)
def test_agents_import_only_the_session_the_standard_library_numpy_and_yaml(path):
    for name in imports(path):
        assert allowed(name), f"{path.name} imports {name}"


@pytest.mark.parametrize("path", COMPOSITE, ids=lambda path: path.name)
def test_the_composite_skills_import_only_the_session_the_standard_library_and_numpy(path):
    for name in imports(path):
        assert allowed(name, ("phygs.skills.composite",)) and not name.startswith("yaml"), f"{path.name} imports {name}"


@pytest.mark.parametrize("path", PLANS, ids=lambda path: f"{path.parent.name}/{path.name}")
def test_episode_plans_keep_the_agent_rule(path):
    for name in imports(path):
        assert allowed(name, ("phygs.skills.composite", "phygs_robot_tests.episodes.episode")), (
            f"{path.name} imports {name}"
        )
