# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The robot tests' tasks: the retrieve task with its evaluator, point navigation and object search.

:func:`load_task` turns a spec's ``task`` block into the task object the Environment takes
(:class:`~phygs.episodes.task.Task`). The framework never imports this package; the robot tests
do, and the entry points reach it through ``--task_loader phygs_robot_tests.tasks:load_task``. A
benchmark ships its own tasks the same way.
"""

from __future__ import annotations

from phygs_robot_tests.tasks.retrieve import RetrieveTask

from phygs.episodes.task import NoTask, Task

GOALS = ("retrieve", "none")


def load_task(block: dict) -> Task:
    """The task object for a spec's ``task`` block: ``{goal: retrieve, description, target}`` or
    ``{goal: none}``."""
    if not isinstance(block, dict) or "goal" not in block:
        raise ValueError("The task block needs a goal.")
    goal = block["goal"]
    if goal not in GOALS:
        raise ValueError(f"Unknown goal {goal!r}; expected one of {GOALS}.")
    if goal == "none":
        if set(block) != {"goal"}:
            raise ValueError("A goal of none has no other fields.")
        return NoTask()
    if set(block) != {"goal", "description", "target"} or not (block["description"] and block["target"]):
        raise ValueError("A retrieve task is {goal, description, target}, all given.")
    return RetrieveTask(str(block["description"]), str(block["target"]))
