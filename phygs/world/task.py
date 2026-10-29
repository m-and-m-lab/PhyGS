# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The seam between the framework and a task: what the World needs from one.

The World never reads a spec's ``task`` block itself; it takes a task object built from it
(:func:`phygs_robot_tests.tasks.load_task`, or any object with this interface). Of the task it
uses only what is here: the public brief for the agent and the page, the one scene object the task
is about (privileged: hints, the arm planner's target, the known map leaves it out), the evaluation
once the robot has stopped, and what the report's provenance records.
"""

from __future__ import annotations

import importlib
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Task(Protocol):
    """A task as the World sees it. ``goal`` and ``description`` are public (the agent's
    ``session.task``); ``target`` is the scene name of the object the task is about, or None."""

    goal: str
    description: str
    target: str | None

    def brief(self) -> str:
        """One sentence telling a person (the robot UI's Start card) what to do."""
        ...

    def result(self, truth: Any, ended_by: str, contacts: list[dict], drawers_open_at_start: dict[str, float]) -> dict:
        """Score the settled end state from ground truth; ``outcome`` is ``success``, ``failure`` or
        ``not_evaluated``. Called once, after the robot has stopped and settled."""
        ...

    def provenance(self) -> dict:
        """What the report records about the evaluation's rules (e.g. a thresholds file's hash)."""
        ...


class NoTask:
    """No task: the agent explores, nothing is evaluated (specs with ``task: {goal: none}``)."""

    goal = "none"
    description = ""
    target = None

    def brief(self) -> str:
        return "Explore."

    def result(self, truth: Any, ended_by: str, contacts: list[dict], drawers_open_at_start: dict[str, float]) -> dict:
        return {"outcome": "not_evaluated", "reason": "this episode has no task"}

    def provenance(self) -> dict:
        return {}


def summary(task: Task) -> dict:
    """The task as a report records it: its public part only (the target's name stays out)."""
    return dict(goal=task.goal, description=task.description)


def load_task(block: dict, loader: str | None) -> Task:
    """The task object for a spec's task block through ``loader`` (``module:function``, taking the
    block), e.g. ``phygs_robot_tests.tasks:load_task``; without a loader only ``{goal: none}`` runs."""
    if loader is None:
        if block != {"goal": "none"}:
            raise ValueError(
                "The spec has a task block; give its loader, e.g. --task_loader phygs_robot_tests.tasks:load_task."
            )
        return NoTask()
    module_name, _, function = loader.partition(":")
    if not module_name or not function:
        raise ValueError(f"A task loader is module:function; got {loader!r}.")
    task = getattr(importlib.import_module(module_name), function)(block)
    if not isinstance(task, Task):
        raise TypeError(f"{loader} returned {type(task).__name__}, not a Task (see phygs.world.task).")
    return task
