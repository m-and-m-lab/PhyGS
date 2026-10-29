# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""What a benchmark drives: a :class:`World` built from an :class:`EpisodeSpec`, a :class:`Task` it is
given, an agent it runs, the report it writes. ``docs/world.md`` is the page for benchmark authors.

    from phygs.world import EpisodeSpec, World     # World after the Isaac Sim app has started.

    spec = EpisodeSpec.load("episode.yaml")
    with World(spec, output, task=my_task, overrides={"end_settle_seconds": 1.0}) as world:
        report = world.run("template")             # report["evaluation"] is the task's result.

Modules: :mod:`.world` (the World; needs Isaac Sim), :mod:`.spec`, :mod:`.task`, :mod:`.report`,
:mod:`.provenance`, :mod:`.build` (pure); :mod:`.truth`, :mod:`.recorder`, :mod:`.navigation`,
:mod:`.viewer`. Names are imported on first use, so ``from phygs.world import EpisodeSpec`` works
before the app starts and ``World`` only after.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

EXPORTS = {
    "World": "phygs.world.world",
    "run_episode": "phygs.world.world",
    "SETTINGS": "phygs.world.build",
    "Truth": "phygs.world.truth",
    "EpisodeSpec": "phygs.world.spec",
    "load_agent": "phygs.world.spec",
    "SCHEMA": "phygs.world.spec",
    "Task": "phygs.world.task",
    "EpisodeEnd": "phygs.world.task",
    "NoTask": "phygs.world.task",
    "load_task": "phygs.world.task",
    "REPORT_SCHEMA": "phygs.world.report",
    "Viewer": "phygs.world.viewer",
    "EpisodeView": "phygs.world.viewer",
}
__all__ = [
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
    "load_agent",
    "load_task",
    "run_episode",
]

if TYPE_CHECKING:
    from phygs.world.build import SETTINGS
    from phygs.world.report import REPORT_SCHEMA
    from phygs.world.spec import SCHEMA, EpisodeSpec, load_agent
    from phygs.world.task import EpisodeEnd, NoTask, Task, load_task
    from phygs.world.truth import Truth
    from phygs.world.viewer import EpisodeView, Viewer
    from phygs.world.world import World, run_episode


def __getattr__(name: str):
    if name not in EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(EXPORTS[name]), name)
