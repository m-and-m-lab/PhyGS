# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""A read-only view of a running episode for whoever watches it (the robot UI), outside the agent.

An :class:`~phygs.world.World` given a :class:`Viewer` captures the
viewer's cameras through its own declared observation consumer (``viewer``, like the video
recorder's ``recorder``), and calls :meth:`Viewer.update` on the main thread once per camera period
with an :class:`EpisodeView`. What a viewer reads is not an agent call: the audit does not count it,
and it grants the agent nothing. A person who also acts does so as an agent, through the session.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


class Viewer(Protocol):
    """Shown alongside an episode: the cameras it needs and its per-frame update."""

    cameras: tuple[str, ...]
    """Camera names to capture for the view, whether or not the agent observes them."""

    def update(self, view: EpisodeView) -> None:
        """Called on the main thread once per camera period; must not block."""
        ...


@dataclass(frozen=True)
class EpisodeView:
    """What a viewer may read now.

    ``observe()`` returns the viewer's observation packet: the viewer's cameras, joint state and the
    ground-truth base pose (a person watching sees the robot). ``costmap()`` is navigation's own map
    (Nav2's global costmap: cost per cell, 253 where the robot's centre may not be, 254 an obstacle,
    255 unknown; its origin and cell size [m]), None without navigation or before it serves.
    ``narration(start)`` returns the skills' step lines from index ``start`` on, as (time [s],
    kind, text); ``holding`` whether the gripper holds an object.
    """

    episode_id: int
    time_seconds: float
    """Simulated time since the episode started [s]."""
    status: str
    """The episode runtime's status: ``running``, then ``completed``, ``interrupted`` or ``failed``."""
    observe: Callable[[], Any]
    costmap: Callable[[], tuple[np.ndarray, tuple[float, float], float] | None]
    narration: Callable[[int], list[tuple[float, str, str]]]
    holding: bool
