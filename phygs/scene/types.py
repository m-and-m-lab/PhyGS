# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""What a scene reader returns, as plain types (no USD or Isaac Sim imports): skills, hints and the
evaluator take them without depending on the reader."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ArticulatedPart:
    """A sliding part (drawer) and its handle bar in the world frame; privileged task knowledge.

    ``handle_center`` [m]; ``handle_axis``: unit vector along the bar; ``handle_length`` and
    ``handle_radius`` [m]; ``direction``: unit vector the part moves in when it opens;
    ``opening`` [m] now, and its ``limits`` (closed, open) [m]; ``floor_center`` [m]: the middle of
    the drawer floor the opening exposes now, where an object can be set down (None if unknown).
    """

    handle_center: tuple[float, float, float]
    handle_axis: tuple[float, float, float]
    handle_length: float
    handle_radius: float
    direction: tuple[float, float, float]
    opening: float
    limits: tuple[float, float]
    floor_center: tuple[float, float, float] | None = None
