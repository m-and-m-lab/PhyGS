# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared navigation goals, feedback and reference-controller configuration."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass


def angle_error(target: float, current: float) -> float:
    """Shortest signed angular displacement [rad]."""
    return math.atan2(math.sin(target - current), math.cos(target - current))


@dataclass(frozen=True)
class NavigationGoal:
    """World-frame destination [m], optional heading and tolerances [rad, m, rad]."""

    x: float
    y: float
    heading: float | None = None
    position_tolerance: float = 0.04
    heading_tolerance: float = 0.04

    def __post_init__(self):
        values = (self.x, self.y, self.position_tolerance, self.heading_tolerance)
        if not all(map(math.isfinite, values)) or self.position_tolerance <= 0 or self.heading_tolerance <= 0:
            raise ValueError("Goal coordinates must be finite and tolerances positive.")
        if self.heading is not None and not math.isfinite(self.heading):
            raise ValueError("Heading must be finite.")


@dataclass(frozen=True)
class NavigationConfig:
    """Navigation settings: command period [s] and speed limits [m/s, rad/s].

    ``timeout`` [s] is the default time budget of one ``navigate_to``; None (the default) sets
    none: navigation runs until it succeeds, fails or is cancelled.

    Folded arm positions and tolerance are [m or rad, depending on joint type].

    Fine approach (:mod:`.fine_approach`): within ``fine_approach_radius`` [m] of a goal whose
    position tolerance is smaller, route following stops and navigation finishes with single-axis moves from
    settled poses, until the goal is reached or the time budget ends. Once started, the fine
    approach keeps the goal until a settled pose is beyond ``fine_approach_exit_radius`` [m]: stop
    drift and turning in place shift the base 1-2 cm, past the entry radius. A heading error above
    ``fine_turn_first`` [rad] is turned before translating, a smaller one after. A move covers
    its error in ``fine_move_seconds`` [s], at no more than ``fine_max_speed`` [m/s] or
    ``fine_max_yaw_speed`` [rad/s] and no less than the lowest walking speed.
    """

    period: float = 0.1
    max_speed: float = 1.0
    max_yaw_speed: float = 1.5
    timeout: float | None = None
    arm_tolerance: float = 0.18
    fine_approach_radius: float = 0.10
    fine_approach_exit_radius: float = 0.25
    fine_turn_first: float = 0.3
    fine_move_seconds: float = 1.0
    fine_max_speed: float = 0.3
    fine_max_yaw_speed: float = 0.5
    # A settled pose outside the goal tolerances beyond the fine approach is corrected by following a
    # new route in; after this many corrections the command fails instead of looping on its budget.
    max_goal_corrections: int = 3
    # The arm's transit posture [m or rad] (the robot's arm.stow), within arm_tolerance while the base
    # moves; none: no arm check.
    arm_posture: tuple[tuple[str, float], ...] = ()

    reference_frame: str = "world"

    def __post_init__(self):
        if not self.reference_frame or self.reference_frame == "body":
            raise ValueError("Navigation requires a fixed reference frame.")
        for value in (
            self.period,
            self.max_speed,
            self.max_yaw_speed,
            self.arm_tolerance,
            self.fine_approach_radius,
            self.fine_approach_exit_radius,
            self.fine_turn_first,
            self.fine_move_seconds,
            self.fine_max_speed,
            self.fine_max_yaw_speed,
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Navigation timing, gains, limits and tolerances must be positive and finite.")
        if self.fine_approach_exit_radius < self.fine_approach_radius:
            raise ValueError("fine_approach_exit_radius must not be smaller than fine_approach_radius.")
        if isinstance(self.max_goal_corrections, bool) or type(self.max_goal_corrections) is not int:
            raise ValueError("max_goal_corrections must be an integer.")
        if self.max_goal_corrections < 1:
            raise ValueError("max_goal_corrections must be at least 1.")
        if self.timeout is not None and not (math.isfinite(self.timeout) and self.timeout > 0):
            raise ValueError("A navigation timeout must be positive and finite, or None.")


@dataclass(frozen=True)
class NavigationFeedback:
    """Skill execution feedback, distinct from independent task success."""

    command_id: int
    episode_id: int
    status: str
    reason: str
    path: tuple[tuple[float, float], ...] = ()
    replans: int = 0


# Navigation step narration (see phygs.tools.narration).
STEPS = logging.getLogger("phygs.skills.navigation")


def describe_goal(goal: NavigationGoal) -> str:
    """A goal for step messages: position [m], heading [rad] and tolerances."""
    heading = "any heading" if goal.heading is None else f"heading {goal.heading:.2f} rad"
    return (
        f"goal ({goal.x:.2f}, {goal.y:.2f}) m, {heading}; tolerance {goal.position_tolerance:.2f} m"
        f" / {goal.heading_tolerance:.2f} rad"
    )


def describe_path(path) -> str:
    """A route for step messages: waypoints and length [m]."""
    length = sum(math.dist(a, b) for a, b in zip(path, path[1:]))
    return f"{len(path)} waypoints, {length:.2f} m"
