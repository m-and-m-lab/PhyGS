# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Single-axis moves that finish a ``navigate_to`` near its goal (navigation internals, not a skill).

The walking policy was trained with small commands on one axis at a time (forward/back, left/right
or turning). Near rest it does not follow small commands that mix axes: 4 cm from a goal it stood
still for 12 s under 0.063 m/s sideways and 0.183 rad/s turning, and goal tolerances below 0.05 m
timed out (2026-10-16, output_check/20261016-1938_nav_tolerance_005_004). Within the fine approach
radius route following therefore stops and, from each settled pose, makes one move on one axis.
Nav2 navigation finishes its goals this way. The moves are single-axis body commands, sideways
included: a holonomic base's.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from phygs.execution.locomotion.base import DeadBand
from phygs.execution.navigation.types import NavigationConfig, NavigationGoal, angle_error


@dataclass(frozen=True)
class FineMove:
    """One single-axis body command [m/s, m/s, rad/s] held for ``duration`` [s]."""

    command: tuple[float, float, float]
    duration: float
    description: str


class FineApproach:
    """The next move from a settled pose: a heading error above
    :attr:`NavigationConfig.fine_turn_first` is turned first; then the base translates along the body
    axis with the larger position error; a smaller heading error is turned once the position is
    within tolerance. A 5-6 cm move turned the base 0.07-0.14 rad, more than a 0.04 rad tolerance,
    so turning first every time alternated turns and moves until the move limit (house, Nav2,
    2026-10-16, output_check/20261016-2035_nav_tests_tol004).

    The speed is what covers the error in :attr:`NavigationConfig.fine_move_seconds`, bounded by the
    lowest walking speed in that direction (the locomotion dead band) and the fine speed limits.
    At the lowest speed a translation lasts as long as the measured :class:`Coverage` says the base
    needs to cover the error (a short move from standing falls short); turns last error / speed.
    The move does not try to be exact: the robot stops, measures the settled pose and asks again.
    """

    @dataclass(frozen=True)
    class Coverage:
        """Distance [m] the base covers from standing at the lowest walking speed held for each of
        ``seconds`` [s], per direction (``forward``, ``backward``, ``left``, ``right``)."""

        seconds: tuple[float, ...]
        distances: dict[str, tuple[float, ...]]

        def __post_init__(self):
            if set(self.distances) != {"forward", "backward", "left", "right"}:
                raise ValueError("Move coverage needs forward, backward, left and right.")
            if any(len(values) != len(self.seconds) for values in self.distances.values()):
                raise ValueError("Move coverage needs one distance per duration.")
            if list(self.seconds) != sorted(self.seconds) or self.seconds[0] <= 0:
                raise ValueError("Move coverage durations must increase from above zero.")

        @classmethod
        def from_moves(cls, moves: dict) -> FineApproach.Coverage:
            """From the base's measured ``lowest_speed_moves`` ({seconds, forward, backward, left,
            right} [s, m]; :class:`~phygs.execution.locomotion.base.BaseController`)."""
            seconds = tuple(float(v) for v in moves["seconds"])
            directions = ("forward", "backward", "left", "right")
            return cls(seconds, {d: tuple(float(v) for v in moves[d]) for d in directions})

        def duration(self, direction: str, distance: float) -> float | None:
            """Duration [s] that covers ``distance`` [m], interpolated (from zero below the first
            measurement); None beyond the longest measured move."""
            distances, seconds = [0.0], [0.0]
            for value, duration in zip(self.distances[direction], self.seconds):
                if value > distances[-1]:  # A longer move that covered no more adds nothing.
                    distances.append(value)
                    seconds.append(duration)
            if distance > distances[-1]:
                return None
            return float(np.interp(distance, distances, seconds))

    def __init__(
        self,
        config: NavigationConfig,
        physics_dt: float,
        dead_band: DeadBand,
        coverage: FineApproach.Coverage | None = None,
    ):
        """``dead_band`` and ``coverage`` are the base's (none: a move at the lowest speed lasts error /
        speed)."""
        self._config, self._dt, self._band = config, physics_dt, dead_band
        self._coverage = coverage

    def applies(self, goal: NavigationGoal, distance: float, started: bool) -> bool:
        """Whether fine moves finish the goal at ``distance`` [m]: its position tolerance is below the
        radius, and the base is within it (within the exit radius once the fine approach started)."""
        radius = self._config.fine_approach_radius
        if goal.position_tolerance >= radius:
            return False
        return distance <= (self._config.fine_approach_exit_radius if started else radius)

    @staticmethod
    def move_free(grid, move: FineMove, position: tuple[float, float], yaw: float) -> bool:
        """Whether ``move`` from the settled pose [m, rad] keeps the unpadded footprint clear on
        ``grid`` (a :class:`~phygs.execution.navigation.map.GridMap`): fine moves stay
        within the goal's padding."""
        if move.command[2]:
            return grid.turn_free(position, yaw, move.command[2] * move.duration, padded=False)
        end = FineApproach.end_point(move, position, yaw)
        return grid.segment_free(position, end, yaw, bare_start=math.inf)

    @staticmethod
    def end_point(move: FineMove, position: tuple[float, float], yaw: float) -> tuple[float, float]:
        """Where the commanded move ends [m], for a map check; a turn ends where it starts."""
        c, s = math.cos(yaw), math.sin(yaw)
        x, y = move.command[0] * move.duration, move.command[1] * move.duration
        return position[0] + c * x - s * y, position[1] + s * x + c * y

    def move(self, goal: NavigationGoal, position: tuple[float, float], yaw: float) -> FineMove | None:
        """The next move from the settled pose [m, rad]; None inside both goal tolerances."""
        heading_error = 0.0 if goal.heading is None else angle_error(goal.heading, yaw)
        dx, dy = goal.x - position[0], goal.y - position[1]
        at_position = math.hypot(dx, dy) <= goal.position_tolerance
        turn_first = abs(heading_error) > self._config.fine_turn_first
        if abs(heading_error) > goal.heading_tolerance and (turn_first or at_position):
            speed = self._speed(2, heading_error, self._config.fine_max_yaw_speed)
            return self._make(2, heading_error, speed, f"turning {heading_error:+.3f} rad")
        if at_position:
            return None
        c, s = math.cos(yaw), math.sin(yaw)
        body = (c * dx + s * dy, -s * dx + c * dy)
        axis = 0 if abs(body[0]) >= abs(body[1]) else 1
        speed = self._speed(axis, body[axis], self._config.fine_max_speed)
        name = ("forward", "sideways")[axis]
        return self._make(axis, body[axis], speed, f"{name} {body[axis]:+.3f} m")

    def _speed(self, axis: int, error: float, limit: float) -> float:
        lowest = (self._band.lowest_positive if error > 0 else self._band.lowest_negative)[axis]
        return max(lowest, min(limit, abs(error) / self._config.fine_move_seconds))

    def _make(self, axis: int, error: float, speed: float, description: str) -> FineMove:
        seconds = abs(error) / speed
        lowest = (self._band.lowest_positive if error > 0 else self._band.lowest_negative)[axis]
        if axis < 2 and math.isclose(speed, lowest):
            direction = (("forward", "backward"), ("left", "right"))[axis][error < 0]
            if self._coverage is not None:
                seconds = self._coverage.duration(direction, abs(error)) or seconds
        ticks = max(1, round(seconds / self._dt))
        command = [0.0, 0.0, 0.0]
        command[axis] = math.copysign(speed, error)
        return FineMove((command[0], command[1], command[2]), ticks * self._dt, description)
