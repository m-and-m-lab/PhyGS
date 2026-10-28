# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Point-navigation task and evaluator independent of navigation skill feedback."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from phygs.execution.navigation.types import NavigationGoal, angle_error


@dataclass(frozen=True)
class PointNavigationTask:
    """Task identity, goal [m, rad], time budget [s], and permitted command interfaces.

    ``minimum_height`` [m]: the base below it is a fall (the robot's ``fallen_height``); None: the
    robot cannot fall (a base on joints), only a tilt below ``minimum_up_z`` is one.
    """

    scene_id: str
    initial_state_id: str
    spawn: tuple[float, float, float]
    goal: NavigationGoal
    time_budget: float = 45.0
    allowed_actions: tuple[str, ...] = ("navigation", "motion")
    localization: str = "ground_truth"
    arrival_hold: float = 0.2
    stopped_speed: float = 0.08
    stopped_yaw_speed: float = 0.08
    collision_force: float = 1.0
    minimum_height: float | None = None
    minimum_up_z: float = 0.7

    def __post_init__(self):
        if (
            not self.scene_id
            or not self.initial_state_id
            or self.localization not in {"none", "ground_truth", "estimated"}
        ):
            raise ValueError("Point-navigation tasks require identities and an explicit localization access mode.")
        if not self.allowed_actions or not set(self.allowed_actions) <= {"navigation", "motion"}:
            raise ValueError("Unsupported task action interfaces.")
        values = (
            self.time_budget,
            self.arrival_hold,
            self.stopped_speed,
            self.stopped_yaw_speed,
            self.collision_force,
            *(() if self.minimum_height is None else (self.minimum_height,)),
            self.minimum_up_z,
        )
        if not all(math.isfinite(value) and value > 0 for value in values):
            raise ValueError("Task budgets and physical thresholds must be positive and finite.")


@dataclass(frozen=True)
class PhysicalSample:
    """Independent state at time [s]: position [m], yaw [rad], speeds [m/s, rad/s], force [N]."""

    time: float
    x: float
    y: float
    z: float
    yaw: float
    speed: float
    yaw_speed: float
    up_z: float
    obstacle_force: float


class PointNavigationEvaluator:
    """Judge actual arrival; never consume the navigation client's success flag.

    Collision events are rising edges of obstacle contact force above the task's
    threshold. Ground contact is excluded by the diagnostic's sensor selection.
    Distance is the sampled XY polyline length; no path-efficiency claim is made.
    """

    def __init__(self, task: PointNavigationTask):
        self.task = task
        self.status = "running"
        self.reason = ""
        self.travel_distance = 0.0
        self.collisions = 0
        self.falls = 0
        self._contact = False
        self._previous: PhysicalSample | None = None
        self._arrival_since: float | None = None

    def update(self, sample: PhysicalSample) -> str:
        """Consume monotonically timed physical measurements and return task status."""
        if self.status != "running":
            return self.status
        if not all(map(math.isfinite, asdict(sample).values())) or sample.time < 0:
            raise ValueError("Physical measurements must be finite with nonnegative time.")
        if self._previous is not None:
            if sample.time <= self._previous.time:
                raise ValueError("Evaluator timestamps must increase.")
            self.travel_distance += math.hypot(sample.x - self._previous.x, sample.y - self._previous.y)
        contact = sample.obstacle_force > self.task.collision_force
        self.collisions += int(contact and not self._contact)
        self._contact = contact
        self._previous = sample
        low = self.task.minimum_height is not None and sample.z < self.task.minimum_height
        if low or sample.up_z < self.task.minimum_up_z:
            self.falls += 1
            self.status = "fallen"
        elif contact:
            self.status = "collision"
        elif sample.time >= self.task.time_budget:
            self.status = "timed_out"
        else:
            goal = self.task.goal
            arrived = (
                math.hypot(sample.x - goal.x, sample.y - goal.y) <= goal.position_tolerance
                and (goal.heading is None or abs(angle_error(goal.heading, sample.yaw)) <= goal.heading_tolerance)
                and sample.speed <= self.task.stopped_speed
                and abs(sample.yaw_speed) <= self.task.stopped_yaw_speed
            )
            if not arrived:
                self._arrival_since = None
            elif self._arrival_since is None:
                self._arrival_since = sample.time
            elif sample.time - self._arrival_since >= self.task.arrival_hold:
                self.status = "succeeded"
        return self.status

    def abort(self, reason: str) -> None:
        """End an unfinished task when its agent stops, without claiming physical success."""
        if self.status == "running":
            self.status = "aborted"
            self.reason = reason

    def result(self) -> dict:
        """Return task/configuration identity and physical outcomes in SI units."""
        return dict(
            task=asdict(self.task),
            status=self.status,
            reason=self.reason,
            travel_distance_m=self.travel_distance,
            collisions=self.collisions,
            falls=self.falls,
            elapsed_seconds=self._previous.time if self._previous else 0.0,
            final=asdict(self._previous) if self._previous else None,
        )
