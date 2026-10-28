# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The robot's ground-truth state for the navigation evaluators (``PointNavigationEvaluator``); the
evaluator's reads, never the agent's. Import after the app started (:mod:`common`)."""

from __future__ import annotations

import math

import torch
from phygs_robot_tests.tasks.point_navigation import PhysicalSample

from phygs.simulation.base import BaseLink


def physical_sample(control, robot, contact_force: float = 0.0) -> PhysicalSample:
    """The robot's ground-truth state for the evaluator, of its base link: pose, planar speed [m/s],
    yaw rate [rad/s], uprightness (the base's z axis up) and the given obstacle ``contact_force`` [N]."""
    base = BaseLink(robot, control.robot.base_link)
    pose = base.pose()
    x, y, z, w = pose[3:].tolist()
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return PhysicalSample(
        control.runtime.time_seconds,
        float(pose[0]),
        float(pose[1]),
        float(pose[2]),
        yaw,
        float(torch.linalg.vector_norm(base.linear_velocity_b()[:2])),
        float(base.angular_velocity_b()[2]),
        1.0 - 2.0 * (x * x + y * y),
        contact_force,
    )
