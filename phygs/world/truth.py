# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Ground truth of a running world, for a task's evaluator and the robot tests; never for agents.

:meth:`World.truth() <phygs.world.world.World.truth>` returns one. A task's ``result`` reads it once,
after the robot has stopped and settled; a robot test may read it at any tick.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from phygs.scene.reader import pose_matrix
from phygs.scene.types import ArticulatedPart

if TYPE_CHECKING:
    from phygs.world.world import World


class Truth:
    """Ground truth now, for the evaluator and robot tests; never given to agents."""

    def __init__(self, world: World) -> None:
        self._scene = world.scene_def
        self._robot = world.scene["robot"]
        self._description = world.robot
        self._base = world.base

    def object_pose(self, name: str) -> np.ndarray:
        """World pose (4x4) [m] of a scene object."""
        return self._scene.object_pose(name)

    def part(self, name: str) -> ArticulatedPart:
        """An articulated part now (its ``opening`` [m])."""
        return self._scene.part(name)

    def held(self, name: str, *, lift_m: float, reach_m: float) -> bool:
        """Whether an object is in the hand: lifted at least ``lift_m`` [m] above where it started
        and its centre within ``reach_m`` [m] of the wrist link (grasped objects are never attached,
        so an object off its support next to the wrist is held). The thresholds are the evaluator's."""
        center = self._scene.object_box(name)[0]
        start = self._scene.initial_box(name)[0]
        wrist_link = self._robot.body_names.index(self._description.arm.wrist_link)
        wrist = self._robot.data.body_link_pose_w.torch[0, wrist_link, :3]
        near = float(np.linalg.norm(center - wrist.cpu().numpy())) <= reach_m
        return bool(center[2] - start[2] >= lift_m and near)

    def part_openings(self) -> dict[str, float]:
        """Every drawer's opening [m] now."""
        return {name: self._scene.part(name).opening for name in self._scene.part_names}

    def body_motion(self) -> dict[str, tuple[float, float]]:
        """Every scene body with colliders: (speed [m/s], distance [m] from where it was placed)."""
        return self._scene.body_motion()

    def inside(self, name: str, part: str) -> bool:
        """Whether an object's centre is inside a part (a drawer)."""
        return self._scene.inside(self._scene.object_box(name)[0], part)

    def base_pose(self) -> np.ndarray:
        """World pose (4x4) [m] of the robot's base."""
        return pose_matrix(self._base.pose().cpu().tolist())

    def arm_joints(self) -> dict[str, float]:
        """Arm joint positions [m or rad], planning order."""
        positions = self._robot.data.joint_pos.torch[0].cpu().tolist()
        return {name: positions[self._robot.joint_names.index(name)] for name in self._description.arm.joints}

    def stow_error_rad(self) -> float:
        """Largest arm joint distance [m or rad, rad for a revolute arm] from the stow posture."""
        stow = self._description.arm.stow
        return float(max(abs(value - goal) for value, goal in zip(self.arm_joints().values(), stow, strict=True)))
