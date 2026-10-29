# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Read-only Isaac Lab differential IK adapter for the Spot wrist link."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.assets import Articulation


class WristPoseServo:
    """Free-space wrist-pose control in world coordinates; no collision avoidance.

    The controlled frame is the configured wrist link, not an inferred grasp center.
    """

    def __init__(self, robot: Articulation, joint_names: tuple[str, ...], wrist_link: str) -> None:
        from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg

        self.robot = robot
        self.joints, resolved = robot.find_joints(list(joint_names), preserve_order=True)
        if tuple(resolved) != joint_names:
            raise ValueError("Arm joint ordering does not match the robot.")
        ids, names = robot.find_bodies(wrist_link)
        if len(ids) != 1 or names[0] != wrist_link:
            raise ValueError("Expected exactly one configured wrist link.")
        self.body = ids[0]
        self.row = self.body - int(robot.is_fixed_base)
        self.columns = [j + robot.num_base_dofs for j in self.joints]
        self.names = joint_names
        self.solver = DifferentialIKController(
            DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            num_envs=1,
            device=robot.device,
        )

    def start(self, pose: tuple[float, ...]) -> None:
        self.initial = self.robot.data.body_link_pose_w.torch[:, self.body].clone()
        self.goal = torch.tensor([pose], device=self.robot.device, dtype=torch.float32)
        self.solver.reset()

    def targets(self, fraction: float) -> dict[str, float]:
        from isaaclab.utils.math import quat_slerp

        desired = self.initial.clone()
        desired[:, :3] += fraction * (self.goal[:, :3] - self.initial[:, :3])
        desired[:, 3:] = quat_slerp(self.initial[0, 3:], self.goal[0, 3:].clone(), fraction)
        self.solver.set_command(desired)
        measured = self.robot.data.body_link_pose_w.torch[:, self.body]
        jacobian = self.robot.data.body_link_jacobian_w.torch[:, self.row][:, :, self.columns]
        joint_pos = self.robot.data.joint_pos.torch[:, self.joints]
        targets = self.solver.compute(measured[:, :3], measured[:, 3:], jacobian, joint_pos)
        return dict(zip(self.names, targets[0].tolist(), strict=True))

    def error(self) -> tuple[float, float]:
        from isaaclab.utils.math import compute_pose_error

        measured = self.robot.data.body_link_pose_w.torch[:, self.body]
        position, rotation = compute_pose_error(measured[:, :3], measured[:, 3:], self.goal[:, :3], self.goal[:, 3:])
        return float(torch.linalg.vector_norm(position)), float(torch.linalg.vector_norm(rotation))
