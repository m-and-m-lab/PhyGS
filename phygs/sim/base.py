# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The robot base's measured state: the framework's ``body`` frame in the world."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from isaaclab.assets import Articulation


class BaseLink:
    """Pose and velocity of the robot's base link (``robot.yaml`` ``base_link``).

    A base link that is the articulation's floating root (Spot) reads the root's state; a base link
    on joints under a fixed root (a base on planar joints) reads its own body's. Reading the root
    there would report the fixed root: a pose that never moves, without an error.
    """

    def __init__(self, articulation: Articulation, name: str) -> None:
        ids, names = articulation.find_bodies(name)
        exact = [index for index, found in zip(ids, names) if found == name]
        if len(exact) != 1:
            raise ValueError(f"The base link {name!r} is not one body of the robot ({articulation.body_names}).")
        self._articulation, self._index = articulation, exact[0]
        self._root = self._index == 0 and not articulation.is_fixed_base

    def pose(self) -> torch.Tensor:
        """World pose [x, y, z [m], qx, qy, qz, qw]."""
        data = self._articulation.data
        return data.root_pose_w.torch[0] if self._root else data.body_link_pose_w.torch[0, self._index]

    def linear_velocity_b(self) -> torch.Tensor:
        """Linear velocity [m/s] in the base frame."""
        data = self._articulation.data
        if self._root:
            return data.root_lin_vel_b.torch[0]
        return self._in_base(data.body_link_lin_vel_w.torch[0, self._index])

    def angular_velocity_b(self) -> torch.Tensor:
        """Angular velocity [rad/s] in the base frame."""
        data = self._articulation.data
        if self._root:
            return data.root_ang_vel_b.torch[0]
        return self._in_base(data.body_link_ang_vel_w.torch[0, self._index])

    def _in_base(self, vector_w: torch.Tensor) -> torch.Tensor:
        """A world-frame vector in the base frame."""
        from isaaclab.utils.math import quat_apply_inverse

        pose = self._articulation.data.body_link_pose_w.torch[0, self._index]
        return quat_apply_inverse(pose[3:].unsqueeze(0), vector_w.unsqueeze(0))[0]
