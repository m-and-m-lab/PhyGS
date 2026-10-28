# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Lower an end-effector trajectory to arm joints before it moves the robot.

The trajectory is densified (at most :data:`STEP_M` and :data:`STEP_RAD` between samples), solved
with inverse kinematics sample by sample, each seeded from the last, and refused when a sample is
out of reach or the arm would change configuration (:data:`JUMP_RAD`). Too fast for the joint
speed limit, it is slowed down uniformly. Joint limits only: collisions are the planners' concern,
and a trajectory an agent submits is executed as given.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from phygs.execution.manipulation.planning import EndEffectorTrajectory, JointTrajectory

STEP_M = 0.02  # Largest position change between solved samples [m].
STEP_RAD = 0.05  # Largest orientation change between solved samples [rad].
JUMP_RAD = 0.3  # Largest joint change between solved samples [rad]; more is another arm configuration.

InverseKinematics = Callable[[list[float], list[float]], "list[float] | None"]
ForwardKinematics = Callable[[list[float]], np.ndarray]


class TrajectoryRefused(ValueError):
    """The trajectory cannot be followed; nothing moved. ``failure`` is a skill failure value
    (``"out_of_reach"``), or None for a malformed request such as a start away from the tool."""

    def __init__(self, failure: str | None, message: str) -> None:
        super().__init__(message)
        self.failure = failure


@dataclass(frozen=True)
class LoweredTrajectory:
    """The joint trajectory to execute and its tool reference.

    ``reference`` holds the body-frame tool poses (4x4) at ``joints.times``; ``time_scale`` is how
    much the trajectory was slowed down to the joint speed limit (1.0: not at all).
    """

    joints: JointTrajectory
    reference: np.ndarray
    tool: str
    time_scale: float

    def tool_pose(self, seconds: float) -> np.ndarray:
        """Body-frame reference tool pose (4x4) at elapsed time [s]: linear position, slerp orientation."""
        times = self.joints.times
        seconds = min(max(seconds, 0.0), float(times[-1]))
        index = min(int(np.searchsorted(times, seconds, side="right")) - 1, len(times) - 2)
        fraction = (seconds - times[index]) / (times[index + 1] - times[index])
        a, b = self.reference[index], self.reference[index + 1]
        pose = np.eye(4)
        pose[:3, 3] = a[:3, 3] + fraction * (b[:3, 3] - a[:3, 3])
        rotations = Rotation.from_matrix(np.stack((a[:3, :3], b[:3, :3])))
        pose[:3, :3] = Slerp([0.0, 1.0], rotations)([fraction]).as_matrix()[0]
        return pose


def pose_error(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Position [m] and orientation [rad] difference of two poses (4x4)."""
    turned = Rotation.from_matrix(a[:3, :3].T @ b[:3, :3]).magnitude()
    return float(np.linalg.norm(a[:3, 3] - b[:3, 3])), float(turned)


def _matrix(pose_xyzw: np.ndarray) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(pose_xyzw[3:]).as_matrix()
    matrix[:3, 3] = pose_xyzw[:3]
    return matrix


def _densified(times: np.ndarray, poses: list[np.ndarray]) -> tuple[np.ndarray, list[np.ndarray]]:
    """Samples at most STEP_M and STEP_RAD apart, interpolated linearly and by slerp."""
    out_times, out_poses = [float(times[0])], [poses[0]]
    for i in range(len(poses) - 1):
        a, b = poses[i], poses[i + 1]
        distance, turned = pose_error(a, b)
        count = max(1, math.ceil(max(distance / STEP_M, turned / STEP_RAD) - 1e-9))  # 0.1 / 0.02 is 5.
        slerp = Slerp([0.0, 1.0], Rotation.from_matrix(np.stack((a[:3, :3], b[:3, :3]))))
        for k in range(1, count + 1):
            f = k / count
            pose = np.eye(4)
            pose[:3, 3] = a[:3, 3] + f * (b[:3, 3] - a[:3, 3])
            pose[:3, :3] = slerp([f]).as_matrix()[0]
            out_times.append(float(times[i] + f * (times[i + 1] - times[i])))
            out_poses.append(pose)
    return np.asarray(out_times), out_poses


def lower(
    trajectory: EndEffectorTrajectory,
    *,
    names: tuple[str, ...],
    seed: list[float],
    body: np.ndarray,
    wrist_from_tool: np.ndarray,
    inverse: InverseKinematics,
    forward: ForwardKinematics,
    max_velocity: float,
    start_tolerance: tuple[float, float],
) -> LoweredTrajectory:
    """Joint trajectory for ``names`` following ``trajectory`` from the arm at ``seed`` [rad].

    ``body`` is the world pose (4x4) of the robot base now; a world-frame trajectory is moved into
    it, a body-frame one is taken as relative to it. ``wrist_from_tool`` (4x4) [m] is the trajectory's
    tool in the wrist link (the robot's tool frame). ``inverse`` and ``forward`` solve the wrist
    link in the body frame [m, xyzw] / (4x4). The first pose must be within ``start_tolerance``
    [m, rad] of the tool where the arm is. Raises :class:`TrajectoryRefused` before anything moves.
    """
    tool_from_wrist = np.linalg.inv(wrist_from_tool)
    body_from_frame = np.linalg.inv(body) if trajectory.frame == "world" else np.eye(4)
    poses = [body_from_frame @ _matrix(p) for p in trajectory.poses]
    current = forward(seed) @ wrist_from_tool
    off, turned = pose_error(current, poses[0])
    if off > start_tolerance[0] or turned > start_tolerance[1]:
        raise TrajectoryRefused(
            None,
            f"The trajectory starts {off:.3f} m and {turned:.3f} rad from the tool; start where the tool is.",
        )
    times, dense = _densified(trajectory.times, poses)
    solved = [list(seed)]
    for k, tool in enumerate(dense[1:], start=1):
        wrist = tool @ tool_from_wrist
        quaternion = Rotation.from_matrix(wrist[:3, :3]).as_quat()
        joints = inverse([*wrist[:3, 3].tolist(), *quaternion.tolist()], solved[-1])
        if joints is None:
            raise TrajectoryRefused(
                "out_of_reach", f"The trajectory leaves the arm's reach at {times[k]:.2f} s; move the base."
            )
        if float(np.max(np.abs(np.subtract(joints, solved[-1])))) > JUMP_RAD:
            raise TrajectoryRefused(
                "out_of_reach",
                f"The trajectory needs another arm configuration at {times[k]:.2f} s; move the base.",
            )
        solved.append(list(joints))
    joint_trajectory = JointTrajectory(names, times, np.asarray(solved), trajectory.episode_id)
    joint_trajectory, scale = joint_trajectory.slowed(max_velocity)
    return LoweredTrajectory(joint_trajectory, np.stack(dense), trajectory.tool, scale)
