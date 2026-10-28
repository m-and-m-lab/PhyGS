# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""What the person chose on the page, as arguments the skill API already takes.

The server has turned a clicked pixel into a world point and surface normal
(:mod:`phygs.ui.targets`); here a point becomes a ``PointTarget``, a
"face the surface" choice a wrist pose, and "move up 5 cm" a world axis. The API is unchanged:
these are the inputs an agent would compute itself, from what the robot's cameras saw.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial.transform import Rotation

from phygs.execution.manipulation.skills.types import PointTarget

UP = np.array([0.0, 0.0, 1.0])
# Directions for Move along, relative to the robot's heading.
DIRECTIONS = ("up", "down", "forward", "back", "left", "right")
LINE_SPEED_M_S = 0.1  # Move along's wrist speed [m/s].
# Wrist orientations for move_to_pose: where the gripper points.
ORIENTATIONS = ("down", "forward", "surface")
# Drawer handle bars: along the drawer front (horizontal) or upright.


def point_target(target: dict) -> PointTarget:
    """A resolved click ``{point, ...}`` [m, world] as a PointTarget."""
    point = [float(value) for value in target["point"]]
    if len(point) != 3 or not all(map(math.isfinite, point)):
        raise ValueError("A target point is three finite world coordinates [m].")
    return PointTarget(tuple(point), "world")


def heading_axes(yaw: float) -> dict[str, np.ndarray]:
    """World unit vectors for the robot-relative directions at heading yaw [rad]."""
    forward = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    left = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
    return dict(up=UP, down=-UP, forward=forward, back=-forward, left=left, right=-left)


def axis(direction: str, yaw: float) -> tuple[float, float, float]:
    """Move along's world axis for a robot-relative direction."""
    if direction not in DIRECTIONS:
        raise ValueError(f"Unknown direction {direction!r}; expected one of {DIRECTIONS}.")
    return tuple(float(value) for value in heading_axes(yaw)[direction])


def line(start: tuple[float, ...], axis: tuple[float, float, float], distance_m: float) -> tuple[list[float], list]:
    """Move along's times [s] and poses [m, xyzw]: from the wrist pose ``start`` straight along
    ``axis`` by ``distance_m`` [m] at :data:`LINE_SPEED_M_S`, keeping the orientation."""
    if not distance_m > 0.0:
        raise ValueError("Move along needs a positive distance.")
    end = [*(np.asarray(start[:3]) + distance_m * np.asarray(axis)).tolist(), *start[3:]]
    return [0.0, distance_m / LINE_SPEED_M_S], [list(start), end]


def gripper_rotation(pointing: np.ndarray, yaw: float) -> Rotation:
    """The gripper's axes (a tool frame's: x the approach, z the jaws' closing axis) with the
    approach along ``pointing``, upright where possible.

    The y axis is horizontal (left of the pointing direction); pointing straight up or down, it is
    the robot's left. Top-down at heading 0 the jaws close along the heading, as the routines grasp.
    """
    x = np.asarray(pointing, dtype=float) / np.linalg.norm(pointing)
    y = np.cross(UP, x)
    if np.linalg.norm(y) < 1e-6:
        y = heading_axes(yaw)["left"]
    y /= np.linalg.norm(y)
    return Rotation.from_matrix(np.stack([x, y, np.cross(x, y)], axis=1))


def wrist_pose(
    point: tuple[float, ...],
    orientation: str,
    *,
    yaw: float,
    standoff_m: float,
    grasp: np.ndarray,
    normal: tuple[float, ...] | None = None,
) -> tuple[float, ...]:
    """move_to_pose's wrist pose [m, xyzw]: the grasp point ``standoff_m`` [m] before ``point``.

    The gripper points down, along the robot's heading, or into the surface (against its
    ``normal``). ``grasp`` is the robot's grasp tool frame in the wrist link (4x4) [m].
    """
    if orientation not in ORIENTATIONS:
        raise ValueError(f"Unknown orientation {orientation!r}; expected one of {ORIENTATIONS}.")
    if not (math.isfinite(standoff_m) and standoff_m >= 0.0):
        raise ValueError("The standoff is a distance of at least 0 m.")
    if orientation == "surface":
        if normal is None:
            raise ValueError("No surface normal at that pixel (an edge); click inside a surface.")
        pointing = -np.asarray(normal, dtype=float)
    else:
        pointing = -UP if orientation == "down" else heading_axes(yaw)["forward"]
    tool = np.eye(4)
    tool[:3, :3] = gripper_rotation(pointing, yaw).as_matrix()
    tool[:3, 3] = np.asarray(point, dtype=float) - tool[:3, 0] * standoff_m
    wrist = tool @ np.linalg.inv(grasp)
    return (
        *(float(value) for value in wrist[:3, 3]),
        *(float(v) for v in Rotation.from_matrix(wrist[:3, :3]).as_quat()),
    )
