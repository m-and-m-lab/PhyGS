# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rotations and poses with numpy only (the routines may not import scipy)."""

from __future__ import annotations

import math

import numpy as np


def rotation_of(xyzw) -> np.ndarray:
    """Rotation matrix of a quaternion [x, y, z, w]."""
    x, y, z, w = np.asarray(xyzw, dtype=float) / np.linalg.norm(xyzw)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def quaternion(rotation: np.ndarray) -> np.ndarray:
    """Quaternion [x, y, z, w] of a rotation matrix."""
    m = rotation
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0:
        s = 2.0 * math.sqrt(trace + 1.0)
        q = [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = [0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s]
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = [(m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s]
    else:
        s = 2.0 * math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = [(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s]
    q = np.asarray(q)
    return q / np.linalg.norm(q)


def rotation_about(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotation matrix about a unit axis by angle [rad] (Rodrigues)."""
    x, y, z = axis
    k = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + math.sin(angle) * k + (1.0 - math.cos(angle)) * (k @ k)


def rotate_z(angle: float) -> np.ndarray:
    """Rotation matrix about world z by angle [rad]."""
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def matrix(pose_xyzw) -> np.ndarray:
    """Pose (4x4) of [x, y, z, qx, qy, qz, qw]."""
    pose = np.eye(4)
    pose[:3, :3] = rotation_of(pose_xyzw[3:])
    pose[:3, 3] = pose_xyzw[:3]
    return pose


def xyzw(pose: np.ndarray) -> tuple[float, ...]:
    """[x, y, z, qx, qy, qz, qw] of a pose (4x4)."""
    return (*pose[:3, 3].tolist(), *quaternion(pose[:3, :3]).tolist())
