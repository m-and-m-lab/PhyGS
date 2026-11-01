# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Arm forward and inverse kinematics from the planning URDF, in the body frame (the URDF's root link).

Joint limits only: self and world collisions are the planners' checks. Inverse kinematics is
bounded least squares from the seed, then from random joint sets, so a solution near the edge
of the workspace is not lost to a poor seed. Solutions stay LIMIT_INSET_RAD inside the URDF
limits: the simulator holds the limits in single precision, which can exclude the exact bound.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

LIMIT_INSET_RAD = 1e-4
NEAR_SEED_RAD = 0.3  # A solution this close to the seed (every joint) ends the search [rad].


class ArmKinematics:
    """Kinematics of the serial chain from the URDF's root link to ``tip`` through ``joints``
    (revolute [rad] or prismatic [m], in order)."""

    @classmethod
    def from_model(cls, model: dict) -> ArmKinematics:
        """The arm of a planning model (:func:`~.planners.model.prepare_model`): its joints to the wrist link."""
        return cls(model["urdf"], tuple(model["joints"]), model["wrist_link"])

    def __init__(
        self,
        urdf: str | Path,
        joints: tuple[str, ...],
        tip: str,
        *,
        restarts: int = 16,
        position_tolerance: float = 1e-3,
        orientation_tolerance: float = 1e-2,
    ) -> None:
        """``restarts`` random joint sets are tried after the seed; tolerances are [m] and [rad]."""
        elements = {j.find("child").get("link"): j for j in ET.parse(urdf).getroot().findall("joint")}
        roots = {j.find("parent").get("link") for j in elements.values()} - set(elements)
        path, link = [], tip
        while link not in roots:
            if link not in elements:
                raise ValueError(f"No joint chain from the root link to {tip!r} in {urdf}")
            path.append(elements[link])
            link = elements[link].find("parent").get("link")
        self._chain = []  # (origin 4x4, axis or None, joint index or None, prismatic), root to tip.
        lower, upper = [], []
        for element in reversed(path):
            origin = np.eye(4)
            if (frame := element.find("origin")) is not None:
                origin[:3, 3] = np.fromstring(frame.get("xyz", "0 0 0"), sep=" ")
                rpy = np.fromstring(frame.get("rpy", "0 0 0"), sep=" ")
                origin[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
            name = element.get("name")
            if element.get("type") == "fixed":
                self._chain.append((origin, None, None, False))
                continue
            if element.get("type") not in {"revolute", "prismatic"} or name not in joints:
                raise ValueError(f"Joint {name!r} on the chain to {tip!r} is not one of the arm joints {joints}")
            limit = element.find("limit")
            lower.append(float(limit.get("lower")))
            upper.append(float(limit.get("upper")))
            axis = np.fromstring(element.find("axis").get("xyz"), sep=" ")
            self._chain.append((origin, axis, joints.index(name), element.get("type") == "prismatic"))
        if len(lower) != len(joints):
            raise ValueError(f"The chain to {tip!r} does not hold exactly the joints {joints}")
        order = [index for _, axis, index, _ in self._chain if axis is not None]
        self.lower, self.upper = np.array(lower)[np.argsort(order)], np.array(upper)[np.argsort(order)]
        self._restarts = restarts
        self._tolerances = (position_tolerance, orientation_tolerance)

    def forward(self, joints: list[float] | np.ndarray) -> np.ndarray:
        """Body-frame pose (4x4) [m] of the tip at joint positions [m or rad]."""
        pose = np.eye(4)
        for origin, axis, index, prismatic in self._chain:
            pose = pose @ origin
            if axis is not None:
                motion = np.eye(4)
                if prismatic:
                    motion[:3, 3] = axis * joints[index]
                else:
                    motion[:3, :3] = Rotation.from_rotvec(axis * joints[index]).as_matrix()
                pose = pose @ motion
        return pose

    def inverse(self, pose_xyzw: list[float], seed: list[float]) -> list[float] | None:
        """Joint angles [rad] placing the tip at a body-frame pose [m, xyzw], or None.

        Tried from the seed first, then from ``restarts`` random joint sets (fixed sequence). A
        solution within NEAR_SEED_RAD of the seed (every joint) is returned at once; otherwise the
        solution nearest the seed: the arm stays in its configuration (e.g. not the one with the
        forearm and wrist rolled by half a turn, whose wrist roll then meets its limit).
        """
        position = np.asarray(pose_xyzw[:3], dtype=float)
        rotation = Rotation.from_quat(pose_xyzw[3:]).as_matrix()

        def residual(joints: np.ndarray) -> np.ndarray:
            tip = self.forward(joints)
            error = Rotation.from_matrix(rotation.T @ tip[:3, :3]).as_rotvec()
            return np.concatenate((tip[:3, 3] - position, 0.2 * error))  # 0.2 m per rad.

        lower, upper = self.lower + LIMIT_INSET_RAD, self.upper - LIMIT_INSET_RAD
        margin = 1e-6
        starts = [np.clip(seed, lower + margin, upper - margin)]
        starts += list(np.random.default_rng(0).uniform(lower, upper, (self._restarts, len(lower))))
        best, nearest = None, np.inf
        for start in starts:
            result = least_squares(residual, start, bounds=(lower, upper), xtol=1e-10, ftol=1e-10)
            tip = self.forward(result.x)
            reached = np.linalg.norm(tip[:3, 3] - position) <= self._tolerances[0]
            turned = Rotation.from_matrix(rotation.T @ tip[:3, :3]).magnitude() <= self._tolerances[1]
            if reached and turned:
                distance = float(np.max(np.abs(result.x - np.asarray(seed, dtype=float))))
                if distance <= NEAR_SEED_RAD:
                    return result.x.tolist()
                if distance < nearest:
                    best, nearest = result.x.tolist(), distance
        return best
