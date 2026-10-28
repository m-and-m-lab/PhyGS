# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Skill targets, failure reasons and configuration shared by the skill backends."""

from __future__ import annotations

import math
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from phygs.execution.manipulation.planning import JointTrajectory
from phygs.observations.types import Stamp

BACKENDS = ("collision_avoidance_planner", "direct")


@dataclass(frozen=True)
class PixelTarget:
    """A pixel [px] in the latest capture of a permitted depth camera."""

    camera: str
    stamp: Stamp
    x: int
    y: int


@dataclass(frozen=True)
class PointTarget:
    """A 3D point [m] in the world frame or the robot body frame."""

    position: tuple[float, float, float]
    frame: str = "world"


@dataclass(frozen=True)
class ObjectTarget:
    """A simulator object; privileged, accepted only when the task grants it."""

    object_id: str


SkillTarget = PixelTarget | PointTarget | ObjectTarget


@dataclass(frozen=True)
class ArticulatedPart:
    """A sliding part (drawer) and its handle bar in the world frame; privileged task knowledge.

    ``handle_center`` [m]; ``handle_axis``: unit vector along the bar; ``handle_length`` and
    ``handle_radius`` [m]; ``direction``: unit vector the part moves in when it opens;
    ``opening`` [m] now, and its ``limits`` (closed, open) [m]; ``floor_center`` [m]: the middle of
    the drawer floor the opening exposes now, where an object can be set down (None if unknown).
    """

    handle_center: tuple[float, float, float]
    handle_axis: tuple[float, float, float]
    handle_length: float
    handle_radius: float
    direction: tuple[float, float, float]
    opening: float
    limits: tuple[float, float]
    floor_center: tuple[float, float, float] | None = None


class SkillFailure(StrEnum):
    """Reasons an agent can act on; values are added as skills start to report them."""

    HOLDING_OBJECT = "holding_object"
    NOT_HOLDING = "not_holding"
    TARGET_INVALID = "target_invalid"
    OUT_OF_REACH = "out_of_reach"
    PATH_UNOBSERVED = "path_unobserved"
    NO_GRASP = "no_grasp"
    NO_PLACE = "no_place"
    GRASP_EMPTY = "grasp_empty"
    GRASP_LOST = "grasp_lost"
    PLANNING_FAILED = "planning_failed"
    OBSTACLE_CHANGED = "obstacle_changed"
    EXECUTION_STALLED = "execution_stalled"
    CANCELLED = "cancelled"


class SkillError(Exception):
    """A skill outcome the agent can act on; any other exception is a defect and propagates."""

    def __init__(self, failure: SkillFailure, message: str) -> None:
        super().__init__(message)
        self.failure, self.message = failure, message


@dataclass(frozen=True)
class HeldObject:
    """Box held by the gripper, for collision checking only; it is never attached in physics.

    ``size`` [m]; ``wrist_from_object`` is the box pose in the wrist link frame (4x4, m); ``name``
    the scene object, which a planner then leaves out of the obstacles (it moves with the robot).
    """

    size: tuple[float, float, float]
    wrist_from_object: np.ndarray
    name: str | None = None

    def box(self) -> dict:
        """Planner box: size [m] and pose [x, y, z, qx, qy, qz, qw] in the wrist link frame."""
        from scipy.spatial.transform import Rotation

        rotation = Rotation.from_matrix(self.wrist_from_object[:3, :3]).as_quat()
        return dict(size=list(self.size), pose=[*self.wrist_from_object[:3, 3].tolist(), *rotation.tolist()])

    def spheres(self, world_from_wrist: np.ndarray, cell: float) -> np.ndarray:
        """Spheres [x, y, z, r] (m) covering the box, one per grid cell no larger than cell [m]."""
        size = np.asarray(self.size, dtype=float)
        counts = np.maximum(1, np.ceil(size / cell)).astype(int)
        step = size / counts
        axes = [(np.arange(n) + 0.5) * s - extent / 2 for n, s, extent in zip(counts, step, size)]
        local = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
        pose = world_from_wrist @ self.wrist_from_object
        centers = local @ pose[:3, :3].T + pose[:3, 3]
        return np.column_stack((centers, np.full(len(centers), np.linalg.norm(step) / 2)))


# Wrist pose [x, y, z, qx, qy, qz, qw] (m) in the robot body frame and seed arm joints [rad]
# -> arm joints [rad] within joint limits (e.g. ArmKinematics.inverse), or None.
InverseKinematics = Callable[[list[float], list[float]], list[float] | None]


# Goal arm joints [rad] -> pending collision-free plan from the measured state.
JointPlanner = Callable[[list[float]], Future[JointTrajectory]]

# Goal world wrist pose [x, y, z, qx, qy, qz, qw] (m) -> pending collision-free plan.
PosePlanner = Callable[[list[float]], Future[JointTrajectory]]


@dataclass(frozen=True)
class SkillConfig:
    """The ``stow`` and ``look_at`` skills' settings.

    ``joints``: the arm joints, in planning order; ``arm_root_link`` and ``wrist_link`` the arm's
    first link and its wrist link. ``move_seconds`` [s]: a straight joint move's duration.
    ``stow_posture`` [m or rad]: arm joints of the folded transit posture; navigation requires it
    and the transit footprint was measured in it. ``prestow_posture`` [m or rad]: the stow posture
    slightly unfolded above the body. Planners reach the stow through it (and leave the stow through
    it): sphere models of the robot call the folded stow itself a self-collision. ``look_camera``:
    the hand camera ``look_at`` aims.
    """

    backend: str
    move_seconds: float
    stow_posture: tuple[float, ...]
    prestow_posture: tuple[float, ...]
    look_camera: str
    joints: tuple[str, ...]
    arm_root_link: str
    wrist_link: str

    @classmethod
    def from_robot(cls, robot, backend: str) -> SkillConfig:
        """The skills of ``robot`` (a robot description) with the skill ``backend``."""
        return cls(
            backend=backend,
            move_seconds=robot.arm.move_seconds,
            stow_posture=robot.arm.stow,
            prestow_posture=robot.arm.prestow,
            look_camera=robot.look_camera,
            joints=robot.arm.joints,
            arm_root_link=robot.arm.root_link,
            wrist_link=robot.arm.wrist_link,
        )

    def __post_init__(self) -> None:
        if self.backend not in BACKENDS:
            raise ValueError(f"Unknown skill backend {self.backend!r}; expected one of {BACKENDS}.")
        if not (math.isfinite(self.move_seconds) and self.move_seconds > 0):
            raise ValueError("A joint move needs a positive, finite duration.")
        postures = (self.stow_posture, self.prestow_posture)
        if not self.joints or any(len(p) != len(self.joints) or not all(map(math.isfinite, p)) for p in postures):
            raise ValueError("The stow and pre-stow postures need one finite value per arm joint.")
        if not self.look_camera:
            raise ValueError("look_at needs a camera name.")

    def planner_stow(self, tolerance: float) -> dict:
        """The arm planner's ``stow`` setting: plans asked within ``tolerance`` [rad] of the stow start
        with a straight move to the pre-stow posture."""
        return dict(
            stow_posture=list(self.stow_posture),
            prestow_posture=list(self.prestow_posture),
            move_seconds=self.move_seconds,
            tolerance=tolerance,
        )


# Low-level failure messages -> skill failure; anything else is an execution stall.
MOTION_FAILURES = (
    ("unknown space", SkillFailure.PATH_UNOBSERVED),
    ("observed map bounds", SkillFailure.PATH_UNOBSERVED),
    ("occupied voxel", SkillFailure.OBSTACLE_CHANGED),
    ("obstacle", SkillFailure.OBSTACLE_CHANGED),
    ("planner", SkillFailure.PLANNING_FAILED),
    ("new plan", SkillFailure.PLANNING_FAILED),
    ("model snapshot", SkillFailure.PLANNING_FAILED),
)


def motion_failure(reason: str) -> SkillError:
    """The skill failure for a low-level failure reason (MOTION_FAILURES; otherwise a stall)."""
    failure = next(
        (f for text, f in MOTION_FAILURES if text in reason),
        SkillFailure.EXECUTION_STALLED,
    )
    return SkillError(failure, reason)


def unit_approach(approach: tuple[float, float, float]) -> np.ndarray:
    """Validated approach direction: a finite world unit vector that does not point upward."""
    vector = np.asarray(approach, dtype=float)
    if vector.shape != (3,) or not np.isfinite(vector).all() or abs(np.linalg.norm(vector) - 1.0) > 1e-3:
        raise ValueError("The approach must be a world unit vector [x, y, z].")
    if vector[2] > 1e-6:
        raise ValueError("The approach must not point upward.")
    return vector / np.linalg.norm(vector)


def entry_depth(size: tuple[float, float, float], approach: np.ndarray) -> float:
    """Distance [m] from a world-aligned box's center back to where the approach line enters it."""
    return float(min(extent / 2 / abs(a) for extent, a in zip(size, approach) if abs(a) > 1e-9))
