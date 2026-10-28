# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manipulation skill backends (``stow``, ``look_at``); every motion is a low-level
ManipulationClient command.

``direct`` moves in straight joint moves and avoids nothing, like Spot's own arm controller;
``collision_avoidance_planner`` plans the view and the stow around every known obstacle. Agents
compose grasps, placements and drawer strokes themselves from ``move_to_pose``, end-effector
trajectories and the gripper (the robot tests' routines, ``phygs_robot_tests/routines/``).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation

from phygs.execution.manipulation.planners.model import link_transforms
from phygs.execution.manipulation.skills.runner import SkillSequence
from phygs.execution.manipulation.skills.types import (  # noqa: I001
    HeldObject,
    InverseKinematics,
    JointPlanner,
    ObjectTarget,
    PixelTarget,
    PointTarget,
    PosePlanner,
    SkillConfig,
    SkillError,
    SkillFailure,
    SkillTarget,
    motion_failure,
)

if TYPE_CHECKING:
    from phygs.observations.types import ObservationReader
    from phygs.skills.manipulation.api import ManipulationClient

LOG = logging.getLogger("phygs.skills.manipulation")

# Within this [rad] (every arm joint) of the stow posture, the arm counts as stowed.
STOW_TOLERANCE_RAD = 0.2


class DirectSkills:
    """Skills built from straight joint moves; no world collision avoidance."""

    name = "direct"

    def __init__(
        self,
        arm: ManipulationClient,
        observations: ObservationReader,
        model: dict,
        config: SkillConfig,
        *,
        ik: InverseKinematics,
        gripper_open: dict[str, float],
        object_boxes: Callable[[str], tuple[np.ndarray, np.ndarray]] | None = None,
    ) -> None:
        self._arm, self._observations, self._model, self._config = (
            arm,
            observations,
            model,
            config,
        )
        self._solve_ik = ik
        self._gripper_open = gripper_open
        self._object_boxes = object_boxes
        self.attempts = 0
        self._plan_held: HeldObject | None = None  # Given to a skill's planned moves.
        self.step = ""  # What the running skill does now, e.g. "look_at: moving to the view".
        self._skill = ""

    def _begin(self, skill: str, held: HeldObject | None = None) -> None:
        """Start a skill; ``held`` is the object its planned moves carry (None: nothing)."""
        self._skill, self.step = skill, ""
        self._plan_held = held

    def _say(self, text: str) -> None:
        """Announce the current step: kept in ``step`` (skill feedback) and logged."""
        self.step = f"{self._skill}: {text}"
        LOG.info(self.step)

    def stow(self, held: HeldObject | None = None) -> SkillSequence:
        """Fold the arm into the transit posture navigation requires, closing an empty gripper.

        A held object stays held and is carried in the stow; the planned part of the fold
        avoids obstacles with ``held`` (its box and name). The transit footprint does not cover it.
        """
        self._begin("stow", held)
        self.attempts = 1
        if not self._arm.holding:
            self._say("closing the empty gripper")
            yield from self._expect(self._arm.close_gripper())
        else:
            self._say("keeping the held object")
        yield from self._fold()
        self._say("stowed")

    def unstow(self) -> SkillSequence:
        """Unfold the arm from the stow straight to the pre-stow posture, ready to plan from; nothing
        when it is not stowed (as Spot's "ready" after "stow")."""
        self._begin("unstow")
        self.attempts = 1
        if self._stowed():
            yield from self._unfold()
            self._say("ready (pre-stow posture)")
        else:
            self._say("not stowed; nothing to do")

    def look_at(
        self, target: SkillTarget, distance_m: float = 0.5, elevation_deg: float = 45.0, held: HeldObject | None = None
    ) -> SkillSequence:
        """Move the hand camera to one view of the target point and aim it there, upright.

        The camera is distance_m [m] from the point, elevation_deg [deg] above the horizontal,
        on the side of the arm's base (its root link). An object target means its center. The
        skill ends when the arm has settled there; it does not check what the camera sees.
        """
        self._begin("look_at", held)
        self.attempts = 0
        if not (math.isfinite(distance_m) and distance_m > 0 and 0.0 <= elevation_deg <= 90.0):
            raise ValueError("The distance must be positive and finite, the elevation within [0, 90] deg.")
        body = self._body_pose()
        point, _ = self._resolve(target, body)
        toward = point - self._link_pose(body, self._config.arm_root_link)[:3, 3]  # From the arm's base, horizontal.
        toward[2] = 0.0
        toward = toward / np.linalg.norm(toward) if np.linalg.norm(toward) > 1e-6 else body[:3, 0]
        wrist_from_camera = self._wrist_from_camera(body)
        pose = self._view_pose(point, toward, distance_m, math.radians(elevation_deg), wrist_from_camera)
        self._say(f"{describe(target)} {fmt(point)} m; view {distance_m:.2f} m, {elevation_deg:.0f} deg up")
        joints = self._ik(pose, self._arm_joints(), body)
        if joints is None:
            raise SkillError(
                SkillFailure.OUT_OF_REACH,
                "This view of the target is out of reach; move the base.",
            )
        self.attempts = 1
        yield from self._leave_stow()
        yield from self._go_to_view(pose, joints)
        self._say("at the view; the arm has settled")

    def _fold(self) -> SkillSequence:
        """Straight joint move into the stow posture."""
        self._say("moving straight into the stow")
        yield from self._expect(self._move(list(self._config.stow_posture)))

    def _leave_stow(self) -> SkillSequence:
        """Straight moves start from the stow like from anywhere else."""
        yield from ()

    def _unfold(self) -> SkillSequence:
        """Straight joint move from the stow to the pre-stow posture."""
        self._say("leaving the stow: moving straight to the pre-stow posture")
        yield from self._expect(self._move(list(self._config.prestow_posture)))

    def _stowed(self) -> bool:
        joints = zip(self._arm_joints(), self._config.stow_posture, strict=True)
        return max(abs(measured - stowed) for measured, stowed in joints) <= STOW_TOLERANCE_RAD

    def _go_to_view(self, pose: list[float], joints: list[float]) -> SkillSequence:
        """Move straight to the view's IK joints [rad]."""
        self._say("moving straight to the view")
        yield from self._expect(self._move(joints))

    def _wrist_from_camera(self, body: np.ndarray) -> np.ndarray:
        """Measured transform (4x4) from the wrist link to the look camera's optical frame."""
        camera = self._observations.observe().cameras.get(self._config.look_camera)
        if camera is None:
            raise PermissionError(f"look_at needs the permitted camera {self._config.look_camera!r}.")
        return np.linalg.inv(self._wrist_pose(body)) @ body @ camera.extrinsics.matrix.cpu().numpy()

    @staticmethod
    def _view_pose(
        point: np.ndarray,
        toward: np.ndarray,
        distance: float,
        elevation: float,
        wrist_from_camera: np.ndarray,
    ) -> list[float]:
        """World wrist pose [m, xyzw] with the camera distance [m] from point, looking at it, upright.

        The camera is on the ``toward`` side's opposite (the robot's side), elevation [rad] above
        the horizontal. Looking straight down, image "down" points along ``toward``.
        """
        forward = toward * math.cos(elevation) - np.array([0.0, 0.0, math.sin(elevation)])
        down = np.array([0.0, 0.0, -1.0]) - forward * -forward[2]
        down = down / np.linalg.norm(down) if np.linalg.norm(down) > 1e-6 else toward
        camera = np.eye(4)
        camera[:3, :3] = np.column_stack((np.cross(down, forward), down, forward))  # Optical: x right, y down.
        camera[:3, 3] = point - forward * distance
        wrist = camera @ np.linalg.inv(wrist_from_camera)
        return [
            *wrist[:3, 3].tolist(),
            *Rotation.from_matrix(wrist[:3, :3]).as_quat().tolist(),
        ]

    def _ik(self, pose: list[float], seed: list[float], body: np.ndarray) -> list[float] | None:
        """Arm joints [rad] for a world wrist pose [m, xyzw], from the injected IK (joint limits only)."""
        world = np.eye(4)
        world[:3, :3] = Rotation.from_quat(pose[3:]).as_matrix()
        world[:3, 3] = pose[:3]
        local = np.linalg.inv(body) @ world
        return self._solve_ik(
            [
                *local[:3, 3].tolist(),
                *Rotation.from_matrix(local[:3, :3]).as_quat().tolist(),
            ],
            seed,
        )

    def _wrist_pose(self, body: np.ndarray) -> np.ndarray:
        """Measured world pose of the wrist link (4x4, m)."""
        return self._link_pose(body, self._config.wrist_link)

    def _link_pose(self, body: np.ndarray, link: str) -> np.ndarray:
        """Measured world pose (4x4, m) of a robot link."""
        joints = {
            **self._model["locked_joints"],
            **dict(zip(self._config.joints, self._arm_joints())),
        }
        return body @ link_transforms(self._model["urdf"], joints)[link]

    def _resolve(self, target: SkillTarget, body: np.ndarray) -> tuple[np.ndarray, tuple[float, float, float] | None]:
        """World point [m] and object size [m]: an object's box center and size, else a point and None.

        The point is the pixel's depth return or the given point.
        """
        if isinstance(target, ObjectTarget):
            if self._object_boxes is None:
                raise PermissionError("Object targets require privileged_targets in the task.")
            center, size = self._object_boxes(target.object_id)
            return np.asarray(center, dtype=float), tuple(map(float, size))
        if isinstance(target, PointTarget):
            if target.frame not in {"world", "body"}:
                raise ValueError("Point targets use the 'world' or 'body' frame.")
            point = np.asarray(target.position, dtype=float)
            return (point if target.frame == "world" else (body @ [*point, 1.0])[:3]), None
        if not isinstance(target, PixelTarget):
            raise TypeError(f"Unsupported skill target {type(target).__name__}.")
        packet = self._observations.observe()
        camera = packet.cameras.get(target.camera)
        if camera is None or camera.depth is None:
            raise PermissionError(f"Camera {target.camera!r} has no permitted depth.")
        if camera.stamp != target.stamp:
            raise SkillError(
                SkillFailure.TARGET_INVALID,
                "The pixel is not from the latest capture; observe again.",
            )
        height, width = camera.depth.shape
        if not (0 <= target.x < width and 0 <= target.y < height) or not bool(camera.depth_valid[target.y, target.x]):
            raise SkillError(SkillFailure.TARGET_INVALID, "The pixel has no valid depth.")
        ray = np.linalg.solve(camera.intrinsics.cpu().numpy(), [target.x, target.y, 1.0])
        world_from_camera = packet.transform("world", camera.frame, camera.stamp).cpu().numpy()
        return (world_from_camera @ [*(ray * float(camera.depth[target.y, target.x])), 1.0])[:3], None

    def _expect(self, command: int, planning: SkillFailure = SkillFailure.PLANNING_FAILED) -> SkillSequence:
        """Wait for a low-level command; turn its failure into a skill failure.

        ``planning`` is reported when the command's planner found no plan.
        """
        feedback = yield command
        if feedback.status != "succeeded":
            error = motion_failure(feedback.reason or feedback.status)
            if error.failure == SkillFailure.PLANNING_FAILED:
                error.failure = planning
            raise error
        return feedback

    def _move(self, joints: list[float]) -> int:
        target = dict(zip(self._config.joints, joints, strict=True))
        return self._arm.move_joints(
            target,
            duration=max(self._config.move_seconds, self._arm.minimum_move_duration(target)),
        )

    def _body_pose(self) -> np.ndarray:
        localization = self._observations.observe().localization
        if localization is None or not localization.valid:
            raise PermissionError("Manipulation skills require permitted localization.")
        return localization.pose.matrix.cpu().numpy()

    def _arm_joints(self) -> list[float]:
        return list(self._measured(self._config.joints).values())

    def _measured(self, names) -> dict[str, float]:
        """Measured positions [m or rad] of the named joints."""
        joints = self._observations.observe().joints
        measured = dict(zip(joints.names, joints.positions.cpu().tolist(), strict=True))
        return {n: measured[n] for n in names}


class CollisionAvoidancePlannerSkills(DirectSkills):
    """Reach start poses, views and the stow along collision-free plans from the planning library."""

    name = "collision_avoidance_planner"

    def __init__(
        self,
        *args,
        joint_planner: JointPlanner,
        pose_planner: PosePlanner,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._joint_planner, self._pose_planner = joint_planner, pose_planner

    def _fold(self) -> SkillSequence:
        """Plan to the pre-stow posture, then move straight into the stow.

        Sphere models of the robot call the folded stow a self-collision, so planners cannot end
        there; the last 0.3 rad stay within the robot's own folded volume, above its body.
        """
        if not self._stowed():
            prestow = list(self._config.prestow_posture)
            self._say("planning and moving to the pre-stow posture")
            yield from self._expect(self._arm.execute_trajectory(self._joint_planner(prestow, self._plan_held)))
        self._say("moving straight into the stow")
        yield from self._expect(self._move(list(self._config.stow_posture)))

    def _leave_stow(self) -> SkillSequence:
        """From the stow, move straight to the pre-stow posture: planners reject the stow as a start."""
        if self._stowed():
            yield from self._unfold()

    def _go_to_view(self, pose: list[float], joints: list[float]) -> SkillSequence:
        """Plan to the view's wrist pose [m, xyzw] and move there."""
        self._say("planning and moving to the view")
        yield from self._expect(self._arm.execute_trajectory(self._pose_planner(pose, self._plan_held)))


def fmt(vector) -> str:
    """A vector for step messages, e.g. (1.25, 0.00, 0.47)."""
    return "(" + ", ".join(f"{float(v):.2f}" for v in vector) + ")"


def describe(target: SkillTarget) -> str:
    """A target for step messages."""
    if isinstance(target, ObjectTarget):
        return f"object {target.object_id!r}"
    if isinstance(target, PixelTarget):
        return f"pixel ({target.x}, {target.y}) of {target.camera!r}"
    return f"point ({target.frame})"


def create_skills(
    arm: ManipulationClient,
    observations: ObservationReader,
    model: dict,
    config: SkillConfig,
    *,
    ik: InverseKinematics,
    gripper_open: dict[str, float],
    object_boxes: Callable[[str], tuple[np.ndarray, np.ndarray]] | None = None,
    joint_planner: JointPlanner | None = None,
    pose_planner: PosePlanner | None = None,
) -> DirectSkills:
    """Build the configured backend; the planner backend requires grasp, joint and pose planners.

    ``object_boxes`` is a privileged source, given only when the task grants it.
    """
    common = dict(ik=ik, gripper_open=gripper_open, object_boxes=object_boxes)
    if config.backend == "direct":
        return DirectSkills(arm, observations, model, config, **common)
    if joint_planner is None or pose_planner is None:
        raise ValueError("The collision_avoidance_planner backend requires grasp, joint and pose planners.")
    planners = dict(
        joint_planner=joint_planner,
        pose_planner=pose_planner,
    )
    return CollisionAvoidancePlannerSkills(arm, observations, model, config, **planners, **common)
