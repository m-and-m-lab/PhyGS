# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Public runtime arm/gripper API: low-level commands and the ``stow`` and ``look_at`` skills
(see docs/manipulation_skills.md and docs/design/manipulation_contract/)."""

from __future__ import annotations

import math
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phygs.execution.manipulation.controller import (
        ManipulationController,
    )
    from phygs.execution.manipulation.kinematics import ArmKinematics
    from phygs.execution.manipulation.skills.backends import DirectSkills
    from phygs.observations.types import ObservationReader
    from phygs.robots.description import RobotDescription
    from phygs.runtime.episode import EpisodeRuntime
    from phygs.skills.navigation.api import NavigationClient

from phygs.execution.manipulation.lowering import TrajectoryRefused as TrajectoryRefused
from phygs.execution.manipulation.lowering import lower, pose_error
from phygs.execution.manipulation.planning import (
    EndEffectorTrajectory as EndEffectorTrajectory,
)
from phygs.execution.manipulation.planning import JointTrajectory as JointTrajectory
from phygs.execution.manipulation.skills.runner import (
    SkillRun,
    SkillSequence,
)
from phygs.execution.manipulation.skills.types import (
    ArticulatedPart as ArticulatedPart,
)
from phygs.execution.manipulation.skills.types import (
    HeldObject as HeldObject,
)
from phygs.execution.manipulation.skills.types import (
    ObjectTarget as ObjectTarget,
)
from phygs.execution.manipulation.skills.types import (
    PixelTarget as PixelTarget,
)
from phygs.execution.manipulation.skills.types import (
    PointTarget as PointTarget,
)
from phygs.execution.manipulation.skills.types import SkillFailure
from phygs.execution.manipulation.skills.types import (
    SkillTarget as SkillTarget,
)
from phygs.runtime.commands import CommandClient, CommandFeedback
from phygs.runtime.timing import duration_ticks


@dataclass(frozen=True)
class MotionFeedback(CommandFeedback):
    """Measured motion outcome; joint error [rad], pose errors [m, rad].

    ``pose_error`` is the end-effector tracking error of an end-effector trajectory (or the
    Cartesian servo's error); ``progress`` the fraction of the reference duration reached;
    ``time_scale`` how much the trajectory was slowed down to the joint speed limit (1.0: not).

    A finished motion (succeeded, failed or cancelled) also reports where it ended:
    ``reached_seconds`` [s] of its reference it got through, ``joint_targets`` [rad] its reference
    then (for a joint move, the interpolated set-points; for a trajectory, its sample at that time),
    ``joint_positions`` [m or rad] the measured arm and gripper, ``gripper_aperture`` [m or rad] the
    measured primary gripper joint alone, ``stalled`` whether the arm stopped
    following its reference (a tracking failure or a timeout, not a cancel), and ``tool_pose``
    [m, xyzw] the measured hand pose (``tool_pose("hand")``; None without kinematics or
    localization). While a motion runs these are empty.
    """

    reason: str = ""
    joint_error: float = 0.0
    pose_error: tuple[float, float] = (0.0, 0.0)
    progress: float = 0.0
    time_scale: float = 1.0
    reached_seconds: float = 0.0
    joint_targets: dict[str, float] = field(default_factory=dict)
    joint_positions: dict[str, float] = field(default_factory=dict)
    gripper_aperture: float | None = None
    stalled: bool = False
    tool_pose: tuple[float, ...] | None = None


@dataclass(frozen=True)
class SkillFeedback(MotionFeedback):
    """Skill outcome from robot-sensed signals; not evidence of task success.

    ``reason`` is one sentence for the agent; ``failure`` is the actionable code.
    """

    skill: str = ""
    failure: SkillFailure | None = None
    attempts: int = 0
    holding: bool = False
    step: str = ""  # What the skill does now (or did last), e.g. "look_at: moving to the view".


@dataclass
class _ActiveSkill:
    command_id: int
    episode_id: int
    name: str
    run: SkillRun


class ManipulationClient(CommandClient):
    """Arm/gripper commands with optional planned trajectories; no grasp-success claim.

    ``robot`` (its description) gives the arm joints, the gripper's primary joint and its open and
    closed positions, and the tool frames.
    """

    def __init__(
        self,
        runtime: EpisodeRuntime,
        controller: ManipulationController,
        robot: RobotDescription,
        *,
        enabled: bool = True,
        pose_planner: Callable[[tuple[float, ...]], JointTrajectory | Future[JointTrajectory]] | None = None,
        joint_planner: Callable[[list[float], HeldObject | None], JointTrajectory | Future[JointTrajectory]]
        | None = None,
    ) -> None:
        super().__init__(runtime, controller.declaration.name)
        self._controller = controller
        self._pose_planner, self._joint_planner = pose_planner, joint_planner
        self._pending_plan: Future | None = None
        self._enabled = enabled
        self._joints, self._tools = robot.arm.joints, robot.gripper.tools
        self._gripper_joints, self._gripper_joint = robot.gripper.joints, robot.gripper.primary
        self._open_angle, self._closed_angle = robot.gripper.open, robot.gripper.closed
        self._outcomes: dict[int, MotionFeedback] = {}
        self._skills: DirectSkills | None = None
        self._skill: _ActiveSkill | None = None
        self._kinematics: ArmKinematics | None = None
        self._localization: ObservationReader | None = None
        self.navigation: NavigationClient | None = None
        controller.validate({self._gripper_joint: self._open_angle})
        controller.validate({self._gripper_joint: self._closed_angle})

    def _require_available(self) -> None:
        if not self._enabled:
            raise PermissionError("Manipulation is not permitted by this task.")
        if self.navigation is not None and self.navigation.active:
            raise RuntimeError("Cancel navigation and wait for stopping before manipulation.")
        if self._skill is not None and not self._skill.run.done and not self._skill.run.advancing:
            raise RuntimeError("Cancel or finish the current manipulation skill first.")
        if self._current is not None and self.feedback(self._current).status in {
            "active",
            "stopping",
        }:
            raise RuntimeError("Cancel or finish the current manipulation command first.")

    @property
    def active_command(self) -> int | None:
        """The arm command executing or stopping now (a skill's or a direct one), or None."""
        if self._skill is not None and not self._skill.run.done:
            return self._skill.command_id
        if self._current is not None and self.feedback(self._current).status in {"active", "stopping"}:
            return self._current
        return None

    @property
    def episode_id(self) -> int:
        """Current episode; a trajectory for :meth:`execute_trajectory` must carry it."""
        return self._runtime.episode_id

    @property
    def controller(self) -> ManipulationController:
        """Trusted diagnostics for robot tests (environment-only; agents cannot reach it): the
        controller's ``mode``, ``reason`` and joint ``reference`` [rad]."""
        return self._controller

    @property
    def gripper_closed(self) -> bool:
        """Whether the gripper is commanded closed: squeezing a held object (whatever the angle a wide
        object stops it at), or its setpoint is nearer the closed than the open angle."""
        controller = self._controller
        setpoint = controller.gripper_setpoint
        if setpoint is None:
            return False
        return controller.object_held or abs(setpoint - self._closed_angle) < abs(setpoint - self._open_angle)

    @property
    def holding(self) -> bool:
        """Whether the gripper closed on an object and still holds it (from measured gripper motion)."""
        return self._controller.holding

    def _measured_tool_pose(self) -> tuple[float, ...] | None:
        """The hand's pose for feedback, when kinematics and localization allow it."""
        if self._kinematics is None or self._localization is None:
            return None
        try:
            return self.tool_pose("hand")
        except PermissionError:
            return None

    def tool_pose(self, tool: str = "hand") -> tuple[float, ...]:
        """World pose [m, xyzw] of the tool (``hand`` between the jaws, or ``wrist``), from the
        measured arm and the permitted localization; for starting end-effector trajectories."""
        if self._kinematics is None:
            raise RuntimeError("No arm kinematics are bound.")
        wrist_from_tool = self._tool(tool)
        localization = self._localization.observe().localization
        if localization is None or not localization.valid:
            raise PermissionError("The tool pose requires permitted localization.")
        from scipy.spatial.transform import Rotation

        measured = self._controller.reader.positions()
        body = localization.pose.matrix.cpu().numpy()
        pose = body @ self._kinematics.forward([measured[j] for j in self._joints]) @ wrist_from_tool
        return (*pose[:3, 3].tolist(), *Rotation.from_matrix(pose[:3, :3]).as_quat().tolist())

    def _tool(self, tool: str):
        """The tool frame in the wrist link (4x4) [m]."""
        if tool not in self._tools:
            raise ValueError(f"Unknown tool {tool!r}; expected one of {sorted(self._tools)}.")
        return self._tools[tool]

    def minimum_move_duration(self, positions: dict[str, float]) -> float:
        """Shortest move_joints duration [s] to positions [rad] within the velocity and acceleration limits."""
        measured, config = self._controller.reader.positions(), self._controller.config
        distance = max(abs(value - measured[joint]) for joint, value in positions.items())
        return max(
            1.875 * distance / config.max_velocity,
            math.sqrt(10 / 3**0.5 * distance / config.max_acceleration),
        )

    def bind_kinematics(self, kinematics: ArmKinematics, observations: ObservationReader) -> None:
        """Trusted setup: the arm's kinematics (planning URDF, wrist link in the body frame) and the
        permitted localization, which end-effector trajectories need; bound once."""
        if self._kinematics is not None:
            raise RuntimeError("Arm kinematics are already bound.")
        self._kinematics, self._localization = kinematics, observations

    def bind_skills(self, skills: DirectSkills) -> None:
        """Trusted setup: bind the skill backend once; agents call the skill methods."""
        if self._skills is not None:
            raise RuntimeError("Manipulation skills are already bound.")
        self._skills = skills

    def stow(self, *, held: HeldObject | None = None) -> int:
        """Fold the arm into the transit posture navigation requires; close an empty gripper.

        A held object stays held and is carried; the planned part avoids obstacles with ``held``
        (its box and name). The navigation footprint does not cover it.
        """
        return self._start_skill("stow", lambda skills: skills.stow(held))

    def unstow(self) -> int:
        """Unfold the arm from the stow to the pre-stow posture, ready to plan from (nothing when it is
        not stowed). Optional: every arm skill, :meth:`move_to_pose` and :meth:`plan` leave the stow
        this way by themselves."""
        return self._start_skill("unstow", lambda skills: skills.unstow())

    def look_at(
        self,
        target: SkillTarget,
        *,
        distance_m: float = 0.5,
        elevation_deg: float = 45.0,
        held: HeldObject | None = None,
    ) -> int:
        """Move the hand camera to one view of the target and aim it there, upright.

        The camera is distance_m [m] from the target, elevation_deg [deg] above the horizontal,
        on the robot's side. Ends when the arm has settled; it does not check what the camera
        sees: observe afterwards to get the view. An object target means its center.
        """
        return self._start_skill("look_at", lambda skills: skills.look_at(target, distance_m, elevation_deg, held))

    def _start_skill(self, name: str, sequence: Callable[[DirectSkills], SkillSequence]) -> int:
        if self._skills is None:
            raise RuntimeError("No manipulation skill backend is bound.")
        self._require_available()
        command_id, self._next_id = self._next_id, self._next_id + 1
        run = SkillRun(sequence(self._skills), self.feedback)
        self._skill = _ActiveSkill(command_id, self._runtime.episode_id, name, run)
        self._record_skill()
        return command_id

    def _record_skill(self) -> None:
        """Freeze the outcome of a finished skill, including whether it ends holding an object."""
        skill = self._skill
        if skill is None or not skill.run.done or skill.command_id in self._outcomes:
            return
        failure = skill.run.failure
        status = "succeeded" if failure is None else "cancelled" if failure == SkillFailure.CANCELLED else "failed"
        measured = self._controller.reader.positions()
        self._outcomes[skill.command_id] = SkillFeedback(
            skill.command_id,
            skill.episode_id,
            status,
            skill.run.message,
            progress=1.0 if status == "succeeded" else 0.0,
            joint_positions={joint: measured[joint] for joint in self._controller.declaration.joints},
            gripper_aperture=measured[self._gripper_joint],
            stalled=self._controller.stalled,
            tool_pose=self._measured_tool_pose(),
            skill=skill.name,
            failure=failure,
            attempts=self._skills.attempts,
            holding=self.holding,
            step=self._skills.step,
        )

    def move_joints(self, positions: dict[str, float], *, duration: float = 3.0) -> int:
        """Move arm/gripper targets [m or rad] over duration [s], then verify settling.

        The duration is rounded up to whole physics steps, so the move is never faster than asked.
        This free-space command does not check collisions. Unspecified joints hold.
        """
        self._require_available()
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("Duration must be positive and finite.")
        step = self._runtime.physics_dt
        duration = math.ceil(duration / step - 1e-9) * step
        controller = self._controller
        controller.validate(positions)
        minimum = self.minimum_move_duration(positions)
        if duration < minimum * (1 - 1e-9):
            raise ValueError(
                f"Requested motion exceeds the joint velocity or acceleration limit; request at least {minimum:.2f} s."
            )
        ticks = duration_ticks(
            duration + controller.config.timeout_seconds + controller.config.stop_timeout_seconds,
            self._runtime.physics_dt,
        )
        command = self._submit(ticks + 1)
        controller.start(positions, duration)
        return command

    def move_to_pose(
        self,
        pose_xyzw: tuple[float, ...],
        *,
        held: HeldObject | None = None,
        tool: str = "hand",
        duration: float | None = None,
    ) -> int:
        """Move the tool (``hand`` between the jaws, or ``wrist``) to a world pose [m, xyzw].

        A configured planner plans around the permitted obstacles and times the motion; ``held`` is
        the object the gripper carries (its box in the wrist frame and name), so the planner avoids
        obstacles with it and leaves it out of them. From the stow the arm first moves straight to
        the pre-stow posture. Otherwise this uses free-space differential IK,
        with duration defaulting to 3 s. Planner failure never switches to differential IK.
        """
        pose_xyzw = self._wrist_pose(pose_xyzw, tool)
        self._require_available()
        if self._pose_planner is not None:
            if duration is not None:
                raise ValueError("The collision planner determines trajectory duration from its motion limits")
            return self.execute_trajectory(self._pose_planner(pose_xyzw, held))
        duration = 3.0 if duration is None else duration
        if self._controller.pose_servo is None:
            raise RuntimeError("No Cartesian servo is configured.")
        self._require_available()
        self._controller.pose_servo.start(pose_xyzw)
        command = self.move_joints(self._held_arm(), duration=duration)
        self._controller.mode = "pose"
        return command

    def _wrist_pose(self, pose_xyzw: tuple[float, ...], tool: str) -> tuple[float, ...]:
        """The wrist pose [m, xyzw] that puts ``tool`` at a validated world pose [m, xyzw]."""
        import math

        import numpy as np
        from scipy.spatial.transform import Rotation

        if len(pose_xyzw) != 7 or not all(math.isfinite(v) for v in pose_xyzw):
            raise ValueError("Expected a finite world pose [x,y,z,qx,qy,qz,qw].")
        if not math.isclose(sum(v * v for v in pose_xyzw[3:]), 1.0, abs_tol=1e-5):
            raise ValueError("Pose quaternion must be normalized.")
        wrist_from_tool = self._tool(tool)
        world_from_tool = np.eye(4)
        world_from_tool[:3, :3] = Rotation.from_quat(pose_xyzw[3:]).as_matrix()
        world_from_tool[:3, 3] = pose_xyzw[:3]
        wrist = world_from_tool @ np.linalg.inv(wrist_from_tool)
        return (*wrist[:3, 3].tolist(), *Rotation.from_matrix(wrist[:3, :3]).as_quat().tolist())

    def plan(
        self,
        *,
        goal: dict[str, float] | None = None,
        pose_xyzw: tuple[float, ...] | None = None,
        tool: str = "hand",
        held: HeldObject | None = None,
    ) -> Future[JointTrajectory]:
        """Plan a collision-free arm path without moving: to every arm joint's position [m or rad]
        (``goal``) or to a world pose of the tool [m, xyzw] (``hand`` between the jaws, or
        ``wrist``). ``held`` is the object the gripper carries, as for :meth:`move_to_pose`.

        Returns a Future of the :class:`JointTrajectory` from the arm's current joints; it fails
        with the planner's reason. From the stow the trajectory first moves straight to the pre-stow
        posture (planners reject the stow as a start), then follows the plan. The plan is made while
        the simulation runs: with the charged clock, sleep (``session.sleep``) until it is done.
        Execute it with :meth:`execute_trajectory`, or drop it. Needs the collision planner.
        """
        self._require_available()
        if (goal is None) == (pose_xyzw is None):
            raise ValueError("Plan to a joint goal or to a tool pose, not both.")
        if goal is not None:
            if self._joint_planner is None:
                raise RuntimeError("No collision planner is configured for joint goals.")
            if set(goal) != set(self._joints):
                raise ValueError(f"A joint goal sets every arm joint: {', '.join(self._joints)}.")
            self._controller.validate(goal)
            return self._as_future(self._joint_planner([float(goal[j]) for j in self._joints], held))
        if self._pose_planner is None:
            raise RuntimeError("No collision planner is configured for tool poses.")
        return self._as_future(self._pose_planner(self._wrist_pose(pose_xyzw, tool), held))

    @staticmethod
    def _as_future(plan: JointTrajectory | Future[JointTrajectory]) -> Future[JointTrajectory]:
        if isinstance(plan, Future):
            return plan
        future: Future[JointTrajectory] = Future()
        future.set_result(plan)
        return future

    def _held_arm(self) -> dict[str, float]:
        """The arm's held targets [rad] after a successful motion, else its measured joints."""
        return dict(self._controller.arm_hold) or self._measured_arm()

    def _measured_arm(self) -> dict[str, float]:
        """Measured arm joints [m or rad]; the gripper is left out so a held object stays squeezed."""
        measured = self._controller.reader.positions()
        return {j: measured[j] for j in self._controller.declaration.joints if j not in self._gripper_joints}

    def execute_trajectory(self, trajectory: JointTrajectory | EndEffectorTrajectory | Future[JointTrajectory]) -> int:
        """Execute a trajectory in this episode as given: no planner, no collision check.

        A :class:`JointTrajectory` holds planner samples [rad, s]. An :class:`EndEffectorTrajectory`
        holds timed tool poses (the hand by default); it is solved with IK first and refused before
        moving (:class:`TrajectoryRefused`) when a pose is out of reach, needs another arm
        configuration, or the first pose is not where the tool is. Either is slowed down uniformly
        where a joint would exceed the speed limit (feedback ``time_scale``). While it runs every
        joint tracks its reference (``tracking_tolerance``) and, for an end-effector trajectory, the
        tool its pose (``tool_position_tolerance``, ``tool_orientation_tolerance``); at the end the
        arm only has to stop. A pending plan (Future) holds the arm until it arrives; it must start
        within ``tracking_tolerance`` of the measured arm. :meth:`move_to_pose` plans around obstacles.
        """
        self._require_available()
        if isinstance(trajectory, EndEffectorTrajectory):
            return self._start_end_effector(trajectory)
        if not isinstance(trajectory, Future):
            return self._start_trajectory(trajectory)
        self._pending_plan = trajectory
        command = self._submit(2)
        self._controller.start(self._held_arm(), 1.0)
        self._controller.mode = "planning"
        return command

    def _start_end_effector(self, trajectory: EndEffectorTrajectory) -> int:
        if self._kinematics is None:
            raise RuntimeError("No arm kinematics are bound for end-effector trajectories.")
        if trajectory.episode_id != self._runtime.episode_id:
            raise ValueError("Trajectory belongs to a different episode")
        localization = self._localization.observe().localization
        if localization is None or not localization.valid:
            raise PermissionError("End-effector trajectories require permitted localization.")
        controller, measured = self._controller, self._controller.reader.positions()
        config = controller.config
        wrist_from_tool = self._tool(trajectory.tool)
        lowered = lower(
            trajectory,
            names=self._joints,
            seed=[measured[j] for j in self._joints],
            body=localization.pose.matrix.cpu().numpy(),
            wrist_from_tool=wrist_from_tool,
            inverse=self._kinematics.inverse,
            forward=self._kinematics.forward,
            max_velocity=config.max_velocity,
            start_tolerance=(config.tool_position_tolerance, config.tool_orientation_tolerance),
        )
        forward, joints = self._kinematics.forward, self._joints

        def tool_error(positions: dict[str, float], seconds: float) -> tuple[float, float]:
            tool = forward([positions[j] for j in joints]) @ wrist_from_tool
            return pose_error(tool, lowered.tool_pose(seconds))

        command = self._start_trajectory(lowered.joints)
        controller.tool_error = tool_error
        controller.time_scale *= lowered.time_scale
        return command

    def _start_trajectory(self, trajectory: JointTrajectory, *, pending: bool = False) -> int:
        import math

        controller = self._controller
        if trajectory.episode_id != self._runtime.episode_id:
            raise ValueError("Trajectory belongs to a different episode")
        if not set(trajectory.names) <= set(controller.declaration.joints):
            raise ValueError("Trajectory commands undeclared joints")
        measured = controller.reader.positions()
        for row in trajectory.positions:
            controller.validate(dict(zip(trajectory.names, row, strict=True)))
        if (
            max(abs(measured[n] - trajectory.positions[0, i]) for i, n in enumerate(trajectory.names))
            > controller.config.tracking_tolerance
        ):
            # A jump, not a precision check: a held object's load holds the arm a few hundredths of
            # a radian off any plan computed from it (refused at 0.03 rad at 250 Hz, 2026-10-17).
            raise ValueError("Trajectory start does not match measured joints")
        trajectory, time_scale = trajectory.slowed(controller.config.max_velocity)
        duration = float(trajectory.times[-1])
        ticks = math.ceil(
            (duration + controller.config.timeout_seconds + controller.config.stop_timeout_seconds)
            / self._runtime.physics_dt
        )
        if pending:
            self._runtime.renew(self._name, duration_ticks=ticks + 1)
            command = self._current
        else:
            command = self._submit(ticks + 1)
        controller.start(dict(zip(trajectory.names, trajectory.positions[-1], strict=True)), duration)
        controller.trajectory = trajectory
        controller.mode = "trajectory"
        controller.time_scale = time_scale
        return command

    def open_gripper(self, *, duration: float = 2.0) -> int:
        """Move the gripper to its configured open position [m or rad] over duration [s]."""
        return self.move_joints({self._gripper_joint: self._open_angle}, duration=duration)

    def close_gripper(self, *, duration: float = 2.0) -> int:
        """Close the gripper toward its configured closed position [m or rad] over duration [s].

        Succeeds when the gripper stops: fully closed, or with reason "Gripper stopped on an
        object". The gripper keeps squeezing through later arm commands until the next
        gripper command. Success does not prove that the object is retained.
        """
        return self.move_joints({self._gripper_joint: self._closed_angle}, duration=duration)

    def cancel(self, command_id: int) -> None:
        """Hold measured joints and retain ownership until physical stopping; skills stop issuing commands."""
        skill = self._skill
        if skill is not None and command_id == skill.command_id and not skill.run.done:
            skill.run.cancel()
            self._record_skill()
            if skill.run.command is not None:
                self.cancel(skill.run.command)
            return
        if self.feedback(command_id).status not in {"active", "stopping"}:
            return
        if self._controller.mode == "stopping":
            return
        if self._pending_plan is not None:
            self._pending_plan.cancel()
            self._pending_plan = None
        self._controller.stop("cancelled", "Cancelled by agent")
        ticks = duration_ticks(self._controller.config.stop_timeout_seconds, self._runtime.physics_dt)
        self._runtime.renew(self._name, duration_ticks=ticks + 1)

    def update(self) -> None:
        """Runtime hook: finish the current command, then let an active skill issue its next one."""
        self._update_command()
        if self._skill is not None and not self._skill.run.done:
            self._skill.run.update()
            self._record_skill()

    def _update_command(self) -> None:
        """Release ownership after measured completion or stopping."""
        if self._current is None or self._runtime.controller_state(self._name) == "idle":
            return
        controller = self._controller
        if controller.mode == "planning":
            self._runtime.renew(self._name, duration_ticks=2)
            if self._pending_plan.done():
                result, self._pending_plan = self._pending_plan, None
                try:
                    self._start_trajectory(result.result(), pending=True)
                except Exception as exc:
                    controller.stop("failed", str(exc))
                    ticks = duration_ticks(controller.config.stop_timeout_seconds, self._runtime.physics_dt)
                    self._runtime.renew(self._name, duration_ticks=ticks + 1)
        if controller.ready:
            record = self._records[self._current]
            progress = 1.0 if controller.outcome == "succeeded" else controller.progress
            measured = controller.reader.positions()
            self._outcomes[self._current] = MotionFeedback(
                record.command_id,
                record.episode_id,
                controller.outcome,
                controller.reason,
                controller.error_rad,
                controller.pose_error,
                progress=progress,
                time_scale=controller.time_scale,
                reached_seconds=progress * controller.duration,
                joint_targets=dict(controller.final_reference),
                joint_positions={joint: measured[joint] for joint in controller.declaration.joints},
                gripper_aperture=measured[self._gripper_joint],
                stalled=controller.stalled,
                tool_pose=self._measured_tool_pose(),
            )
            self._records[self._current] = self._outcomes[self._current]
            self._runtime.release(self._name)

    def _end_episode(self) -> None:
        if self._skill is not None and not self._skill.run.done:
            self._skill.run.cancel()
            self._record_skill()
        self._controller.clear_hold()
        if self._pending_plan is not None:
            self._pending_plan.cancel()
            self._pending_plan = None
        super()._end_episode()

    def feedback(self, command_id: int) -> MotionFeedback:
        """Read execution outcome without advancing simulation; skills return SkillFeedback."""
        skill = self._skill
        if skill is not None and command_id == skill.command_id:
            if not skill.run.done:
                return SkillFeedback(
                    command_id,
                    skill.episode_id,
                    "active",
                    skill=skill.name,
                    step=self._skills.step,
                )
            inner = skill.run.command
            if self._outcomes[command_id].status == "cancelled" and inner is not None:
                if self.feedback(inner).status in {"active", "stopping"}:
                    return replace(self._outcomes[command_id], status="stopping")
        if command_id in self._outcomes:
            return self._outcomes[command_id]
        record = super().feedback(command_id)
        status = "stopping" if record.status == "active" and self._controller.mode == "stopping" else record.status
        return MotionFeedback(record.command_id, record.episode_id, status)
