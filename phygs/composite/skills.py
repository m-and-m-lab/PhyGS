# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The composite skills over one session: walking to stances, the arm's moves, picks, places and
drawer strokes, composed from the robot's commands.

The moves are planned moves (``move_to_pose``) to where a straight motion starts, straight
end-effector lines (``execute_trajectory``), the gripper, and the ``look_at`` and ``stow`` skills.
They place the robot's tool frames (``grasp``, ``handle``, ``fixed_jaw``), which share the
gripper's axes: x the approach, z the jaws' closing axis.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import numpy as np

from phygs.composite.geometry import matrix, rotate_z, rotation_about, xyzw
from phygs.composite.tuning import TUNING, Tuning
from phygs.session import (
    AgentFailed,
    ArticulatedPart,
    EndEffectorTrajectory,
    HeldObject,
    PointTarget,
    SkillFailure,
    TrajectoryRefused,
    entry_depth,
    motion_failure,
    unit_approach,
)

# Failures refused before the arm moved: the next option may work.
REFUSED = frozenset({SkillFailure.OUT_OF_REACH, SkillFailure.PLANNING_FAILED})
LOOK_REFUSED = REFUSED | {SkillFailure.OBSTACLE_CHANGED, SkillFailure.PATH_UNOBSERVED}
DOWN = np.array([0.0, 0.0, -1.0])
UP = -DOWN
# The gripper's axes pointing straight down at heading 0: x (the approach) down, z (the jaws' closing
# axis) along the heading.
TOP_DOWN = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])


@dataclass(frozen=True)
class Outcome:
    """What a composite skill did: the option that worked (``option``) and what the robot measured
    after it (``measured``), and how many options were refused before it (``refused``). Only
    :meth:`CompositeSkills.look_at` returns ``succeeded`` False (no view was reachable); the others
    raise :class:`~phygs.session.AgentFailed` when every option failed."""

    skill: str
    succeeded: bool = True
    reason: str = ""
    option: dict[str, Any] = field(default_factory=dict)
    measured: dict[str, Any] = field(default_factory=dict)
    refused: int = 0


class Motion(NamedTuple):
    """One command's result; ``refused``: it failed before the arm moved (try another option)."""

    status: str
    reason: str = ""
    refused: bool = False


class Unreachable(AgentFailed):
    """navigate_to refused a stance before walking: no path to it, or the robot does not fit there."""


def _step(name: str | Callable[..., str]):
    """Run the decorated skill as one step of the agent's plan (``session.step``), named ``name`` or
    ``name(...)`` of the skill's arguments, so a run's timeline shows it beside the robot's commands."""

    def decorate(skill):
        @functools.wraps(skill)
        def run(self, *args, **kwargs):
            with self.session.step(name(*args, **kwargs) if callable(name) else name):
                return skill(self, *args, **kwargs)

        return run

    return decorate


def outward(part: ArticulatedPart) -> np.ndarray:
    """The horizontal unit vector a part opens toward."""
    direction = np.asarray(part.direction, dtype=float)[:2]
    return direction / np.linalg.norm(direction)


class CompositeSkills:
    """Composite skills for the robot of ``session`` (an ``AgentSession``), with ``tuning``.

    Robot facts come from ``session.robot``; the base pose from the observed localization
    (``gt_localization``). :attr:`held` is the object the last pick grasped; the arm's later plans
    carry it while the gripper holds it.
    """

    def __init__(self, session, tuning: Tuning = TUNING) -> None:
        self.session, self.tuning = session, tuning
        self.tools = {name: matrix(pose) for name, pose in session.robot.tools.items()}
        self.held: HeldObject | None = None

    # Robot facts and readings.

    def tool(self, name: str) -> np.ndarray:
        """The robot's tool frame ``name`` in the wrist link (4x4) [m]."""
        if name not in self.tools:
            raise AgentFailed(f"The robot has no {name!r} tool frame (robot.yaml gripper.tools).")
        return self.tools[name]

    def wrist_for(self, name: str, pose: np.ndarray) -> np.ndarray:
        """The world wrist pose (4x4) that puts the tool frame ``name`` at the world ``pose`` (4x4)."""
        return pose @ np.linalg.inv(self.tool(name))

    def gripper_axes(self, wrist: np.ndarray) -> np.ndarray:
        """The gripper's axes (3x3, world: x the approach, z the jaws' closing axis) of a wrist pose."""
        return wrist[:3, :3] @ self.tool("grasp")[:3, :3]

    def base_pose(self) -> np.ndarray:
        """The base's world pose (4x4) from the localization the session observes."""
        localization = self.session.observe().localization
        if localization is None or not localization.valid:
            raise AgentFailed("The composite skills need the base pose (gt_localization).")
        return np.asarray(localization.pose.matrix.tolist())

    def heading(self) -> float:
        """The base's heading [rad]."""
        pose = self.base_pose()
        return math.atan2(pose[1, 0], pose[0, 0])

    def joints(self) -> dict[str, float]:
        """The measured joint positions [m or rad] by name (empty without joint_state)."""
        joints = self.session.observe().joints
        return {} if joints is None else dict(zip(joints.names, joints.positions.tolist(), strict=True))

    def stowed(self) -> bool:
        """Whether every arm joint is within ``stow_tolerance_rad`` of the stow posture."""
        robot, measured = self.session.robot, self.joints()
        if not measured:
            return False
        return max(abs(measured[j] - v) for j, v in zip(robot.arm_joints, robot.stow)) <= self.tuning.stow_tolerance_rad

    def _held(self) -> HeldObject | None:
        """The picked object while the gripper holds it: the arm's plans carry it."""
        return self.held if self.session.manipulation.holding else None

    def _ahead(self, part: ArticulatedPart) -> float:
        """How far the part's handle is ahead of the base centre, into the part [m]."""
        handle = np.asarray(part.handle_center, dtype=float)[:2]
        return float(np.dot(handle - self.base_pose()[:2, 3], -outward(part)))

    # Stances: (x, y, heading) [m, m, rad], from the affordance's geometry and the grip range.

    def open_stance(self, part: ArticulatedPart, attempt: int = 0) -> tuple[float, float, float]:
        """Facing the drawer with its handle at the far end of ``grip_range_m`` less the walk's
        position tolerance ahead of the base centre, so wherever the base stops within that tolerance
        the handle is in reach; each later ``attempt`` one tolerance nearer."""
        out = outward(part)
        ahead = self.tuning.grip_range_m[1] - self.tuning.goal_tolerance[0] * (1 + attempt)
        x, y = np.asarray(part.handle_center, dtype=float)[:2] + out * ahead
        return float(x), float(y), math.atan2(-out[1], -out[0])

    def close_stance(self, part: ArticulatedPart, distance: float) -> tuple[float, float, float]:
        """Facing the drawer so a push of ``distance`` [m] (the hand ``push_back_m`` short of the
        handle) is centred in ``grip_range_m`` ahead of the base centre."""
        out = outward(part)
        near, far = self.tuning.grip_range_m
        ahead = self.tuning.push_back_m + (near + far - distance) / 2
        x, y = np.asarray(part.handle_center, dtype=float)[:2] + out * ahead
        return float(x), float(y), math.atan2(-out[1], -out[0])

    def reach_stances(
        self, point, facing=None, part: ArticulatedPart | None = None, turn_deg: float = 0.0
    ) -> list[tuple[float, float, float]]:
        """Where to stand to reach ``point`` [m]: facing it with the point ``reach_ahead_m`` ahead of
        the base centre. ``facing`` (a horizontal direction) is the direction the robot faces; in an
        open drawer ``part``, square to it; else along the line from where the base stands.
        ``turn_deg`` turns the stance about the point (counterclockwise from above) [deg]. In a
        drawer, when the stance puts the body over it (its handle nearer than ``reach_in_clear_m``), a
        clear stance with the handle that far ahead comes first, on the same line."""
        point = np.asarray(point, dtype=float)[:2]
        if part is not None:
            facing = -outward(part)
        elif facing is None:
            pose = self.base_pose()
            facing = point - pose[:2, 3]
            if np.linalg.norm(facing) < 1e-6:
                facing = pose[:2, 0]
        facing = np.asarray(facing, dtype=float)[:2]
        facing = rotate_z(math.radians(turn_deg))[:2, :2] @ (facing / np.linalg.norm(facing))
        heading = math.atan2(facing[1], facing[0])
        stance = point - facing * self.tuning.reach_ahead_m
        stances = [(float(stance[0]), float(stance[1]), heading)]
        if part is not None:
            handle_ahead = float(np.dot(np.asarray(part.handle_center, dtype=float)[:2] - stance, facing))
            if handle_ahead < self.tuning.reach_in_clear_m:
                clear = stance - facing * (self.tuning.reach_in_clear_m - handle_ahead)
                stances.insert(0, (float(clear[0]), float(clear[1]), heading))
        return stances

    # Walking.

    @_step("walk")
    def walk(self, stance: tuple[float, float, float], settle: bool = True) -> Outcome:
        """Stow the arm if it is out (the base walks with the arm folded), navigate_to ``stance``
        (x, y, heading) [m, m, rad] within ``goal_tolerance``, then stand while the legs settle.

        Each attempt has its own time limit (``walk_timeout_s``); a timed-out attempt is issued again,
        up to ``walk_tries`` attempts (near a goal the follower can stand still just outside the
        tolerance, 2026-10-15). Any other failure ends the skill.
        """
        if not self.stowed():
            self.stow()
        position_tolerance, heading_tolerance = self.tuning.goal_tolerance
        for attempt in range(1, self.tuning.walk_tries + 1):
            feedback = self.session.wait(
                self.session.navigation.navigate_to(
                    *stance,
                    position_tolerance=position_tolerance,
                    heading_tolerance=heading_tolerance,
                    timeout=self.tuning.walk_timeout_s,
                )
            )
            if feedback.status == "succeeded":
                break
            if feedback.status != "timed_out":
                error = Unreachable if feedback.status == "unreachable" else AgentFailed
                raise error(f"navigate_to {stance}: {feedback.status}: {feedback.reason}")
            print(f"composite: navigate_to {stance} timed out ({feedback.reason}); trying again", flush=True)
        else:
            raise AgentFailed(f"navigate_to {stance}: {feedback.status}: {feedback.reason}")
        if settle:
            self.session.sleep(self.tuning.settle_seconds)
        pose = self.base_pose()
        base = [float(pose[0, 3]), float(pose[1, 3]), math.atan2(pose[1, 0], pose[0, 0])]
        return Outcome("walk", option=dict(stance=list(stance), attempt=attempt), measured=dict(base=base))

    @_step("go_to_reach")
    def go_to_reach(self, point, facing=None, part: ArticulatedPart | None = None) -> Outcome:
        """Walk to where the arm reaches ``point`` [m] (:meth:`reach_stances`), through the clear
        stance when there is one. While navigate_to refuses the stance as unreachable, it turns about
        the point by the next of ``reach_turns_deg`` (square to a drawer ``part`` it does not turn)."""
        turns = self.tuning.reach_turns_deg if part is None else (0.0,)
        for refused, turn in enumerate(turns):
            stances = self.reach_stances(point, facing, part, turn)
            try:
                for stance in stances[:-1]:
                    self.walk(stance, settle=False)
                walked = self.walk(stances[-1])
                break
            except Unreachable as error:
                if refused + 1 == len(turns):
                    raise
                print(f"composite: {error}; turning the stance", flush=True)
        option = dict(stances=[list(s) for s in stances], turn_deg=turn)
        return Outcome("go_to_reach", option=option, measured=walked.measured, refused=refused)

    # The arm.

    def run(self, command) -> Motion:
        """Issue ``command()`` and wait; a trajectory refused before moving is a refused motion."""
        try:
            issued = command()
        except TrajectoryRefused as error:
            return Motion("refused", str(error), True)
        feedback = self.session.wait(issued)
        if feedback.status == "succeeded":
            return Motion("succeeded", feedback.reason)
        unmoved = getattr(feedback, "progress", 0.0) == 0.0
        refused = unmoved and motion_failure(feedback.reason or feedback.status).failure in REFUSED
        return Motion(feedback.status, feedback.reason, refused)

    def gripper(self, close: bool) -> None:
        """Close or open the gripper and wait."""
        manipulation = self.session.manipulation
        feedback = self.session.wait(manipulation.close_gripper() if close else manipulation.open_gripper())
        if feedback.status != "succeeded":
            raise AgentFailed(f"gripper: {feedback.reason}")

    @_step("stow")
    def stow(self) -> Outcome:
        """Stow the arm, carrying the held object so the planned part avoids obstacles with it. When
        no stow plan starts from where the arm is, lift the hand ``unstick_m`` and stow once more."""
        manipulation, option = self.session.manipulation, "stow"
        feedback = self.session.wait(manipulation.stow(held=self._held()))
        if feedback.status != "succeeded" and getattr(feedback, "failure", None) in REFUSED:
            if self.move_straight(UP, self.tuning.unstick_m).status == "succeeded":
                option = "lift, then stow"
                feedback = self.session.wait(manipulation.stow(held=self._held()))
        if feedback.status != "succeeded":
            raise AgentFailed(f"stow: {feedback.reason}")
        return Outcome("stow", option=dict(how=option), measured=dict(holding=manipulation.holding))

    def leave_stow(self) -> None:
        """From the stow, move straight to the pre-stow posture: planners reject the stow as a start."""
        if not self.stowed():
            return
        robot = self.session.robot
        prestow = dict(zip(robot.arm_joints, robot.prestow))
        feedback = self.session.wait(self.session.manipulation.move_joints(prestow, duration=robot.move_seconds))
        if feedback.status != "succeeded":
            raise AgentFailed(f"leaving the stow: {feedback.reason}")

    @_step(lambda point, views="target": f"look_at({views if isinstance(views, str) else 'views'})")
    def look_at(self, point, views: str | Sequence[tuple[float, float]] = "target") -> Outcome:
        """Aim the look camera at ``point`` [m] from each view (distance [m], elevation [deg]) in turn:
        ``look_views[views]``, or the views given. Not succeeded when every view was refused; a
        failure while moving ends the skill."""
        listed = self.tuning.look_views[views] if isinstance(views, str) else views
        target = PointTarget(tuple(map(float, point)))
        last = "no view"
        for refused, (distance_m, elevation_deg) in enumerate(listed):
            feedback = self.session.wait(
                self.session.manipulation.look_at(
                    target, distance_m=distance_m, elevation_deg=elevation_deg, held=self._held()
                )
            )
            if feedback.status == "succeeded":
                view = dict(distance_m=distance_m, elevation_deg=elevation_deg)
                return Outcome("look_at", option=view, refused=refused)
            if feedback.failure not in LOOK_REFUSED:
                raise AgentFailed(f"look_at: {feedback.reason}")
            last = feedback.reason
        print(f"composite: no view of {tuple(np.round(target.position, 3))} was reachable", flush=True)
        return Outcome("look_at", succeeded=False, reason=f"no view was reachable ({last})", refused=len(listed))

    def approach_straight(self, end: np.ndarray, axis, distance: float, held: HeldObject | None = None) -> Motion:
        """A planned move to ``distance`` [m] back from the wrist pose ``end`` (4x4, world) along the
        world unit ``axis``, then straight to ``end``; the straight part may end in contact."""
        self.leave_stow()
        start = end.copy()
        start[:3, 3] = end[:3, 3] - np.asarray(axis, dtype=float) * distance
        manipulation = self.session.manipulation
        planned = self.run(lambda: manipulation.move_to_pose(xyzw(start), held=held, tool="wrist"))
        if planned.status != "succeeded":
            return planned
        return self.move_straight_to(end[:3, 3])

    def move_straight(self, axis, distance: float, tool: str = "wrist") -> Motion:
        """A straight end-effector line of ``distance`` [m] along a world unit ``axis`` from where the
        tool is, keeping its orientation."""
        here = np.asarray(self.session.manipulation.tool_pose(tool), dtype=float)
        return self.move_straight_to(here[:3] + np.asarray(axis, dtype=float) * float(distance), tool)

    def move_straight_to(self, position, tool: str = "wrist") -> Motion:
        """A straight end-effector line to a world position [m] from where the tool is, keeping its
        orientation, at ``line_speed_m_s``."""
        manipulation = self.session.manipulation
        here = np.asarray(manipulation.tool_pose(tool), dtype=float)
        there = np.concatenate((np.asarray(position, dtype=float), here[3:]))
        seconds = max(float(np.linalg.norm(there[:3] - here[:3])) / self.tuning.line_speed_m_s, 0.1)
        line = EndEffectorTrajectory([0.0, seconds], [here, there], manipulation.episode_id, tool=tool)
        return self.run(lambda: manipulation.execute_trajectory(line))

    @_step(lambda center, size, name=None, **options: f"pick({name})" if name else "pick")
    def pick(self, center, size, name: str | None = None, approaches=None, look: bool = True) -> Outcome:
        """Grasp the object box (``center``, ``size`` [m], world-aligned) from where the base stands:
        look at it (``look``), then for each pick approach and each grasp yaw in turn, a planned move
        to the pre-grasp, straight in, close, straight back out. An option refused before moving tries
        the next one. :attr:`held` is then the object; ``name`` the scene object's, if known.

        ``approaches``: world unit vectors (None: straight down), else ``pick_approaches`` (robot frame).
        """
        tuning, manipulation = self.tuning, self.session.manipulation
        if look:
            self.look_at(center, "target")
        if approaches is None:
            turn = rotate_z(self.heading())
            approaches = [a if a is None else tuple(turn @ np.asarray(a)) for a in tuning.pick_approaches]
        center, size = np.asarray(center, dtype=float), tuple(map(float, size))
        refused, last = 0, "no approach"
        for approach in approaches:
            direction = DOWN if approach is None else unit_approach(tuple(map(float, approach)))
            entry = center - direction * entry_depth(size, direction)
            self.gripper(close=False)
            for yaw in tuning.grasp_yaws_deg:
                wrist = self.grasp_wrist_pose(entry, direction, math.radians(yaw))
                reached = self.approach_straight(wrist, direction, tuning.approach_m)
                if reached.status == "succeeded":
                    break
                if not reached.refused:
                    raise AgentFailed(f"pick approach: {reached.reason}")
                refused, last = refused + 1, reached.reason
            else:
                continue
            self.gripper(close=True)
            gripped = manipulation.holding
            held = self.held_object(name, center, size)  # The wrist measured at the grasp.
            lifted = self.move_straight(-direction, tuning.approach_m)
            if lifted.status != "succeeded":
                raise AgentFailed(f"pick lift: {lifted.reason}")
            if not manipulation.holding:
                raise AgentFailed(
                    "pick: the object slipped while lifting." if gripped else "pick: the gripper closed on nothing."
                )
            self.session.sleep(tuning.hold_seconds)
            self.held = held
            option = dict(approach=None if approach is None else [round(float(v), 4) for v in direction], yaw_deg=yaw)
            gripper = self.joints().get(self.session.robot.gripper_joints[0])
            measured = dict(holding=manipulation.holding, gripper=gripper)
            return Outcome("pick", option=option, measured=measured, refused=refused)
        raise AgentFailed(f"pick: every approach was refused ({refused} options); last: {last}")

    @_step("place")
    def place(self, point) -> Outcome:
        """Set the held object down with its bottom center at a world point [m]: a planned move above
        it, straight down along the gripper axis (the wrist keeps its orientation, so the object keeps
        its yaw), release ``place_clearance_m`` above the surface, straight back up, close the
        gripper. The gripper must not hold the object afterwards."""
        held, manipulation = self.held, self.session.manipulation
        if held is None:
            raise AgentFailed("place: no picked object")
        wrist = matrix(manipulation.tool_pose("wrist"))
        bottom = (wrist @ held.wrist_from_object @ [0.0, 0.0, -held.size[2] / 2, 1.0])[:3]
        end = wrist.copy()
        end[:3, 3] = wrist[:3, 3] + (
            np.asarray(point, dtype=float) + [0.0, 0.0, self.tuning.place_clearance_m] - bottom
        )
        axis, distance = self.gripper_axes(end)[:, 0], self.tuning.approach_m
        reached = self.approach_straight(end, axis, distance, held=held)
        if reached.status != "succeeded":
            raise AgentFailed(f"place: {reached.reason}")
        there = matrix(manipulation.tool_pose("wrist")) @ held.wrist_from_object @ [0.0, 0.0, -held.size[2] / 2, 1.0]
        self.gripper(close=False)
        out = self.move_straight(-axis, distance)
        if out.status != "succeeded":
            raise AgentFailed(f"place: moving out: {out.reason}")
        self.gripper(close=True)  # The planners model a closed gripper.
        if manipulation.holding:
            raise AgentFailed("place: the gripper still holds the object")
        self.held = None
        return Outcome("place", measured=dict(released_at=[round(float(v), 4) for v in there[:3]]))

    @_step("place_upright")
    def place_upright(self, point) -> Outcome:
        """Place an object picked straight down so it stands upright: look at the point (this also
        leaves the stow), turn the gripper top-down with the wrist ``place_wrist_height_m`` above it
        (after a stow the object lies sideways, and place keeps the wrist's orientation), then
        :meth:`place`."""
        if not self.look_at(point, "place").succeeded:
            raise AgentFailed("place: no view of the place spot was reachable")
        upright = rotate_z(self.heading()) @ TOP_DOWN @ self.tool("grasp")[:3, :3].T  # The wrist's rotation.
        above = np.eye(4)
        above[:3, :3] = upright
        above[:3, 3] = np.asarray(point, dtype=float) + [0.0, 0.0, self.tuning.place_wrist_height_m]
        moved = self.session.wait(self.session.manipulation.move_to_pose(xyzw(above), held=self.held, tool="wrist"))
        if moved.status != "succeeded":
            raise AgentFailed(f"move_to_pose above the place spot: {moved.reason}")
        return self.place(point)

    def place_spot(self, part: ArticulatedPart, size) -> np.ndarray:
        """Where to set an object of ``size`` [m] down in the open drawer ``part`` [m]: on the exposed
        floor, ``place_carcass_clearance_m`` from the carcass front (the opened jaw reaches toward it),
        but no closer than ``place_front_margin_m`` to the drawer front. The exposed floor is about the
        opening long, centred on the part's ``floor_center``."""
        if part.floor_center is None:
            raise AgentFailed("place_spot: the drawer's floor is unknown")
        out = np.array([*outward(part), 0.0])
        half = float(np.dot(np.abs(out), size)) / 2  # The object's half depth along the opening.
        wanted = self.tuning.place_carcass_clearance_m - part.opening / 2
        room = part.opening / 2 - half - self.tuning.place_front_margin_m
        return np.asarray(part.floor_center, dtype=float) + out * float(np.clip(wanted, 0.0, max(room, 0.0)))

    def held_object(self, name: str | None, center: np.ndarray, size: tuple[float, float, float]) -> HeldObject:
        """The object box [m] at ``center``, world-aligned, in the measured wrist frame, moved along the
        jaws' closing axis to where they hold it: resting on the fixed jaw (``fixed_jaw``; the closing
        jaw pushes it there), or centred on the grasp point when both jaws close."""
        wrist = matrix(self.session.manipulation.tool_pose("wrist"))
        across = self.gripper_axes(wrist)[:, 2]
        if "fixed_jaw" in self.tools:
            width = float(np.dot(np.asarray(size), np.abs(across)))
            rest = float(np.dot((wrist @ self.tools["fixed_jaw"])[:3, 3] - center, across)) + width / 2
        else:
            rest = float(np.dot((wrist @ self.tool("grasp"))[:3, 3] - center, across))
        world_from_object = np.eye(4)
        world_from_object[:3, 3] = center + across * rest
        return HeldObject(tuple(size), np.linalg.inv(wrist) @ world_from_object, name=name)

    def grasp_wrist_pose(self, point: np.ndarray, approach: np.ndarray, roll: float) -> np.ndarray:
        """World wrist pose (4x4) that puts the robot's ``grasp`` tool frame at ``point``, its approach
        along ``approach``: the top-down gripper turned by the shortest rotation from down to the
        approach, then by ``roll`` [rad] about it (a yaw about world z for a top-down grasp)."""
        axis = np.cross(DOWN, approach)
        angle = math.acos(float(np.clip(np.dot(DOWN, approach), -1.0, 1.0)))
        tilt = rotation_about(axis / np.linalg.norm(axis), angle) if angle > 1e-9 else np.eye(3)
        pose = np.eye(4)
        pose[:3, :3] = rotation_about(-approach, roll) @ tilt @ TOP_DOWN
        pose[:3, 3] = point
        return self.wrist_for("grasp", pose)

    # Drawers.

    @_step("open_drawer")
    def open_drawer(self, part: ArticulatedPart, distance: float | None = None) -> Outcome:
        """Stand at the open stance, look at the handle, grasp it and pull: ``distance`` [m], else as
        far as the part's travel allows (less the limit margin), and never past where the handle is at
        the near end of ``grip_range_m`` from where the base stopped. When every way to take the
        handle is refused, stand again one walk tolerance nearer, up to ``stance_tries`` stances."""
        tuning = self.tuning
        travel = part.limits[1] - part.opening - tuning.limit_margin_m
        wanted = travel if distance is None else min(distance, travel)

        def stroke() -> float:
            return min(wanted, self._ahead(part) - tuning.grip_range_m[0])

        return self._stroke_from_stances(part, "open", lambda attempt: self.open_stance(part, attempt), stroke)

    @_step("close_drawer")
    def close_drawer(self, part: ArticulatedPart, distance: float | None = None) -> Outcome:
        """Stand at the close stance and push the drawer with the closed gripper: ``distance`` [m], else
        to its closed end (less the limit margin). When every way to push is refused, stand there
        again, up to ``stance_tries`` stances."""
        length = part.opening - part.limits[0] - self.tuning.limit_margin_m if distance is None else distance
        return self._stroke_from_stances(part, "close", lambda attempt: self.close_stance(part, length), lambda: length)

    def _stroke_from_stances(self, part: ArticulatedPart, kind: str, stance, planned) -> Outcome:
        """Walk to ``stance(attempt)``, look at the handle and stroke ``planned()`` [m] (asked from where
        the base stopped), until a stroke is not refused before moving."""
        refused = 0
        for attempt in range(self.tuning.stance_tries):
            self.walk(stance(attempt))
            self.look_at(part.handle_center, "handle")
            distance = planned()
            if distance <= 0:
                raise AgentFailed(f"{kind}_drawer: nothing to move from here ({distance:.3f} m).")
            stroked = self._stroke(part, kind, distance)
            if stroked is not None:
                break
            refused += len(self.tuning.handle_pitch_deg)
            print(f"composite: no {kind} stroke from stance {attempt + 1}; standing again", flush=True)
        else:
            raise AgentFailed(f"{kind}_drawer: every handle pitch was refused from {self.tuning.stance_tries} stances.")
        moved, option, measured = stroked
        if moved.status != "succeeded":
            raise AgentFailed(f"{kind}_drawer: the move along the slide: {moved.reason}")
        self.session.sleep(self.tuning.settle_seconds)
        option = dict(stance=attempt, stroke_m=round(float(distance), 4), **option)
        return Outcome(f"{kind}_drawer", option=option, measured=measured, refused=refused)

    def _stroke(self, part: ArticulatedPart, kind: str, distance: float):
        """One stroke: ``open`` grasps the handle and pulls; ``close`` pushes with the closed gripper.

        Each handle pitch is tried in turn while the approach or the slide move is refused before
        moving. Returns (the slide move, the option, the measurements), or None when every pitch was
        refused. Measured: how far the hand moved along the slide [m] and, pulling, whether the
        gripper still held the handle at the end.
        """
        manipulation, push = self.session.manipulation, kind == "close"
        motion = (-1.0 if push else 1.0) * np.asarray(part.direction, dtype=float)
        for pitch in self.tuning.handle_pitch_deg:
            self.gripper(close=push)
            grasp = self._approach_handle(part, pitch, self.tuning.push_back_m if push else 0.0)
            if grasp is None:
                continue
            axes, option = grasp
            if not push:
                self.gripper(close=True)
                if not manipulation.holding:
                    raise AgentFailed(f"{kind}_drawer: the gripper closed on nothing at the handle.")
            start = np.asarray(manipulation.tool_pose("wrist"), dtype=float)[:3]
            moved = self.move_straight(motion, distance)
            end = np.asarray(manipulation.tool_pose("wrist"), dtype=float)[:3]
            measured = dict(moved_m=round(float(np.dot(end - start, motion)), 4))
            if not push:
                measured["held_handle"] = manipulation.holding
                self.gripper(close=False)
            self._release(axes, self.tuning.push_release if push else self.tuning.handle_release)
            if moved.status == "succeeded" or not moved.refused:
                return moved, dict(pitch_deg=pitch, **option), measured  # Moved, or failed while moving.
        return None

    def _release(self, axes: np.ndarray, moves) -> None:
        """The release moves in order, each at its distance or else half of it; when a move is refused
        both times the rest are skipped (each move starts where the one before ended)."""
        for move, length in moves:
            axis = _release_axis(axes, move)
            for distance in (float(length), float(length) / 2):
                outcome = self.move_straight(axis, distance)
                if outcome.status == "succeeded" or not outcome.refused:
                    break
            if outcome.status != "succeeded":
                print(f"composite: release move {move} refused ({outcome.reason}); skipping the rest", flush=True)
                return

    def _approach_handle(self, part: ArticulatedPart, pitch_deg: float, back_m: float):
        """Move onto the handle at a pitch: each approach distance, each bar direction in turn. Returns
        the gripper's axes (3x3, world) of the grasp reached and the option, or None when all were
        refused."""
        for distance in self.tuning.handle_approach_m:
            for side in (1.0, -1.0):
                grasp = self.handle_grasp_pose(part, side, pitch_deg)
                grasp[:3, 3] -= grasp[:3, 0] * back_m
                reached = self.approach_straight(self.wrist_for("handle", grasp), grasp[:3, 0], float(distance))
                if reached.status == "succeeded":
                    return grasp[:3, :3], dict(approach_m=distance, side=side)
                if not reached.refused:
                    raise AgentFailed(f"handle approach: {reached.reason}")
        return None

    def handle_grasp_pose(self, part: ArticulatedPart, side: float, pitch_deg: float) -> np.ndarray:
        """World pose (4x4) of the robot's ``handle`` tool frame around a part's handle bar: the bar
        along the gripper's y (``side`` picks its direction), the approach into the part, pitched down
        by ``pitch_deg`` [deg] about the bar; the bar's axis on the frame, or, with a fixed jaw, the
        bar resting on it (its axis a radius above the frame along the closing axis)."""
        into = -np.asarray(part.direction, dtype=float)
        bar = np.asarray(part.handle_axis, dtype=float) * side
        y = bar - into * np.dot(bar, into)
        y /= np.linalg.norm(y)
        level_z = np.cross(into, y)
        pitch = math.radians(pitch_deg) * (1.0 if level_z[2] >= 0 else -1.0)  # Down, whichever way y points.
        approach = rotation_about(y, pitch) @ into
        z = np.cross(approach, y)
        rest = part.handle_radius if "fixed_jaw" in self.tools else 0.0
        pose = np.eye(4)
        pose[:3, :3] = np.column_stack((approach, y, z))
        pose[:3, 3] = np.asarray(part.handle_center, dtype=float) - z * rest
        return pose


def _release_axis(rotation: np.ndarray, move: str) -> np.ndarray:
    """World axis of a release move for the gripper's axes at ``rotation`` (straight-line moves keep
    them): ``wrist_down`` against the closing axis, ``gripper_back`` against the approach."""
    if move == "wrist_down":
        return -rotation[:, 2]
    if move == "gripper_back":
        return -rotation[:, 0]
    if move == "up":
        return UP.copy()
    if move == "forward":
        level = np.array([rotation[0, 0], rotation[1, 0], 0.0])
        return level / np.linalg.norm(level)
    raise ValueError(f"Unknown release move {move!r}")
