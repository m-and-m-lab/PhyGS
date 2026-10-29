# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Arm motions composed from the robot's commands: planned moves (``move_to_pose``) to where a
straight motion starts, straight end-effector lines (``execute_trajectory``), the gripper, and the
``look_at`` and ``stow`` skills; picks and places built from them."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from phygs_robot_tests.routines import ARM, PARAMS, TOOLS, base
from phygs_robot_tests.routines.geometry import matrix, rotate_z, rotation_about, xyzw

from phygs.session import (
    AgentFailed,
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
# The gripper's axes pointing straight down at heading 0: x (the approach) down, z (the jaws' closing
# axis) along the heading.
TOP_DOWN = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])


def tool(name: str) -> np.ndarray:
    """The robot's tool frame ``name`` in the wrist link (4x4) [m]."""
    if name not in TOOLS:
        raise AgentFailed(f"The robot has no {name!r} tool frame (robot.yaml gripper.tools).")
    return TOOLS[name]


def wrist_for(name: str, pose: np.ndarray) -> np.ndarray:
    """The world wrist pose (4x4) that puts the tool frame ``name`` at the world ``pose`` (4x4)."""
    return pose @ np.linalg.inv(tool(name))


def gripper_axes(wrist: np.ndarray) -> np.ndarray:
    """The gripper's axes (3x3, world: x the approach, z the jaws' closing axis) of a wrist pose."""
    return wrist[:3, :3] @ tool("grasp")[:3, :3]


@dataclass(frozen=True)
class Outcome:
    """A composed motion's result; ``refused``: it failed before the arm moved (try another option)."""

    status: str
    reason: str = ""
    refused: bool = False


def run(session, command) -> Outcome:
    """Issue ``command()`` and wait; a trajectory refused before moving is a refused outcome."""
    try:
        issued = command()
    except TrajectoryRefused as error:
        return Outcome("refused", str(error), True)
    feedback = session.wait(issued)
    if feedback.status == "succeeded":
        return Outcome("succeeded", feedback.reason)
    unmoved = getattr(feedback, "progress", 0.0) == 0.0
    refused = unmoved and motion_failure(feedback.reason or feedback.status).failure in REFUSED
    return Outcome(feedback.status, feedback.reason, refused)


def gripper(session, command) -> None:
    """Open or close the gripper (``command``: the client's method) and wait."""
    feedback = session.wait(command())
    if feedback.status != "succeeded":
        raise AgentFailed(f"gripper: {feedback.reason}")


def stow(session, held: HeldObject | None = None) -> None:
    """Stow the arm; ``held``: the object in the gripper, so the planned part avoids obstacles with it."""
    feedback = session.wait(session.manipulation.stow(held=held))
    if feedback.status != "succeeded":
        raise AgentFailed(f"stow: {feedback.reason}")


def leave_stow(session) -> None:
    """From the stow, move straight to the pre-stow posture: planners reject the stow as a start."""
    joints = session.observe().joints
    measured = dict(zip(joints.names, joints.positions.tolist(), strict=True))
    if max(abs(measured[j] - v) for j, v in zip(ARM["joints"], ARM["stow"])) > PARAMS["stow_tolerance_rad"]:
        return
    prestow = dict(zip(ARM["joints"], ARM["prestow"]))
    feedback = session.wait(session.manipulation.move_joints(prestow, duration=ARM["move_seconds"]))
    if feedback.status != "succeeded":
        raise AgentFailed(f"leaving the stow: {feedback.reason}")


def look_at(session, point, views: str, held: HeldObject | None = None) -> bool:
    """Try the ``look_views[views]`` in order; False when every view was refused. A failure while
    moving ends the plan."""
    for distance_m, elevation_deg in PARAMS["look_views"][views]:
        target = PointTarget(tuple(map(float, point)))
        feedback = session.wait(
            session.manipulation.look_at(target, distance_m=distance_m, elevation_deg=elevation_deg, held=held)
        )
        if feedback.status == "succeeded":
            return True
        if feedback.failure not in LOOK_REFUSED:
            raise AgentFailed(f"look_at {views}: {feedback.reason}")
    print(f"routines: no {views} view was reachable", flush=True)
    return False


def approach_straight(session, end: np.ndarray, axis, distance: float, held: HeldObject | None = None) -> Outcome:
    """A planned move to ``distance`` [m] back from the wrist pose ``end`` (4x4, world) along the
    world unit ``axis``, then straight to ``end``; the straight part may end in contact."""
    leave_stow(session)
    start = end.copy()
    start[:3, 3] = end[:3, 3] - np.asarray(axis, dtype=float) * distance
    planned = run(session, lambda: session.manipulation.move_to_pose(xyzw(start), held=held, tool="wrist"))
    if planned.status != "succeeded":
        return planned
    return move_straight_to(session, end[:3, 3])


def move_straight(session, axis, distance: float, tool: str = "wrist") -> Outcome:
    """A straight end-effector line of ``distance`` [m] along a world unit ``axis`` from where the
    tool is, keeping its orientation."""
    here = np.asarray(session.manipulation.tool_pose(tool), dtype=float)
    return move_straight_to(session, here[:3] + np.asarray(axis, dtype=float) * float(distance), tool)


def move_straight_to(session, position, tool: str = "wrist") -> Outcome:
    """A straight end-effector line to a world position [m] from where the tool is, keeping its
    orientation, at ``line_speed_m_s``."""
    here = np.asarray(session.manipulation.tool_pose(tool), dtype=float)
    there = np.concatenate((np.asarray(position, dtype=float), here[3:]))
    seconds = max(float(np.linalg.norm(there[:3] - here[:3])) / PARAMS["line_speed_m_s"], 0.1)
    line = EndEffectorTrajectory([0.0, seconds], [here, there], session.manipulation.episode_id, tool=tool)
    return run(session, lambda: session.manipulation.execute_trajectory(line))


def pick(session, target, approaches=None, look: bool = True) -> HeldObject:
    """Look at the target (a ``TargetHint``), then grasp it: for each pick approach and each grasp
    yaw in turn, a planned move to the pre-grasp, straight in, close, straight back out. An option
    refused before moving tries the next one. Returns the held object.

    ``approaches``: world unit vectors (None: straight down), else ``pick_approaches`` (robot frame).
    """
    if look:
        look_at(session, target.center, "target")
    if approaches is None:
        turn = rotate_z(base.heading(session))
        approaches = [a if a is None else tuple(turn @ np.asarray(a)) for a in PARAMS["pick_approaches"]]
    center, size = np.asarray(target.center, dtype=float), tuple(map(float, target.size))
    distance = PARAMS["approach_m"]
    last = "no approach"
    for approach in approaches:
        direction = DOWN if approach is None else unit_approach(tuple(map(float, approach)))
        entry = center - direction * entry_depth(size, direction)
        gripper(session, session.manipulation.open_gripper)
        for yaw in PARAMS["grasp_yaws_deg"]:
            reached = approach_straight(
                session, grasp_wrist_pose(entry, direction, math.radians(yaw)), direction, distance
            )
            if reached.status == "succeeded":
                break
            if not reached.refused:
                raise AgentFailed(f"pick approach: {reached.reason}")
            last = reached.reason
        else:
            continue
        gripper(session, session.manipulation.close_gripper)
        gripped = session.manipulation.holding
        held = held_object(session, target.name, center, size)  # The wrist measured at the grasp.
        lifted = move_straight(session, -direction, distance)
        if lifted.status != "succeeded":
            raise AgentFailed(f"pick lift: {lifted.reason}")
        if session.manipulation.holding:
            session.sleep(PARAMS["hold_seconds"])
            return held
        raise AgentFailed(
            "pick: the object slipped while lifting." if gripped else "pick: the gripper closed on nothing."
        )
    raise AgentFailed(f"pick: every approach was refused; last: {last}")


def place(session, point, held: HeldObject) -> None:
    """Set the held object down with its bottom center at a world point [m]: a planned move above
    it, straight down along the gripper axis (the wrist keeps its orientation, so the object keeps
    its yaw), release ``place_clearance_m`` above the surface, straight back up, close the gripper."""
    wrist = matrix(session.manipulation.tool_pose("wrist"))
    bottom = (wrist @ held.wrist_from_object @ [0.0, 0.0, -held.size[2] / 2, 1.0])[:3]
    end = wrist.copy()
    end[:3, 3] = wrist[:3, 3] + (np.asarray(point, dtype=float) + [0.0, 0.0, PARAMS["place_clearance_m"]] - bottom)
    axis, distance = gripper_axes(end)[:, 0], PARAMS["approach_m"]
    reached = approach_straight(session, end, axis, distance, held=held)
    if reached.status != "succeeded":
        raise AgentFailed(f"place: {reached.reason}")
    gripper(session, session.manipulation.open_gripper)
    out = move_straight(session, -axis, distance)
    if out.status != "succeeded":
        raise AgentFailed(f"place: moving out: {out.reason}")
    gripper(session, session.manipulation.close_gripper)  # The planners model a closed gripper.


def place_upright(session, point, held: HeldObject) -> None:
    """Place an object picked straight down so it stands upright: look at the point (this also
    leaves the stow), turn the gripper top-down with the wrist ``place_wrist_height_m`` above it
    (after a stow the object lies sideways, and place keeps the wrist's orientation), then
    :func:`place`."""
    if not look_at(session, point, "place", held=held):
        raise AgentFailed("place: no view of the place spot was reachable")
    upright = rotate_z(base.heading(session)) @ TOP_DOWN @ tool("grasp")[:3, :3].T  # The wrist's rotation.
    above = np.eye(4)
    above[:3, :3], above[:3, 3] = upright, np.asarray(point, dtype=float) + [0.0, 0.0, PARAMS["place_wrist_height_m"]]
    moved = session.wait(session.manipulation.move_to_pose(xyzw(above), held=held, tool="wrist"))
    if moved.status != "succeeded":
        raise AgentFailed(f"move_to_pose above the place spot: {moved.reason}")
    place(session, point, held)
    if session.manipulation.holding:
        raise AgentFailed("place: the gripper still holds the object")


def held_object(session, name: str, center: np.ndarray, size: tuple[float, float, float]) -> HeldObject:
    """The object box [m] at ``center``, world-aligned, in the measured wrist frame, moved along the
    jaws' closing axis to where they hold it: resting on the fixed jaw (``fixed_jaw``; the closing
    jaw pushes it there), or centred on the grasp point when both jaws close."""
    wrist = matrix(session.manipulation.tool_pose("wrist"))
    across = gripper_axes(wrist)[:, 2]
    if "fixed_jaw" in TOOLS:
        width = float(np.dot(np.asarray(size), np.abs(across)))
        rest = float(np.dot((wrist @ TOOLS["fixed_jaw"])[:3, 3] - center, across)) + width / 2
    else:
        rest = float(np.dot((wrist @ tool("grasp"))[:3, 3] - center, across))
    world_from_object = np.eye(4)
    world_from_object[:3, 3] = center + across * rest
    return HeldObject(tuple(size), np.linalg.inv(wrist) @ world_from_object, name=name)


def grasp_wrist_pose(point: np.ndarray, approach: np.ndarray, roll: float) -> np.ndarray:
    """World wrist pose (4x4) that puts the robot's ``grasp`` tool frame at ``point``, its approach
    along ``approach``: the top-down gripper turned by the shortest rotation from down to the
    approach, then by ``roll`` [rad] about it (a yaw about world z for a top-down grasp)."""
    axis = np.cross(DOWN, approach)
    angle = math.acos(float(np.clip(np.dot(DOWN, approach), -1.0, 1.0)))
    tilt = rotation_about(axis / np.linalg.norm(axis), angle) if angle > 1e-9 else np.eye(3)
    pose = np.eye(4)
    pose[:3, :3] = rotation_about(-approach, roll) @ tilt @ TOP_DOWN
    pose[:3, 3] = point
    return wrist_for("grasp", pose)
