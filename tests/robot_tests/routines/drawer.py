# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Drawer strokes: grasp the handle and pull, or push with the closed gripper, then let go."""

from __future__ import annotations

import math

import numpy as np
from phygs_robot_tests.routines import PARAMS, TOOLS, arm, base
from phygs_robot_tests.routines.geometry import rotation_about

from phygs.session import AgentFailed


def place_spot(session, name: str, size: tuple[float, float, float]) -> np.ndarray:
    """Where to set an object of ``size`` [m] down in the open drawer ``name`` [m]: on the exposed
    floor, ``place_carcass_clearance_m`` from the carcass front (the opened jaw reaches toward it),
    but no closer than ``place_front_margin_m`` to the drawer front. The exposed floor is about the
    opening long, centred on the part's ``floor_center``."""
    part = session.hints.part(name)
    outward = np.array([*base.outward(part), 0.0])
    half = float(np.dot(np.abs(outward), size)) / 2  # The object's half depth along the opening.
    wanted = PARAMS["place_carcass_clearance_m"] - part.opening / 2
    room = part.opening / 2 - half - PARAMS["place_front_margin_m"]
    return np.asarray(part.floor_center, dtype=float) + outward * float(np.clip(wanted, 0.0, max(room, 0.0)))


def open_drawer(session, name: str) -> float:
    """Pull the drawer by its handle from where the robot stands (:func:`pull_distance`); walk to
    the open stance again when every pitch is refused. Returns the planned pull [m]."""
    return _stroke_from_stance(session, name, "open", lambda: pull_distance(session, name))


def close_drawer(session, name: str) -> None:
    """Push the drawer shut (to its closed end, less the limit margin) with the closed gripper,
    from the ``close`` stance the robot stands at."""
    drawer = session.hints.part(name)
    distance = drawer.opening - drawer.limits[0] - PARAMS["limit_margin_m"]
    _stroke_from_stance(session, name, "close", lambda: distance)


def pull_distance(session, drawer: str) -> float:
    """How far to pull a drawer from where the base stands now [m]: until its handle is at the
    near end of ``grip_range_m`` ahead of the base centre (the hand then ends in front of the
    body), and no further than its travel allows (less the limit margin)."""
    part = session.hints.part(drawer)
    ahead = float(
        np.dot(np.asarray(part.handle_center, dtype=float)[:2] - base.pose(session)[:2, 3], -base.outward(part))
    )
    return min(ahead - PARAMS["grip_range_m"][0], part.limits[1] - part.opening - PARAMS["limit_margin_m"])


def _stroke_from_stance(session, name: str, kind: str, planned) -> float:
    """One stroke of ``planned()`` [m] (asked again after a re-walk); returns the distance moved."""
    for attempt in range(1, PARAMS["stance_tries"] + 1):
        distance = planned()
        if distance <= 0:
            raise AgentFailed(f"{kind}: nothing to move from here ({distance:.3f} m).")
        moved = stroke(session, session.hints.part(name), kind, distance)
        if moved is not None:
            break
        if attempt == PARAMS["stance_tries"]:
            raise AgentFailed(f"{kind}: every handle pitch was refused from the stance.")
        print(f"routines: no {kind} stroke from here; stowing and walking to the stance again", flush=True)
        arm.stow(session)
        base.go_to(session, name, kind)
    if moved.status != "succeeded":
        raise AgentFailed(f"{kind} move along the slide: {moved.reason}")
    session.sleep(PARAMS["settle_seconds"])
    return distance


def stroke(session, drawer, kind: str, distance: float):
    """One stroke: ``open`` grasps the handle and pulls; ``close`` pushes with the closed gripper.

    Each handle pitch is tried in turn while the approach or the slide move is refused before
    moving. Returns the slide move's outcome, or None when every pitch was refused.
    """
    manipulation, push = session.manipulation, kind == "close"
    motion = (-1.0 if push else 1.0) * np.asarray(drawer.direction, dtype=float)
    for pitch in PARAMS["handle_pitch_deg"]:
        arm.gripper(session, manipulation.close_gripper if push else manipulation.open_gripper)
        grasp = _approach_handle(session, drawer, pitch, PARAMS["push_back_m"] if push else 0.0)
        if grasp is None:
            continue
        if not push:
            arm.gripper(session, manipulation.close_gripper)
            if not manipulation.holding:
                raise AgentFailed(f"{kind}: the gripper closed on nothing at the handle.")
        moved = arm.move_straight(session, motion, distance)
        if not push:
            arm.gripper(session, manipulation.open_gripper)
        _release(session, grasp, PARAMS["push_release" if push else "handle_release"])
        if moved.status == "succeeded" or not moved.refused:
            return moved  # Moved, or failed while moving.
    return None


def _release(session, grasp: np.ndarray, moves) -> None:
    """The release moves in order, each at its distance or else half of it; when a move is refused
    both times the rest are skipped (each move starts where the one before ended)."""
    for move, length in moves:
        axis = _release_axis(grasp, move)
        for distance in (float(length), float(length) / 2):
            outcome = arm.move_straight(session, axis, distance)
            if outcome.status == "succeeded" or not outcome.refused:
                break
        if outcome.status != "succeeded":
            print(f"routines: release move {move} refused ({outcome.reason}); skipping the rest", flush=True)
            return


def _approach_handle(session, drawer, pitch_deg: float, back_m: float) -> np.ndarray | None:
    """Move onto the handle at a pitch: each approach distance, each bar direction in turn. Returns
    the gripper's axes (3x3, world) of the grasp reached, or None when all were refused."""
    for distance in PARAMS["handle_approach_m"]:
        for side in (1.0, -1.0):
            grasp = handle_grasp_pose(drawer, side, pitch_deg)
            grasp[:3, 3] -= grasp[:3, 0] * back_m
            reached = arm.approach_straight(session, arm.wrist_for("handle", grasp), grasp[:3, 0], float(distance))
            if reached.status == "succeeded":
                return grasp[:3, :3]
            if not reached.refused:
                raise AgentFailed(f"handle approach: {reached.reason}")
    return None


def handle_grasp_pose(drawer, side: float, pitch_deg: float) -> np.ndarray:
    """World pose (4x4) of the robot's ``handle`` tool frame around a part's handle bar: the bar along
    the gripper's y (``side`` picks its direction), the approach into the part, pitched down by
    ``pitch_deg`` [deg] about the bar; the bar's axis on the frame, or, with a fixed jaw, the bar
    resting on it (its axis a radius above the frame along the closing axis)."""
    into = -np.asarray(drawer.direction, dtype=float)
    bar = np.asarray(drawer.handle_axis, dtype=float) * side
    y = bar - into * np.dot(bar, into)
    y /= np.linalg.norm(y)
    level_z = np.cross(into, y)
    pitch = math.radians(pitch_deg) * (1.0 if level_z[2] >= 0 else -1.0)  # Down, whichever way y points.
    approach = rotation_about(y, pitch) @ into
    z = np.cross(approach, y)
    rest = drawer.handle_radius if "fixed_jaw" in TOOLS else 0.0
    pose = np.eye(4)
    pose[:3, :3], pose[:3, 3] = (
        np.column_stack((approach, y, z)),
        np.asarray(drawer.handle_center, dtype=float) - z * rest,
    )
    return pose


def _release_axis(rotation: np.ndarray, move: str) -> np.ndarray:
    """World axis of a release move for the gripper's axes at ``rotation`` (straight-line moves keep
    them): ``wrist_down`` against the closing axis, ``gripper_back`` against the approach."""
    if move == "wrist_down":
        return -rotation[:, 2]
    if move == "gripper_back":
        return -rotation[:, 0]
    if move == "up":
        return np.array([0.0, 0.0, 1.0])
    if move == "forward":
        level = np.array([rotation[0, 0], rotation[1, 0], 0.0])
        return level / np.linalg.norm(level)
    raise ValueError(f"Unknown release move {move!r}")
