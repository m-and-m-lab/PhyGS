# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The skills the UI can run, as sent to the page in ``hello`` (docs/ui.md, Running a skill).

Each entry names the public API call, the tile, the channel (base or arm), the picker the page
switches to, and the inputs with units and defaults. The page builds its tiles and command bar
from this list; the session checks requests against it.
"""

from __future__ import annotations

from phygs.execution.navigation.types import NavigationGoal

# The tolerance defaults are the public API's (NavigationClient.navigate_to takes them from NavigationGoal).
NAVIGATE_TO = dict(
    name="navigate_to",
    tile="Navigate to",
    call="navigation.navigate_to(x, y, heading, position_tolerance, heading_tolerance)",
    channel="base",
    picker="map",
    doc="Walk to a map position and, if given, turn to a heading. Plans on the observed map.",
    inputs=[
        dict(name="x", unit="m", picker="map"),
        dict(name="y", unit="m", picker="map"),
        dict(name="heading", unit="rad", picker="map drag", optional=True),
        dict(name="position_tolerance", unit="m", default=NavigationGoal.position_tolerance, min=0.02, max=1.0),
        dict(name="heading_tolerance", unit="rad", default=NavigationGoal.heading_tolerance, min=0.02, max=1.0),
    ],
)
DRIVE = dict(
    name="drive",
    tile="Drive",
    call="locomotion.command_velocity(v_x, v_y, v_rot, duration=0.5), renewed while a key is held",
    channel="base",
    picker=None,
    doc="Hold W/S (forward, back), A/D (left, right), Q/E (turn). Releasing stops within 0.5 s.",
    inputs=[
        dict(name="v_x", unit="m/s"),
        dict(name="v_y", unit="m/s"),
        dict(name="v_rot", unit="rad/s"),
    ],
)
STOP = dict(
    name="stop",
    tile="Stop",
    call="locomotion.stop()",
    channel="base",
    picker=None,
    doc="Controlled stop of direct motion.",
    inputs=[],
)


ARM_JOINTS = dict(
    name="move_joints",
    tile="Arm joints",
    call="manipulation.move_joints(positions, duration)",
    channel="arm",
    picker="hand",
    doc="Set the arm joints and press Go; the arm moves there in a straight joint move. "
    "Watch the hand camera to frame a view. Save poses to play them as a Joint path.",
    inputs=[dict(name="positions", unit="rad"), dict(name="duration", unit="s", optional=True)],
)
JOINT_PATH = dict(
    name="execute_trajectory",
    tile="Joint path",
    call="manipulation.execute_trajectory(JointTrajectory(arm joints, times, positions))",
    channel="arm",
    picker="hand",
    doc="Play saved arm poses in order, from the current pose, each reached after the time you set.",
    inputs=[dict(name="waypoints", unit="rad, s")],
)
STOW = dict(
    name="stow",
    tile="Stow",
    call="manipulation.stow()",
    channel="arm",
    picker=None,
    doc="Fold the arm into the transit posture navigation needs; a held object stays held.",
    inputs=[],
)
UNSTOW = dict(
    name="unstow",
    tile="Unstow",
    call="manipulation.unstow()",
    channel="arm",
    picker=None,
    doc="Unfold the arm from the stow to the ready (pre-stow) posture. Optional: arm moves leave the "
    "stow by themselves.",
    inputs=[],
)
OPEN_GRIPPER = dict(
    name="open_gripper",
    tile="Open",
    call="manipulation.open_gripper(duration=2.0)",
    channel="arm",
    picker=None,
    doc="Open the gripper.",
    inputs=[],
)
CLOSE_GRIPPER = dict(
    name="close_gripper",
    tile="Close",
    call="manipulation.close_gripper(duration=2.0)",
    channel="arm",
    picker=None,
    doc="Close the gripper; it reports holding when it closes on something.",
    inputs=[],
)
LOOK_AT = dict(
    name="look_at",
    tile="Look at",
    call="manipulation.look_at(PointTarget(point), distance_m, elevation_deg)",
    channel="arm",
    picker="camera",
    doc="Aim the hand camera at a point you click in a camera view, from a distance and height you set.",
    inputs=[
        dict(name="target", unit="m", picker="camera"),
        dict(name="distance_m", unit="m", default=0.5, min=0.2, max=1.0),
        dict(name="elevation_deg", unit="deg", default=45.0, min=0.0, max=90.0),
    ],
)
MOVE_TO_POSE = dict(
    name="move_to_pose",
    tile="Move to pose",
    call="manipulation.move_to_pose(wrist pose from the clicked point, direction and standoff)",
    channel="arm",
    picker="camera",
    doc="Bring the gripper to a point you click, pointing down, forward, or into the surface, "
    "stopping the standoff before it.",
    inputs=[
        dict(name="target", unit="m", picker="camera"),
        dict(name="orientation", choices=["down", "forward", "surface"], default="down"),
        dict(name="standoff_m", unit="m", default=0.10, min=0.0, max=0.4),
    ],
)
MOVE_ALONG = dict(
    name="move_along",
    tile="Move along",
    call="manipulation.execute_trajectory(EndEffectorTrajectory: a wrist line along the direction)",
    channel="arm",
    picker=None,
    doc="Move the wrist straight up, down, forward, back, left or right of the robot at 0.1 m/s, "
    "keeping its orientation.",
    inputs=[
        dict(name="direction", choices=["up", "down", "forward", "back", "left", "right"], default="up"),
        dict(name="distance_m", unit="m", default=0.05, min=0.01, max=0.5),
    ],
)
# Skills that run at once on their tile: they take no inputs.
IMMEDIATE = ("stop", "stow", "unstow", "open_gripper", "close_gripper")
# The action interface (RobotControl allowed_actions) each base skill's client needs; the arm
# skills need "manipulation".
ACTION = {"navigate_to": "navigation", "drive": "motion", "stop": "motion"}
# Skills the UI can plan first, show, then run as that very trajectory (manipulation.execute_trajectory).
PREVIEW = ("move_joints", "move_to_pose")


def skills(
    drive_limits: dict,
    arm_limits: dict | None = None,
    targets: bool = False,
    preview: bool = False,
    actions: tuple[str, ...] = ("navigation", "motion", "manipulation"),
    hand_view: str = "hand",
) -> list[dict]:
    """The runnable skills. ``drive_limits`` (max_speed [m/s], max_yaw_speed [rad/s]) bound Drive;
    ``arm_limits`` ({joint: [lower, upper]} [rad]) enables the joint-space arm skills; ``targets``
    enables the skills that take a clicked target or a direction (they need IK and a planner);
    ``preview`` marks the skills whose plan the page can show before running it (a planner is
    bound). Only skills whose client the episode's ``actions`` allow are listed. ``hand_view`` is the
    view of the robot's look (hand) camera, where the arm skills' targets are picked."""
    drive = dict(DRIVE, limits=dict(drive_limits))
    listed = [NAVIGATE_TO, drive, STOP]
    if arm_limits is not None:
        joints = dict(ARM_JOINTS, limits={name: list(map(float, limit)) for name, limit in arm_limits.items()})
        listed += [joints, JOINT_PATH, STOW, UNSTOW, OPEN_GRIPPER, CLOSE_GRIPPER]
        if targets:
            listed += [LOOK_AT, MOVE_TO_POSE, MOVE_ALONG]
    listed = [skill for skill in listed if ACTION.get(skill["name"], "manipulation") in actions]
    return [
        dict(
            skill,
            picker=hand_view if skill.get("picker") == "hand" else skill.get("picker"),
            immediate=skill["name"] in IMMEDIATE,
            preview=preview and skill["name"] in PREVIEW,
        )
        for skill in listed
    ]
