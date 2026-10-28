# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Building blocks the robot tests' scripted plans share: walking (``base``), arm motions, picks
and places (``arm``) and drawer strokes (``drawer``).

They drive the robot through the agent session only (they import the session, the standard
library, numpy and yaml; ``tests/test_agent_imports.py`` checks it), read privileged facts
only through ``session.hints``, and raise ``AgentFailed`` when they run out of options. They check
nothing about the world: the tests judge. The strategy is in ``params.yaml``; the robot's tool
frames, arm joints and stow postures come from ``session.robot`` (:func:`use_robot`); where to
stand in a scene is the routines' own data, ``configs/stances/<scene>.yaml`` (:func:`use_scene`).
The routines place the robot's tool frames (``grasp``, ``handle``, ``fixed_jaw``), which share the
gripper's axes: x the approach, z the jaws' closing axis. The strategy and the stances were tuned
for Spot.
"""

from __future__ import annotations

from pathlib import Path

import yaml

_HERE = Path(__file__).resolve().parent
PARAMS = yaml.safe_load((_HERE / "params.yaml").read_text())
# The robot in use (use_robot): its tool frames (name -> pose in the wrist link, 4x4 [m]) and its arm
# (joints, stow and pre-stow postures [m or rad], a straight move's duration [s]).
TOOLS: dict = {}
ARM: dict = {}
STANCES: dict[str, dict[str, list[float]]] = {}  # place -> use -> [x, y, heading] [m, m, rad], of the scene in use.


def use_robot(robot) -> None:
    """Take the robot's facts from ``robot`` (``session.robot``, a ``RobotInfo``)."""
    from phygs_robot_tests.routines.geometry import matrix

    TOOLS.clear()
    TOOLS.update({name: matrix(pose) for name, pose in robot.tools.items()})
    ARM.clear()
    ARM.update(joints=robot.arm_joints, stow=robot.stow, prestow=robot.prestow, move_seconds=robot.move_seconds)


def use_scene(name: str) -> None:
    """Load the stances for scene ``name`` (``configs/stances/<name>.yaml``; none if it has no file)."""
    path = _HERE.parent / "configs" / "stances" / f"{name}.yaml"
    STANCES.clear()
    if path.is_file():
        STANCES.update(yaml.safe_load(path.read_text())["stances"])
