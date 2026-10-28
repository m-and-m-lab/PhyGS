# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Page choices translated into the skill API's own arguments."""

import math

import numpy as np
import pytest
from fakes import SPOT
from scipy.spatial.transform import Rotation

from phygs.ui import translate

AHEAD = np.eye(4)
AHEAD[0, 3] = 0.2  # A grasp point 0.2 m along the wrist x, the gripper's axes the wrist's.


def test_downward_wrist_puts_the_grasp_point_above_the_point_with_the_jaws_along_the_heading():
    pose = translate.wrist_pose((1.0, 0.2, 0.4), "down", yaw=0.0, standoff_m=0.1, grasp=SPOT.tool("grasp"))
    wrist = np.eye(4)
    wrist[:3, :3], wrist[:3, 3] = Rotation.from_quat(pose[3:]).as_matrix(), pose[:3]
    grasp = wrist @ SPOT.tool("grasp")
    assert grasp[:3, 3] == pytest.approx((1.0, 0.2, 0.5))  # The standoff above the point.
    assert grasp[:3, 0] == pytest.approx((0.0, 0.0, -1.0)) and grasp[:3, 2] == pytest.approx((1.0, 0.0, 0.0))
    assert pose[:3] == pytest.approx((1.0 - 0.029, 0.2, 0.5 + 0.183))  # Spot: the jaw gap's middle on the point.


def test_a_hand_that_approaches_along_its_z_points_its_z_down():
    from phygs.robots.description import load_robot

    franka = load_robot("ridgeback_franka").tool("grasp")
    pose = translate.wrist_pose((1.0, 0.2, 0.4), "down", yaw=0.0, standoff_m=0.0, grasp=franka)
    hand = Rotation.from_quat(pose[3:])
    assert hand.apply([0.0, 0.0, 1.0]) == pytest.approx((0.0, 0.0, -1.0), abs=1e-9)  # The approach, down.
    assert hand.apply([0.0, 1.0, 0.0]) == pytest.approx((1.0, 0.0, 0.0), abs=1e-9)  # Fingers close along the heading.
    assert pose[:3] == pytest.approx((1.0, 0.2, 0.4 + franka[2, 3]))  # The wrist its grasp offset above.


def test_forward_and_surface_wrists_point_along_the_heading_and_into_the_surface():
    yaw = math.pi / 2  # The robot faces +y.
    forward = translate.wrist_pose((0.0, 1.0, 0.5), "forward", yaw=yaw, standoff_m=0.0, grasp=AHEAD)
    gripper = Rotation.from_quat(forward[3:]).apply([1.0, 0.0, 0.0])
    assert gripper == pytest.approx([0.0, 1.0, 0.0], abs=1e-9)
    assert Rotation.from_quat(forward[3:]).apply([0.0, 0.0, 1.0]) == pytest.approx(
        [0.0, 0.0, 1.0], abs=1e-9
    )  # Upright.
    assert forward[:3] == pytest.approx((0.0, 0.8, 0.5))
    wall = translate.wrist_pose(
        (2.0, 0.0, 0.6), "surface", yaw=0.0, standoff_m=0.05, grasp=AHEAD, normal=(-1.0, 0.0, 0.0)
    )
    assert Rotation.from_quat(wall[3:]).apply([1.0, 0.0, 0.0]) == pytest.approx([1.0, 0.0, 0.0], abs=1e-9)
    assert wall[:3] == pytest.approx((1.75, 0.0, 0.6))
    with pytest.raises(ValueError, match="normal"):
        translate.wrist_pose((2.0, 0.0, 0.6), "surface", yaw=0.0, standoff_m=0.05, grasp=AHEAD)


def test_directions_follow_the_robot_and_a_line_keeps_the_orientation():
    yaw = math.pi / 2
    assert translate.axis("forward", yaw) == pytest.approx((0.0, 1.0, 0.0), abs=1e-9)
    assert translate.axis("right", yaw) == pytest.approx((1.0, 0.0, 0.0), abs=1e-9)
    assert translate.axis("up", yaw) == (0.0, 0.0, 1.0)
    times, poses = translate.line((1.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 1.0), 0.05)
    assert times == pytest.approx([0.0, 0.05 / translate.LINE_SPEED_M_S])
    assert poses[1] == pytest.approx([1.0, 0.0, 0.55, 0.0, 0.0, 0.0, 1.0])
    with pytest.raises(ValueError):
        translate.line((1.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 1.0), 0.0)
    with pytest.raises(ValueError):
        translate.axis("sideways", 0.0)


def test_point_targets_are_world_points():
    target = translate.point_target({"point": [1.0, 2.0, 0.5], "view": "hand"})
    assert target.position == (1.0, 2.0, 0.5) and target.frame == "world"
    with pytest.raises(ValueError):
        translate.point_target({"point": [1.0, float("nan"), 0.5]})
