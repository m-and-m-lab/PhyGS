# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""End-effector trajectories: validation, IK lowering, refusals before moving, slowing down."""

import numpy as np
import pytest
from fakes import SPOT

from phygs.execution.manipulation.lowering import (
    STEP_M,
    TrajectoryRefused,
    lower,
    pose_error,
)
from phygs.execution.manipulation.planning import EndEffectorTrajectory

NAMES = ("j0", "j1", "j2", "j3", "j4", "j5")
HAND = SPOT.tool("hand")  # Spot's hand (between the jaws) in the wrist frame (4x4) [m].
HAND_X, _, HAND_Z = HAND[:3, 3]


def forward(joints):
    """Fake arm: the first three joints are the wrist position [m], orientation fixed."""
    pose = np.eye(4)
    pose[:3, 3] = joints[:3]
    return pose


def inverse(pose, seed, reach=1.0):
    if np.linalg.norm(pose[:3]) > reach or not np.allclose(pose[3:], (0, 0, 0, 1), atol=1e-6):
        return None
    return [*pose[:3], 0.0, 0.0, 0.0]


def hand_line(start, end, seconds, frame="world"):
    poses = [[*start, 0, 0, 0, 1], [*end, 0, 0, 0, 1]]
    return EndEffectorTrajectory([0.0, seconds], poses, episode_id=1, frame=frame)


def lowered(trajectory, seed=(0.3, 0.0, 0.4, 0, 0, 0), body=np.eye(4), **kwargs):
    options = dict(
        names=NAMES,
        seed=list(seed),
        body=body,
        wrist_from_tool=HAND,
        inverse=inverse,
        forward=forward,
        max_velocity=1.5,
        start_tolerance=(0.02, 0.05),
    )
    return lower(trajectory, **{**options, **kwargs})


def test_a_hand_line_is_densified_and_solved_for_the_wrist():
    hand = (0.3 + HAND_X, 0.0, 0.4 + HAND_Z)
    result = lowered(hand_line(hand, (hand[0] + 0.1, 0.0, hand[2]), 1.0))
    times, joints = result.joints.times, result.joints.positions
    assert len(times) - 1 >= 0.1 / STEP_M and result.time_scale == 1.0
    np.testing.assert_allclose(joints[:, 0], 0.3 + (times / 1.0) * 0.1, atol=1e-9)  # The wrist, 0.2 m behind the hand.
    k = 2  # The reference at a solved sample is the tool pose of its joints.
    assert pose_error(result.tool_pose(times[k]), forward(joints[k]) @ HAND)[0] < 1e-9


def test_too_fast_is_slowed_down_to_the_speed_limit():
    hand = (0.3 + HAND_X, 0.0, 0.4 + HAND_Z)
    result = lowered(hand_line(hand, (hand[0] + 0.5, 0.0, hand[2]), 0.1))
    assert result.time_scale == pytest.approx((0.5 / 0.1) / 1.5)
    assert result.joints.peak_speed() == pytest.approx(1.5)


def test_refusals_before_moving():
    hand = (0.3 + HAND_X, 0.0, 0.4 + HAND_Z)
    with pytest.raises(TrajectoryRefused, match="leaves the arm's reach") as error:
        lowered(hand_line(hand, (hand[0] + 1.0, 0.0, hand[2]), 2.0))
    assert error.value.failure == "out_of_reach"

    def flipping(pose, seed):  # Another configuration from the second sample on.
        joints = inverse(pose, seed)
        return joints if pose[0] < 0.31 else [*joints[:3], 1.0, 0.0, 0.0]

    with pytest.raises(TrajectoryRefused, match="another arm configuration"):
        lowered(hand_line(hand, (hand[0] + 0.1, 0.0, hand[2]), 1.0), inverse=flipping)
    with pytest.raises(TrajectoryRefused, match="starts 0.050 m") as error:
        lowered(hand_line((hand[0], 0.05, hand[2]), (hand[0] + 0.1, 0.05, hand[2]), 1.0))
    assert error.value.failure is None


def test_world_and_body_frames():
    body = np.eye(4)
    body[:3, 3] = (2.0, 1.0, 0.0)  # The base, 2 m and 1 m away from the world origin.
    hand = (0.3 + HAND_X, 0.0, 0.4 + HAND_Z)
    world = [hand[0] + 2.0, 1.0, hand[2]]
    in_world = lowered(hand_line(world, (world[0] + 0.1, 1.0, hand[2]), 1.0), body=body)
    in_body = lowered(hand_line(hand, (hand[0] + 0.1, 0.0, hand[2]), 1.0, frame="body"), body=body)
    np.testing.assert_allclose(in_world.joints.positions, in_body.joints.positions, atol=1e-9)


def test_trajectory_validation():
    with pytest.raises(ValueError, match="normalized"):
        EndEffectorTrajectory([0.0, 1.0], [[0, 0, 0, 0, 0, 0, 2], [0, 0, 0, 0, 0, 0, 1]], 1)
    with pytest.raises(ValueError, match="Frame"):
        EndEffectorTrajectory([0.0, 1.0], [[0, 0, 0, 0, 0, 0, 1]] * 2, 1, frame="odom")
    with pytest.raises(ValueError, match="increase strictly"):
        EndEffectorTrajectory([0.0, 0.0], [[0, 0, 0, 0, 0, 0, 1]] * 2, 1)
