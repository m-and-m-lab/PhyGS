# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""End-effector trajectories through execute_trajectory: lowering, tool tracking, slowing down."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from fakes import SPOT, FakeWorld, advance, manipulation_stack

from phygs.execution.manipulation.planning import EndEffectorTrajectory, JointTrajectory
from phygs.skills.manipulation.api import TrajectoryRefused

JOINTS, GRIPPER, TOOLS = SPOT.arm.joints, SPOT.gripper.primary, SPOT.gripper.tools
START = {"arm_sh0": 0.3, "arm_sh1": 0.0, "arm_el0": 0.4}  # The fake wrist at (0.3, 0, 0.4) m.


class Kinematics:
    """Fake arm: arm_sh0, arm_sh1, arm_el0 are the wrist position [m]; orientation fixed."""

    def forward(self, joints):
        pose = np.eye(4)
        pose[:3, 3] = joints[:3]
        return pose

    def inverse(self, pose, seed):
        if np.linalg.norm(pose[:3]) > 1.0:
            return None
        return [*pose[:3], 0.0, 0.0, 0.0]


class Observations:
    def observe(self):
        return SimpleNamespace(localization=SimpleNamespace(valid=True, pose=SimpleNamespace(matrix=torch.eye(4))))


def setup(lag=0.0):
    """The arm at 100 Hz; ``lag`` [rad] keeps arm_sh0 (the wrist x) short of its target."""
    world = FakeWorld(
        (*JOINTS, GRIPPER),
        initial=START,
        physics_dt=0.01,
        follow=lambda target: {**target, "arm_sh0": target["arm_sh0"] - lag},
    )
    limits = {**dict.fromkeys(JOINTS, (-2.0, 2.0)), GRIPPER: (-1.6, 0.0)}
    return manipulation_stack(
        world,
        limits,
        gripper=GRIPPER,
        open_angle=-1.4,
        closed_angle=-0.05,
        kinematics=(Kinematics(), Observations()),
        tools=TOOLS,
    )


def hand_line(client, dx, seconds):
    hand = np.array([0.3, 0.0, 0.4]) + TOOLS["hand"][:3, 3]
    poses = [[*hand, 0, 0, 0, 1], [hand[0] + dx, hand[1], hand[2], 0, 0, 0, 1]]
    return EndEffectorTrajectory([0.0, seconds], poses, client.episode_id)


def test_an_end_effector_line_runs_and_reports_progress():
    world, controller, runtime, client = setup()
    command = client.execute_trajectory(hand_line(client, 0.1, 1.0))
    advance(runtime, client, 2.0)
    feedback = client.feedback(command)
    assert feedback.status == "succeeded" and feedback.progress == 1.0 and feedback.time_scale == 1.0
    assert world.state["arm_sh0"] == pytest.approx(0.4)  # The wrist followed the hand 0.1 m along x.


def test_too_fast_is_slowed_down_instead_of_refused():
    world, controller, runtime, client = setup()
    command = client.execute_trajectory(hand_line(client, 0.3, 0.1))  # 3 m/s on one joint.
    advance(runtime, client, 3.0)
    feedback = client.feedback(command)
    assert feedback.status == "succeeded" and feedback.time_scale == pytest.approx(3.0 / 1.5)
    world, controller, runtime, client = setup()  # A joint trajectory too: slowed, not refused.
    fast = JointTrajectory(JOINTS, np.array([0.0, 0.1]), np.array([[0.3, 0, 0.4, 0, 0, 0], [0.6, 0, 0.4, 0, 0, 0]]), 1)
    command = client.execute_trajectory(fast)
    advance(runtime, client, 3.0)
    assert client.feedback(command).time_scale == pytest.approx(2.0)


def test_the_tool_leaving_its_path_fails_while_moving():
    world, controller, runtime, client = setup(lag=0.03)  # 3 cm behind: over the 2 cm tool limit.
    command = client.execute_trajectory(hand_line(client, 0.1, 1.0))
    advance(runtime, client, 3.0)
    feedback = client.feedback(command)
    assert feedback.status == "failed" and "End-effector tracking" in feedback.reason
    assert feedback.pose_error[0] > 0.02 and 0.0 < feedback.progress < 1.0


def test_tool_pose_is_the_hand_ahead_of_the_wrist():
    world, controller, runtime, client = setup()
    offset = TOOLS["hand"][:3, 3]
    np.testing.assert_allclose(client.tool_pose(), (0.3 + offset[0], 0.0, 0.4 + offset[2], 0, 0, 0, 1), atol=1e-9)
    np.testing.assert_allclose(client.tool_pose("wrist")[:3], (0.3, 0.0, 0.4), atol=1e-9)


def test_refused_before_moving():
    world, controller, runtime, client = setup()
    with pytest.raises(TrajectoryRefused, match="reach"):
        client.execute_trajectory(hand_line(client, 1.0, 2.0))
    assert runtime.controller_state("manipulation") == "idle"
    assert world.state["arm_sh0"] == pytest.approx(0.3)
