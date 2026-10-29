# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""A made-up second robot (tests/fixtures/robots/fake_mm) through the framework's robot-neutral parts,
without Isaac Sim: what Spot does not exercise. A prismatic torso lift in the arm chain (kinematics
and the planning model), a gripper of two fingers that follow one primary joint (the manipulation
controller), a base on planar joints (its base controller), and the planning model's robot facts."""

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from fakes import FakeWorld, advance
from scipy.spatial.transform import Rotation

from phygs.execution.locomotion.planar import PlanarJointBase
from phygs.execution.manipulation.kinematics import ArmKinematics
from phygs.execution.manipulation.planners.model import gripper_positions, link_transforms, prepare_model
from phygs.robots.description import load_robot
from phygs.runtime.episode import EpisodeRuntime, RunMode

ROBOT = load_robot(Path(__file__).parent / "fixtures" / "robots" / "fake_mm")


def test_a_prismatic_lift_moves_the_wrist_along_its_axis_and_ik_solves_it():
    arm = ArmKinematics(ROBOT.urdf, ROBOT.arm.joints, ROBOT.arm.wrist_link)
    low, high = [0.0, 0.0, -1.0, 1.5, 0.0], [0.3, 0.0, -1.0, 1.5, 0.0]
    np.testing.assert_allclose(arm.forward(high)[:3, 3] - arm.forward(low)[:3, 3], (0.0, 0.0, 0.3), atol=1e-12)
    fingers = ROBOT.gripper.positions(0.0)
    frames = link_transforms(str(ROBOT.urdf), {**dict(zip(ROBOT.arm.joints, high)), **fingers})  # The model's FK.
    np.testing.assert_allclose(frames[ROBOT.arm.wrist_link], arm.forward(high), atol=1e-12)
    target = arm.forward([0.2, 0.4, -0.8, 1.2, 0.3])
    pose = [*target[:3, 3], *Rotation.from_matrix(target[:3, :3]).as_quat()]
    solved = arm.inverse(pose, [0.1, 0.3, -0.9, 1.3, 0.2])
    np.testing.assert_allclose(arm.forward(solved)[:3, 3], target[:3, 3], atol=1e-3)
    assert 0.0 <= solved[0] <= 0.5  # The lift [m] stays within its limits.


def test_the_planning_model_carries_the_robots_links_and_its_two_finger_gripper(tmp_path):
    locked = {"finger_left_joint": 0.04, "finger_right_joint": 0.04}
    sphere = [{"center": [0.0, 0.0, 0.0], "radius": 0.05}]
    model = prepare_model(ROBOT, tmp_path, geometry={"forearm": sphere}, locked_joints=locked, usd_boxes={})
    assert model["joints"] == list(ROBOT.arm.joints) and model["base_link"] == "base_link"
    assert model["wrist_link"] == "wrist" and model["held_contact_links"] == ["wrist", "finger_left", "finger_right"]
    assert model["locked_joints"] == locked and gripper_positions(model, 0.01) == {
        "finger_left_joint": 0.01,
        "finger_right_joint": 0.01,
    }
    with pytest.raises(ValueError, match="missing"):  # Every joint outside the arm is measured.
        prepare_model(ROBOT, tmp_path, geometry={"forearm": sphere}, locked_joints={"x": 0.0}, usd_boxes={})


def test_follower_fingers_take_the_primarys_targets_and_cannot_be_commanded_alone():
    joints = (*ROBOT.arm.joints, *ROBOT.gripper.joints)
    world = FakeWorld(joints, initial={"finger_left_joint": 0.04, "finger_right_joint": 0.04})
    limits = {
        **dict.fromkeys(ROBOT.arm.joints, (-3.0, 3.0)),
        "finger_left_joint": (0.0, 0.04),
        "finger_right_joint": (0.0, 0.04),
    }
    from phygs.execution.manipulation.controller import ManipulationController, MotionConfig
    from phygs.skills.manipulation.api import ManipulationClient

    controller = ManipulationController(
        joints,
        world,
        limits,
        MotionConfig(**ROBOT.arm.motion),
        source=RunMode.AUTONOMOUS,
        gripper=ROBOT.gripper.primary,
        gripper_followers={"finger_right_joint": 1.0},
    )
    runtime = EpisodeRuntime(world, (controller,), mode=RunMode.AUTONOMOUS)
    client = ManipulationClient(runtime, controller, ROBOT)
    runtime.reset()
    closed = client.close_gripper(duration=0.5)
    advance(runtime, client, 1.5)
    assert client.feedback(closed).status == "succeeded"
    assert world.state["finger_left_joint"] == pytest.approx(0.0) and world.state[
        "finger_right_joint"
    ] == pytest.approx(0.0)
    with pytest.raises(ValueError, match="follow the gripper"):
        client.move_joints({"finger_right_joint": 0.02}, duration=0.5)


def test_a_planar_base_turns_the_body_command_into_world_joint_velocities():
    yaw = [math.pi / 2]
    base = PlanarJointBase(
        ("base_x", "base_y", "base_yaw"),
        lambda: (1.0, 2.0, yaw[0]),
        dead_band=SimpleNamespace(apply=lambda command: command),
        deceleration=SimpleNamespace(apply=lambda now, goal, dt: goal, ramp_seconds=lambda now: 0.0),
        stop_seconds=0.2,
        physics_dt=0.004,
    )
    base.command = (0.5, 0.0, 0.3)  # Forward, facing +y: the base moves along world +y.
    targets = base.step(0.004)
    assert targets.positions == {"base_x": 1.0, "base_y": 2.0, "base_yaw": yaw[0]}
    np.testing.assert_allclose(
        [targets.velocities[j] for j in ("base_x", "base_y", "base_yaw")], (0.0, 0.5, 0.3), atol=1e-12
    )
    base.stop()
    assert base.stop_duration_ticks() == 50 and base.step(0.004).velocities == {
        "base_x": 0.0,
        "base_y": 0.0,
        "base_yaw": 0.0,
    }
    assert base.hold_targets() == {} and base.kinematics == "holonomic"


def test_a_planar_base_builds_from_its_robot_yaml():
    names = ("base_x", "base_y", "base_yaw")
    articulation = SimpleNamespace(
        find_joints=lambda wanted, preserve_order: ([0, 1, 2], list(wanted)),
        data=SimpleNamespace(joint_pos=SimpleNamespace(torch=torch.tensor([[0.5, -0.5, 0.1]]))),
    )
    base = PlanarJointBase.from_robot(ROBOT, articulation, physics_dt=0.004)
    assert base.declaration.joints == names and base.declaration.stop_ticks == 50
    assert base.standing_targets().positions == pytest.approx({"base_x": 0.5, "base_y": -0.5, "base_yaw": 0.1})
    assert PlanarJointBase.files(ROBOT) == {"base_config_sha256": ROBOT.base.config}


def test_a_base_on_joints_under_a_fixed_root_reads_its_own_pose_not_the_roots():
    """Isaac Lab's Ridgeback-Franka: the root stays at the origin while the base link moves. Reading
    the root (as observations did before robot descriptions) reports a base that never moves."""
    from phygs.sim.base import BaseLink

    def articulation(fixed: bool):
        turned = [0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)]  # 90 deg about z.
        return SimpleNamespace(
            is_fixed_base=fixed,
            body_names=["world_root", "base_link"] if fixed else ["base_link"],
            find_bodies=lambda name: ([1], ["base_link"]) if fixed else ([0], ["base_link"]),
            data=SimpleNamespace(
                root_pose_w=SimpleNamespace(torch=torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])),
                root_lin_vel_b=SimpleNamespace(torch=torch.tensor([[0.3, 0.0, 0.0]])),
                body_link_pose_w=SimpleNamespace(
                    torch=torch.tensor([[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [2.0, 1.0, 0.0, *turned]]])
                ),
                body_link_lin_vel_w=SimpleNamespace(torch=torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.4, 0.0]]])),
            ),
        )

    planar = BaseLink(articulation(fixed=True), "base_link")
    torch.testing.assert_close(planar.pose()[:3], torch.tensor([2.0, 1.0, 0.0]))
    torch.testing.assert_close(planar.linear_velocity_b(), torch.tensor([0.4, 0.0, 0.0]), atol=1e-6, rtol=0)
    floating = BaseLink(articulation(fixed=False), "base_link")  # Spot: the root is the base.
    torch.testing.assert_close(floating.pose(), torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]))
    torch.testing.assert_close(floating.linear_velocity_b(), torch.tensor([0.3, 0.0, 0.0]))
