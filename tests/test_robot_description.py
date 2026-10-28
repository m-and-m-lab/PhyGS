# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Robot descriptions: every robot of the repository passes the description check (the first rung of
the robot-test ladder, no simulator), and a made-up second robot (tests/fixtures/robots/fake_mm: a
torso lift in the arm, two prismatic fingers, a base on planar joints) goes through the same code."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from phygs.robots.description import ROBOTS, RobotDescription, load_robot

FAKE = Path(__file__).parent / "fixtures" / "robots" / "fake_mm"


ROBOT_FOLDERS = sorted(path.parent for path in ROBOTS.glob("[!_]*/robot.yaml"))  # _template is not a robot.


@pytest.mark.parametrize("folder", ROBOT_FOLDERS, ids=lambda p: p.name)
def test_every_robot_of_the_repository_matches_its_urdf_and_usd(folder):
    try:
        robot = load_robot(folder.name)
    except FileNotFoundError as missing:  # Assets fetched on demand (robot.yaml fetch).
        pytest.skip(str(missing))
    robot.validate_urdf()
    pytest.importorskip("pxr")
    robot.validate_usd()
    assert robot.name == folder.name and robot.folder == folder
    assert robot.look_camera in robot.cameras and set(robot.cameras_for("planning")) <= set(robot.cameras)


def test_spot_is_found_by_name_folder_or_file():
    by_name = load_robot("spot")
    assert load_robot(ROBOTS / "spot").sha256 == load_robot(ROBOTS / "spot" / "robot.yaml").sha256 == by_name.sha256
    assert by_name.arm.joints == ("arm_sh0", "arm_sh1", "arm_el0", "arm_el1", "arm_wr0", "arm_wr1")
    assert by_name.gripper.primary == "arm_f1x" and by_name.base.kinematics == "holonomic"
    assert by_name.arm_joint_links[-1] == by_name.arm.wrist_link
    with pytest.raises(FileNotFoundError, match="spot"):  # The error lists the robots there are.
        load_robot("no_such_robot")


def test_a_second_robot_loads_from_anywhere_with_its_own_plugin():
    robot = load_robot(FAKE)
    robot.validate_urdf()
    assert robot.arm.joints[0] == "lift" and robot.gripper.positions(0.02) == {
        "finger_left_joint": 0.02,
        "finger_right_joint": 0.02,
    }
    assert robot.arm_links == ("torso", "shoulder", "upper_arm", "forearm", "wrist", "finger_left", "finger_right")
    assert robot.plugin(robot.actuators)(0.004) == {"all": dict(physics_dt=0.004)}  # A file of its folder.
    assert robot.cameras_for("front") == ("head",) and robot.look_camera == "wrist_cam"
    np.testing.assert_allclose(robot.tool("hand")[:3, 3], (0.15, 0.0, 0.0))
    info = robot.info()
    assert (
        info.arm_joints == robot.arm.joints
        and info.base_kinematics == "holonomic"
        and info.tools["grasp"] == (0.14, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    )


def _broken(tmp_path, change) -> Path:
    """A copy of the fake robot's folder with ``change(data)`` applied to its robot.yaml."""
    for path in FAKE.iterdir():
        if path.is_file():
            (tmp_path / path.name).write_bytes(path.read_bytes())
    data = yaml.safe_load((FAKE / "robot.yaml").read_text())
    change(data)
    (tmp_path / "robot.yaml").write_text(yaml.safe_dump(data))
    return tmp_path


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda d: d.update(schema=0), "schema"),
        (lambda d: d.update(colour="red"), "unknown keys"),
        (lambda d: d["base"].update(kinematics="legged"), "kinematics"),
        (lambda d: d["gripper"]["tools"].pop("grasp"), "tools need"),
        (
            lambda d: d["gripper"]["tools"]["grasp"].update(rotation_xyzw=[0.0, 0.0, 0.7071068, 0.7071068]),
            "gripper's axes",
        ),
        (lambda d: d["gripper"].update(joints={"finger_left_joint": 0.5}), "primary"),
        (lambda d: d["camera_roles"].update(look=["head", "wrist_cam"]), "'look' is one camera"),
        (lambda d: d["camera_roles"].update(sideways=["head"]), "unknown camera roles"),
        (lambda d: d["camera_roles"].update(navigation=["periscope"]), "unknown cameras"),
        (lambda d: d["arm"].update(stow=[0.0, 0.0]), "one value per arm joint"),
        (lambda d: d["initial_joints"].update(elbow=0.3), "must not set arm joints"),
        (lambda d: d["navigation"].update(footprint_profile="missing.json"), "does not exist"),
        (lambda d: d["arm"]["motion"].pop("gripper_stopped_velocity"), "arm.motion: missing keys"),
    ],
)
def test_a_broken_robot_yaml_is_refused_at_load(tmp_path, change, message):
    with pytest.raises((ValueError, FileNotFoundError), match=message):
        load_robot(_broken(tmp_path, change))


@pytest.mark.parametrize(
    "change, message",
    [
        (
            lambda r: replace(r, arm=replace(r.arm, joints=("shoulder_yaw", "lift", *r.arm.joints[2:]))),
            "not the movable",
        ),
        (lambda r: replace(r, arm=replace(r.arm, wrist_link="gripper")), "no link 'gripper'"),
        (lambda r: replace(r, arm=replace(r.arm, stow=(0.9, *r.arm.stow[1:]))), r"arm.stow: lift=0.9 outside"),
        (lambda r: replace(r, gripper=replace(r.gripper, open=0.05)), "gripper.open"),
        (lambda r: replace(r, base_link="torso"), "root link is 'base_link', not base_link 'torso'"),
    ],
)
def test_the_description_check_finds_what_the_urdf_contradicts(change, message):
    robot: RobotDescription = change(load_robot(FAKE))
    with pytest.raises(ValueError, match=message):
        robot.validate_urdf()


def test_every_robot_s_tools_follow_the_gripper_axes():
    # x the approach, z the jaws' closing axis: Spot's are the wrist's; the Franka hand approaches
    # along its z and closes its fingers along its y.
    np.testing.assert_allclose(load_robot("spot").tool("grasp")[:3, :3], np.eye(3))
    franka = load_robot("ridgeback_franka").tool("grasp")[:3, :3]
    np.testing.assert_allclose(franka[:, 0], (0.0, 0.0, 1.0), atol=1e-7)
    np.testing.assert_allclose(franka[:, 2], (0.0, 1.0, 0.0), atol=1e-7)


def test_the_motion_fields_robot_yaml_reads_are_motion_configs():
    from dataclasses import fields

    from phygs.execution.manipulation.controller import MotionConfig
    from phygs.robots.description import JOINT_MOTION, OTHER_MOTION

    assert sorted(JOINT_MOTION + OTHER_MOTION) == sorted(f.name for f in fields(MotionConfig))
    assert load_robot(FAKE).arm.transit_posture[0] == ("lift", 0.0)


def test_two_folders_with_one_robot_name_keep_their_own_plugins(tmp_path):
    copy = _broken(tmp_path, lambda d: None)  # Same name, fake_mm.
    (copy / "actuators.py").write_text("def actuator_cfgs(physics_dt):\n    return {'copy': physics_dt}\n")
    original, copied = load_robot(FAKE), load_robot(copy)
    assert original.name == copied.name
    assert original.plugin(original.actuators)(0.004) == {"all": dict(physics_dt=0.004)}
    assert copied.plugin(copied.actuators)(0.004) == {"copy": 0.004}


def test_a_differential_base_cannot_be_commanded_sideways():
    from types import SimpleNamespace

    from phygs.execution.locomotion.base import DeadBand
    from phygs.runtime.episode import ControllerDeclaration
    from phygs.skills.locomotion.api import LocomotionClient

    controller = SimpleNamespace(
        declaration=ControllerDeclaration("locomotion", ("left_wheel", "right_wheel")),
        kinematics="differential",
        dead_band=DeadBand((0.0,) * 3, (0.0,) * 3, (0.0,) * 3),
    )
    client = LocomotionClient(SimpleNamespace(physics_dt=0.004), controller)
    with pytest.raises(ValueError, match="cannot move sideways"):
        client.command_velocity(0.2, 0.1, 0.0)


def test_the_template_has_every_field_the_schema_reads(tmp_path):
    """robots/_template/robot.yaml, the starting point for a new robot, loads once its files exist."""
    (tmp_path / "robot.yaml").write_bytes((ROBOTS / "_template" / "robot.yaml").read_bytes())
    for name in ("my_robot.urdf", "base.yaml", "actuators.py"):
        (tmp_path / name).write_bytes((FAKE / {"my_robot.urdf": "robot.urdf"}.get(name, name)).read_bytes())
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "my_robot.usd").write_text("#usda 1.0\n")
    robot = load_robot(tmp_path)
    assert robot.name == "my_robot" and robot.base.kind == "planar_joints"


def test_every_measured_footprint_still_matches_the_files_it_pins():
    """A robot's transit footprint (``navigation/transit.json``) pins the robot's USD and the files
    its base's motion depends on (the walking policy and its training config) by hash; a byte
    changed in any of them, even a comment, invalidates the measurement and every navigation run."""
    from phygs.execution.navigation.footprint import load_footprint
    from phygs.robots.control import base_controller_class

    for folder in ROBOT_FOLDERS:
        robot = load_robot(folder.name)
        profile = robot.navigation.footprint_profile
        if profile is None:
            continue
        files = base_controller_class(robot).files(robot)
        footprint = load_footprint(profile, robot.usd, files)
        assert footprint, folder.name
