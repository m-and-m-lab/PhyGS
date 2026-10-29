# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""A new robot's first simulator check (docs/adding_a_robot.md): it spawns from its ``robot.yaml``
with the arm stowed, the base drives forward, sideways and turns, then stops, the arm stays stowed
while driving and reaches the pre-stow, the gripper closes and opens, every camera gives depth, and
localization follows the base link. Robot-neutral: every value comes from the description.

    ./isaaclab.sh -p scripts/PhyGS/tests/robot_tests/components/check_bring_up.py \
        --robot ridgeback_franka --output scripts/PhyGS/output_check/$(date +%Y%m%d-%H%M)_bring_up
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # phygs from the checkout.
from phygs_robot_tests import common  # noqa: E402

flags = common.parser(__doc__)
flags.set_defaults(enable_cameras=True)
args = flags.parse_args()
check = common.Check("check_bring_up", args)
robot_description = common.robot(args)  # --robot (default spot).

import isaaclab.sim as sim_utils  # noqa: E402
import numpy as np  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402
from PIL import Image  # noqa: E402

from phygs.observations.types import CameraPermission, ObservationConfig  # noqa: E402
from phygs.robots.control import RobotControl  # noqa: E402
from phygs.robots.scene import make_scene_cfg, simulation_cfg  # noqa: E402
from phygs.runtime.episode import RunMode  # noqa: E402
from phygs.runtime.timing import duration_ticks  # noqa: E402
from phygs.sim.base import BaseLink  # noqa: E402

CAMERA_PERIOD = 0.032  # [s]
DRIVE_SECONDS, STOP_SECONDS = 2.0, 5.0  # A drive command [s] and the most a stop may take after it [s].
DRIVE_FRACTION = 0.6  # Of the robot's navigation speed limits.
STOW_TOLERANCE = 0.05  # Arm joints off the stow [m or rad].


def run(report):
    dt = common.PHYSICS_DT
    robot = robot_description
    cameras = tuple(robot.cameras)
    render = duration_ticks(CAMERA_PERIOD, dt)
    sim = sim_utils.SimulationContext(simulation_cfg(dt=dt, render_interval=render, device=args.device))
    scene = InteractiveScene(
        make_scene_cfg(**common.robot_scene(robot, cameras=cameras, imu=True))(
            num_envs=1, env_spacing=3.0, replicate_physics=False
        )
    )
    sim.reset()
    scene.update(dt)
    control = RobotControl.create(
        sim,
        scene,
        robot,
        mode=RunMode.AUTONOMOUS,
        manipulation=True,
        allowed_actions=("motion", "manipulation", "arm"),
        observations=ObservationConfig(
            joint_state=True,
            imu=True,
            localization="ground_truth",
            state_period=dt,
            camera_period=CAMERA_PERIOD,
            cameras=tuple(CameraPermission(name) for name in cameras),
        ),
    )
    check.record(control, scene["robot"])
    base = BaseLink(scene["robot"], robot.base_link)
    control.reset(settling_seconds=1.0)

    def stow_error() -> float:
        joints = control.observe().joints
        measured = dict(zip(joints.names, joints.positions.tolist(), strict=True))
        return round(max(abs(measured[j] - v) for j, v in zip(robot.arm.joints, robot.arm.stow, strict=True)), 4)

    def planar() -> np.ndarray:
        x, y, _, qx, qy, qz, qw = base.pose().tolist()
        return np.array([x, y, math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))])

    def step(seconds: float) -> None:
        for _ in range(duration_ticks(seconds, dt)):
            control.step()

    check.measure("spawned_stowed_m_or_rad", stow_error(), stow_error() <= STOW_TOLERANCE, "the arm is not stowed")
    speed = DRIVE_FRACTION * robot.navigation.limits["max_speed"]
    yaw_speed = DRIVE_FRACTION * robot.navigation.limits["max_yaw_speed"]
    for name, command, axis in (
        ("forward", (speed, 0.0, 0.0), 0),
        ("left", (0.0, speed, 0.0), 1),
        ("turn", (0.0, 0.0, yaw_speed), 2),
    ):
        if axis == 1 and robot.base.kinematics == "differential":
            continue
        start = planar()
        command_id = control.locomotion.command_velocity(*command, duration=DRIVE_SECONDS)
        step(DRIVE_SECONDS)
        stopping = 0.0  # The base slows down and settles (a legged base stands up) [s].
        while control.locomotion.feedback(command_id).status != "done" and stopping < STOP_SECONDS:
            step(0.1)
            stopping += 0.1
        moved, expected = planar() - start, command[axis] * DRIVE_SECONDS
        along = float(moved[axis])
        across = float(np.abs(np.delete(moved[:2], axis)).max()) if axis < 2 else float(np.abs(moved[:2]).max())
        report.setdefault("drives", {})[name] = dict(command=command, moved=moved.round(4).tolist(), stop_s=stopping)
        check.measure(f"{name}_moved", round(along, 3), 0.5 * expected <= along <= 1.5 * expected, "not as commanded")
        check.measure(f"{name}_straight_m", round(across, 3), across <= 0.25 * abs(expected) + 0.05, "drifted")
        status = control.locomotion.feedback(command_id).status
        check.measure(f"{name}_stopped", f"{status} after {stopping:.1f} s", status == "done", "the base did not stop")
    check.measure("stowed_while_driving_m_or_rad", stow_error(), stow_error() <= STOW_TOLERANCE, "the arm moved")
    client = control.manipulation
    prestow = dict(zip(robot.arm.joints, robot.arm.prestow, strict=True))
    command_id = client.move_joints(prestow, duration=robot.arm.move_seconds)
    step(robot.arm.move_seconds + 2.0)
    feedback = client.feedback(command_id)
    check.measure("arm_to_prestow", feedback.status, feedback.status == "succeeded", feedback.reason)
    for name, command, value in (
        ("close", client.close_gripper, robot.gripper.closed),
        ("open", client.open_gripper, robot.gripper.open),
    ):
        command_id = command()  # The API's duration: within every gripper's speed limit.
        step(4.0)
        feedback = client.feedback(command_id)
        error = round(abs(feedback.joint_positions[robot.gripper.primary] - value), 4)
        check.measure(
            f"gripper_{name}", f"{feedback.status}, {error} off", feedback.status == "succeeded", feedback.reason
        )
    packet = control.observe()
    for name in cameras:
        camera = packet.cameras[name]
        valid = round(float(camera.depth_valid.float().mean()), 3)
        check.measure(f"camera_{name}_depth_fraction", valid, camera.valid and valid > 0.2, "no depth")
        Image.fromarray(camera.rgb.cpu().numpy()[..., :3]).save(args.output / f"camera_{name}.png")
    seen = packet.localization.pose.matrix[:2, 3].tolist()
    error = round(float(np.linalg.norm(np.subtract(seen, planar()[:2]))), 4)
    check.measure("localization_is_the_base_link_m", error, error < 1e-3, "localization is not the base link")


check.run(lambda: run(check.data))
