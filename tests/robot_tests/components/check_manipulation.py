# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Basic arm and gripper control: raise, open and close the gripper, sweep, a free-space wrist pose,
cancel mid-motion and reset. ``--manual`` drives it from the keyboard; ``--record`` saves a
third-person video. No planner, collision avoidance, object grasping or physical attachment.

    ./isaaclab.sh -p scripts/PhyGS/tests/robot_tests/components/check_manipulation.py \
        --output scripts/PhyGS/output_check/$(date +%Y%m%d-%H%M)_manipulation
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # phygs from the checkout.
from phygs_robot_tests import common  # noqa: E402

flags = common.parser(__doc__, manual=True)
flags.add_argument("--config", type=Path, default=common.CONFIGS / "manipulation.yaml")
flags.set_defaults(enable_cameras=True)
args = flags.parse_args()
config = yaml.safe_load(args.config.read_text())  # The check's own settings and inspection camera.
check = common.Check("check_manipulation", args)
robot_description = common.robot(args)  # --robot (default spot).

import isaaclab.sim as sim_utils  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402

from phygs.observations.types import CameraPermission, ObservationConfig  # noqa: E402
from phygs.robots.control import RobotControl  # noqa: E402
from phygs.robots.scene import make_scene_cfg, simulation_cfg  # noqa: E402
from phygs.runtime.episode import RunMode  # noqa: E402
from phygs.runtime.timing import duration_ticks  # noqa: E402
from phygs.simulation.base import BaseLink  # noqa: E402

require = common.require


def run(report):
    dt = common.PHYSICS_DT
    period = config["camera"]["period"]
    camera_ticks = duration_ticks(period, dt)
    eye, target = config["camera"]["eye"], config["camera"]["target"]  # The Kit viewport's view.
    sim = sim_utils.SimulationContext(simulation_cfg(dt=dt, render_interval=camera_ticks, device=args.device))
    scene_arguments = common.robot_scene(robot_description)
    inspection = dict(config["sensors"]["cameras"]["inspection"])  # On the robot's base link.
    inspection["prim_path"] = inspection["prim_path"].format(
        base_link=robot_description.base_link, ENV_REGEX_NS="{ENV_REGEX_NS}"
    )
    scene_arguments["cameras"]["inspection"] = inspection
    scene = InteractiveScene(make_scene_cfg(**scene_arguments)(num_envs=1, env_spacing=3.0, replicate_physics=False))
    sim.reset()
    scene.update(dt)
    sim.set_camera_view(eye, target)
    robot = scene["robot"]
    control = RobotControl.create(
        sim,
        scene,
        robot_description,
        mode=RunMode.HUMAN if args.manual else RunMode.AUTONOMOUS,
        manipulation=True,
        allowed_actions=("motion", "manipulation"),
        observations=ObservationConfig(
            joint_state=True,
            imu=False,
            localization="ground_truth",
            camera_period=period,
            cameras=(CameraPermission("inspection"),),
        ),
    )
    check.record(control, scene["robot"], eye=(1.8, -2.0, 1.2), target=(0.4, 0.0, 0.6))
    client = control.manipulation
    names = (*robot_description.arm.joints, *robot_description.gripper.joints)
    indices = [robot.joint_names.index(n) for n in names]
    wrist = robot.body_names.index(robot_description.arm.wrist_link)
    settings = config["check"]
    # The arm raised (the robot's planning default), and swept: the first arm joint turned from there.
    raised = dict(zip(robot_description.arm.joints, robot_description.planning.default_joints, strict=True))
    first = robot_description.arm.joints[0]
    sweep = {**raised, first: raised[first] + settings["sweep_first_joint_rad"]}
    base = BaseLink(robot, robot_description.base_link)
    frames = []
    action = "standing"
    command = None
    clock = dict(keep_alive=-1.0, printed=0.0)
    report.update(
        config=config,
        collision_avoidance=False,
        grasping=False,
        steps=[],
        camera_resolution=[640, 480],
        camera_fps_sim=1 / period,
        wrist_frame=robot_description.arm.wrist_link,
    )

    def measured():
        return dict(
            joint_positions=dict(zip(names, robot.data.joint_pos.torch[0, indices].tolist(), strict=True)),
            joint_velocities=robot.data.joint_vel.torch[0, indices].tolist(),
            wrist_pose_world_xyzw=robot.data.body_link_pose_w.torch[0, wrist].tolist(),
            base_pose_world_xyzw=robot.data.root_pose_w.torch[0].tolist(),
        )

    def step(seconds):
        for _ in range(duration_ticks(seconds, dt)):
            require(check.app.is_running(), "Simulation closed before completion")
            now = control.runtime.time_seconds
            if now - clock["keep_alive"] >= 0.4:  # The legs stand on.
                control.locomotion.command_velocity(0, 0, 0, duration=0.5)
                clock["keep_alive"] = now
            control.step()
            if control.runtime.tick % camera_ticks == 0:
                packet = control.observe()
                camera = packet.cameras["inspection"]
                require(camera.valid and camera.depth.shape == (480, 640), "Invalid 640x480 depth observation")
                require(camera.stamp.episode_id == control.runtime.episode_id, "Stale camera after reset")
                if frames and frames[-1][0] == camera.stamp.episode_id:
                    require(
                        abs(camera.stamp.time_seconds - frames[-1][1] - period) < 1e-9,
                        "Depth capture skipped a simulated camera interval",
                    )
                frames.append((camera.stamp.episode_id, camera.stamp.time_seconds))
                if robot_description.fallen_height is not None:  # A base on joints cannot fall.
                    require(float(base.pose()[2]) > robot_description.fallen_height, "The robot fell")
                    require(
                        float(robot.data.projected_gravity_b.torch[0, 2]) < settings["maximum_tilt_gravity_z"],
                        "The robot tilted",
                    )
            if control.runtime.time_seconds - clock["printed"] >= 2.0:
                clock["printed"] = control.runtime.time_seconds
                print(
                    json.dumps(dict(action=action, sim_seconds=control.runtime.time_seconds, state=measured())),
                    flush=True,
                )

    def finish(label, ident, expected="succeeded"):
        nonlocal action, command
        action, command = label, ident
        while client.feedback(ident).status in {"active", "stopping"}:
            step(period)
        result = dict(action=label, feedback=asdict(client.feedback(ident)), state=measured())
        report["steps"].append(result)
        print(json.dumps(result), flush=True)
        require(result["feedback"]["status"] == expected, f"{label}: {result['feedback']}")
        require(
            max(abs(v) for v in result["state"]["joint_velocities"])
            <= robot_description.arm.motion["stopped_velocity"],
            "Manipulation reported completion while joints were moving",
        )
        step(0.32)

    started = time.monotonic()
    keyboard = None
    try:
        torch.manual_seed(7)
        control.reset(settling_seconds=settings["settling_seconds"])
        initial = measured()["joint_positions"]
        if not args.manual:
            finish("raise arm", client.move_joints(raised, duration=settings["move_seconds"]))
            finish("open claw", client.open_gripper())
            finish("close claw (no object)", client.close_gripper())
            finish("sweep arm", client.move_joints(sweep, duration=settings["move_seconds"]))
            pose = measured()["wrist_pose_world_xyzw"]
            pose[:3] = (np.array(pose[:3]) + settings["wrist_offset_world_m"]).tolist()
            finish(
                "free-space wrist pose",
                client.move_to_pose(tuple(pose), tool="wrist", duration=settings["move_seconds"]),
            )
            action = "cancel mid-motion"
            ident = client.move_joints(raised, duration=settings["move_seconds"])
            step(0.64)
            before = measured()["joint_positions"]
            client.cancel(ident)
            finish(action, ident, "cancelled")
            drift = max(abs(before[j] - measured()["joint_positions"][j]) for j in names)
            require(drift <= settings["maximum_cancel_drift_rad"], f"Excessive motion after cancel: {drift}")
            report["cancel_drift_rad"] = drift
            ident = client.open_gripper()
            step(0.32)
            action = "reset"
            torch.manual_seed(7)
            control.reset(settling_seconds=settings["settling_seconds"])
            require(client.feedback(ident).status == "cancelled", "Reset did not cancel pending command")
            error = max(abs(initial[j] - measured()["joint_positions"][j]) for j in names)
            report["reset_joint_error_rad"] = error
            require(error <= settings["reset_tolerance_rad"], f"Joint state leaked across reset: {error}")
            finish("raise after reset", client.move_joints(raised, duration=settings["move_seconds"]))
            finish("open after reset", client.open_gripper())
            control.runtime.complete("Basic manipulation checks passed")
        else:
            keyboard = common.Keyboard()
            print(
                "1 raise, 2 sweep, 3 wrist +Z, 4 open, 5 close, SPACE cancel, R reset, "
                "ESC emergency stop. No collision avoidance.",
                flush=True,
            )
            while check.app.is_running():
                key = keyboard.take()
                busy = command is not None and client.feedback(command).status in {"active", "stopping"}
                if key == "ESCAPE":
                    control.runtime.emergency_stop("Human emergency stop")
                    break
                if key == "R":
                    control.reset(settling_seconds=settings["settling_seconds"])
                    command, action = None, "reset"
                elif key == "SPACE" and busy:
                    client.cancel(command)
                elif key in {"KEY_1", "KEY_2", "KEY_3", "KEY_4", "KEY_5"}:
                    if busy:
                        print("Command active; cancel or wait for completion.", flush=True)
                    else:
                        action = key
                        if key in {"KEY_1", "KEY_2"}:
                            command = client.move_joints(
                                raised if key == "KEY_1" else sweep, duration=settings["move_seconds"]
                            )
                        elif key == "KEY_3":
                            pose = measured()["wrist_pose_world_xyzw"]
                            pose[:3] = (np.array(pose[:3]) + settings["wrist_offset_world_m"]).tolist()
                            command = client.move_to_pose(tuple(pose), tool="wrist", duration=settings["move_seconds"])
                        else:
                            command = client.open_gripper() if key == "KEY_4" else client.close_gripper()
                step(period)
            return "manual_finished"
    finally:
        if keyboard is not None:
            keyboard.close()
        control.close()
        gaps = [b[1] - a[1] for a, b in zip(frames, frames[1:]) if a[0] == b[0]]
        report.update(
            camera_frames=len(frames),
            camera_max_gap_sim=max(gaps) if gaps else None,
            wall_seconds=time.monotonic() - started,
            runtime=control.runtime.result(),
        )


check.data["mode"] = "human" if args.manual else "autonomous"
check.run(lambda: run(check.data))
