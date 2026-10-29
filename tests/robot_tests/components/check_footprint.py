# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Transit footprint (rung 03): measure the robot's standing outline with the arm stowed
(navigation's rectangle footprint, plus a padding), the collision envelope it sweeps walking, and its
velocity tracking and stopping. A passed run writes the profile navigation needs to
``<output>/transit.json``: copy it to the robot's ``navigation.footprint_profile`` (Spot's:
``phygs/robots/spot/navigation/transit.json``). ``--robot`` picks the robot (default
spot); ``--manual`` drives from a window.

    ./isaaclab.sh -p scripts/PhyGS/tests/robot_tests/components/check_footprint.py \
        --output scripts/PhyGS/output_check/$(date +%Y%m%d-%H%M)_footprint
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # phygs from the checkout.
from phygs_robot_tests import common  # noqa: E402

flags = common.parser(__doc__, manual=True)
flags.add_argument("--config", type=Path, default=common.CONFIGS / "footprint.yaml")
args = flags.parse_args()
config = yaml.safe_load(args.config.read_text())  # The test's settings.
check = common.Check("check_footprint", args)
robot_description = common.robot(args)  # --robot (default spot).

import isaaclab.sim as sim_utils  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402

from phygs.execution.navigation.footprint import FootprintSweep  # noqa: E402
from phygs.robots.control import RobotControl, base_controller_class  # noqa: E402
from phygs.robots.scene import make_scene_cfg, simulation_cfg  # noqa: E402
from phygs.runtime.episode import RunMode  # noqa: E402
from phygs.runtime.timing import duration_ticks  # noqa: E402
from phygs.sim.base import BaseLink  # noqa: E402
from phygs.sim.footprint import CollisionFootprint  # noqa: E402


def run(report):
    dt = common.PHYSICS_DT
    sim = sim_utils.SimulationContext(simulation_cfg(dt=dt, render_interval=10, device=args.device))
    sim.set_camera_view((5, 5, 4), (0, 0, 0.4))
    scene_cfg = make_scene_cfg(**common.robot_scene(robot_description))
    scene = InteractiveScene(scene_cfg(num_envs=1, env_spacing=10, replicate_physics=False))
    sim.reset()
    scene.update(dt)
    mode = RunMode.HUMAN if args.manual else RunMode.AUTONOMOUS
    control = RobotControl.create(sim, scene, robot_description, mode=mode)
    check.record(control, scene["robot"])
    robot = scene["robot"]
    base = BaseLink(robot, robot_description.base_link)
    geometry = CollisionFootprint(sim.stage, robot, "/World/envs/env_0/Robot", base=base)
    sweep = FootprintSweep()
    standing = FootprintSweep()  # The standing outline only.
    arm_posture = robot_description.arm.transit_posture
    arm_indices = [robot.joint_names.index(name) for name, _ in arm_posture]
    arm_reference = torch.tensor([value for _, value in arm_posture], device=robot.device)
    base_files = base_controller_class(robot_description).files(robot_description)
    report.update(
        schema=2,
        configuration=dict(config, robot=dict(name=robot_description.name, usd_path=str(robot_description.usd))),
        geometry_sha256=geometry.geometry_sha256,
        geometry_method="enabled_collision_local_extent_boxes_at_live_link_poses",
        collision_shapes=geometry.colliders,
        physics_dt=dt,
        cases=[],
        arm_posture=arm_posture,
        asset_sha256=hashlib.sha256(robot_description.usd.read_bytes()).hexdigest(),
        # The files the base's motion depends on (a walking policy and its configuration), by key.
        **{key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in base_files.items()},
    )

    def sample():
        points = geometry.points()
        sweep.update(points)
        data = robot.data
        pose = base.pose()
        arm_error = float((data.joint_pos.torch[0, arm_indices] - arm_reference).abs().max())
        fallen = robot_description.fallen_height  # None: the robot cannot fall (a base on joints).
        if fallen is not None and (
            float(pose[2]) < fallen or -float(data.projected_gravity_b.torch[0, 2]) < config["criteria"]["minimum_up_z"]
        ):
            raise AssertionError("Robot fell during footprint measurement")
        if arm_error > config["criteria"]["arm_tolerance"]:
            raise AssertionError("Arm left its declared transit posture")
        velocity = [*base.linear_velocity_b()[:2].tolist(), float(base.angular_velocity_b()[2])]
        return dict(
            time=control.runtime.time_seconds, position=pose[:3].tolist(), velocity=velocity, arm_error=arm_error
        )

    def reset():
        torch.manual_seed(config["seed"])
        control.reset(settling_seconds=config["settling_seconds"])
        sample()

    if args.manual:
        import omni.ui as ui

        window = ui.Window("Footprint measurement", width=420, height=250)
        with window.frame:
            with ui.VStack():
                ui.Label("Body vx, vy [m/s] within +/-1; yaw [rad/s] within +/-1.5")
                fields = [ui.FloatField().model for _ in range(3)]
                ui.Button("Stop", clicked_fn=lambda: [field.set_value(0.0) for field in fields])
                label = ui.Label("Drive to sample the folded-arm envelope")
        reset()
        shown = 0.0
        while check.app.is_running():
            command = tuple(field.as_float for field in fields)
            if not np.isfinite(command).all() or max(map(abs, command[:2])) > 1 or abs(command[2]) > 1.5:
                raise ValueError("Manual measurement commands exceed tested speed limits")
            if control.runtime.tick % duration_ticks(0.1, dt) == 0:
                control.locomotion.command_velocity(*command, duration=0.2)
            control.step()
            state = sample()
            if control.runtime.time_seconds - shown >= 2.0:
                shown = control.runtime.time_seconds
                label.text = f"Radius: {sweep.radius:.3f} m; samples: {sweep.samples}"
                print(json.dumps(dict(state=state, envelope=sweep.result())), flush=True)
        report.update(envelope=sweep.result())
        return "manual_finished"

    def stand(seconds):
        """Sample the standing outline for ``seconds`` [s] with no command."""
        for _ in range(duration_ticks(seconds, dt)):
            control.step()
            sample()
            standing.update(geometry.points())

    reset()
    stand(config["standing_seconds"])
    for name, command in config["commands"].items():
        reset()
        entry = dict(name=name, command=command, samples=[])
        report["cases"].append(entry)
        print(json.dumps(dict(case=name, command=command, initial=sample())), flush=True)
        control.locomotion.command_velocity(*command, duration=config["motion_seconds"] + dt)
        errors, every = [], duration_ticks(config["sample_seconds"], dt)
        for _ in range(duration_ticks(config["motion_seconds"], dt)):
            control.step()
            state = sample()
            if state["time"] >= config["steady_after_seconds"]:
                errors.append(np.asarray(state["velocity"]) - command)
            if control.runtime.tick % every == 0:
                entry["samples"].append(state)
        stop_position = np.asarray(state["position"][:2])
        control.locomotion.stop()
        stop_distance = 0.0
        previous = stop_position
        for _ in range(duration_ticks(config["stop_seconds"], dt)):
            control.step()
            state = sample()
            position = np.asarray(state["position"][:2])
            stop_distance += float(np.linalg.norm(position - previous))
            previous = position
        if (
            np.linalg.norm(state["velocity"][:2]) > config["criteria"]["stopped_speed"]
            or abs(state["velocity"][2]) > config["criteria"]["stopped_yaw_speed"]
        ):
            raise AssertionError(f"{name}: robot did not stop")
        stand(config["standing_seconds"])
        entry.update(
            velocity_rmse=np.sqrt(np.mean(np.square(errors), axis=0)).tolist(),
            velocity_max_abs_error=np.max(np.abs(errors), axis=0).tolist(),
            stop_distance_m=stop_distance,
            final=state,
        )
        print(json.dumps({key: value for key, value in entry.items() if key != "samples"}), flush=True)
    outline = standing.result()
    hull = np.asarray(outline["polygon"])
    report.update(
        envelope=sweep.result(),
        standing=dict(
            bounds=[float(hull[:, 0].min()), float(hull[:, 0].max()), float(hull[:, 1].min()), float(hull[:, 1].max())],
            padding=config["padding"],
            polygon=outline["polygon"],
            samples=outline["samples"],
        ),
    )
    calibration = {key: value for key, value in report.items() if key != "cases"}
    calibration["status"] = "passed"
    calibration["tracking"] = [
        {key: value for key, value in case.items() if key != "samples"} for case in report["cases"]
    ]
    (args.output / "transit.json").write_text(json.dumps(calibration, indent=2) + "\n")
    return "passed"


check.run(lambda: run(check.data))
