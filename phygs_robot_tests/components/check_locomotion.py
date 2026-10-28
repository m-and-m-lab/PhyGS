# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Locomotion: forward, lateral and turn tracking, the arm held while walking, stops (expiry and
explicit), and contact with a barrier. ``--manual`` drives Spot from the keyboard while the arm
changes pose from a library.

    ./isaaclab.sh -p scripts/PhyGS/phygs_robot_tests/components/check_locomotion.py \
        --output scripts/PhyGS/output_check/$(date +%Y%m%d-%H%M)_locomotion
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # The PhyGS checkout, without pip install.
from phygs_robot_tests import common  # noqa: E402

flags = common.parser(__doc__, manual=True)
flags.add_argument("--config", type=Path, default=common.CONFIGS / "locomotion.yaml")
args = flags.parse_args()
config = yaml.safe_load(args.config.read_text())  # The test's settings.
check = common.Check("check_locomotion", args)
robot_description = common.robot(args)  # --robot (default spot); its walking policy is checked.
asset = robot_description.usd

import isaaclab.sim as sim_utils  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.assets import AssetBaseCfg  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402
from isaaclab.sensors import ContactSensorCfg  # noqa: E402

from phygs.robots.control import RobotControl  # noqa: E402
from phygs.robots.scene import make_scene_cfg, simulation_cfg  # noqa: E402
from phygs.runtime.episode import RunMode  # noqa: E402
from phygs.runtime.timing import duration_ticks  # noqa: E402

require = common.require
_ARM_POSE_LIBRARY_PATH = common.ROOT / "spot_arm_pose_library.npz"
_ARM_POSE_JOINT_ORDER = ("arm_sh0", "arm_sh1", "arm_el0", "arm_el1", "arm_wr0", "arm_wr1")
_ARM_POSE_PERIOD_SECONDS = 10.0


def _load_arm_pose_library(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        joint_order = tuple(np.asarray(data["joint_order"]).astype(np.str_).tolist())
        joint_pos = np.asarray(data["joint_pos"], dtype=np.float32)

    if joint_order != _ARM_POSE_JOINT_ORDER:
        raise ValueError(f"Unexpected arm-pose joint order in {path}: {joint_order}")
    if joint_pos.ndim != 2 or joint_pos.shape[1] != len(_ARM_POSE_JOINT_ORDER):
        raise ValueError(f"Unexpected arm-pose library shape in {path}: {joint_pos.shape}")
    return joint_pos


def teleop_keys(linear_speed: float, strafe_speed: float, yaw_speed: float) -> dict[str, np.ndarray]:
    """Key -> base velocity (vx, vy, yaw rate) [m/s, m/s, rad/s]; held keys add up."""
    return {
        "NUMPAD_8": np.array([linear_speed, 0.0, 0.0], dtype=np.float32),
        "UP": np.array([linear_speed, 0.0, 0.0], dtype=np.float32),
        "NUMPAD_2": np.array([-linear_speed, 0.0, 0.0], dtype=np.float32),
        "DOWN": np.array([-linear_speed, 0.0, 0.0], dtype=np.float32),
        "NUMPAD_6": np.array([0.0, -strafe_speed, 0.0], dtype=np.float32),
        "RIGHT": np.array([0.0, -strafe_speed, 0.0], dtype=np.float32),
        "NUMPAD_4": np.array([0.0, strafe_speed, 0.0], dtype=np.float32),
        "LEFT": np.array([0.0, strafe_speed, 0.0], dtype=np.float32),
        "NUMPAD_7": np.array([0.0, 0.0, yaw_speed], dtype=np.float32),
        "N": np.array([0.0, 0.0, yaw_speed], dtype=np.float32),
        "NUMPAD_9": np.array([0.0, 0.0, -yaw_speed], dtype=np.float32),
        "M": np.array([0.0, 0.0, -yaw_speed], dtype=np.float32),
    }


def observe(robot):
    return {
        "root_pose_xyzw": robot.data.root_pose_w.torch[0].tolist(),
        "velocity_body": robot.data.root_lin_vel_b.torch[0].tolist(),
        "angular_velocity_body": robot.data.root_ang_vel_b.torch[0].tolist(),
        "projected_gravity_body": robot.data.projected_gravity_b.torch[0].tolist(),
        "joint_positions": robot.data.joint_pos.torch[0].tolist(),
        "joint_velocities": robot.data.joint_vel.torch[0].tolist(),
    }


def create_control():
    dt = common.PHYSICS_DT
    sim = sim_utils.SimulationContext(simulation_cfg(dt=dt, render_interval=10, device=args.device))
    scene_cfg = make_scene_cfg(**common.robot_scene(robot_description))(
        num_envs=1, env_spacing=5.0, replicate_physics=False
    )
    # Immutable kinematic geometry permits contact reporting without dynamic reset writes.
    scene_cfg.barrier = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Barrier",
        spawn=sim_utils.CuboidCfg(
            size=(0.2, 4.0, 1.5),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(config["barrier_x_m"], 0.0, 0.75)),
    )
    scene_cfg.barrier_contact = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Barrier", update_period=dt)
    robot_pos = np.array([0.0, 0.0, robot_description.spawn_height])
    camera = config["camera"]
    sim.set_camera_view(
        eye=(robot_pos + np.asarray(camera["eye_offset"])).tolist(),
        target=(robot_pos + [0.0, 0.0, camera["target_height_offset"]]).tolist(),
    )
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    scene.update(dt)
    robot = scene["robot"]
    require(not robot.is_fixed_base, "The locomotion check requires a floating-base walking robot")
    mode = RunMode.HUMAN if args.manual else RunMode.AUTONOMOUS
    return RobotControl.create(sim, scene, robot_description, mode=mode), scene


def run_auto(control, scene, report):
    dt = control.runtime.physics_dt
    robot = scene["robot"]
    policy = control.locomotion.controller.policy
    report.update(
        config=config,
        physics_dt=dt,
        inference_dt=policy.inference_dt,
        joint_order=policy.leg_joint_names,
        policy_sha256=hashlib.sha256(Path(policy.policy_path).read_bytes()).hexdigest(),
        robot_sha256=hashlib.sha256(asset.read_bytes()).hexdigest(),
        device=args.device,
        information_source="simulator body velocity, gravity, joint state; privileged diagnostic measurements",
        scenarios=[],
    )
    criteria = config["criteria"]
    baseline = None
    for name, command in config["commands"].items():
        torch.manual_seed(config["seed"])
        control.reset(settling_seconds=config["settling_seconds"])
        initial = observe(robot)
        if baseline is None:
            baseline = initial
        error = max(abs(a - b) for key in initial for a, b in zip(initial[key], baseline[key], strict=True))
        require(error <= criteria["reset_max_abs_error"], f"Reset changed settled initial state: {error}")
        entry = {"name": name, "initial": initial, "reset_max_abs_error": error, "command": command, "samples": []}
        report["scenarios"].append(entry)
        motion_seconds = config["collision_seconds"] if name == "collision" else config["motion_seconds"]
        command_id = control.locomotion.command_velocity(*command, duration=motion_seconds)
        control.arm.command_joint_positions({"arm_sh0": config["arm_target_rad"]}, duration=1.0)
        ticks, sample_ticks = duration_ticks(motion_seconds, dt), duration_ticks(config["sample_seconds"], dt)
        peak_contact_force = 0.0
        measurements = []
        for tick in range(ticks):
            control.step()
            force = float(torch.linalg.vector_norm(scene["barrier_contact"].data.net_forces_w.torch))
            peak_contact_force = max(peak_contact_force, force)
            if tick % sample_ticks == 0:
                observation = observe(robot)
                entry["samples"].append({"tick": control.runtime.tick, **observation})
                require(
                    all(torch.isfinite(torch.tensor(values)).all() for values in observation.values()),
                    f"Nonfinite state during {name}",
                )
                require(observation["root_pose_xyzw"][2] > criteria["minimum_height_m"], f"Robot fell during {name}")
                require(
                    observation["projected_gravity_body"][2] < criteria["maximum_projected_gravity_z"],
                    f"Robot tilted excessively during {name}",
                )
                if tick >= ticks // 2:
                    measurements.append(observation["velocity_body"][:2] + observation["angular_velocity_body"][2:3])
        mean = torch.tensor(measurements).mean(dim=0).tolist()
        entry["mean_velocity_body"] = mean
        entry["velocity_error"] = [abs(actual - wanted) for actual, wanted in zip(mean, command, strict=True)]
        entry["peak_barrier_contact_force_n"] = peak_contact_force
        if name == "collision":
            require(peak_contact_force > config["minimum_collision_force_n"], "Robot did not contact the barrier")
            furthest_x = max(sample["root_pose_xyzw"][0] for sample in entry["samples"])
            require(furthest_x < config["barrier_x_m"] - 0.1, f"Robot crossed the barrier: {furthest_x}")
        else:
            require(max(entry["velocity_error"]) <= criteria["velocity_error"], f"Tracking failed for {name}: {mean}")
            moving_axis = next(index for index, value in enumerate(command) if value)
            require(mean[moving_axis] * command[moving_axis] > 0, f"Wrong motion direction for {name}")
        arm_position = float(robot.data.joint_pos.torch[0, robot.joint_names.index("arm_sh0")])
        entry["arm_position_rad"] = arm_position
        require(
            abs(arm_position - config["arm_target_rad"]) < criteria["arm_error_rad"],
            f"Arm target was not maintained while walking: {arm_position}",
        )
        # Forward tests expiration; the remaining scenarios request an explicit stop.
        stop_id = command_id if name == "forward" else control.locomotion.stop()
        for _ in range(duration_ticks(config["stop_observation_seconds"], dt)):
            control.step()
        final = observe(robot)
        speed = float(torch.linalg.vector_norm(torch.tensor(final["velocity_body"][:2])))
        yaw_speed = abs(final["angular_velocity_body"][2])
        entry.update(
            final=final,
            stopped_linear_speed_mps=speed,
            stopped_yaw_speed_radps=yaw_speed,
            feedback=asdict(control.locomotion.feedback(stop_id)),
        )
        require(speed < criteria["stopped_linear_speed_mps"], f"Failed to stop linear motion: {speed}")
        require(yaw_speed < criteria["stopped_yaw_speed_radps"], f"Failed to stop turning: {yaw_speed}")
        require(entry["feedback"]["status"] == "done", f"Stop sequence did not complete: {entry['feedback']}")
        control.runtime.complete(f"{name} diagnostic passed")
        entry["result"] = control.runtime.result()
        print(
            json.dumps(
                {key: value for key, value in entry.items() if key not in {"initial", "final", "samples", "result"}}
            ),
            flush=True,
        )


def run_manual(control, scene, report):
    control.reset(settling_seconds=config["settling_seconds"])
    keyboard, keys = common.Keyboard(), teleop_keys(**config["teleop"])

    def command() -> tuple[float, float, float]:
        total = sum((keys[name] for name in keyboard.held if name in keys), np.zeros(3, dtype=np.float32))
        return tuple(float(value) for value in total)

    arm_poses = _load_arm_pose_library(_ARM_POSE_LIBRARY_PATH)
    rng = np.random.default_rng(config["seed"])
    dt = control.runtime.physics_dt
    arm_ticks = duration_ticks(_ARM_POSE_PERIOD_SECONDS, dt)
    refresh_ticks = duration_ticks(0.1, dt)
    report.update(mode="human", initial=observe(scene["robot"]))
    print("Controls: Up/Down forward-back, Left/Right strafe, N/M yaw. Release keys to stop.", flush=True)
    print("Alternate keys: NUMPAD 8/2, 4/6, 7/9. Arm pose changes every 10 s.", flush=True)
    print("Manual inspection: check motion direction, arm stability, stopping and barrier contact; close to end.")
    print(json.dumps({"initial": report["initial"]}), flush=True)
    try:
        while check.app.is_running():
            if control.runtime.tick % refresh_ticks == 0:
                control.locomotion.command_velocity(*command(), duration=0.5)
            if control.runtime.tick % arm_ticks == 0:
                pose = arm_poses[int(rng.integers(len(arm_poses)))]
                control.arm.command_joint_positions(dict(zip(_ARM_POSE_JOINT_ORDER, pose.tolist(), strict=True)))
            control.step()
            if control.runtime.tick % duration_ticks(1.0, dt) == 0:
                print(
                    json.dumps(
                        {
                            "tick": control.runtime.tick,
                            "command": command(),
                            "observation": observe(scene["robot"]),
                        }
                    ),
                    flush=True,
                )
    finally:
        keyboard.close()
    return "manual_finished"


def main(report):
    report["mode"] = "human" if args.manual else "autonomous"
    control, scene = create_control()
    check.record(control, scene["robot"])
    return run_manual(control, scene, report) if args.manual else run_auto(control, scene, report)


check.run(lambda: main(check.data))
