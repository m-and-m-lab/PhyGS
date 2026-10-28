# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""World and runtime lifecycle: reset restores the state, ownership and acquire, command expiry,
emergency stop and episode isolation, over several episodes (no walking policy: the legs hold a
fixed posture).

    ./isaaclab.sh -p scripts/PhyGS/phygs_robot_tests/components/check_world_lifecycle.py \
        --output scripts/PhyGS/output_check/$(date +%Y%m%d-%H%M)_world_lifecycle
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # The PhyGS checkout, without pip install.
from phygs_robot_tests import common  # noqa: E402

flags = common.parser(__doc__)
flags.add_argument("--config", type=Path, default=common.CONFIGS / "locomotion.yaml")
flags.add_argument("--cycles", type=int, default=3)
args = flags.parse_args()
if args.cycles < 2:
    flags.error("--cycles must be at least two to test episode isolation.")
config = yaml.safe_load(args.config.read_text())  # The test's settings.
check = common.Check("check_world_lifecycle", args)
robot_description = common.robot(args)  # --robot (default spot).
asset = robot_description.usd

import isaaclab.sim as sim_utils  # noqa: E402
import torch  # noqa: E402
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402

from phygs.robots.scene import make_scene_cfg, simulation_cfg  # noqa: E402
from phygs.runtime.episode import ControllerDeclaration, EpisodeRuntime, RunMode  # noqa: E402
from phygs.runtime.timing import duration_ticks  # noqa: E402
from phygs.simulation.world import IsaacLabWorld  # noqa: E402

require = common.require
# Durations [s]: settling after reset, the arm command, how long it is observed (past its expiry),
# and the whole-body command before the emergency stop.
SETTLE_S, COMMAND_S, OBSERVE_S, WHOLE_S = 1.0, 0.6, 0.7, 0.02


class ArmSweep:
    """Small joint-space diagnostic initialized from current arm position."""

    def __init__(self, name, joints, swept: str, source=RunMode.AUTONOMOUS):
        self.declaration = ControllerDeclaration(name, joints, source)
        self.swept = swept  # The joint it moves (the arm's first).

    def reset(self, positions):
        self.initial = positions
        self.elapsed = 0.0

    def step(self, dt):
        self.elapsed += dt
        targets = self.initial.copy()
        if self.swept in targets:
            targets[self.swept] += 0.2 * min(self.elapsed / 0.5, 1.0)
        return targets


def json_state(scene):
    return {
        kind: {name: {field: value.tolist() for field, value in fields.items()} for name, fields in entities.items()}
        for kind, entities in scene.get_state().items()
    }


def run(report):
    dt = common.PHYSICS_DT
    sim = sim_utils.SimulationContext(simulation_cfg(dt=dt, device=args.device))
    cfg = make_scene_cfg(**common.robot_scene(robot_description))(num_envs=1, env_spacing=5.0, replicate_physics=False)
    # Deterministic posture holding isolates lifecycle checks from learned walking.
    for name in ("spot_legs", "spot_hip", "spot_knee"):  # Either actuator layout of Spot's (robots/spot).
        if name not in cfg.robot.actuators:
            continue
        cfg.robot.actuators[name].min_delay = 0
        cfg.robot.actuators[name].max_delay = 0
        cfg.robot.actuators[name].stiffness = 800.0
        cfg.robot.actuators[name].damping = 25.0
    cfg.obstacle = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle",
        spawn=sim_utils.CuboidCfg(
            size=(0.2, 0.2, 0.2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(2.0, 0.0, 0.5)),
    )
    for name, y in (("wall_left", 1.0), ("wall_right", -1.0)):
        setattr(
            cfg,
            name,
            AssetBaseCfg(
                prim_path=f"{{ENV_REGEX_NS}}/{name}",
                spawn=sim_utils.CuboidCfg(size=(4.0, 0.1, 1.0), collision_props=sim_utils.CollisionPropertiesCfg()),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(1.0, y, 0.5)),
            ),
        )
    scene = InteractiveScene(cfg)
    sim.reset()
    scene.update(sim.get_physics_dt())
    robot = scene["robot"]
    require(not robot.is_fixed_base, "Lifecycle diagnostic requires a floating-base robot")
    world = IsaacLabWorld(sim, scene)
    arm, swept = robot_description.arm.joints, robot_description.arm.joints[0]
    require(
        set(arm) <= set(world.joint_names), f"The robot lacks arm joints {sorted(set(arm) - set(world.joint_names))}"
    )
    controllers = (
        ArmSweep("arm", arm, swept),
        ArmSweep("whole", world.joint_names, swept),
        ArmSweep("human", arm, swept, RunMode.HUMAN),
    )
    runtime = EpisodeRuntime(world, controllers, mode=RunMode.AUTONOMOUS)
    command, observed = duration_ticks(COMMAND_S, dt), duration_ticks(OBSERVE_S, dt)
    report.update(
        physics_dt=world.physics_dt,
        physics_backend=scene.physics_backend,
        device=args.device,
        scene_config=config,
        frames={
            "world": "right-handed, Z up",
            "root_pose": "xyz [m], quaternion xyzw",
            "joint_positions": "rad",
            "joint_velocities": "rad/s",
        },
        posture_gains={"leg_stiffness": 800.0, "leg_damping": 25.0, "delay_ticks": 0},
        robot_asset=str(asset),
        robot_sha256=hashlib.sha256(asset.read_bytes()).hexdigest(),
        joint_names=world.joint_names,
        raw_initial_state=json_state(scene),
        criteria={
            "reset_max_abs_error": 1e-6,
            "obstacle_height_m": [0.08, 0.13],
            "robot_min_height_m": 0.25,
            "arm_min_motion_rad": 0.05,
        },
        episodes=[],
    )
    baseline = scene.get_state()
    for cycle in range(args.cycles):
        runtime.reset()
        restored = scene.get_state()
        error = max(
            float((restored[kind][name][field] - value).abs().max())
            for kind, entities in baseline.items()
            for name, fields in entities.items()
            for field, value in fields.items()
        )
        require(error <= 1e-6, f"Reset mismatch: {error}")
        require(all(controller.elapsed == 0 for controller in controllers), "Controller memory survived reset")
        runtime.reset(settling_ticks=duration_ticks(SETTLE_S, dt))
        entry = {"cycle": cycle, "reset_max_abs_error": error, "settled_state": json_state(scene)}
        report["episodes"].append(entry)
        runtime.acquire("arm", duration_ticks=command)
        try:
            runtime.acquire("whole", duration_ticks=command)
        except RuntimeError:
            entry["conflict_rejected"] = True
        else:
            raise AssertionError("Overlapping whole-body command was accepted")
        try:
            runtime.acquire("human", duration_ticks=command)
        except PermissionError:
            entry["human_rejected"] = True
        else:
            raise AssertionError("Human command was accepted during autonomous run")
        start_arm = world.positions()[swept]
        entry["samples"] = []
        sample = duration_ticks(0.2, dt)
        for tick in range(observed):
            runtime.step()
            if tick % sample == 0 or tick == observed - 1:
                entry["samples"].append(
                    {
                        "tick": runtime.tick,
                        "time_seconds": runtime.time_seconds,
                        "joint_positions_rad": world.positions(),
                        "arm_controller_elapsed_seconds": controllers[0].elapsed,
                    }
                )
        entry["action"] = {
            "controller": "arm",
            "joint": swept,
            "delta_rad": 0.2,
            "ramp_seconds": 0.5,
            "duration_seconds": COMMAND_S,
        }
        state = scene.get_state()
        require(
            all(
                torch.isfinite(value).all().item()
                for entities in state.values()
                for fields in entities.values()
                for value in fields.values()
            ),
            "Nonfinite physical state",
        )
        obstacle_z = float(state["rigid_object"]["obstacle"]["root_pose"][0, 2])
        robot_z = float(state["articulation"]["robot"]["root_pose"][0, 2])
        arm_motion = abs(world.positions()[swept] - start_arm)
        entry.update(obstacle_height_m=obstacle_z, robot_height_m=robot_z, arm_motion_rad=arm_motion)
        require(0.08 <= obstacle_z <= 0.13, f"Gravity/ground collision failed: obstacle z={obstacle_z}")
        require(robot_z > 0.25, f"Robot fell: z={robot_z}")
        require(arm_motion > 0.05, f"Arm did not move: {arm_motion}")
        require(
            any(event.kind == "expired" for event in runtime.events if event.episode_id == runtime.episode_id),
            "Command did not expire",
        )
        whole = duration_ticks(WHOLE_S, dt)
        runtime.acquire("whole", duration_ticks=whole)
        for _ in range(whole):
            runtime.step()
        entry["final_state"] = json_state(scene)
        runtime.emergency_stop("diagnostic emergency stop")
        stopped_tick = runtime.tick
        try:
            runtime.step()
        except RuntimeError:
            entry["resume_rejected"] = True
        else:
            raise AssertionError("Interrupted episode resumed")
        require(runtime.tick == stopped_tick, "Clock advanced after emergency stop")
        entry["result"] = runtime.result()
        # Deliberately perturb the object to prove the next episode restores it.
        pose = scene["obstacle"].data.root_pose_w.torch.clone()
        pose[:, 0] += 0.7
        scene["obstacle"].write_root_pose_to_sim_index(root_pose=pose)
        print(
            json.dumps(
                {key: value for key, value in entry.items() if key not in ("settled_state", "final_state", "samples")}
            ),
            flush=True,
        )


check.run(lambda: run(check.data))
