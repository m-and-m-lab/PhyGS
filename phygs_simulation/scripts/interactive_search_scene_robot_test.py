#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Interactive Search Scene + Robot Load Test (Standalone)

Loads a USD scene, spawns a robot, applies rigid-body/collision setup to
scene assets, and runs a continuous simulation loop.

This script intentionally excludes manipulation skills and CuRobo logic so
scene + robot performance can be isolated.

.. code-block:: bash

  ./isaaclab.sh -p scripts/interactive-search/scripts/interactive_search_scene_robot_test.py \\
    --enable_cameras \\
    --scene_usd scripts/bedroom1/export_scene.blend/export_scene.usdc \\
    --robot_usd scripts/interactive-search/spot_model/spot_arm_w_cam.usd
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import time
from pathlib import Path

from isaaclab.app import AppLauncher


_DEFAULT_CUROBO_RETRACT_JOINT_POS = {
    "arm_sh0": 0.0,
    "arm_sh1": -3.14,
    "arm_el0": 3.14,
    "arm_el1": 0.0,
    "arm_wr0": 0.0,
    "arm_wr1": 0.0,
    "arm_f1x": -1.56,
}


def _resolve_usd_path(path_str: str) -> str:
    """Resolve local file paths; keep non-local (e.g., Nucleus) paths unchanged."""
    if not path_str:
        return path_str
    candidate = Path(path_str)
    if candidate.exists():
        return str(candidate.resolve())
    return path_str


parser = argparse.ArgumentParser(description="Load a USD scene and spawn a robot for FPS isolation tests.")

parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--env_spacing", type=float, default=20.0)

# Scene/robot inputs
parser.add_argument(
    "--scene_usd",
    type=str,
    default="/workspace/isaaclab/scripts/.scene_usd/indoor_2_highres/indoor_2/export_scene.blend/export_scene_sorted_physics.usdc",
    help="USD/USDC file to reference as the environment scene (local path or Nucleus path).",
)
parser.add_argument(
    "--robot_usd",
    type=str,
    default="scripts/interactive-search/spot_model/spot_arm_w_cam.usd",
    help="Robot USD file to reference (local path or Nucleus path).",
)
parser.add_argument("--robot_prim_path", type=str, default="/World/Robot", help="Prim path for the robot articulation.")
parser.add_argument(
    "--robot_pos",
    type=float,
    nargs=3,
    default=(10.2, 3.2, 0.75),
    metavar=("X", "Y", "Z"),
    help="Robot base position in world frame (meters).",
)
parser.add_argument(
    "--robot_rot",
    type=float,
    nargs=4,
    default=(1.0, 0.0, 0.0, 0.0),
    metavar=("W", "X", "Y", "Z"),
    help="Robot base orientation quaternion in world frame (w, x, y, z).",
)

# Simulation toggles
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable Fabric (use USD I/O for state reads)."
)
parser.add_argument(
    "--fix_root",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Fix the robot root link to the world.",
)
parser.add_argument(
    "--robot_disable_gravity",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Disable gravity on the robot rigid bodies.",
)
parser.add_argument(
    "--perf_log_interval",
    type=float,
    default=2.0,
    help="Wall-clock seconds between FPS/RTF performance log lines.",
)
parser.add_argument(
    "--perf_warmup_steps",
    type=int,
    default=120,
    help="Number of initial simulation steps to exclude from FPS/RTF measurements.",
)
parser.add_argument(
    "--max_steps",
    type=int,
    default=0,
    help="Stop after this many simulation steps. Use 0 to run until the app exits.",
)
parser.add_argument(
    "--render_in_headless",
    action="store_true",
    default=False,
    help="Call sim.step(render=True) even when running with --headless.",
)

# Robot articulation config
parser.add_argument(
    "--arm_joints",
    type=str,
    nargs="+",
    default=["arm_sh0", "arm_sh1", "arm_el0", "arm_el1", "arm_wr0", "arm_wr1"],
    help="Arm joint names (order matters) to configure arm actuators.",
)
parser.add_argument(
    "--gripper_joints",
    type=str,
    nargs="+",
    default=["arm_f1x"],
    help="Gripper joint names to configure gripper actuators.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import carb

import isaaclab.sim as isaaclab_sim
import isaaclab.sim.utils as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.utils import configclass
from pxr import PhysxSchema, UsdGeom, UsdPhysics



def _make_scene_cfg() -> type[InteractiveSceneCfg]:
    scene_usd = _resolve_usd_path(args_cli.scene_usd)
    robot_usd = _resolve_usd_path(args_cli.robot_usd)

    @configclass
    class InteractiveSearchSceneRobotTestCfg(InteractiveSceneCfg):
        # ground plane
        ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=isaaclab_sim.GroundPlaneCfg())

        light = AssetBaseCfg(
            prim_path="/World/Light",
            spawn=isaaclab_sim.DomeLightCfg(intensity=2500.0, color=(0.75, 0.75, 0.75)),
        )

        robot = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=isaaclab_sim.UsdFileCfg(
                usd_path=robot_usd,
                activate_contact_sensors=True,
                rigid_props=isaaclab_sim.RigidBodyPropertiesCfg(disable_gravity=args_cli.robot_disable_gravity),
                articulation_props=isaaclab_sim.ArticulationRootPropertiesCfg(fix_root_link=args_cli.fix_root),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=tuple(args_cli.robot_pos),
                rot=tuple(args_cli.robot_rot),
                # Keep Spot knees in a valid range to avoid startup articulation validation failures.
                joint_pos={
                    ".*_knee": -1.5,
                    **_DEFAULT_CUROBO_RETRACT_JOINT_POS,
                },
            ),
            actuators={
                "arm": ImplicitActuatorCfg(
                    joint_names_expr=args_cli.arm_joints,
                    effort_limit_sim=200.0,
                    stiffness=400.0,
                    damping=80.0,
                ),
                "gripper": ImplicitActuatorCfg(
                    joint_names_expr=args_cli.gripper_joints,
                    effort_limit_sim=200.0,
                    stiffness=2e3,
                    damping=1e2,
                ),
            },
        )

        if scene_usd:
            environment = AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/Scene",
                spawn=isaaclab_sim.UsdFileCfg(usd_path=scene_usd),
            )

    return InteractiveSearchSceneRobotTestCfg


def _reset_robot(scene: InteractiveScene) -> None:
    robot = scene["robot"]
    root_state = robot.data.default_root_state.clone()
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.reset()


# Set up rigid body and collision APIs; SDF provides better collision handling for complex meshes and physics stability.
def _set_rigid_body_and_colliders(
    prim_path: str, objects_collision_approximation: str = "sdf", structure_collision_approximation: str = "none") -> None:
    stage = get_current_stage()
    objects = stage.GetPrimAtPath(f"{prim_path}/objects")
    if objects and objects.IsValid():
        for child_prim in objects.GetChildren():
            if child_prim.IsA(UsdGeom.Xform):
                child_path = child_prim.GetPath().pathString
                print(f"Defining rigid body for child xform prim: {child_path}")
                isaaclab_sim.schemas.define_rigid_body_properties(
                    child_path,
                    isaaclab_sim.schemas.RigidBodyPropertiesCfg(rigid_body_enabled=True, disable_gravity=False),
                )
                # isaaclab_sim.schemas.define_mass_properties(
                #     prim_path,
                #     isaaclab_sim.schemas.MassPropertiesCfg(mass=10.0),
                # )

        objects_mesh_prims = sim_utils.get_all_matching_child_prims(
            objects.GetPath().pathString,
            predicate=lambda prim: prim.IsA(UsdGeom.Mesh),
        )

        for mesh_prim in objects_mesh_prims:
            if "exterior" in mesh_prim.GetName():
                continue
            print(f"Defining collision for mesh prim: {mesh_prim.GetPath().pathString}")
            isaaclab_sim.schemas.define_collision_properties(
                mesh_prim.GetPath(),
                isaaclab_sim.schemas.CollisionPropertiesCfg(collision_enabled=True),
            )
            mesh_collider = UsdPhysics.MeshCollisionAPI.Apply(mesh_prim)
            mesh_collider.CreateApproximationAttr().Set(objects_collision_approximation)

            if objects_collision_approximation == "convexDecomposition":
                collision_api = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(mesh_prim)
                collision_api.CreateHullVertexLimitAttr().Set(64)
                collision_api.CreateMaxConvexHullsAttr().Set(64)
                collision_api.CreateMinThicknessAttr().Set(0.001)
                collision_api.CreateShrinkWrapAttr().Set(True)
                collision_api.CreateErrorPercentageAttr().Set(0.1)
            elif objects_collision_approximation == "convexHull":
                collision_api = PhysxSchema.PhysxConvexHullCollisionAPI.Apply(mesh_prim)
                collision_api.CreateHullVertexLimitAttr().Set(64)
                collision_api.CreateMinThicknessAttr().Set(0.00001)
            elif objects_collision_approximation == "sdf":
                collision_api = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(mesh_prim)
                collision_api.CreateSdfResolutionAttr().Set(64)
            elif objects_collision_approximation == "none":
                PhysxSchema.PhysxTriangleMeshCollisionAPI.Apply(mesh_prim)

    structures = stage.GetPrimAtPath(f"{prim_path}/structure")
    if structures and structures.IsValid():
        structure_mesh_prims = sim_utils.get_all_matching_child_prims(
            structures.GetPath().pathString,
            predicate=lambda prim: prim.IsA(UsdGeom.Mesh),
        )

        for mesh_prim in structure_mesh_prims:
            if "exterior" in mesh_prim.GetName():
                continue
            print(f"Defining collision for mesh prim: {mesh_prim.GetPath().pathString}")
            isaaclab_sim.schemas.define_collision_properties(
                mesh_prim.GetPath(),
                isaaclab_sim.schemas.CollisionPropertiesCfg(collision_enabled=True),
            )
            mesh_collider = UsdPhysics.MeshCollisionAPI.Apply(mesh_prim)
            mesh_collider.CreateApproximationAttr().Set(structure_collision_approximation)

            if structure_collision_approximation == "convexDecomposition":
                collision_api = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(mesh_prim)
                collision_api.CreateHullVertexLimitAttr().Set(64)
                collision_api.CreateMaxConvexHullsAttr().Set(64)
                collision_api.CreateMinThicknessAttr().Set(0.001)
                collision_api.CreateShrinkWrapAttr().Set(True)
                collision_api.CreateErrorPercentageAttr().Set(0.1)
            elif structure_collision_approximation == "convexHull":
                collision_api = PhysxSchema.PhysxConvexHullCollisionAPI.Apply(mesh_prim)
                collision_api.CreateHullVertexLimitAttr().Set(64)
                collision_api.CreateMinThicknessAttr().Set(0.00001)
            elif structure_collision_approximation == "sdf":
                collision_api = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(mesh_prim)
                collision_api.CreateSdfResolutionAttr().Set(256)
            elif structure_collision_approximation == "none":
                PhysxSchema.PhysxTriangleMeshCollisionAPI.Apply(mesh_prim)


def run_simulator(sim: isaaclab_sim.SimulationContext, scene: InteractiveScene) -> None:
    sim_dt = sim.get_physics_dt()
    render_this_step = (not args_cli.headless) or args_cli.render_in_headless
    log_interval = max(args_cli.perf_log_interval, 1.0e-6)
    warmup_steps = max(args_cli.perf_warmup_steps, 0)
    max_steps = max(args_cli.max_steps, 0)

    print("[INFO] Running scene + robot load test. Close the app window (or stop process) to exit.")
    print(
        "[DEBUG] Runtime mode: "
        f"headless={args_cli.headless}, render_each_step={render_this_step}, "
        f"device={args_cli.device}, fabric_enabled={not args_cli.disable_fabric}"
    )
    print(
        "[DEBUG] Timing config: "
        f"physics_dt={sim_dt:.6f}s ({1.0 / sim_dt:.1f} Hz), "
        f"warmup_steps={warmup_steps}, log_interval={log_interval:.2f}s, max_steps={max_steps or 'unlimited'}"
    )

    total_steps = 0
    measured_steps = 0
    window_steps = 0
    perf_start_time = time.perf_counter()
    window_start_time = perf_start_time

    try:
        while simulation_app.is_running():
            scene.write_data_to_sim()
            sim.step(render=render_this_step)
            scene.update(sim_dt)

            total_steps += 1
            now = time.perf_counter()

            if total_steps <= warmup_steps:
                if total_steps == warmup_steps:
                    perf_start_time = now
                    window_start_time = now
                    print(f"[DEBUG] Perf warmup complete at step {total_steps}; starting FPS/RTF measurement.")
                if max_steps and total_steps >= max_steps:
                    print(f"[INFO] Reached --max_steps={max_steps}; exiting simulation loop.")
                    break
                continue

            measured_steps += 1
            window_steps += 1
            window_wall_time = now - window_start_time

            if window_wall_time >= log_interval:
                avg_wall_time = max(now - perf_start_time, 1.0e-9)
                actual_fps = window_steps / window_wall_time
                actual_rtf = (window_steps * sim_dt) / window_wall_time
                avg_fps = measured_steps / avg_wall_time
                avg_rtf = (measured_steps * sim_dt) / avg_wall_time
                print(
                    "[PERF] "
                    f"step={total_steps} measured_steps={measured_steps} "
                    f"actual_fps={actual_fps:.2f} actual_rtf={actual_rtf:.3f} "
                    f"avg_fps={avg_fps:.2f} avg_rtf={avg_rtf:.3f} "
                    f"sim_time={measured_steps * sim_dt:.2f}s wall_time={avg_wall_time:.2f}s"
                )
                window_start_time = now
                window_steps = 0

            if max_steps and total_steps >= max_steps:
                print(f"[INFO] Reached --max_steps={max_steps}; exiting simulation loop.")
                break
    finally:
        if measured_steps > 0:
            now = time.perf_counter()
            avg_wall_time = max(now - perf_start_time, 1.0e-9)
            print(
                "[PERF] Final summary: "
                f"total_steps={total_steps} measured_steps={measured_steps} "
                f"avg_fps={measured_steps / avg_wall_time:.2f} "
                f"avg_rtf={(measured_steps * sim_dt) / avg_wall_time:.3f} "
                f"sim_time={measured_steps * sim_dt:.2f}s wall_time={avg_wall_time:.2f}s"
            )
        else:
            print(f"[PERF] Final summary: no measured steps after warmup_steps={warmup_steps}.")


def main() -> None:
    sim_cfg = isaaclab_sim.SimulationCfg(dt=1.0 / 120.0, render_interval=1, device=args_cli.device)
    sim_cfg.use_fabric = not args_cli.disable_fabric
    sim = isaaclab_sim.SimulationContext(sim_cfg)
    carb.settings.get_settings().set_int("/persistent/simulation/minFrameRate", 15)
    physics_context = sim.get_physics_context()
    physics_context.enable_gpu_dynamics(True)

    cam_target = list(args_cli.robot_pos)
    cam_target[2] += 0.6
    sim.set_camera_view(eye=[cam_target[0] + 2.0, cam_target[1] + 2.0, cam_target[2] + 1.5], target=cam_target)

    scene_cfg = _make_scene_cfg()(num_envs=args_cli.num_envs, env_spacing=args_cli.env_spacing, replicate_physics=False)
    scene = InteractiveScene(scene_cfg)
    print(
        "[DEBUG] Scene config: "
        f"num_envs={args_cli.num_envs}, env_spacing={args_cli.env_spacing}, "
        f"scene_usd={_resolve_usd_path(args_cli.scene_usd)}, robot_usd={_resolve_usd_path(args_cli.robot_usd)}"
    )

    if _resolve_usd_path(args_cli.scene_usd):
        for i in range(args_cli.num_envs):
            prim_path = f"/World/envs/env_{i}/Scene"
            _set_rigid_body_and_colliders(prim_path)
            print("Set rigid body and colliders for env:", prim_path)

    sim.reset()
    print("[INFO] Setup complete.")
    _reset_robot(scene)
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
