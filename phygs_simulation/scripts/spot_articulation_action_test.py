#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
from omni.isaac.kit import SimulationApp


ARM_JOINTS_DEFAULT = ["arm_sh0", "arm_sh1", "arm_el0", "arm_el1", "arm_wr0", "arm_wr1"]


def _resolve_usd_path(path_str: str) -> str:
    """Resolve local paths; leave non-local (e.g., Nucleus) paths unchanged."""
    if not path_str:
        return path_str
    candidate = Path(path_str)
    if candidate.is_absolute() and candidate.exists():
        return str(candidate)
    if candidate.exists():
        return str(candidate.resolve())
    repo_root = Path(__file__).resolve().parents[3]
    repo_candidate = repo_root / candidate
    if repo_candidate.exists():
        return str(repo_candidate.resolve())
    return path_str


def _build_targets(base: np.ndarray, delta: float) -> list[np.ndarray]:
    signs = np.ones_like(base, dtype=float)
    signs[1::2] = -1.0
    offsets = signs * float(delta)
    return [base, base + offsets, base - offsets]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Load Spot USD and apply ArticulationAction at fixed intervals to test direct control."
        )
    )
    parser.add_argument(
        "--usd_path",
        type=str,
        default="scripts/interactive-search/spot_model/spot_arm_w_cam.usd",
        help="Robot USD file to reference (local path or Nucleus path).",
    )
    parser.add_argument(
        "--robot_prim_path",
        type=str,
        default="/World/spot_arm_w_cam",
        help="Prim path for the robot articulation.",
    )
    parser.add_argument(
        "--joint_names",
        type=str,
        nargs="+",
        default=ARM_JOINTS_DEFAULT,
        help="Joint names to command (order matters).",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=0.35,
        help="Joint offset (radians) applied around the current pose.",
    )
    parser.add_argument(
        "--interval_s",
        type=float,
        default=1.0,
        help="Seconds to hold each target before switching.",
    )
    parser.add_argument(
        "--num_cycles",
        type=int,
        default=3,
        help="Number of full target cycles to run.",
    )
    parser.add_argument(
        "--physics_dt",
        type=float,
        default=1.0 / 240.0,
        help="Physics timestep.",
    )
    parser.add_argument(
        "--rendering_dt",
        type=float,
        default=1.0 / 60.0,
        help="Rendering timestep.",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=60,
        help="Steps to run before sending commands.",
    )
    parser.add_argument("--headless", action="store_true", help="Run without a GUI window.")
    parser.add_argument(
        "--renderer",
        type=str,
        default="RayTracedLighting",
        help="Renderer for Isaac Sim (ignored in headless).",
    )
    args = parser.parse_args()

    simulation_app = SimulationApp(launch_config={"renderer": args.renderer, "headless": args.headless})

    try:
        from omni.isaac.core import World
        from omni.isaac.core.articulations import Articulation
        from omni.isaac.core.utils.stage import add_reference_to_stage
        from omni.isaac.core.utils.types import ArticulationAction

        usd_path = _resolve_usd_path(args.usd_path)

        world = World(physics_dt=args.physics_dt, rendering_dt=args.rendering_dt)
        world.scene.add_default_ground_plane()
        add_reference_to_stage(usd_path, args.robot_prim_path)
        world.reset()

        robot = Articulation(args.robot_prim_path)
        robot.initialize()

        print("Articulation initialized")
        print("   DOFs:", robot.num_dof)
        print("   DOF names:", robot.dof_names)

        name_to_idx = {name: idx for idx, name in enumerate(robot.dof_names)}
        missing = [name for name in args.joint_names if name not in name_to_idx]
        if missing:
            print("Missing joint names:", missing)
            print("   Available DOFs:", robot.dof_names)
            return

        joint_indices = np.array([name_to_idx[name] for name in args.joint_names], dtype=int)
        base = robot.get_joint_positions()[joint_indices]
        targets = _build_targets(base, args.delta)

        interval_steps = max(1, int(round(args.interval_s / args.physics_dt)))
        total_steps = interval_steps * args.num_cycles * len(targets)

        print("Command joints:", args.joint_names)
        print("   Base pose:", np.round(base, 3))
        print("   Interval steps:", interval_steps)
        print("   Total steps:apply_action", total_steps)

        for _ in range(args.warmup_steps):
            world.step(render=not args.headless)

        step = 0
        action = None
        while simulation_app.is_running() and step < total_steps:
            if step % interval_steps == 0:
                target_idx = (step // interval_steps) % len(targets)
                q_full = robot.get_joint_positions()
                q_full[joint_indices] = targets[target_idx]
                action = ArticulationAction(joint_positions=q_full)
                print(f"[step {step}] apply target {target_idx}: {np.round(targets[target_idx], 3)}")

            if action is not None:
                robot.apply_action(action)
            world.step(render=not args.headless)
            step += 1

    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
