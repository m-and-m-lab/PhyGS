#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import time
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


def _resolve_usd_path(path_str: str) -> str:
    """Resolve local file paths and keep non-local (e.g. Nucleus) paths unchanged."""
    if not path_str:
        return path_str
    candidate = Path(path_str).expanduser()
    if candidate.exists():
        return str(candidate.resolve())
    return path_str


parser = argparse.ArgumentParser(
    description="Open an existing USD stage directly in Isaac Lab for GUI-vs-standalone repro."
)
parser.add_argument("--physics_dt", type=float, default=1.0 / 120.0, help="Physics time-step.")
parser.add_argument("--solver", choices=("tgs", "pgs"), default="tgs", help="PhysX constraint solver.")
parser.add_argument("--rendering_dt", type=float, default=1.0 / 60.0, help="Rendering time-step.")
parser.add_argument(
    "--camera_eye",
    type=float,
    nargs=3,
    default=(4.0, 4.0, 3.0),
    metavar=("X", "Y", "Z"),
    help="Viewport camera eye.",
)
parser.add_argument(
    "--camera_target",
    type=float,
    nargs=3,
    default=(0.0, 0.0, 0.5),
    metavar=("X", "Y", "Z"),
    help="Viewport camera target.",
)
parser.add_argument(
    "--no_render_every_step",
    action="store_true",
    help="Step physics with render=False and call sim.render() manually at render_interval.",
)
parser.add_argument(
    "--render_mode",
    type=str,
    default="full",
    choices=("full", "partial", "none"),
    help="full: render normally; none: physics only with --viz none; partial: deprecated and unsupported.",
)
parser.add_argument(
    "--stats_interval",
    type=int,
    default=200,
    help="Print performance stats every N simulation steps.",
)

parser.add_argument("--num_steps", type=int, default=None, help="Exit after N physics steps.")

parser.add_argument(
    "--validate_scene", action="store_true", help="Check rigid-body rest stability each simulated second."
)
parser.add_argument(
    "--exercise_joints", action="store_true", help="Apply forces to open/close every joint after stability checks."
)
parser.add_argument(
    "--drop_objects", action="store_true", help="Force-lift and release table, shelf, and drawer objects."
)
parser.add_argument("--screenshot", type=Path, help="Save the final Kit viewport image.")
parser.add_argument("--report", type=Path, help="Write validation and timing measurements as JSON.")
parser.add_argument(
    "--warmup_steps", type=int, default=360, help="Exclude these initial steps from performance measurements."
)
parser.add_argument(
    "--min_fps", type=float, default=0.0, help="Required measured rendered frames/s; zero disables the threshold."
)

AppLauncher.add_app_launcher_args(parser)
scene_source = parser.add_mutually_exclusive_group(required=True)
scene_source.add_argument("--scene_record", type=Path, help="Prepared ForageBench scene record (offline).")
scene_source.add_argument(
    "--usd_path",
    type=str,
    help="Path to the saved USD/USDC stage to open directly.",
)
args_cli = parser.parse_args()
scene_provenance = None
if args_cli.scene_record:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # phygs from the checkout.
    from phygs.scene.fetch import load_record

    scene_provenance = load_record(args_cli.scene_record)
    args_cli.usd_path = scene_provenance["usd_path"]
    if "--physics_dt" not in sys.argv and not any(arg.startswith("--physics_dt=") for arg in sys.argv):
        args_cli.physics_dt = scene_provenance["physics"]["timestep_seconds"]

if not all(math.isfinite(dt) and dt > 0 for dt in (args_cli.physics_dt, args_cli.rendering_dt)):
    parser.error("--physics_dt and --rendering_dt must be finite and positive.")
if args_cli.stats_interval <= 0 or (args_cli.num_steps is not None and args_cli.num_steps <= 0):
    parser.error("--stats_interval and --num_steps must be positive.")
if args_cli.render_mode == "partial":
    parser.error("--render_mode partial is no longer supported by Isaac Lab; use full or none with --viz none.")
if args_cli.render_mode == "none" and args_cli.visualizer not in (None, ["none"]):
    parser.error("--render_mode none requires --viz none.")

if (args_cli.exercise_joints or args_cli.drop_objects) and not args_cli.validate_scene:
    parser.error("--exercise_joints and --drop_objects require --validate_scene.")
if args_cli.validate_scene and (args_cli.num_steps is None or args_cli.num_steps * args_cli.physics_dt < 5):
    parser.error("--validate_scene requires --num_steps covering at least five simulated seconds.")
if args_cli.warmup_steps < 0 or not math.isfinite(args_cli.min_fps) or args_cli.min_fps < 0:
    parser.error("--warmup_steps and --min_fps must be nonnegative and finite.")
if args_cli.min_fps and args_cli.render_mode == "none":
    parser.error("--min_fps requires rendering.")

# Launch Omniverse app
app_launcher = AppLauncher(args_cli, multi_gpu=False)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab_physx.physics import PhysxCfg


def main() -> None:
    usd_path = _resolve_usd_path(args_cli.usd_path)

    # Open the saved stage directly instead of spawning it as a sub-asset.
    stage = sim_utils.open_stage(usd_path)

    render_interval = max(1, round(args_cli.rendering_dt / args_cli.physics_dt))

    # Create a simulation context over the opened stage.
    sim_cfg = sim_utils.SimulationCfg(
        device=args_cli.device,
        dt=args_cli.physics_dt,
        render_interval=render_interval,
        physics=PhysxCfg(solver_type=1 if args_cli.solver == "tgs" else 0),
    )
    sim = sim_utils.SimulationContext(sim_cfg)

    # Optional viewport camera
    sim.set_camera_view(eye=args_cli.camera_eye, target=args_cli.camera_target)

    if sim.stage is not stage:
        raise RuntimeError("SimulationContext did not use the opened USD stage.")
    sim.reset()

    checks = None
    if args_cli.validate_scene:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scene_generation" / "usd_tools"))
        from scene_lib.scene_checks import SceneChecks

        checks = SceneChecks(sim)
    if args_cli.min_fps and not sim.visualizers:
        raise RuntimeError("FPS validation requires an active renderer; use --viz kit.")

    print("[INFO] Stage ready.")
    print(f"[INFO] Opened stage: {usd_path}")
    print(f"[INFO] Device: {args_cli.device}")
    if args_cli.device.startswith("cuda"):
        print("[INFO] GPU physics uses the tensor API; for viewport mouse dragging relaunch with --device cpu.")
    print(f"[INFO] physics_dt: {args_cli.physics_dt}")
    print(f"[INFO] rendering_dt: {args_cli.rendering_dt}")
    print(f"[INFO] render_interval: {render_interval}")
    print(f"[INFO] render_mode: {args_cli.render_mode}")
    print(f"[INFO] no_render_every_step: {args_cli.no_render_every_step}")
    print(f"[INFO] stats_interval: {args_cli.stats_interval}")

    frame_count = 0
    last_t = time.perf_counter()
    benchmark_start = None
    rendered_frames = 0

    while simulation_app.is_running() and (args_cli.num_steps is None or frame_count < args_cli.num_steps):
        if frame_count == args_cli.warmup_steps:
            benchmark_start = time.perf_counter()
        rendered = False
        if args_cli.render_mode == "none":
            sim.step(render=False)
        elif args_cli.no_render_every_step:
            sim.step(render=False)
            if (frame_count + 1) % render_interval == 0:
                sim.render()
                rendered = sim.is_rendering
        else:
            sim.step(render=True)
            rendered = sim.is_rendering

        if benchmark_start is not None and rendered:
            rendered_frames += 1
        frame_count += 1
        if checks is not None and frame_count % round(1 / args_cli.physics_dt) == 0:
            checks.sample(frame_count * args_cli.physics_dt)

        if frame_count % args_cli.stats_interval == 0:
            now = time.perf_counter()
            elapsed = now - last_t
            step_rate = args_cli.stats_interval / elapsed
            print(f"[INFO] step rate: {step_rate:.2f} steps/s over last {args_cli.stats_interval} steps")
            last_t = now

    elapsed = time.perf_counter() - benchmark_start if benchmark_start is not None else 0.0
    fps = rendered_frames / elapsed if elapsed else 0.0
    if args_cli.num_steps is not None and frame_count != args_cli.num_steps:
        raise RuntimeError(f"Simulation stopped early after {frame_count} of {args_cli.num_steps} steps.")
    report = dict(
        solver=args_cli.solver,
        scene_provenance=scene_provenance,
        physics_steps=frame_count,
        physics_dt=args_cli.physics_dt,
        warmup_steps=args_cli.warmup_steps,
        measured_seconds=elapsed,
        rendered_frames=rendered_frames,
        rendered_fps=fps,
        steps_per_second=(frame_count - args_cli.warmup_steps) / elapsed if elapsed else 0.0,
        device=args_cli.device,
        render_mode=args_cli.render_mode,
    )
    if checks is not None:
        if checks.settled_seconds is None or checks.stable_samples < 3:
            raise AssertionError("Scene did not produce three consecutive stable samples.")
        report.update(settled_wall_seconds=checks.settled_seconds, stability=checks.samples)
        if args_cli.exercise_joints:
            report["joints"] = checks.exercise_joints()
        if args_cli.drop_objects:
            report["drops"] = checks.drop_objects()
    if args_cli.screenshot:
        from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport

        viewport = get_active_viewport()
        if viewport is None:
            raise RuntimeError("--screenshot requires an active Kit viewport.")
        args_cli.screenshot.parent.mkdir(parents=True, exist_ok=True)
        capture = capture_viewport_to_file(viewport, str(args_cli.screenshot))
        task = asyncio.ensure_future(capture.wait_for_result())
        deadline = time.perf_counter() + 30
        while not task.done():
            if time.perf_counter() > deadline:
                raise TimeoutError("Viewport capture did not finish in 30 seconds.")
            sim.render()
        task.result()
        report["viewport_resolution"] = list(viewport.resolution)
    if args_cli.report:
        args_cli.report.parent.mkdir(parents=True, exist_ok=True)
        args_cli.report.write_text(json.dumps(report, indent=2) + "\n")
    print(f"[RESULT] {report}", flush=True)
    if fps < args_cli.min_fps:
        raise AssertionError(f"Rendered FPS {fps:.2f} is below required {args_cli.min_fps:.2f}.")
    print(f"[INFO] Completed {frame_count} physics steps.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close(exit_code=1 if sys.exc_info()[0] is not None else 0)
