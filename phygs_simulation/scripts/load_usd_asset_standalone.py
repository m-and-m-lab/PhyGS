#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import argparse
import time
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
parser.add_argument(
    "--usd_path",
    type=str,
    required=True,
    help="Path to the saved USD/USDC stage to open directly.",
)
parser.add_argument("--physics_dt", type=float, default=1.0 / 120.0, help="Physics time-step.")
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
    help="GUI render mode after opening the stage.",
)
parser.add_argument(
    "--stats_interval",
    type=int,
    default=200,
    help="Print performance stats every N simulation steps.",
)

# Append AppLauncher args from IsaacLab
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils


def main() -> None:
    usd_path = _resolve_usd_path(args_cli.usd_path)

    # Open the saved stage directly instead of spawning it as a sub-asset.
    ok = sim_utils.open_stage(usd_path)
    if not ok:
        raise RuntimeError(f"Failed to open stage: {usd_path}")

    render_interval = max(1, round(args_cli.rendering_dt / args_cli.physics_dt))

    # Create a simulation context over the opened stage.
    sim_cfg = sim_utils.SimulationCfg(
        device=args_cli.device,
        dt=args_cli.physics_dt,
        render_interval=render_interval,
    )
    sim = sim_utils.SimulationContext(sim_cfg)

    # Optional viewport camera
    sim.set_camera_view(eye=args_cli.camera_eye, target=args_cli.camera_target)

    # Select render mode explicitly.
    if args_cli.render_mode == "full":
        sim.set_render_mode(sim.RenderMode.FULL_RENDERING)
    elif args_cli.render_mode == "partial":
        sim.set_render_mode(sim.RenderMode.PARTIAL_RENDERING)
    elif args_cli.render_mode == "none":
        sim.set_render_mode(sim.RenderMode.NO_RENDERING)

    sim.reset()

    print("[INFO] Stage ready.")
    print(f"[INFO] Opened stage: {usd_path}")
    print(f"[INFO] Device: {args_cli.device}")
    print(f"[INFO] physics_dt: {args_cli.physics_dt}")
    print(f"[INFO] rendering_dt: {args_cli.rendering_dt}")
    print(f"[INFO] render_interval: {render_interval}")
    print(f"[INFO] render_mode: {args_cli.render_mode}")
    print(f"[INFO] no_render_every_step: {args_cli.no_render_every_step}")
    print(f"[INFO] stats_interval: {args_cli.stats_interval}")

    frame_count = 0
    last_t = time.perf_counter()

    while simulation_app.is_running():
        if args_cli.no_render_every_step:
            sim.step(render=False)
            if frame_count % render_interval == 0:
                sim.render()
        else:
            sim.step(render=True)

        frame_count += 1

        if frame_count % args_cli.stats_interval == 0:
            now = time.perf_counter()
            elapsed = now - last_t
            step_rate = args_cli.stats_interval / elapsed
            print(f"[INFO] step rate: {step_rate:.2f} steps/s over last {args_cli.stats_interval} steps")
            last_t = now


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()