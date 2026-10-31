# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render a finished run again from its state log: the run itself renders nothing it does not need.

    ./isaaclab.sh -p scripts/PhyGS/examples/replay_run.py \
        scripts/PhyGS/output_check/<YYYYMMDD-HHMM>_grid_table --headless

Writes into ``<run>/replay`` (or ``--output``):

- ``replay.mp4``, as a recording looks (recording.video): a camera that follows the robot with the
  panels on its right (front camera, hand camera, navigation, their depth, arm planner), and the
  overview strip under both: the clock, what the gripper holds, the arm's contact force, the agent's
  step and the skills' latest line, the floor plan with the robot's path and Nav2's route, and the
  timeline of the agent's steps and the robot's commands;
- with ``--views``, one clip per view, frame for frame with it: ``outside``, ``front``,
  ``front_depth``, ``hand``, ``hand_depth``, ``navigation``, ``arm_planner``;
- ``trace.json``: the timeline and the path the strip is drawn from.

The run folder needs ``state.npz`` (every run since PhyGS 0.7.0 writes it), ``report.json`` and
``audit.jsonl``. The navigation and arm planner panels need a state log of PhyGS 0.9.0 or later
(Nav2's costmap and the arm plans); an older one shows them without the costmap and the plans.
The floor plan is the scene release's, else one rendered once and cached (docs/world.md).
``phygs replay`` is this command.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]  # The PhyGS checkout.
sys.path.insert(0, str(PROJECT))  # Runs from a checkout without pip install.

from isaaclab.app import AppLauncher

VIEWS = ("outside", "front", "front_depth", "hand", "hand_depth", "navigation", "arm_planner")

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("run", type=Path, help="A finished run's output folder")
parser.add_argument("--output", type=Path, default=None, help="Where to write (default: <run>/replay)")
parser.add_argument("--views", nargs="*", choices=VIEWS, default=[], help="Also write these views as clips")
parser.add_argument(
    "--camera", choices=("chase", "fixed"), default="chase", help="Outside: behind the robot, or at the record view"
)
parser.add_argument(
    "--size", type=int, nargs=2, default=None, metavar=("WIDTH", "HEIGHT"), help="Outside view [px]; default: record"
)
parser.add_argument("--speed", type=int, default=1, help="Render every n-th sample: n times real time")
parser.add_argument("--no_panels", action="store_true", help="replay.mp4 without the panels on the right")
parser.add_argument("--no_overview", action="store_true", help="replay.mp4 without the strip under it")
parser.add_argument("--robot", default=None, help="Replace the run's robot (same joints): a name or robot.yaml")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(device="cpu", enable_cameras=True)
args = parser.parse_args()
for name in ("report.json", "state.npz"):
    if not (args.run / name).is_file():
        parser.error(f"{args.run} has no {name}; replay needs a finished run with a state log")
if args.speed < 1:
    parser.error("--speed takes a whole number of samples per frame, 1 or more")
output = args.output or args.run / "replay"
app = AppLauncher(args).app

from phygs.recording.trace import build as build_trace  # noqa: E402
from phygs.recording.video import Video  # noqa: E402
from phygs.world.replay import VIEWS as ROBOT_VIEWS  # noqa: E402
from phygs.world.replay import Clip, Replay  # noqa: E402 (after the app starts)


def main() -> int:
    output.mkdir(parents=True, exist_ok=True)
    trace = build_trace(args.run)  # From the logs alone.
    (output / "trace.json").write_text(json.dumps(trace) + "\n")
    views = ROBOT_VIEWS if not args.no_panels else tuple(view for view in args.views if view != "outside")
    size = "record" if args.size is None else tuple(args.size)  # The settings' record size, as the recording's.
    replay = Replay(args.run, robot=args.robot, views=views, outside=size, fixed=args.camera == "fixed")
    states = replay.states
    fps = 1.0 / (replay.period * args.speed)
    video = Video(
        output / "replay.mp4",
        fps=fps,
        plan=None if args.no_overview else replay.plan(),
        panels=not args.no_panels,
        overview=not args.no_overview,
    )
    clips = {name: Clip(output / f"{name}.mp4", fps) for name in args.views}
    try:
        for sample, bodies in enumerate(states.bodies()):
            if sample % args.speed:
                continue
            replay.apply(sample, bodies)
            replay.aim()
            outside, sources = replay.render(sample)
            video.frame(outside, sources=sources, trace=trace, sim_time=float(states.time[sample]))
            if args.views:
                images = replay.views(sources, outside)
                for name in args.views:
                    clips[name].write(images.get(name))
            if sample % (50 * args.speed) == 0:
                print(f"[REPLAY] {states.time[sample]:7.2f} / {states.time[-1]:.2f} s", flush=True)
    finally:
        video.close()
        for clip in clips.values():
            clip.close()
        replay.close()
    print(f"[REPLAY] {len(states)} samples at {fps:.2f} fps: {output}", flush=True)
    return 0


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    except BaseException:
        traceback.print_exc()
        sys.stderr.flush()
    finally:
        app.close(exit_code=code)
