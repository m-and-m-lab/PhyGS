# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render a finished run again from its state log: the run itself renders nothing it does not need.

    ./isaaclab.sh -p scripts/PhyGS/examples/replay_run.py \
        scripts/PhyGS/output_check/<YYYYMMDD-HHMM>_grid_table --headless

Writes into ``<run>/replay`` (or ``--output``), frame for frame:

- ``outside.mp4``: a camera that follows the robot, with the overview strip under it: the clock,
  what the gripper holds, the arm's contact force, the agent's step, the floor plan with the
  robot's path and Nav2's route, and the timeline of the agent's steps and the robot's commands;
- ``front.mp4``, ``front_depth.mp4``, ``hand.mp4``, ``hand_depth.mp4``: the robot's own views;
- ``trace.json``: the timeline and the path the strip is drawn from.

The run folder needs ``state.npz`` (every run since PhyGS 0.7.0 writes it), ``report.json`` and
``audit.jsonl``. The floor plan is the scene release's, else one rendered once and cached
(docs/world.md). ``phygs replay`` is this command.
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

VIEWS = ("front", "front_depth", "hand", "hand_depth")

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("run", type=Path, help="A finished run's output folder")
parser.add_argument("--output", type=Path, default=None, help="Where to write (default: <run>/replay)")
parser.add_argument("--views", nargs="*", choices=VIEWS, default=list(VIEWS), help="The robot's views to write")
parser.add_argument(
    "--camera", choices=("chase", "fixed"), default="chase", help="Outside: behind the robot, or at the record view"
)
parser.add_argument("--width", type=int, default=1280, help="Outside view width [px]")
parser.add_argument("--height", type=int, default=720, help="Outside view height [px]")
parser.add_argument("--speed", type=int, default=1, help="Render every n-th sample: n times real time")
parser.add_argument("--no_overview", action="store_true", help="The outside view without the strip under it")
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

import numpy as np  # noqa: E402 (numpy before Kit can crash its start)

from phygs.recording.overview import Overview  # noqa: E402
from phygs.recording.trace import build as build_trace  # noqa: E402
from phygs.world.replay import Clip, Replay  # noqa: E402 (after the app starts)

OVERVIEW_HEIGHT = 300  # [px]


def main() -> int:
    output.mkdir(parents=True, exist_ok=True)
    trace = build_trace(args.run)  # From the logs alone.
    (output / "trace.json").write_text(json.dumps(trace) + "\n")
    replay = Replay(args.run, robot=args.robot, views=tuple(args.views), outside=(args.width, args.height))
    states = replay.states
    fps = 1.0 / (replay.period * args.speed)
    overview = None if args.no_overview else Overview(trace, replay.plan(), width=args.width, height=OVERVIEW_HEIGHT)
    clips = {"outside": Clip(output / "outside.mp4", fps)}
    clips.update({name: Clip(output / f"{name}.mp4", fps) for name in args.views})
    try:
        for sample, bodies in enumerate(states.bodies()):
            if sample % args.speed:
                continue
            replay.apply(sample, bodies)
            replay.follow(args.camera, smooth=sample > 0)
            outside, views = replay.render()
            if overview is not None:
                outside = np.concatenate([outside, overview.draw(float(states.time[sample]))], axis=0)
            clips["outside"].write(outside)
            for name in args.views:
                clips[name].write(views.get(name))
            if sample % (50 * args.speed) == 0:
                print(f"[REPLAY] {states.time[sample]:7.2f} / {states.time[-1]:.2f} s", flush=True)
    finally:
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
