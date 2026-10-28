# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run one world: build the episode from its spec, run an agent, write the report.

    ./isaaclab.sh -p scripts/PhyGS/examples/run_world.py \
        --spec scripts/PhyGS/phygs_robot_tests/configs/episodes/dev_grid_table.yaml \
        --agent template --task_loader phygs_robot_tests.tasks:load_task \
        --output scripts/PhyGS/output_check/$(date +%Y%m%d-%H%M)_grid_table

``--task_loader`` names the ``module:function`` that turns the spec's task block into the task
object (its brief, its privileged target, its evaluation); PhyGS itself knows no task. A spec with
``task: {goal: none}`` needs none.
The robot is the spec's (``robot.model``); ``--robot`` runs the same episode with another one: a robot
name (``phygs/robots/<name>``) or the path of a ``robot.yaml`` (docs/adding_a_robot.md).

The exit code says the episode ran to its end (0) or the run broke (1); the outcome is in
<output>/report.json. Design: docs/design/episode_harness/index.html.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]  # The PhyGS checkout.
sys.path.insert(0, str(PROJECT))  # Runs from a checkout without pip install.

from isaaclab.app import AppLauncher

from phygs.episodes.spec import EpisodeSpec, load_agent
from phygs.episodes.task import load_task

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--spec", type=Path, required=True, help="Episode spec YAML")
parser.add_argument("--agent", required=True, help="A name in phygs.agents, or path/to/file.py:function")
parser.add_argument("--output", type=Path, required=True, help="A fresh output folder (date-time prefix)")
parser.add_argument("--clock", choices=("charged", "realtime"), default="charged")
parser.add_argument("--record", action="store_true", help="Write a third-person video")
parser.add_argument("--seed", type=int, default=None, help="Override the spec's seed")
parser.add_argument("--robot", default=None, help="Override the spec's robot: a robot name or a robot.yaml path")
parser.add_argument(
    "--task_loader",
    default=None,
    help="module:function that builds the task object from the spec's task block (None: the spec must have goal none)",
)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(device="cpu", enable_cameras=True)
args = parser.parse_args()
spec = EpisodeSpec.load(args.spec)  # Validated before Isaac Sim starts.
task = load_task(spec.task, args.task_loader)
if args.seed is not None:
    spec = spec.with_seed(args.seed)
agent_name, agent = load_agent(args.agent)
if (args.output / "report.json").exists():
    parser.error(f"{args.output} holds a finished episode; choose a fresh output folder")
app = AppLauncher(args).app

from phygs.episodes.environment import Environment  # noqa: E402 (after the app starts)
from phygs.robots.description import load_robot  # noqa: E402 (numpy before Kit can crash its start)

robot = load_robot(args.robot or spec.robot.model)
robot.validate_urdf()


def main() -> int:
    with Environment(
        spec, args.output, clock=args.clock, record=args.record, device=args.device, robot=robot, task=task
    ) as environment:
        report = environment.run(agent, name=agent_name)
    print(
        f"Episode {spec.id}: ended by {report['ended_by']} ({report['agent_outcome']['outcome']}),"
        f" evaluation {report['evaluation']['outcome']}, {report['sim_seconds']} s simulated;"
        f" {args.output}/report.json",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    except BaseException:
        traceback.print_exc()  # Closing Isaac Sim ends the process before Python would print it.
        sys.stderr.flush()
    finally:
        app.close(exit_code=code)
