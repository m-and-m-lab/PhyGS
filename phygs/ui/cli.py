# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run a headless robot episode with the robot UI (docs/design/ui/design.html).

The simulation runs headless. Open the printed URL in a browser on this machine (remote:
``ssh -L 8765:127.0.0.1:8765 <host>``). The page shows the task; the simulation starts when
the person presses Start. The UI shows the robot's front cameras (stitched into the Front view when
there are two) and its look (hand) camera, navigation's own map (Nav2's costmap), the robot state,
the skill step lines and the commands. ``--robot`` overrides the spec's robot.

The episode is an episode spec (docs/design/episode_harness) run by the same
:class:`~phygs.world.World` as every agent's: the person is the
agent (:class:`~phygs.ui.session.UiSession`, every action an audited session call
within the spec's actions and grants) and the page is its viewer. The World writes the episode
report (``report.json``, with the task's evaluation) and ``audit.jsonl``; the UI adds
``ui_summary.json``, ``ui_log.jsonl``, ``telemetry.jsonl`` and a ``recording`` that ``replay_cli.py``
plays back. ``phygs/configs/ui/house_drawer.yaml`` runs the Bear house. The episode ends with End, the
budget or E-stop. ``--demo tour`` starts at once, drives a short fixed route through the public
locomotion API and then navigates back with Nav2, so there is something to watch.

    ./isaaclab.sh -p scripts/PhyGS/phygs/ui/cli.py --task_loader phygs_robot_tests.tasks:load_task \\
      --output scripts/PhyGS/output_check/$(date +%Y%m%d-%H%M)_ui_grid
"""

from __future__ import annotations

import argparse
import json
import math
import secrets
import signal
import sys
import threading
import time
import traceback
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]  # The PhyGS checkout (phygs/ui/cli.py).
sys.path.insert(0, str(PROJECT))  # Runs from a checkout without pip install.
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
parser.add_argument("--config", type=Path, default=PROJECT / "phygs" / "configs" / "ui" / "episode.yaml")
parser.add_argument(
    "--spec",
    type=Path,
    default=PROJECT / "phygs/configs/ui/grid_drawer.yaml",
    help="Episode spec (docs/episode_harness)",
)
parser.add_argument("--host", default="127.0.0.1", help="Address to serve on; keep it local and tunnel for remote use")
parser.add_argument("--port", type=int, default=8765)
parser.add_argument("--token", default=None, help="Access token (default: a new random one)")
parser.add_argument("--demo", choices=("none", "tour"), default="none", help="Drive a fixed route to watch")
parser.add_argument("--robot", default=None, help="Override the spec's robot: a robot name or a robot.yaml path")
parser.add_argument(
    "--task_loader",
    default="phygs_robot_tests.tasks:load_task",
    help="module:function that builds the task object from the spec's task block (None: the spec must have goal none)",
)
parser.add_argument("--autostart", action="store_true", help="Start the episode without waiting for the page")
parser.add_argument("--max_wall_seconds", type=float, default=None, help="End after this long (for scripted checks)")
parser.add_argument(
    "--linger_seconds",
    type=float,
    default=None,
    help="After the episode ends, keep serving its last state this long, then exit (default: until Ctrl+C)",
)
parser.add_argument("--output", type=Path, required=True, help="Fresh folder for the episode and the UI logs")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(device="cpu", visualizer=[], enable_cameras=True, headless=True)
args = parser.parse_args()
if args.output.exists() and any(args.output.iterdir()):
    parser.error("Choose a fresh output folder")
args.output.mkdir(parents=True, exist_ok=True)
app = AppLauncher(args).app

import yaml

from phygs.execution.manipulation.planners.model import link_transforms
from phygs.robots.description import load_robot
from phygs.ui import schema
from phygs.ui.bridge import Bridge
from phygs.ui.recording import Recorder
from phygs.ui.server import UiServer
from phygs.ui.session import RUNNING, UiSession
from phygs.world import World
from phygs.world.spec import EpisodeSpec
from phygs.world.task import Task, load_task


def ui_cameras(robot) -> tuple[str, ...]:
    """The cameras the page streams: the robot's front cameras and its look (hand) camera."""
    return (*robot.cameras_for("front"), *robot.cameras_for("look"))


def check_spec(spec: EpisodeSpec, robot, cameras: tuple[str, ...]) -> None:
    """The page streams ``cameras`` of ``robot``, shows the true pose, and offers no arm joint-target client."""
    missing = set(cameras) - set(spec.observations.for_robot(robot))
    if missing:
        raise ValueError(f"{args.spec}: the UI streams {cameras}; the spec does not observe {sorted(missing)}.")
    if "gt_localization" not in spec.grants:
        raise ValueError(f"{args.spec}: the UI shows the true pose; grant gt_localization.")
    if "arm" in spec.actions:
        raise ValueError(f"{args.spec}: the UI has no arm joint-target client; use the skills or motion tier.")


def page(spec: EpisodeSpec, task: Task) -> dict:
    """What the Start card shows about the episode."""
    return dict(
        spec=spec.id,
        goal=task.goal,
        description=task.description,
        budget=spec.sim_seconds,
        actions=list(spec.actions),
        grants_used=list(spec.grants),
    )


def tour(ui: UiSession, session):
    """A fixed route through the public locomotion API (turn left, walk, turn back, walk back, stop),
    then Nav2 back to where the walk ended (a place the robot stood, so free)."""
    route = (
        ("turn left", (0.0, 0.0, 0.5), 5.0),
        ("walk forward", (0.4, 0.0, 0.0), 3.0),
        ("turn right", (0.0, 0.0, -0.5), 5.0),
        ("walk back", (-0.3, 0.0, 0.0), 3.0),
    )
    walked = None
    for label, velocity, seconds in route:
        command = session.locomotion.command_velocity(*velocity, duration=seconds)
        inputs = dict(label=label, velocity=velocity, duration=seconds)
        ui.track("locomotion", "command_velocity", command, inputs, session)
        while session.locomotion.feedback(command).status in {"active", "stopping"}:
            yield
        if label == "walk forward":
            pose = session.observe().localization.pose.matrix
            heading = math.atan2(float(pose[1, 0]), float(pose[0, 0]))
            walked = dict(x=float(pose[0, 3]), y=float(pose[1, 3]), heading=heading)
    ui.track("locomotion", "stop", session.locomotion.stop(), session=session)
    command = session.navigation.navigate_to(**walked)
    ui.track("navigation", "navigate_to", command, walked, session)
    while session.navigation.feedback(command).status in {"active", "stopping"}:
        yield


def main() -> int:
    """Run the episode until it ends, then serve its result; returns the exit code (0: ended normally)."""
    config = yaml.safe_load(args.config.read_text())
    spec = EpisodeSpec.load(args.spec)
    task = load_task(spec.task, args.task_loader)  # The page's brief and the World's evaluation come from it.
    robot = load_robot(args.robot or spec.robot.model)
    cameras, front = ui_cameras(robot), robot.cameras_for("front")
    front = front if len(front) == 2 else ()  # Two cameras make the stitched Front view.
    check_spec(spec, robot, cameras)
    hello = dict(
        episode_id=None,
        task=task.brief(),
        episode=page(spec, task),
        views=[*(["front"] if front else []), *cameras],
        max_depth=config["max_depth"],
        scene=spec.scene.name,
        skills=None,
    )
    bridge = Bridge(hello, log_path=args.output / "ui_log.jsonl")
    ui = UiSession(
        bridge,
        robot=robot,
        cameras=cameras,
        drive_limits=config["drive"],
        held_object_size_m=tuple(config["held_object_size_m"]),
        task=dict(goal=task.goal, description=task.description),
        budget_seconds=spec.sim_seconds,
        telemetry_path=args.output / "telemetry.jsonl",
        demo=tour if args.demo == "tour" else None,
    )
    world = World(
        spec, args.output, clock="realtime", device=args.device, viewer=ui, hold_clock=True, robot=robot, task=task
    )
    recorder = Recorder(args.output / "recording")
    server = None
    exit_code = 1
    started = time.monotonic()
    try:
        world.reset()  # Settled, the planner up and the skills bound before the page opens.
        planner = world.planner
        if planner is not None:
            urdf = planner.model["urdf"]
            ui.bind(forward=lambda joints: link_transforms(urdf, joints))
        ui.bind(
            health=lambda: dict(
                pipeline=world.control.pipeline.metrics,
                planner=None if planner is None else planner.status(),
            )
        )
        joint_limits = world.scene["robot"].data.soft_joint_pos_limits.torch[0].cpu().tolist()
        names = world.scene["robot"].joint_names
        arm_limits = {name: joint_limits[names.index(name)] for name in robot.arm.joints}
        hello.update(
            episode_id=world.control.runtime.episode_id,
            skills=schema.skills(
                config["drive"],
                arm_limits,
                targets=True,
                preview=planner is not None,
                actions=spec.actions,
                hand_view=robot.look_camera,
            ),
            arm_backend=world.settings["planner"]["backend"] if planner is not None else "none",
        )
        token = args.token or secrets.token_urlsafe(9)
        server = UiServer(
            bridge,
            host=args.host,
            port=args.port,
            token=token,
            front_cameras=front,
            max_depth=config["max_depth"],
            recorder=recorder,
        )
        server.start()
        print(
            f"\nRobot UI: {server.url}\n"
            "  Open this full link (with its token) in a browser on this machine.\n"
            f"  From another computer, forward port {server.port} first, then open the same link there:\n"
            f"    ssh -L {server.port}:127.0.0.1:{server.port} <user>@<this machine>"
            "   (or VS Code: Ports > Forward a Port)\n",
            flush=True,
        )
        (args.output / "url.txt").write_text(server.url + "\n")

        def stop(*_):  # Ctrl+C, SIGTERM or the time limit: End when running, else E-stop.
            bridge.request(dict(type="episode", action="end") if ui.phase == RUNNING else dict(type="estop"))

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        if args.max_wall_seconds is not None:
            threading.Timer(args.max_wall_seconds, stop).start()
        if args.autostart or args.demo != "none":
            bridge.request(dict(type="episode", action="start"))
        report = world.run(ui.agent, name="robot_ui")
        shown = ui.show_report(report)
        (args.output / "ui_summary.json").write_text(json.dumps(shown, indent=1))
        timing = dict(
            sim_seconds=report["sim_seconds"],
            wall_seconds=time.monotonic() - started,
            episode_wall_seconds=report["wall_seconds"],
            map_updates=ui.map_updates,
            ended_by=report["ended_by"],
            truth=dict(  # For reviewers; never on the page.
                drawer_openings_m={
                    name: float(world.scene_def.part(name).opening) for name in world.scene_def.part_names
                },
                target_position_m=None
                if task.target is None
                else world.scene_def.object_pose(task.target)[:3, 3].tolist(),
            ),
        )
        (args.output / "timing.json").write_text(json.dumps(timing, indent=1))
        print("timing:", json.dumps(timing), flush=True)
        linger = math.inf if args.linger_seconds is None else args.linger_seconds
        signal.signal(signal.SIGINT, signal.default_int_handler)  # Ended: Ctrl+C now exits.
        signal.signal(signal.SIGTERM, signal.default_int_handler)
        ended = time.monotonic()
        while app.is_running() and time.monotonic() - ended < linger:
            time.sleep(0.1)  # Keep serving the result until the time is up or Ctrl+C.
        exit_code = 0
    except KeyboardInterrupt:
        exit_code = 0
    finally:
        if server is not None:
            server.stop()
        recorder.close()
        ui.close()
        world.close()
        bridge.close()
    return exit_code


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    except BaseException:
        traceback.print_exc()  # Closing Isaac Sim ends the process before Python would print it.
        sys.stderr.flush()
    finally:
        app.close(exit_code=code)  # Without the code a crash would exit 0.
