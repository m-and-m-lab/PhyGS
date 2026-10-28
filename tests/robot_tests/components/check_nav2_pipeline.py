# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Real Nav2 servers and the house search component test together (``check_house.py``): Nav2's
voxel layer maps the camera's points, explore_lite picks frontiers, localization is ground truth. No
SLAM, AMCL or BT navigator is started. The simulation's result is ``<output>/simulation``. Navigation
and obstacle avoidance are agent tests (``episodes/check_navigation.py``,
``episodes/check_dynamic_obstacle.py``), with Nav2 started by the episode harness.

    ./isaaclab.sh -p scripts/PhyGS/tests/robot_tests/components/check_nav2_pipeline.py \
        --output scripts/PhyGS/output_check/$(date +%Y%m%d-%H%M)_house_search
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

ROBOT_TESTS = Path(__file__).resolve().parents[1]
PROJECT = ROBOT_TESTS.parents[1]  # The PhyGS checkout.


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", action="store_true", help="The simulation's video (simulation/third_person.mp4)")
    parser.add_argument("--rviz", action="store_true", help="Open a configured RViz2 view on this test's ROS domain")
    parser.add_argument("--viz", choices=("kit", "none"), default="none", help="kit: watch the simulation")
    parser.add_argument(
        "--ros_domain_id", type=int, default=47, help="ROS domain for Nav2, Spot and optional RViz (default: 47)"
    )
    parser.add_argument("--output", type=Path, required=True, help="Fresh folder for check.json and the logs")
    args = parser.parse_args(argv)
    if not 0 <= args.ros_domain_id <= 101:
        parser.error("--ros_domain_id must be between 0 and 101")
    return args


def commands(args: argparse.Namespace) -> tuple[list[str], list[str]]:
    launch = [
        "ros2",
        "launch",
        str(PROJECT / "phygs/execution/navigation/launch/nav2.launch.py"),
        f"experiment:={ROBOT_TESTS / 'configs' / 'house.yaml'}",
        "sensor_mode:=pointcloud",
    ]
    simulation = [
        sys.executable,
        str(ROBOT_TESTS / "components" / "check_house.py"),
        "--output",
        str(args.output.resolve() / "simulation"),
    ]
    if args.viz == "kit":
        simulation.extend(("--viz", "kit"))
    if args.record:
        simulation.append("--record")
    return launch, simulation


def stop(process: subprocess.Popen) -> None:
    """Stop the owned process group, including ROS launch's child servers."""
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            break
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            continue
        # The launcher may have exited before a descendant; terminate the group
        # on the next iteration as well rather than leaving stale ROS publishers.
    process.wait()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if shutil.which("ros2") is None:
        raise RuntimeError("Source /opt/ros/jazzy/setup.bash before running this test")
    if args.rviz and shutil.which("rviz2") is None:
        raise RuntimeError("Install ros-jazzy-rviz2 and source /opt/ros/jazzy/setup.bash")
    if args.rviz and not os.environ.get("DISPLAY"):
        raise RuntimeError("--rviz requires a graphical display (DISPLAY is unset)")
    environment = dict(os.environ, ROS_DOMAIN_ID=str(args.ros_domain_id))
    subprocess.run(
        [sys.executable, "-c", "import rclpy; from nav2_msgs.action import ComputePathToPose, FollowPath"],
        env=environment,
        check=True,
    )
    for package in ("nav2_planner", "nav2_controller", "nav2_lifecycle_manager", "nav2_common"):
        subprocess.run(["ros2", "pkg", "prefix", package], env=environment, check=True, stdout=subprocess.DEVNULL)
    args.output = args.output.resolve()
    if (args.output / "check.json").exists():
        raise FileExistsError(f"{args.output} holds a finished check; choose a fresh output folder")
    args.output.mkdir(parents=True, exist_ok=True)
    physical_report = args.output / "simulation" / "check.json"
    launch, simulation = commands(args)
    name = "check_nav2_pipeline house search"
    report = {
        "test": name,
        "status": "failed",
        "error": None,
        "ros_domain_id": args.ros_domain_id,
        "localization": "ground_truth",
        "rviz": args.rviz,
        "servers": ["planner_server", "controller_server"],
        "actions": ["ComputePathToPose", "FollowPath"],
        "mapping_owner": "nav2_voxel_layer",
        "sensor_message": "PointCloud2",
        "depth_unprojection_owner": "framework",
        "launch_command": launch,
        "simulation_command": simulation,
    }
    processes = []
    try:
        with (args.output / "nav2.log").open("w") as ros_log, (args.output / "simulation.log").open("w") as sim_log:
            if args.rviz:
                rviz_log = (args.output / "rviz.log").open("w")
                try:
                    rviz = subprocess.Popen(
                        [
                            "rviz2",
                            "-d",
                            str(PROJECT / "phygs/configs/navigation/spot_nav2.rviz"),
                            "--ros-args",
                            "-p",
                            "use_sim_time:=true",
                        ],
                        env=environment,
                        stdout=rviz_log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                finally:
                    rviz_log.close()
                processes.append(rviz)
            nav2 = subprocess.Popen(
                launch, env=environment, stdout=ros_log, stderr=subprocess.STDOUT, start_new_session=True
            )
            processes.append(nav2)
            sim = subprocess.Popen(
                simulation, env=environment, stdout=sim_log, stderr=subprocess.STDOUT, start_new_session=True
            )
            processes.append(sim)
            print(f"Real Nav2 + Spot started. Logs and reports: {args.output}", flush=True)
            print(
                "The adapter waits for lifecycle activation and validates costmap geometry before navigation.",
                flush=True,
            )
            progress = time.monotonic()
            while sim.poll() is None:
                if nav2.poll() is not None:
                    raise RuntimeError(f"Nav2 launch exited ({nav2.returncode}); inspect {args.output / 'nav2.log'}")
                if args.rviz and rviz.poll() is not None:
                    raise RuntimeError(f"RViz exited ({rviz.returncode}); inspect {args.output / 'rviz.log'}")
                time.sleep(0.5)
                if time.monotonic() - progress >= 30:
                    print(f"Spot/Nav2 running; inspect {args.output / 'simulation.log'}", flush=True)
                    progress = time.monotonic()
            if sim.returncode:
                raise RuntimeError(
                    f"Spot diagnostic exited ({sim.returncode}); inspect {args.output / 'simulation.log'}"
                )
            if nav2.poll() is not None:
                raise RuntimeError("Nav2 exited before the diagnostic completed")
            physical = json.loads(physical_report.read_text())
            if physical["status"] != "passed" or physical["backend"] != "nav2":
                raise RuntimeError(f"Unexpected physical report: {physical['status']}; backend={physical['backend']}")
            report["status"] = "passed"
    except KeyboardInterrupt:
        report["status"] = "interrupted"
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        print(report["error"], file=sys.stderr, flush=True)
    finally:
        for process in reversed(processes):
            stop(process)
        (args.output / "check.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"{name}: {report['status']}; {args.output}", flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
