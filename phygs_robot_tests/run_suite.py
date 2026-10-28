# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run the robot tests in ladder order, or a group of them, and print a summary.

The ladder is every robot test in the order the framework is built: scene, sensors, locomotion,
footprint, arm (component tests, below the episode harness), then navigation, obstacle avoidance,
arm planning, pick and place, drawers, search, retrieve and the public entry point (agent tests:
an episode spec and a scripted plan through the agent session). ``phygs_robot_tests/README.md``
has the table. A test whose requirement is missing (ROS 2 Jazzy sourced, explore_lite's overlay,
cuRobo's dependencies) is skipped, not failed.

Each test writes ``<folder>/<name>/check.json`` and its log ``<folder>/<name>.log``; the summary is
``<folder>/summary.json`` (each test's measured values too) and ``<folder>/index.html`` (results, with
``--record`` the videos). The folder is ``output_check/<YYYYMMDD-HHMM>_<label>``. The last line names
the first failing rung.

    ./isaaclab.sh -p scripts/PhyGS/phygs_robot_tests/run_suite.py ladder --upto 10 --record
    ./isaaclab.sh -p scripts/PhyGS/phygs_robot_tests/run_suite.py manipulation --only pick_place_known
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent  # The PhyGS checkout.
SPECS = ROOT / "configs" / "episodes"
ARM, PICK = "episodes/check_arm_planning.py", "episodes/check_pick_place.py"
NAVIGATION, RETRIEVE, STROKE = (
    "episodes/check_navigation.py",
    "episodes/check_retrieve.py",
    "episodes/check_drawer_stroke.py",
)


def spec(name: str) -> list:
    return ["--spec", SPECS / f"{name}.yaml"]


# (rung, name, script, arguments, requirements), in build order.
LADDER = [
    (1, "scene_reset", "components/check_world_lifecycle.py", [], ()),
    (2, "sensors", "components/check_observations.py", [], ()),
    (3, "locomotion", "components/check_locomotion.py", [], ()),
    (4, "footprint", "components/check_footprint.py", [], ()),
    (5, "arm", "components/check_manipulation.py", [], ()),
    (6, "navigation_course", NAVIGATION, spec("nav_course"), ("ros2",)),
    (6, "navigation_house", NAVIGATION, spec("nav_house"), ("ros2",)),
    (7, "obstacle_avoidance", "episodes/check_dynamic_obstacle.py", spec("nav_dynamic"), ("ros2",)),
    (8, "arm_planning_known", ARM, spec("table_reach"), ("curobo",)),
    (8, "arm_planning_known_moveit", ARM, [*spec("table_reach"), "--planner", "moveit"], ("ros2",)),
    (9, "arm_planning_depth", ARM, spec("table_reach_depth"), ("curobo",)),
    (9, "arm_planning_depth_moveit", ARM, [*spec("table_reach_depth"), "--planner", "moveit"], ("ros2",)),
    (10, "pick_place_known", PICK, spec("table_reach"), ("curobo",)),
    (10, "pick_place_known_moveit", PICK, [*spec("table_reach"), "--planner", "moveit"], ("ros2",)),
    (10, "pick_place_depth", PICK, spec("table_reach_depth"), ("curobo",)),
    (10, "pick_place_depth_moveit", PICK, [*spec("table_reach_depth"), "--planner", "moveit"], ("ros2",)),
    (11, "drawer_grid", STROKE, spec("dev_grid_drawer"), ("ros2", "curobo")),
    (11, "drawer_house", STROKE, spec("dev_house_drawer"), ("ros2", "curobo")),
    (12, "drawer_place", "episodes/check_drawer_place.py", [], ("ros2", "curobo")),
    (13, "search_house", "components/check_nav2_pipeline.py", [], ("explore",)),
    (14, "retrieve_grid", RETRIEVE, spec("dev_grid_drawer"), ("ros2", "curobo")),
    (14, "retrieve_house", RETRIEVE, spec("dev_house_drawer"), ("ros2", "curobo")),
    (15, "entry_point", "episodes/check_entry_point.py", [], ("ros2", "curobo")),
]
# Groups by subsystem, as before the ladder (2026-10-28): rungs of the ladder.
GROUPS = {
    "ladder": range(1, 16),
    "base": (1, 2, 3, 4),
    "nav2": (6, 7, 13),
    "manipulation": (5, 8, 9, 10),
    "episodes": (11, 12, 14, 15),
}


def missing(requirements: tuple[str, ...]) -> list[str]:
    """The requirements this shell lacks: ``ros2`` (ROS 2 Jazzy sourced), ``explore`` (explore_lite's
    overlay sourced, which brings ROS 2 too), ``curobo`` (cuRobo's dependency folder)."""
    lacking = []
    prefixes = os.environ.get("AMENT_PREFIX_PATH", "").split(os.pathsep)
    ros = bool(os.environ.get("ROS_DISTRO")) and shutil.which("ros2") is not None
    for requirement in requirements:
        if requirement == "ros2" and not ros:
            lacking.append("ROS 2 Jazzy sourced")
        elif requirement == "explore" and not (
            ros and any(Path(p, "lib", "explore_lite", "explore").exists() for p in prefixes if p)
        ):
            lacking.append("explore_lite overlay (/opt/explore_ws) sourced")
        elif requirement == "curobo":
            settings = yaml.safe_load((PROJECT / "phygs" / "configs" / "environment.yaml").read_text())
            if not Path(settings["planner"]["curobo_dependencies"]).is_dir():
                lacking.append(f"cuRobo dependencies in {settings['planner']['curobo_dependencies']}")
    return lacking


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("groups", nargs="+", choices=sorted(GROUPS))
    parser.add_argument("--only", nargs="+", help="Run only these test names of the groups")
    parser.add_argument("--from", dest="first", type=int, default=1, help="First rung to run")
    parser.add_argument("--upto", type=int, default=15, help="Last rung to run")
    parser.add_argument("--label", default=None, help="Folder label (default: the groups)")
    parser.add_argument("--timeout", type=float, default=3600.0, help="Per test [s wall]")
    parser.add_argument("--record", action="store_true", help="Each test's video (third_person.mp4)")
    args = parser.parse_args()
    rungs = {rung for group in args.groups for rung in GROUPS[group] if args.first <= rung <= args.upto}
    tests = [test for test in LADDER if test[0] in rungs and (not args.only or test[1] in args.only)]
    label = args.label or "_".join(args.groups)
    folder = PROJECT / "output_check" / f"{datetime.now():%Y%m%d-%H%M}_{label}"
    folder.mkdir(parents=True, exist_ok=False)
    print(f"{len(tests)} robot tests into {folder}", flush=True)
    summary = []
    for rung, name, script, extra, requirements in tests:
        lacking = missing(requirements)
        if lacking:
            summary.append(
                dict(rung=rung, name=name, status="skipped", exit=None, minutes=0.0, error=", ".join(lacking))
            )
            print(f"{rung} {name}: skipped (needs {', '.join(lacking)})", flush=True)
            continue
        command = [sys.executable, str(ROOT / script), *map(str, extra), "--output", str(folder / name)]
        command += ["--record"] if args.record else []
        started = time.monotonic()
        with (folder / f"{name}.log").open("w") as log:
            try:
                code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout).returncode
            except subprocess.TimeoutExpired:
                code = "timeout"
        result = folder / name / "check.json"
        record = json.loads(result.read_text()) if result.exists() else {}
        status = record.get("status", "no result")
        entry = dict(
            rung=rung, name=name, status=status, exit=code, minutes=round((time.monotonic() - started) / 60, 1)
        )
        entry.update(error=record.get("error"), checks=record.get("checks", {}))
        summary.append(entry)
        error = f"  {record['error'][:200]}" if record.get("error") else ""
        print(f"{rung} {name}: {status} (exit {code}, {entry['minutes']} min){error}", flush=True)
    (folder / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    (folder / "index.html").write_text(index(folder, summary))
    ran = [entry for entry in summary if entry["status"] != "skipped"]
    passed = sum(entry["status"] == "passed" for entry in ran)
    failed = [entry for entry in ran if entry["status"] != "passed"]
    first = f"; first failure: rung {failed[0]['rung']} {failed[0]['name']}" if failed else ""
    skipped = len(summary) - len(ran)
    print(f"{passed}/{len(ran)} passed, {skipped} skipped{first}; {folder}", flush=True)
    return 0 if not failed else 1


def index(folder: Path, summary: list[dict]) -> str:
    """A local page with each test's result and video (opened from the folder; no network)."""
    rows = []
    for entry in summary:
        name, status = entry["name"], entry["status"]
        if status == "skipped":
            title = f"{entry['rung']} {name}: skipped"
            rows.append(f'<section class="skipped"><h2>{title}</h2><p>needs {entry["error"]}</p></section>')
            continue
        videos = sorted(path.relative_to(folder) for path in (folder / name).glob("**/third_person.mp4"))
        video = "".join(f'<video src="{path}" controls preload="metadata"></video>' for path in videos) or "no video"
        rows.append(
            f'<section class="{status}"><h2>{entry["rung"]} {name}: {status}</h2>'
            f'<p>exit {entry["exit"]}, {entry["minutes"]} min; <a href="{name}/check.json">check.json</a>, '
            f'<a href="{name}.log">log</a></p>{video}</section>'
        )
    ran = [entry for entry in summary if entry["status"] != "skipped"]
    passed = sum(entry["status"] == "passed" for entry in ran)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Robot Tests</title><style>
body {{ margin: 0; font: 15px/1.5 system-ui, sans-serif; background: #fbfbfa; color: #1d1d1b; }}
main {{ max-width: 1000px; margin: 0 auto; padding: 24px 16px; }}
section {{ border-left: 4px solid #2f7a4a; padding: 4px 12px; margin: 18px 0; }}
section:not(.passed) {{ border-left-color: #9a3b2f; }}
section.skipped {{ border-left-color: #9a9a96; color: #6b6b67; }}
h2 {{ font-size: 1.05rem; margin: 4px 0; }} video {{ width: 100%; max-width: 960px; margin: 6px 0; }}
</style></head><body><main><h1>Robot tests: {passed}/{len(ran)} passed, {len(summary) - len(ran)} skipped</h1>
<p>{folder.name}</p>{"".join(rows)}</main></body></html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
