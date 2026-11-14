# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run the robot tests in ladder order, or a group of them, and print a summary.

The ladder is every robot test in the order the framework is built: scene, sensors, locomotion,
footprint, arm (component tests, below the episode harness), then navigation, obstacle avoidance,
arm planning, pick and place, drawers, search, retrieve and the public entry point (agent tests:
an episode spec and a scripted plan through the agent session). ``ladder.json`` holds the
entries (rung, script, spec, requirements, groups, criteria file and last evidence); this runner
reads it and never writes it. ``tests/robot_tests/README.md`` describes each rung. A test whose
requirement is missing (ROS 2 Jazzy sourced, explore_lite's overlay, cuRobo's dependencies) is
skipped, not failed.

Each test writes ``<folder>/<name>/check.json`` and its log ``<folder>/<name>.log``; the summary is
``<folder>/summary.json`` (each test's measured values too) and ``<folder>/index.html`` (results, with
``--record`` the videos). With ``--record`` a component test films itself (``third_person.mp4``). An
agent test always records its episodes (``episode.rrd``); its videos are rendered from them
(``phygs replay``: ``replay/replay.mp4`` beside each ``episode.rrd``) after every test has run and the
verdict is written, so a rung's ``minutes`` is the test alone and its ``replay_minutes`` the render.
``--replay`` picks which agent tests are rendered: ``all`` (the default with ``--record``), ``failed``
(those that did not pass) or ``none`` (the default without it); for a recording left unrendered the page
gives the ``phygs replay`` command (its absolute path, with ``--kit_args``). The folder is
``output_check/<YYYYMMDD-HHMM>_<label>``. The verdict line names the first failing rung; it is printed
before the replays and again as the last line.

    isaaclab -p scripts/PhyGS/tests/robot_tests/run_suite.py ladder --upto 10 --record
    isaaclab -p scripts/PhyGS/tests/robot_tests/run_suite.py manipulation --only pick_place_known

``--kit_args`` goes to every test, for example to leave GPU 0 to another run (``phygs/sim/gpu.py``):
``--kit_args "--/renderer/activeGpu=1 --/physics/cudaDevice=1 --/renderer/multiGpu/enabled=false"``.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]  # The PhyGS checkout.
SPECS = ROOT / "configs" / "episodes"
LADDER = ROOT / "ladder.json"  # The ladder: every robot test in build order; read here, never written.
REPLAY_CHOICES = ("all", "failed", "none")  # --replay: which agent tests' recordings are rendered.


def ladder(path: Path = LADDER) -> list[dict]:
    """The ladder's entries, in build order: each one's rung, suite name, script, spec, arguments,
    requirements and groups, its criteria file and its last evidence (``ladder.json``)."""
    return json.loads(path.read_text())["rungs"]


def select(entries: list[dict], groups: list[str], first: int = 1, upto: int | None = None, only=None) -> list[dict]:
    """The entries of ``groups`` whose rung is in [``first``, ``upto``] (None: the top rung), in ladder
    order; with ``only``, just those suite names."""
    return [
        entry
        for entry in entries
        if set(groups) & set(entry["groups"])
        and first <= entry["rung"] <= (entry["rung"] if upto is None else upto)
        and (not only or entry["name"] in only)
    ]


def arguments(entry: dict) -> list:
    """The test script's arguments: its spec (``configs/episodes/<spec>.yaml``) and flags."""
    return [*(["--spec", SPECS / f"{entry['spec']}.yaml"] if entry["spec"] else []), *entry["args"]]


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
            settings = yaml.safe_load((PROJECT / "phygs" / "world" / "environment.yaml").read_text())
            if not Path(settings["planner"]["curobo_dependencies"]).is_dir():
                lacking.append(f"cuRobo dependencies in {settings['planner']['curobo_dependencies']}")
    return lacking


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    entries = ladder()
    parser.add_argument("groups", nargs="+", choices=sorted({group for entry in entries for group in entry["groups"]}))
    parser.add_argument("--only", nargs="+", help="Run only these test names of the groups")
    parser.add_argument("--from", dest="first", type=int, default=1, help="First rung to run")
    parser.add_argument("--upto", type=int, default=None, help="Last rung to run (default: the top rung)")
    parser.add_argument("--label", default=None, help="Folder label (default: the groups)")
    parser.add_argument("--timeout", type=float, default=3600.0, help="Per test [s wall]")
    parser.add_argument(
        "--record",
        action="store_true",
        help="Each test's video (components film live; agent tests: rendered from recordings, see --replay)",
    )
    parser.add_argument(
        "--replay",
        choices=REPLAY_CHOICES,
        default=None,
        help="Agent tests whose recordings are rendered after the verdict (default: all with --record, else none)",
    )
    parser.add_argument("--kit_args", default=None, help="Kit arguments for every test (one string)")
    args = parser.parse_args()
    tests = select(entries, args.groups, args.first, args.upto, args.only)
    label = args.label or "_".join(args.groups)
    folder = PROJECT / "output_check" / f"{datetime.now():%Y%m%d-%H%M}_{label}"
    folder.mkdir(parents=True, exist_ok=False)
    print(f"{len(tests)} robot tests into {folder}", flush=True)
    summary, agent_tests = [], set()
    for test in tests:
        rung, name, script, extra = test["rung"], test["name"], test["test"], arguments(test)
        lacking = missing(tuple(test["requires"]))
        if lacking:
            summary.append(
                dict(rung=rung, name=name, status="skipped", exit=None, minutes=0.0, error=", ".join(lacking))
            )
            print(f"{rung} {name}: skipped (needs {', '.join(lacking)})", flush=True)
            continue
        episodes = script.startswith("episodes/")  # Agent tests: their videos come from their recordings.
        if episodes:
            agent_tests.add(name)
        command = [sys.executable, str(ROOT / script), *map(str, extra), "--output", str(folder / name)]
        command += ["--record"] if args.record and not episodes else []
        command += [f"--kit_args={args.kit_args}"] if args.kit_args else []
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
    write(folder, summary, args.kit_args)
    print(verdict(folder, summary), flush=True)
    rendering = to_render(summary, agent_tests, args.replay or ("all" if args.record else "none"))
    for entry in rendering:
        started = time.monotonic()
        replay(folder / entry["name"], folder / f"{entry['name']}.replay.log", args.kit_args)
        entry["replay_minutes"] = round((time.monotonic() - started) / 60, 1)
        print(f"{entry['rung']} {entry['name']}: replay rendered ({entry['replay_minutes']} min)", flush=True)
    if rendering:
        write(folder, summary, args.kit_args)
        total = round(sum(entry["replay_minutes"] for entry in rendering), 1)
        print(f"{len(rendering)} replays rendered in {total} min", flush=True)
        print(verdict(folder, summary), flush=True)
    return 0 if all(entry["status"] in ("passed", "skipped") for entry in summary) else 1


def to_render(summary: list[dict], agent_tests: set[str], choice: str) -> list[dict]:
    """The summary entries whose recordings are rendered: agent tests that ran, all of them or (``failed``)
    those that did not pass; ``none``, no entry."""
    ran = [entry for entry in summary if entry["name"] in agent_tests and entry["status"] != "skipped"]
    return {"all": ran, "failed": [entry for entry in ran if entry["status"] != "passed"], "none": []}[choice]


def write(folder: Path, summary: list[dict], kit_args: str | None = None) -> None:
    """``summary.json`` and ``index.html``, rewritten as replays are added."""
    (folder / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    (folder / "index.html").write_text(index(folder, summary, kit_args))


def verdict(folder: Path, summary: list[dict]) -> str:
    """The line that names the first failing rung."""
    ran = [entry for entry in summary if entry["status"] != "skipped"]
    passed = sum(entry["status"] == "passed" for entry in ran)
    failed = [entry for entry in ran if entry["status"] != "passed"]
    first = f"; first failure: rung {failed[0]['rung']} {failed[0]['name']}" if failed else ""
    return f"{passed}/{len(ran)} passed, {len(summary) - len(ran)} skipped{first}; {folder}"


def replay(folder: Path, log: Path, kit_args: str | None) -> None:
    """Render every episode recorded under ``folder`` again (``phygs replay``), into ``replay/`` beside it."""
    with log.open("w") as out:
        for recording in sorted(folder.rglob("episode.rrd")):
            command = [sys.executable, str(PROJECT / "examples" / "replay_run.py"), str(recording.parent)]
            command += [f"--kit_args={kit_args}"] if kit_args else []
            subprocess.run(command, stdout=out, stderr=subprocess.STDOUT)


def replay_command(run: Path, kit_args: str | None) -> str:
    """The shell command that renders ``run``'s recording, from any folder, on the suite's GPU."""
    command = f"phygs replay {shlex.quote(str(run.resolve()))}"
    return command + (f" --kit_args={shlex.quote(kit_args)}" if kit_args else "")


def index(folder: Path, summary: list[dict], kit_args: str | None = None) -> str:
    """A local page with each test's result and video (opened from the folder; no network). A recording
    left unrendered gets its ``phygs replay`` command: the run's absolute path and the suite's Kit arguments."""
    rows = []
    for entry in summary:
        name, status = entry["name"], entry["status"]
        if status == "skipped":
            title = f"{entry['rung']} {name}: skipped"
            rows.append(f'<section class="skipped"><h2>{title}</h2><p>needs {entry["error"]}</p></section>')
            continue
        found = [*(folder / name).glob("**/third_person.mp4"), *(folder / name).glob("**/replay/replay.mp4")]
        videos = sorted(path.relative_to(folder) for path in found)
        video = "".join(f'<video src="{path}" controls preload="metadata"></video>' for path in videos)
        unrendered = sorted(
            r.parent for r in (folder / name).rglob("episode.rrd") if not (r.parent / "replay" / "replay.mp4").exists()
        )
        video += "".join(
            f"<p>not rendered: <code>{html.escape(replay_command(run, kit_args))}</code></p>" for run in unrendered
        )
        replayed = f", replay {entry['replay_minutes']} min" if "replay_minutes" in entry else ""
        rows.append(
            f'<section class="{status}"><h2>{entry["rung"]} {name}: {status}</h2>'
            f'<p>exit {entry["exit"]}, {entry["minutes"]} min{replayed}; <a href="{name}/check.json">check.json</a>, '
            f'<a href="{name}.log">log</a></p>{video or "no video"}</section>'
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
