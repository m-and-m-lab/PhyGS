# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The public entry point, as a user first runs it: ``examples/run_world.py`` with the template agent
(``phygs.agents.template``) on a skills-tier spec (``dev_grid_table``: walk, stow) and a
joints-tier spec (``joints_grid``: a body velocity, an arm joint move). Each run is its own process
with its own simulator; this test starts none itself and reads only what a user gets: the exit code
and ``report.json`` (and the video with ``--record``).

    ./isaaclab.sh -p scripts/PhyGS/phygs_robot_tests/episodes/check_entry_point.py \
        --output scripts/PhyGS/output_check/$(date +%Y%m%d-%H%M)_entry_point
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent  # The PhyGS checkout.
SPECS = ROOT / "configs" / "episodes"
# Spec -> (the session levels the template uses there, the clients whose commands must succeed).
RUNS = {
    "dev_grid_table": (["skills"], ("navigation", "manipulation")),
    "joints_grid": (["joints", "motion"], ("locomotion", "arm")),
}
REPORT_KEYS = (
    "spec",
    "provenance",
    "ended_by",
    "agent_outcome",
    "evaluation",
    "stopped",
    "usage",
    "commands",
    "contacts",
    "runtime",
)


def run(spec: str, output: Path, record: bool) -> tuple[int, dict]:
    """One episode through the public entry point; its exit code and report (empty if none)."""
    command = [sys.executable, str(PROJECT / "examples" / "run_world.py"), "--spec", str(SPECS / f"{spec}.yaml")]
    command += ["--agent", "template", "--task_loader", "phygs_robot_tests.tasks:load_task"]
    command += ["--output", str(output), *(["--record"] if record else [])]
    with (output.parent / f"{spec}.log").open("w") as log:
        code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=1800).returncode
    path = output / "report.json"
    return code, json.loads(path.read_text()) if path.exists() else {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True, help="Fresh folder for check.json and the runs")
    parser.add_argument("--record", action="store_true", help="Each run's video (third_person.mp4)")
    args, _ = parser.parse_known_args()  # The suite's app flags have no use here: no simulator in this process.
    if (args.output / "check.json").exists():
        raise SystemExit(f"{args.output} holds a finished check; choose a fresh output folder")
    args.output.mkdir(parents=True, exist_ok=True)
    checks, error = {}, None

    def measure(name: str, value, ok: bool) -> None:
        checks[name] = dict(value=value, passed=bool(ok))
        print(f"check {name}: {value} ({'passed' if ok else 'FAILED'})", flush=True)

    try:
        for spec, (levels, clients) in RUNS.items():
            code, report = run(spec, args.output / spec, args.record)
            measure(f"{spec}_exit", code, code == 0)
            missing = [key for key in REPORT_KEYS if key not in report]
            measure(f"{spec}_report", missing or "complete", not missing)
            if missing:
                continue
            outcome = (report["ended_by"], report["agent_outcome"]["outcome"])
            measure(f"{spec}_ended", outcome, outcome == ("finish", "agent_returned"))
            used = report["usage"]["levels_used"]
            measure(f"{spec}_levels_used", used, used == levels)
            for client in clients:
                # A command's final feedback is recorded as {feedback type: its fields}.
                waits = [
                    next(iter(c["feedback"].values()))["status"]
                    for c in report["commands"]
                    if c["call"] == f"{client}.wait"
                ]
                # Skills end "succeeded"; a velocity or joint command that ran its duration ends "done".
                ok = bool(waits) and all(status in {"succeeded", "done"} for status in waits)
                measure(f"{spec}_{client}_commands", waits, ok)
            if args.record:
                video = args.output / spec / "third_person.mp4"
                measure(f"{spec}_video", video.exists(), video.exists())
    except Exception as exc:  # noqa: BLE001 - reported in check.json like every robot test.
        traceback.print_exc()
        error = f"{type(exc).__name__}: {exc}"
    status = "passed" if error is None and all(entry["passed"] for entry in checks.values()) else "failed"
    record = dict(test="check_entry_point", status=status, error=error, checks=checks)
    (args.output / "check.json").write_text(json.dumps(record, indent=2, default=str) + "\n")
    print(f"check_entry_point: {status}; {args.output}", flush=True)
    return 0 if status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
