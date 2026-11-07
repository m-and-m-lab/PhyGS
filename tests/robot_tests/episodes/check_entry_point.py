# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The public entry point, as a user first runs it: ``examples/run_world.py`` with the template agent
(``examples/agents/template.py``) on a skills-tier spec (``dev_grid_table``: walk, stow) and a
locomotion-only spec (``motion_grid``: a body velocity). Each run is its own process
with its own simulator; this test starts none itself and reads only what a user gets: the exit code,
``report.json``, the recording ``episode.rrd`` (complete, of this run, holding no secret from the
environment) and the video with ``--record``. A third run (``motion_grid``) is killed with SIGKILL a
second into its episode: its report still says it ran and its recording still reads back.

    isaaclab -p scripts/PhyGS/tests/robot_tests/episodes/check_entry_point.py \
        --output scripts/PhyGS/output_check/$(date +%Y%m%d-%H%M)_entry_point
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

from phygs.recording.reader import Recording

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]  # The PhyGS checkout.
SPECS = ROOT / "configs" / "episodes"
# Spec -> (the session levels the template uses there, the clients whose commands must succeed).
RUNS = {
    "dev_grid_table": (["skills"], ("navigation", "manipulation")),
    "motion_grid": (["motion"], ("locomotion",)),
}
REPORT_KEYS = (
    "status",
    "start_check",
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
    "run_id",
    "recording",
)
KILL_AFTER = 1.0  # [s wall] into the killed run's episode (motion_grid's agent runs for about 3 s).
SECRET = re.compile(r"KEY|TOKEN|SECRET|PASSWORD", re.IGNORECASE)  # Environment variables never recorded.


def entry_point(spec: str, output: Path, record: bool, kit_args: str | None) -> list[str]:
    """The command line of one episode through the public entry point."""
    command = [sys.executable, str(PROJECT / "examples" / "run_world.py"), "--spec", str(SPECS / f"{spec}.yaml")]
    command += ["--agent", "template", "--task_loader", "phygs_robot_tests.tasks:load_task"]
    command += ["--output", str(output), *(["--record"] if record else [])]
    return command + ([f"--kit_args={kit_args}"] if kit_args else [])


def run(spec: str, output: Path, record: bool, kit_args: str | None = None) -> tuple[int, dict]:
    """One episode through the public entry point; its exit code and report (empty if none)."""
    with (output.parent / f"{output.name}.log").open("w") as log:
        command = entry_point(spec, output, record, kit_args)
        code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=1800).returncode
    path = output / "report.json"
    return code, json.loads(path.read_text()) if path.exists() else {}


def killed(spec: str, output: Path, kit_args: str | None) -> dict:
    """One episode killed (SIGKILL, its whole process group) :data:`KILL_AFTER` into the episode, once
    its report says it runs; the report it left (empty if none)."""
    path = output / "report.json"
    with (output.parent / f"{output.name}.log").open("w") as log:
        command = entry_point(spec, output, False, kit_args)
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic() + 900
        while process.poll() is None and time.monotonic() < deadline:
            if path.exists() and json.loads(path.read_text()).get("status") == "running":
                break
            time.sleep(0.2)
        time.sleep(KILL_AFTER)
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()
    return json.loads(path.read_text()) if path.exists() else {}


def secrets_in(recording: Path) -> list[str]:
    """The secret-named environment variables whose values appear in any text of the recording."""
    import pyarrow as pa
    import rerun as rr

    values = {name: value for name, value in os.environ.items() if SECRET.search(name) and len(value) >= 8}
    found = set()
    for chunk in rr.chunk.RrdReader(recording).stream().to_chunks():
        for column in chunk.to_record_batch().columns:
            while pa.types.is_list(column.type) or pa.types.is_large_list(column.type):
                column = column.flatten()
            if pa.types.is_string(column.type) or pa.types.is_large_string(column.type):
                texts = [text for text in column.to_pylist() if text]
                found.update(name for name, value in values.items() if any(value in text for text in texts))
    return sorted(found)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True, help="Fresh folder for check.json and the runs")
    parser.add_argument("--record", action="store_true", help="Each run's video (third_person.mp4)")
    parser.add_argument("--kit_args", default=None, help="Kit arguments for each run (one string)")
    args, _ = parser.parse_known_args()  # The suite's other app flags have no use here: no simulator in this process.
    if (args.output / "check.json").exists():
        raise SystemExit(f"{args.output} holds a finished check; choose a fresh output folder")
    args.output.mkdir(parents=True, exist_ok=True)
    checks, error = {}, None

    def measure(name: str, value, ok: bool) -> None:
        checks[name] = dict(value=value, passed=bool(ok))
        print(f"check {name}: {value} ({'passed' if ok else 'FAILED'})", flush=True)

    try:
        for spec, (levels, clients) in RUNS.items():
            code, report = run(spec, args.output / spec, args.record, args.kit_args)
            measure(f"{spec}_exit", code, code == 0)
            missing = [key for key in REPORT_KEYS if key not in report]
            measure(f"{spec}_report", missing or "complete", not missing)
            if missing:
                continue
            measure(f"{spec}_status", report["status"], report["status"] == "completed")
            outcome = (report["ended_by"], report["agent_outcome"]["outcome"])
            measure(f"{spec}_ended", outcome, outcome == ("finish", "agent_returned"))
            used = report["usage"]["levels_used"]
            measure(f"{spec}_levels_used", used, used == levels)
            recording = Recording(args.output / spec)
            entry = (report["recording"]["status"], recording.properties.get("run_id") == report["run_id"])
            measure(f"{spec}_recording", (*entry, len(recording.states())), entry == ("complete", True))
            leaked = secrets_in(args.output / spec / "episode.rrd")
            measure(f"{spec}_recording_secrets", leaked or "none", not leaked)
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
        stub = killed("motion_grid", args.output / "killed", args.kit_args)
        measure("killed_report", (stub.get("status"), bool(stub.get("run_id"))), stub.get("status") == "running")
        recording = Recording(args.output / "killed")
        phases = [row[2]["TextLog:text"][0] for row in recording.rows("events", "phase")]
        samples = len(recording.states())
        same = recording.properties.get("run_id") == stub.get("run_id")
        measure("killed_recording", (same, samples, phases), same and samples > 0 and "agent" in phases)
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
