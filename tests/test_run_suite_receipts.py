# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The robot test runner's run receipts (``manifest.json``), with a stand-in receipt script and stub test
processes: no simulator starts."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest
from phygs_robot_tests import receipt, run_suite

# A stand-in for write_manifest.py with its interface: it keeps its arguments and the command, adds the
# exit status on --finish, and logs each call (folder, kind) to $FAKE_RECEIPT_LOG.
FAKE_SCRIPT = r"""
import json, os, sys
argv = sys.argv[1:]
log = open(os.environ["FAKE_RECEIPT_LOG"], "a")
if "--finish" in argv:
    folder = argv[argv.index("--finish") + 1]
    path = os.path.join(folder, "manifest.json")
    manifest = json.load(open(path))
    manifest.update(end_time="now", exit_status=int(argv[argv.index("--exit-status") + 1]))
    json.dump(manifest, open(path, "w"))
    log.write(json.dumps([folder, "finish"]) + "\n")
    sys.exit(0)
split = argv.index("--")
options, command = argv[:split], argv[split + 1:]
folder = options[options.index("--out") + 1]
os.makedirs(folder, exist_ok=True)
json.dump(dict(options=options, command=command), open(os.path.join(folder, "manifest.json"), "w"))
log.write(json.dumps([folder, "start"]) + "\n")
"""

# A stub test process: writes its check.json and exits with its own code (Kit can exit 0 on a failure,
# so the receipt must take the process's code, not the check's status).
STUB_TEST = r"""
import json, os, sys
out = sys.argv[sys.argv.index("--output") + 1]
os.makedirs(out, exist_ok=True)
name = os.path.basename(out)
json.dump({"status": "passed" if name == "scene" else "failed"}, open(os.path.join(out, "check.json"), "w"))
sys.exit({"scene": 0, "nav": 3}[name])
"""


def entry(rung: int, name: str, test: str, requires=()) -> dict:
    return dict(rung=rung, name=name, test=test, spec=None, args=[], requires=list(requires), groups=["ladder"])


ENTRIES = [
    entry(1, "scene", "components/check_scene.py"),
    entry(2, "needs_ros", "components/check_ros.py", requires=["ros2"]),
    entry(6, "nav", "episodes/check_navigation.py"),
    entry(9, "slow", "episodes/check_pick_place.py"),
]


@pytest.fixture
def runner(tmp_path, monkeypatch):
    """main() over a fake ladder with stub test processes; returns a function giving (exit code, folder,
    receipt calls). ``slow`` times out; ``needs_ros`` is skipped."""
    script, log, stub = tmp_path / "write_manifest.py", tmp_path / "calls.jsonl", tmp_path / "stub_test.py"
    script.write_text(FAKE_SCRIPT)
    stub.write_text(STUB_TEST)
    monkeypatch.setenv(receipt.ENV, str(script))
    monkeypatch.setenv("FAKE_RECEIPT_LOG", str(log))
    monkeypatch.setattr(run_suite, "ladder", lambda path=None: ENTRIES)
    monkeypatch.setattr(run_suite, "missing", lambda requirements: ["ROS 2 Jazzy sourced"] if requirements else [])
    monkeypatch.setattr(run_suite, "PROJECT", tmp_path)
    real = subprocess.run

    def run(command, **kwargs):
        command = [str(part) for part in command]
        if not command[1].startswith(str(run_suite.ROOT)):  # The receipt script's calls.
            return real(command, **kwargs)
        if command[1].endswith("check_pick_place.py"):
            raise subprocess.TimeoutExpired(command, kwargs.get("timeout"))
        return real([sys.executable, str(stub), *command[2:]], **kwargs)

    monkeypatch.setattr(run_suite.subprocess, "run", run)

    def go(*flags):
        monkeypatch.setattr(sys, "argv", ["run_suite.py", "ladder", *flags])
        code = run_suite.main()
        (folder,) = (tmp_path / "output_check").iterdir()
        calls = [tuple(json.loads(line)) for line in log.read_text().splitlines()] if log.exists() else []
        return code, folder, calls

    return go


def manifest(folder) -> dict:
    return json.loads((folder / "manifest.json").read_text())


def test_the_suite_and_each_test_that_runs_get_a_receipt_finished_with_the_real_exit_code(runner):
    code, folder, calls = runner("--tag", "dev")
    assert code == 1  # nav failed, slow timed out.
    assert manifest(folder / "scene")["exit_status"] == 0
    assert manifest(folder / "nav")["exit_status"] == 3  # The process's code, not the check's status.
    assert manifest(folder / "slow")["exit_status"] == 124  # Stopped at --timeout.
    assert not (folder / "needs_ros").exists()  # A skipped test has no receipt.
    root = manifest(folder)
    assert root["exit_status"] == 1 and root["command"][0] == sys.executable
    options = root["options"]
    assert options[options.index("--repos") + 1] == str(run_suite.PROJECT)
    assert f"--config={run_suite.LADDER}" in options and options[options.index("--tag") + 1] == "dev"
    test = manifest(folder / "nav")
    assert test["command"][1].endswith("episodes/check_navigation.py") and "--output" in test["command"]
    assert calls[0] == (str(folder), "start") and calls[-1] == (str(folder), "finish")
    tests = [(path.rsplit("/", 1)[1], kind) for path, kind in calls[1:-1]]
    assert tests == [(name, kind) for name in ("scene", "nav", "slow") for kind in ("start", "finish")]


def test_a_suite_stopped_during_a_test_finishes_both_receipts(runner, monkeypatch):
    real = run_suite.subprocess.run

    def stopped(command, **kwargs):
        if str(command[1]).endswith("check_navigation.py"):
            raise KeyboardInterrupt
        return real(command, **kwargs)

    monkeypatch.setattr(run_suite.subprocess, "run", stopped)
    with pytest.raises(KeyboardInterrupt):
        runner()
    (folder,) = (run_suite.PROJECT / "output_check").iterdir()
    assert manifest(folder / "scene")["exit_status"] == 0
    assert manifest(folder / "nav")["exit_status"] == 130 and manifest(folder)["exit_status"] == 130


def test_without_the_script_the_tests_run_with_one_warning(runner, monkeypatch, capsys):
    monkeypatch.delenv(receipt.ENV)
    monkeypatch.setattr(receipt, "_warned", False)
    code, folder, calls = runner()
    assert calls == [] and not list(folder.rglob("manifest.json"))
    assert (folder / "scene" / "check.json").is_file() and (folder / "summary.json").is_file()
    assert capsys.readouterr().err.count(f"no run receipts ({receipt.ENV}") == 1


@pytest.mark.skipif(not os.environ.get(receipt.ENV), reason=f"{receipt.ENV} is not set (the real receipt script)")
def test_the_real_script_writes_and_finishes_a_receipt(tmp_path):
    """The receipt script this machine has, through the same calls: the PhyGS checkout with its commit
    and dirty flag, the tags, and the exit status after finishing."""
    folder = tmp_path / "rung"
    project = run_suite.PROJECT
    assert receipt.start(folder, [sys.executable, "-c", "pass"], repos=[project], purpose="test", tags=["dev"])
    receipt.finish(folder, 7)
    written = manifest(folder)
    (repo,) = [item for item in written["repos"] if item["root"] == str(project)]
    assert len(repo["sha"]) == 40 and "dirty" in repo
    assert written["tags"] == ["dev"] and written["exit_status"] == 7 and written["end_time"]
