# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""run_suite.py's order and replay choices, with the test and replay processes replaced (no Kit starts)."""

import json
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
from phygs_robot_tests import run_suite


def entry(rung: int, name: str, test: str) -> dict:
    return dict(rung=rung, name=name, test=test, spec=None, args=[], requires=[], groups=["ladder"])


# A component test, an agent test that passes and one that fails.
ENTRIES = [
    entry(1, "scene", "components/check_scene.py"),
    entry(6, "nav_ok", "episodes/check_navigation.py"),
    entry(9, "pick_bad", "episodes/check_pick_place.py"),
]
STATUS = {"scene": "passed", "nav_ok": "passed", "pick_bad": "failed"}


@pytest.fixture
def suite(tmp_path, monkeypatch):
    """Run main() with a fake ladder; returns a function giving (folder, the commands run, exit code)."""
    monkeypatch.setattr(run_suite, "ladder", lambda path=None: ENTRIES)
    monkeypatch.setattr(run_suite, "missing", lambda requirements: [])
    monkeypatch.setattr(run_suite, "PROJECT", tmp_path)
    monkeypatch.delenv("RP_WRITE_MANIFEST", raising=False)  # No receipts here (tests/test_run_suite_receipts.py).
    calls = []

    def fake_run(command, stdout=None, stderr=None, timeout=None):
        command = [str(c) for c in command]
        folder = tmp_path / "output_check"
        (run_folder,) = folder.iterdir()
        calls.append((command, (run_folder / "summary.json").exists()))
        if command[1].endswith("replay_run.py"):  # A replay: writes its video beside the recording.
            (Path(command[2]) / "replay").mkdir()
            (Path(command[2]) / "replay" / "replay.mp4").write_bytes(b"")
        else:  # A test: its verdict and, for an agent test, its recording.
            out = Path(command[command.index("--output") + 1])
            name = out.name
            (out / "run").mkdir(parents=True)
            check = {"status": STATUS[name], "checks": {"ok": STATUS[name] == "passed"}}
            (out / "check.json").write_text(json.dumps(check))
            if "episodes/" in command[1]:
                (out / "run" / "episode.rrd").write_bytes(b"")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(run_suite.subprocess, "run", fake_run)

    def run(*flags):
        monkeypatch.setattr(sys, "argv", ["run_suite.py", "ladder", *flags])
        code = run_suite.main()
        (folder,) = (tmp_path / "output_check").iterdir()
        return folder, calls, code

    return run


def replays(calls):
    return [Path(command[2]).parent.name for command, _ in calls if command[1].endswith("replay_run.py")]


def test_verdict_is_written_before_any_replay(suite, capsys):
    folder, calls, code = suite("--record")
    assert code == 1  # pick_bad failed
    assert replays(calls) == ["nav_ok", "pick_bad"]  # --record defaults to all agent tests, none for components
    first_replay = next(i for i, (command, _) in enumerate(calls) if command[1].endswith("replay_run.py"))
    assert all(not had for _, had in calls[:first_replay])  # no summary while the tests ran
    assert all(had for _, had in calls[first_replay:])  # the summary existed before every replay
    tests = [command for command, _ in calls[:first_replay]]
    assert ["--record" in command for command in tests] == [True, False, False]  # components film live
    lines = capsys.readouterr().out.splitlines()
    verdicts = [i for i, line in enumerate(lines) if " passed, " in line]
    first_rendered = next(i for i, line in enumerate(lines) if "replay rendered" in line)
    assert verdicts[0] < first_rendered and verdicts[-1] == len(lines) - 1  # verdict first, and again last


def test_summary_records_replay_minutes_and_keeps_statuses(suite):
    folder, calls, code = suite("--record")
    summary = json.loads((folder / "summary.json").read_text())
    assert [(e["name"], e["status"]) for e in summary] == list(STATUS.items())
    assert ["replay_minutes" in e for e in summary] == [False, True, True]
    page = (folder / "index.html").read_text()
    assert page.count("<video") == 2 and "not rendered" not in page


def test_replay_failed_renders_only_failed_agent_tests(suite):
    folder, calls, code = suite("--record", "--replay", "failed")
    assert replays(calls) == ["pick_bad"]
    page = (folder / "index.html").read_text()
    assert f"not rendered: <code>phygs replay {folder.resolve() / 'nav_ok' / 'run'}</code>" in page
    assert page.count("not rendered") == 1


def test_unrendered_command_carries_the_kit_args(suite):
    kit = "--/renderer/activeGpu=1 --/physics/cudaDevice=1"
    folder, calls, code = suite("--record", "--replay", "none", f"--kit_args={kit}")
    page = (folder / "index.html").read_text()
    run = folder.resolve() / "nav_ok" / "run"
    assert f"<code>phygs replay {run} --kit_args={shlex.quote(kit)}</code>".replace("'", "&#x27;") in page
    assert all(f"--kit_args={kit}" in command for command, _ in calls)  # every test ran on the same GPU


@pytest.mark.parametrize("flags", [("--record", "--replay", "none"), ()])
def test_replay_none_renders_nothing_and_names_the_command(suite, flags):
    folder, calls, code = suite(*flags)
    assert replays(calls) == []  # without --record the default is none
    summary = json.loads((folder / "summary.json").read_text())
    assert not any("replay_minutes" in e for e in summary)
    page = (folder / "index.html").read_text()
    assert page.count("not rendered: <code>phygs replay") == 2 and "<video" not in page


def test_to_render_selection():
    summary = [
        dict(name="scene", status="passed"),
        dict(name="nav_ok", status="passed"),
        dict(name="pick_bad", status="failed"),
        dict(name="look", status="skipped"),
    ]
    agents = {"nav_ok", "pick_bad", "look"}
    assert [e["name"] for e in run_suite.to_render(summary, agents, "all")] == ["nav_ok", "pick_bad"]
    assert [e["name"] for e in run_suite.to_render(summary, agents, "failed")] == ["pick_bad"]
    assert run_suite.to_render(summary, agents, "none") == []
