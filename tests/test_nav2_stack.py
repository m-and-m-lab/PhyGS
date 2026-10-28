# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The Nav2 server process group: refused without ROS, stopped with every child (no ROS needed)."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

from phygs.execution.navigation.backends import nav2_stack
from phygs.execution.navigation.backends.nav2_stack import Nav2Stack, stop_group


def test_the_stack_needs_ros_and_a_known_sensor_mode(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="sensor mode"):
        Nav2Stack({}, tmp_path, sensor_mode="lidar")
    monkeypatch.setattr(nav2_stack.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="ROS 2 Jazzy"):
        Nav2Stack({}, tmp_path)


def test_the_stack_writes_its_experiment_and_stops_its_whole_group(tmp_path, monkeypatch):
    launched, real = [], subprocess.Popen

    def popen(command, **options):
        launched.append(command)
        return real(["sh", "-c", "sleep 60 & sleep 60"], **options)  # A launcher with a child server.

    monkeypatch.setattr(nav2_stack.shutil, "which", lambda name: "/usr/bin/ros2")
    monkeypatch.setattr(nav2_stack.subprocess, "Popen", popen)
    stack = Nav2Stack(dict(nav2=dict(namespace="spot_navigation")), tmp_path / "nav2")
    assert yaml.safe_load(stack.experiment_path.read_text()) == dict(nav2=dict(namespace="spot_navigation"))
    assert launched[0][:6] == ["setpriv", "--pdeathsig", "TERM", "ros2", "launch", str(nav2_stack.LAUNCH)]
    assert nav2_stack.LAUNCH.exists()
    assert launched[0][-1] == "sensor_mode:=occupancy_grid" and stack.running
    monkeypatch.undo()
    process = stack._process
    stack.close()
    assert not stack.running and process.returncode is not None
    group = subprocess.run(["pgrep", "-g", str(process.pid)], capture_output=True, text=True)
    assert group.stdout.strip() == ""  # The child sleep stopped with the launcher.
    stack.close()  # A second close does nothing.


def test_stop_group_returns_for_a_process_that_already_ended():
    process = subprocess.Popen(["true"], start_new_session=True)
    process.wait()
    stop_group(process)


def test_the_launcher_stops_when_the_process_that_started_it_is_killed(tmp_path):
    """A run killed without closing its stack (SIGKILL, a crash) must not leave Nav2 on its domain."""
    script = tmp_path / "run.py"
    script.write_text(
        "import subprocess, sys, time\n"
        "child = subprocess.Popen(['setpriv', '--pdeathsig', 'TERM', 'sleep', '60'], start_new_session=True)\n"
        "print(child.pid, flush=True)\n"
        "time.sleep(60)\n"
    )
    run = subprocess.Popen([sys.executable, str(script)], stdout=subprocess.PIPE, text=True)
    child = int(run.stdout.readline())

    def state():  # The child's process state; None once it is gone.
        try:
            return Path(f"/proc/{child}/stat").read_text().split()[2]
        except FileNotFoundError:
            return None

    for _ in range(50):  # Until setpriv has armed the signal and become sleep.
        if Path(f"/proc/{child}/cmdline").read_bytes().startswith(b"sleep"):
            break
        time.sleep(0.02)
    run.kill()  # As a timeout or crash would: no cleanup runs.
    run.wait()
    for _ in range(50):
        if state() in (None, "Z"):  # Gone, or ended and not reaped yet.
            break
        time.sleep(0.1)
    else:
        subprocess.run(["kill", "-9", str(child)])
        raise AssertionError("The launcher outlived the killed run")
