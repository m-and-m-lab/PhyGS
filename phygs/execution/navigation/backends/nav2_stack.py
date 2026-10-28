# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The Nav2 servers as a child process group: launched from a navigation experiment, stopped on close.

``phygs/execution/navigation/launch/nav2.launch.py`` builds the planner, controller and costmap parameters from
an experiment file (the map, the transit footprint profile, the navigation limits and the Nav2
namespace). :class:`Nav2Stack` writes that file, launches the servers with ``ros2 launch`` in
their own process group and stops the whole group (the launcher and its servers) on close. If this
process dies without closing (killed, crashed), the launcher gets SIGTERM (a parent-death signal)
and stops its servers, which also stop when the launcher dies: a stale stack on the same ROS domain
answers the next run's requests. The process needs a sourced ROS 2 Jazzy environment.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
from pathlib import Path

import yaml

LAUNCH = Path(__file__).resolve().parents[1] / "launch" / "nav2.launch.py"


class Nav2Stack:
    """Nav2's planner and controller servers for one run, logged to ``folder/nav2.log``.

    ``experiment`` is the launcher's experiment (written to ``folder/nav2_experiment.yaml``);
    ``sensor_mode`` is ``occupancy_grid`` (the bridge publishes the map) or ``pointcloud``
    (Nav2's obstacle layer maps the published camera points).
    """

    SENSOR_MODES = ("occupancy_grid", "pointcloud")

    def __init__(self, experiment: dict, folder: Path, *, sensor_mode: str = "occupancy_grid") -> None:
        if sensor_mode not in self.SENSOR_MODES:
            raise ValueError(f"Unknown Nav2 sensor mode {sensor_mode!r}; expected one of {self.SENSOR_MODES}.")
        if shutil.which("ros2") is None:
            raise RuntimeError("Nav2 needs a sourced ROS 2 Jazzy environment: source /opt/ros/jazzy/setup.bash.")
        folder.mkdir(parents=True, exist_ok=True)
        self.experiment_path = folder / "nav2_experiment.yaml"
        self.experiment_path.write_text(yaml.safe_dump(experiment, sort_keys=False))
        # The parent-death signal follows the thread that starts the launcher: build the stack on a
        # thread that lives as long as the run (the main thread).
        self.command = [
            *(["setpriv", "--pdeathsig", "TERM"] if shutil.which("setpriv") else []),
            "ros2",
            "launch",
            str(LAUNCH),
            f"experiment:={self.experiment_path}",
            f"sensor_mode:={sensor_mode}",
        ]
        with (folder / "nav2.log").open("w") as log:
            self._process: subprocess.Popen | None = subprocess.Popen(
                self.command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def close(self) -> None:
        """Stop the launcher and every server it started (its process group)."""
        if self._process is None:
            return
        stop_group(self._process)
        self._process = None


def stop_group(process: subprocess.Popen, timeout: float = 10.0) -> None:
    """Stop a process started with ``start_new_session`` and its descendants: SIGINT, SIGTERM, then SIGKILL."""
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            break
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            continue
    process.wait()
