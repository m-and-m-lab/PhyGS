# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Planner-independent timed trajectories and isolated experiment workers."""

from __future__ import annotations

import json
import os
import select
import subprocess
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np

from phygs.execution.manipulation.planners.interface import (
    IkRequest,
    IkResult,
    PlanRequest,
    PlanResult,
)


class JointTrajectory:
    """Named position samples [rad] at increasing times [s], bound to one episode."""

    def __init__(
        self,
        names: tuple[str, ...],
        times: np.ndarray,
        positions: np.ndarray,
        episode_id: int,
    ):
        self.names = tuple(names)
        self.times = np.array(times, dtype=float, copy=True)
        self.positions = np.array(positions, dtype=float, copy=True)
        self.episode_id = episode_id
        if len(set(names)) != len(names) or not names:
            raise ValueError("Trajectory joint names must be unique and nonempty")
        if self.times.ndim != 1 or len(self.times) < 2 or self.positions.shape != (len(self.times), len(names)):
            raise ValueError("Trajectory dimensions must be [time] and [time, joint]")
        if not np.isfinite(self.positions).all() or not np.isfinite(self.times).all():
            raise ValueError("Trajectory must be finite")
        if self.times[0] != 0 or np.any(np.diff(self.times) <= 0):
            raise ValueError("Trajectory times must start at zero and increase strictly")
        self.positions.flags.writeable = self.times.flags.writeable = False

    def sample(self, seconds: float) -> dict[str, float]:
        """Interpolate planner samples at elapsed simulation time [s]."""
        return {n: float(np.interp(seconds, self.times, self.positions[:, i])) for i, n in enumerate(self.names)}

    def velocity(self, seconds: float) -> dict[str, float]:
        """Return the derivative of the piecewise-linear reference [rad/s]."""
        if seconds >= self.times[-1]:
            return dict.fromkeys(self.names, 0.0)
        index = max(0, int(np.searchsorted(self.times, seconds, side="right")) - 1)
        values = (self.positions[index + 1] - self.positions[index]) / (self.times[index + 1] - self.times[index])
        return dict(zip(self.names, map(float, values), strict=True))

    def peak_speed(self) -> float:
        """Largest joint speed of the piecewise-linear reference [rad/s]."""
        return float(np.max(np.abs(np.diff(self.positions, axis=0) / np.diff(self.times)[:, None])))

    def slowed(self, maximum: float) -> tuple[JointTrajectory, float]:
        """This trajectory with its times stretched uniformly so no joint exceeds maximum [rad/s],
        and the time scale (1.0 when it already does not)."""
        scale = max(1.0, self.peak_speed() / maximum)
        if scale == 1.0:
            return self, 1.0
        return JointTrajectory(self.names, self.times * scale, self.positions, self.episode_id), scale


class EndEffectorTrajectory:
    """Timed tool poses [m, quaternion xyzw], bound to one episode.

    ``frame`` is ``"world"``, or ``"body"``: the robot base pose when the trajectory is submitted
    (base sway does not move the target). ``tool`` names the controlled point, one of the robot's
    tool frames (``robot.yaml`` ``gripper.tools``; default ``"hand"``, between the jaws). Producers
    time it; execution slows it down to the joint speed limit.
    """

    FRAMES = ("world", "body")

    def __init__(
        self,
        times: np.ndarray,
        poses: np.ndarray,
        episode_id: int,
        *,
        frame: str = "world",
        tool: str = "hand",
    ):
        self.times = np.array(times, dtype=float, copy=True)
        self.poses = np.array(poses, dtype=float, copy=True)
        self.episode_id, self.frame, self.tool = episode_id, frame, tool
        if frame not in self.FRAMES:
            raise ValueError(f"Frame {frame!r} is not one of {self.FRAMES}")
        if self.times.ndim != 1 or len(self.times) < 2 or self.poses.shape != (len(self.times), 7):
            raise ValueError("End-effector trajectory dimensions must be [time] and [time, 7]")
        if not np.isfinite(self.poses).all() or not np.isfinite(self.times).all():
            raise ValueError("Trajectory must be finite")
        if self.times[0] != 0 or np.any(np.diff(self.times) <= 0):
            raise ValueError("Trajectory times must start at zero and increase strictly")
        if np.any(np.abs(np.linalg.norm(self.poses[:, 3:], axis=1) - 1.0) > 1e-5):
            raise ValueError("Pose quaternions [x, y, z, w] must be normalized")
        self.poses.flags.writeable = self.times.flags.writeable = False


class PlannerProcess:
    """Persistent planner process; private pipe keeps vendor logs out of responses.

    Use submit() from simulation code and keep stepping while its future is pending.
    call() is the blocking transport operation (used for IK). Worker
    startup is synchronous; construct workers before starting a timed episode.
    """

    def __init__(self, command: list[str], config: dict, output: Path, env: dict[str, str]):
        output.mkdir(parents=True, exist_ok=True)
        path = output / "planner_config.json"
        path.write_text(json.dumps(config))
        read_fd, write_fd = os.pipe()
        self.log = (output / "planner.log").open("w")
        self.responses = os.fdopen(read_fd)
        started = time.perf_counter()
        self.process = subprocess.Popen(
            [*command, "--config", str(path), "--response_fd", str(write_fd)],
            stdin=subprocess.PIPE,
            stdout=self.log,
            stderr=self.log,
            text=True,
            pass_fds=(write_fd,),
            env=env,
        )
        os.close(write_fd)
        try:
            self.ready = self._read()
        except BaseException:
            self.close()
            raise
        self.startup_seconds = time.perf_counter() - started
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="manipulation-planner")
        self._pending: Future | None = None

    def _read(self) -> dict:
        if not select.select([self.responses], [], [], 180)[0]:
            raise TimeoutError("Planner response exceeded 180 wall seconds; inspect planner.log")
        line = self.responses.readline()
        if not line:
            raise RuntimeError("Planner exited without a response; inspect planner.log")
        result = json.loads(line)
        if "error" in result:
            raise RuntimeError(result["error"])
        return result

    def call(self, request: PlanRequest | IkRequest) -> PlanResult | IkResult:
        """Send one request and wait for its result (blocking transport)."""
        started = time.perf_counter()
        self.process.stdin.write(json.dumps(request.to_json(), allow_nan=False) + "\n")
        self.process.stdin.flush()
        result = self._read()
        result["roundtrip_seconds"] = time.perf_counter() - started
        return result

    @property
    def busy(self) -> bool:
        """True while a request runs, including one whose caller cancelled waiting for it."""
        return self._pending is not None and not self._pending.done()

    def submit(self, request: PlanRequest) -> Future:
        """Plan off the simulation thread; reject overlapping requests."""
        if self._pending is not None and not self._pending.done():
            raise RuntimeError("A planner request is already pending")
        json.dumps(request.to_json(), allow_nan=False)  # Reject non-finite values before the thread starts.
        self._pending = self._executor.submit(self.call, request)
        return self._pending

    def ik(self, request: IkRequest) -> IkResult:
        """Solve IK now; IK is fast, so the simulation thread waits. Fails while a plan runs."""
        if self.busy:
            raise RuntimeError("A planner request is pending; IK must wait until it finishes")
        return self.call(request)

    def close(self, wait_seconds: float = 10.0) -> None:
        """Stop the worker without raising, so cleanup never hides the error that ended a run."""
        if self.process.poll() is None:
            self.process.stdin.close()
            for stop in (lambda: None, self.process.terminate, self.process.kill):  # A busy planner may not exit.
                stop()
                try:
                    self.process.wait(timeout=wait_seconds)
                    break
                except subprocess.TimeoutExpired:
                    continue
        if hasattr(self, "_executor"):
            self._executor.shutdown(wait=True, cancel_futures=True)
        self.responses.close()
        self.log.close()
