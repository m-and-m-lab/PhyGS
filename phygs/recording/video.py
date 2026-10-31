# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Videos of robot runs, one look for the recording (``--record``) and the replay (``phygs replay``).

A frame is the outside view (a camera following the robot, :class:`Chase`), the panels to its right
(:mod:`phygs.recording.panels`: the front and hand cameras, their depth, navigation and the arm
planner) and the overview strip under both (:mod:`phygs.recording.overview`: the clock, what the
gripper holds, the arm's contact force, the agent's step and the skills' latest line, the floor
plan, and the timeline of steps, commands, grips and bumps from a trace, :mod:`phygs.recording.trace`).
Nothing is drawn over the camera image. A recording draws as the run goes, with the trace so far
and a time axis that grows (:func:`live_axis`); a replay knows the whole run. Frames are drawn and
encoded on a writer thread, so recording slows the simulation little.
"""

from __future__ import annotations

import math
import queue
import threading
from pathlib import Path

import numpy as np

from phygs.recording.overview import Overview
from phygs.recording.panels import Panels, PanelSources, layout
from phygs.scene.plan import Plan

CHASE = (-2.4, -1.4, 1.7)  # [m]: the chase camera from its target, in the robot's heading frame.
CHASE_SECONDS = 0.6  # [s]: the chase camera's smoothing time.
CHASE_KEEP = 1.6  # [m]: the chase camera keeps what is this close to its target, clipping nearer.
FOCAL_LENGTH = 20.0  # [mm] on a 20.955 mm aperture: the outside camera's lens.
AXIS_MIN, AXIS_STEP = 60.0, 30.0  # [s]: a recording's time axis spans at least AXIS_MIN, in AXIS_STEP steps.


def live_axis(time_seconds: float) -> float:
    """A recording's time axis [s] at ``time_seconds``: the run so far, rounded up to whole steps."""
    return max(AXIS_MIN, AXIS_STEP * math.ceil(time_seconds / AXIS_STEP))


def strip_height(height: int) -> int:
    """The overview strip's height [px] under an outside view ``height`` [px] high."""
    return int(min(320, max(230, 0.42 * height)))


class Chase:
    """Where the outside camera looks from: behind and to the right of the robot at :data:`CHASE`
    (its heading frame), turning with it, smoothed over :data:`CHASE_SECONDS` of camera periods
    (``period`` [s]), with the near clip :data:`CHASE_KEEP` before the robot so walls in between are
    cut away. With ``eye`` (world [m]) the camera stays there and only turns to the robot."""

    def __init__(self, period: float, *, eye=None) -> None:
        self.period = period
        self.eye = None if eye is None else np.asarray(eye, dtype=float)
        self._last: tuple[np.ndarray, np.ndarray, float] | None = None

    def reset(self) -> None:
        """Jump to the robot at the next :meth:`aim` (a new episode)."""
        self._last = None

    def aim(self, pose) -> tuple[np.ndarray, np.ndarray, float]:
        """The eye and target (world [m]) and the near clip [m] for the robot's base ``pose``
        [x, y, z, qx, qy, qz, qw] [m]."""
        pose = np.asarray(pose, dtype=float)
        x, y, z, w = pose[3:]
        heading = math.atan2(2 * (x * y + w * z), 1 - 2 * (y * y + z * z))
        alpha = 1.0 - math.exp(-self.period / CHASE_SECONDS)
        if self._last is not None:
            turn = math.atan2(math.sin(heading - self._last[2]), math.cos(heading - self._last[2]))
            heading = self._last[2] + alpha * turn
        c, s = math.cos(heading), math.sin(heading)
        target = np.array([pose[0] + 0.35 * c, pose[1] + 0.35 * s, 0.45])
        if self.eye is not None:
            eye = self.eye
        else:
            back, side, up = CHASE
            eye = target + np.array([c * back - s * side, s * back + c * side, up])
        if self._last is not None:
            eye = self._last[0] + alpha * (eye - self._last[0])
            target = self._last[1] + alpha * (target - self._last[1])
        self._last = (eye, target, heading)
        near = 0.05 if self.eye is not None else max(0.05, float(np.linalg.norm(target - eye)) - CHASE_KEEP)
        return eye, target, near


class Video:
    """An mp4 at ``fps`` [frames per simulated second] in the one look (see the module docstring).

    ``plan`` is the scene's floor plan for the strip (None: a 1 m grid); ``panels`` and ``overview``
    leave the panels or the strip out. Every frame of a video has the same size.
    """

    def __init__(
        self, path: Path, *, fps: float, plan: Plan | None = None, panels: bool = True, overview: bool = True
    ) -> None:
        self.path, self.fps = Path(path), fps
        self._plan, self._panels, self._overview = plan, panels, overview
        self._frames: queue.Queue = queue.Queue(maxsize=64)
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._write, name="video", daemon=True)
        self._thread.start()

    def frame(
        self,
        outside: np.ndarray,
        *,
        sources: PanelSources | None,
        trace: dict | None,
        sim_time: float,
        duration: float | None = None,
    ) -> None:
        """Queue a frame: the outside image, the panels' ``sources`` and the ``trace`` (copied data,
        not changed afterwards) at ``sim_time`` [s]; ``duration`` [s] is the time axis (default: the
        trace's)."""
        if self._error is not None:
            raise RuntimeError(f"Video writer failed: {self._error}") from self._error
        self._frames.put((np.array(outside[..., :3]), sources, trace, float(sim_time), duration))

    def _write(self) -> None:
        import imageio.v2 as imageio

        writer, drawer, strip = None, Panels(), None
        try:
            while (item := self._frames.get()) is not None:
                outside, sources, trace, sim_time, duration = item
                frame = outside
                if self._panels:
                    frame = drawer.compose(outside, sources) if sources is not None else layout(outside, {})
                if self._overview and trace is not None:
                    if strip is None:
                        strip = Overview(self._plan, width=frame.shape[1], height=strip_height(outside.shape[0]))
                    frame = np.concatenate([frame, strip.draw(trace, sim_time, duration)], axis=0)
                frame = np.ascontiguousarray(frame[: frame.shape[0] // 2 * 2, : frame.shape[1] // 2 * 2])
                if writer is None:
                    writer = imageio.get_writer(
                        self.path,
                        fps=self.fps,
                        codec="libx264",
                        quality=8,
                        macro_block_size=1,
                        pixelformat="yuv420p",
                        ffmpeg_params=["-movflags", "+faststart"],
                    )
                writer.append_data(frame)
        except BaseException as error:  # noqa: BLE001 - reported on the next frame or at close.
            self._error = error
            while self._frames.get() is not None:  # Let the producer finish; its frames are dropped.
                pass
        finally:
            if writer is not None:
                writer.close()

    def close(self) -> None:
        if self._thread.is_alive():
            self._frames.put(None)
            self._thread.join()
        if self._error is not None:
            raise RuntimeError(f"Video writer failed: {self._error}") from self._error
