# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Videos of robot runs: each camera frame between a text strip and a timeline strip.

The top strip shows a title, the simulation time, the real-time factor (RTF, simulated seconds per
wall second) and the frame rates (simulated, and written per wall second) over the last few wall
seconds, and the latest skill step of each kind (narration). The bottom strip shows every command
on a base and an arm lane (:mod:`phygs.recording.timeline`): one colour per command
(legend below the lanes), its name inside the span when it fits, a red outline when it failed, a
white one while it runs, faded when cancelled, and a playhead. Nothing is drawn over the image.
With panel sources (:class:`~phygs.recording.panels.PanelSources`), the robot's
cameras, navigation's costmap and the last arm plan are drawn in a grid to the image's right.
Frames are drawn and encoded on a writer thread, so recording slows the simulation little.
"""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

from phygs.recording.panels import Panels, PanelSources

# One colour (RGB) per command; others get a grey.
COMMAND_COLORS = {
    "navigate_to": (66, 133, 244),
    "command_velocity": (0, 172, 193),
    "stop": (120, 144, 156),
    "move_to_pose": (171, 71, 188),
    "execute_trajectory": (255, 152, 0),
    "move_joints": (236, 64, 122),
    "open_gripper": (205, 220, 57),
    "close_gripper": (141, 110, 99),
    "stow": (67, 160, 71),
    "look_at": (38, 198, 218),
    "command_joint_positions": (255, 202, 40),
}
OTHER = (158, 158, 158)
FAILED = (229, 57, 53)
CANCELLED = frozenset({"cancelled", "overridden"})
LANES = ("base", "arm")
BACKGROUND = (24, 26, 30)
TEXT, MUTED = (240, 240, 240), (170, 174, 180)
# Fixed strip sizes, so every frame of a video has the same size: narration lines (one per kind:
# manipulation, navigation, locomotion, check) and legend rows.
LINES, LEGEND_ROWS = 4, 2


def _font(size: int):
    from PIL import ImageFont

    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1: one bitmap size.
        return ImageFont.load_default()


def _fit(draw, text: str, font, width: float) -> str:
    """The text, shortened with an ellipsis to fit the width [px]."""
    if draw.textlength(text, font=font) <= width:
        return text
    while text and draw.textlength(text + "…", font=font) > width:
        text = text[:-1]
    return text + "…"


def _legend(draw, names: list[str], font, width: int) -> list[tuple[int, int, str]]:
    """Legend entries (x, row, name) wrapped to the width [px]."""
    entries, x, row = [], 10, 0
    for name in names:
        step = int(22 + draw.textlength(name, font=font) + 16)
        if x + step > width - 10 and x > 10:
            x, row = 10, row + 1
        entries.append((x, row, name))
        x += step
    return entries


def compose(rgb: np.ndarray, *, title: str, stats: str, sim_time: float, lines: list[str], spans: list) -> np.ndarray:
    """The frame: the text strip, the image and the timeline strip, stacked (the image unchanged)."""
    from PIL import Image, ImageDraw

    image = Image.fromarray(np.ascontiguousarray(rgb[..., :3]))
    width = image.width
    font, small = _font(16), _font(14)
    rows = [title, stats, *lines[:LINES]]
    top = 8 + 21 * (2 + LINES)
    lane_h = 22
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    legend = _legend(probe, list(dict.fromkeys(span.name for span in spans)), small, width)
    legend = [entry for entry in legend if entry[1] < LEGEND_ROWS]
    axis_y = 8 + len(LANES) * (lane_h + 4)
    bottom = axis_y + 24 + 20 * LEGEND_ROWS + 26
    frame = Image.new("RGB", (width, top + image.height + bottom), BACKGROUND)
    frame.paste(image, (0, top))
    draw = ImageDraw.Draw(frame)
    for row, line in enumerate(rows):
        color = (255, 214, 0) if row == 0 else TEXT
        draw.text((10, 5 + 21 * row), _fit(draw, line, font, width - 20), fill=color, font=font)
    y0 = top + image.height
    span_s = max(60.0, sim_time * 1.05)
    left, right = 54, width - 10
    to_x = lambda t: left + (t / span_s) * (right - left)  # noqa: E731
    lane_y = {lane: y0 + 8 + i * (lane_h + 4) for i, lane in enumerate(LANES)}
    for lane, y in lane_y.items():
        draw.text((8, y + 3), lane.upper(), fill=MUTED, font=small)
        draw.rectangle((left, y, right, y + lane_h), fill=(44, 47, 54))
    for span in spans:
        y = lane_y.get(span.lane)
        if y is None:
            continue
        x0 = to_x(span.start)
        x1 = max(to_x(sim_time if span.end is None else span.end), x0 + 3)
        color = COMMAND_COLORS.get(span.name, OTHER)
        outline = None
        if span.status in CANCELLED:
            color = tuple(c // 2 + 22 for c in color)
        elif span.status == "active":
            outline = (255, 255, 255)
        elif span.status != "succeeded":
            outline = FAILED
        draw.rectangle((x0, y, x1, y + lane_h), fill=color, outline=outline, width=3 if outline else 0)
        if draw.textlength(span.name, font=small) + 8 <= x1 - x0:
            draw.text((x0 + 4, y + 3), span.name, fill=(0, 0, 0), font=small)
    axis = y0 + axis_y
    step = 120 if span_s > 600 else 60 if span_s > 300 else 30 if span_s > 120 else 10
    for t in range(0, int(span_s) + 1, step):
        x = to_x(t)
        draw.line((x, axis, x, axis + 4), fill=MUTED)
        if x + 36 < right:
            draw.text((x + 3, axis + 2), f"{t} s", fill=MUTED, font=small)
    x = to_x(sim_time)
    draw.rectangle((x - 1, y0 + 2, x + 1, axis), fill=(255, 214, 0))
    for x, row, name in legend:
        y = axis + 24 + 20 * row
        draw.rectangle((x, y + 2, x + 16, y + 16), fill=COMMAND_COLORS.get(name, OTHER))
        draw.text((x + 22, y), name, fill=TEXT, font=small)
    status = "outline: red failed, white running; faded: cancelled"
    draw.text((10, frame.height - 22), _fit(draw, status, small, width - 20), fill=MUTED, font=small)
    return np.asarray(frame)


class Video:
    """An mp4 at ``fps`` [frames per simulated second]; frames are composed and encoded on a thread."""

    WINDOW_S = 5.0  # RTF and wall frame rate over the last this many wall seconds.

    def __init__(self, path: Path, *, fps: float) -> None:
        self.path, self.fps = Path(path), fps
        self._recent: deque[tuple[float, float]] = deque()  # (wall [s], sim [s]) per frame.
        self._frames: queue.Queue = queue.Queue(maxsize=64)
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._write, name="video", daemon=True)
        self._thread.start()

    def frame(
        self,
        rgb: np.ndarray,
        *,
        title: str,
        sim_time: float,
        lines: list[str],
        spans: list,
        panels: PanelSources | None = None,
    ) -> None:
        """Queue a frame; ``panels`` (copied data, not changed afterwards) are drawn beside the image."""
        if self._error is not None:
            raise RuntimeError(f"Video writer failed: {self._error}") from self._error
        wall = time.perf_counter()
        if self._recent and sim_time < self._recent[-1][1]:
            self._recent.clear()  # A new episode: simulated time started over.
        self._recent.append((wall, sim_time))
        while len(self._recent) > 2 and wall - self._recent[0][0] > self.WINDOW_S:
            self._recent.popleft()
        start, sim0 = self._recent[0]
        elapsed = wall - start
        rtf = (sim_time - sim0) / elapsed if elapsed > 0 else 0.0
        rate = (len(self._recent) - 1) / elapsed if elapsed > 0 else 0.0
        stats = f"sim {sim_time:7.2f} s   RTF {rtf:4.2f}   {self.fps:.2f} fps sim   {rate:4.1f} fps wall"
        spans = [type(span)(**vars(span)) for span in spans]  # Copies: the thread draws them later.
        overlay = dict(title=title, stats=stats, sim_time=sim_time, lines=list(lines), spans=spans)
        self._frames.put((np.array(rgb[..., :3]), panels, overlay))

    def _write(self) -> None:
        import imageio.v2 as imageio

        writer, drawer = None, None
        try:
            while (item := self._frames.get()) is not None:
                rgb, panels, overlay = item
                if panels is not None:
                    drawer = drawer or Panels()
                    rgb = drawer.compose(rgb, panels)
                frame = compose(rgb, **overlay)
                if writer is None:
                    writer = imageio.get_writer(self.path, fps=self.fps, codec="libx264", quality=8)
                writer.append_data(frame)
        except BaseException as error:  # noqa: BLE001 - reported on the next frame or at close.
            self._error = error
        finally:
            if writer is not None:
                writer.close()

    def close(self) -> None:
        if self._thread.is_alive():
            self._frames.put(None)
            self._thread.join()
        if self._error is not None:
            raise RuntimeError(f"Video writer failed: {self._error}") from self._error
