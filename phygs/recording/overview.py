# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The overview strip under a run's video (:mod:`.video`), drawn from a trace (:mod:`.trace`) alone.

Two header lines: the simulated time, what the gripper holds, the arm's strongest contact now [N]
and the agent's step; then the run's title and the skills' latest narrated line. Below them, on the left, the floor plan
(:mod:`phygs.scene.plan`) in a window :data:`WINDOW` wide that follows the robot: Nav2's route
(blue), the path walked so far (orange), the robot (a yellow arrow) and, when the agent read it, the
target (a green ring); without a plan, the same on a 1 m grid. On the right, the trace's lanes with
a playhead. A recording draws it with the trace so far, a replay with the whole run's; both look the
same.
"""

from __future__ import annotations

import math

import numpy as np

from phygs.scene.plan import Plan

BACKGROUND, BOX, OUTLINE, GRID = (10, 15, 21), (18, 24, 31), (60, 72, 88), (30, 38, 48)
TEXT, MUTED, DIM = (255, 255, 255), (143, 160, 179), (102, 120, 140)
ROUTE, WALKED, ROBOT, TARGET = (47, 107, 255), (255, 106, 26), (255, 203, 5), (70, 195, 131)
HOLDING, CONTACT = (110, 231, 166), (255, 155, 126)
LANE_COLORS = {"Base": (110, 168, 255), "Arm": (255, 178, 79), "Gripper": (197, 155, 255)}
STATUS_COLORS = {
    "failed": (240, 106, 94),
    "aborted": (240, 106, 94),
    "rejected": (240, 106, 94),
    "timed_out": (240, 106, 94),
    "cancelled": (90, 100, 112),
    "overridden": (90, 100, 112),
    "grip": (70, 195, 131),
    "touch": (255, 122, 89),
}
AGENT_NOW, AGENT_PAST = (255, 203, 5), (52, 60, 70)
RUNNING = frozenset({"active", "open"})  # Items still going when the trace was composed: they last until now.
HEADER = 52  # [px]: two lines.
LABEL = 66  # [px] for the lane names.
WINDOW = 12.0  # [m]: the plan window's width.


def _font(size: int, bold: bool = False):
    from PIL import ImageFont

    name = "DejaVuSansMono-Bold.ttf" if bold else "DejaVuSansMono.ttf"
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        try:
            return ImageFont.load_default(size=size)
        except TypeError:  # Pillow < 10.1: one bitmap size.
            return ImageFont.load_default()


class Overview:
    """Draws the strip, ``width`` x ``height`` [px]: ``draw(trace, t)`` is an RGB image [height, width, 3].

    ``plan`` is the scene's plan (None: a grid); the floor shown is the one under the trace's start
    height (``path.z``).
    """

    def __init__(self, plan: Plan | None, *, width: int = 1280, height: int = 300) -> None:
        self.plan, self.width, self.height = plan, width, height
        self._fonts = _font(13), _font(13, bold=True), _font(11)
        side = height - HEADER - 12
        self.box = (16, HEADER, 16 + int(side * 16 / 7.6), HEADER + side)  # The plan window.
        self.lanes_box = (self.box[2] + 24, HEADER, width - 16, height - 14)
        self.scale = (self.box[2] - self.box[0]) / WINDOW  # [px/m] in the window.
        self._image = None  # The floor's plan at the window's scale, and its floor (_scaled).
        self._trace, self._poses, self._times = None, np.zeros((0, 3)), np.zeros(0)
        self._center = (0.0, 0.0)  # The window's centre [m] now.

    # Framing.
    def _use(self, trace: dict) -> None:
        """Read a new trace's path (a recording hands a new trace every so often)."""
        if trace is self._trace:
            return
        self._trace = trace
        poses = trace["path"]["poses"]
        self._poses = np.asarray([pose[1:] for pose in poses], dtype=float).reshape(-1, 3)
        self._times = np.asarray([pose[0] for pose in poses], dtype=float)
        if self.plan is not None and self._image is None:
            self._scaled(self.plan.floor_at(float(trace["path"].get("z", 0.0))))

    def _scaled(self, floor) -> None:
        """The floor's image at the window's scale, with a window's margin of background around it."""
        from PIL import Image

        image = Image.open(self.plan.image_path(floor)).convert("RGB")
        factor = self.scale / floor.ppm
        size = (max(1, round(image.width * factor)), max(1, round(image.height * factor)))
        margin = (self.box[2] - self.box[0], self.box[3] - self.box[1])
        padded = Image.new("RGB", (size[0] + 2 * margin[0], size[1] + 2 * margin[1]), BOX)
        padded.paste(image.resize(size, Image.LANCZOS), margin)
        self._image, self._floor, self._factor, self._margin = padded, floor, factor, margin

    def _to_window(self, x: float, y: float) -> tuple[float, float]:
        """The window's pixel of world (x, y) [m]: +x right, +y up, the window centred on :attr:`_center`."""
        cx, cy = self._center
        return (
            (self.box[2] - self.box[0]) / 2 + (x - cx) * self.scale,
            (self.box[3] - self.box[1]) / 2 - (y - cy) * self.scale,
        )

    def _window(self):
        """The plan (or a 1 m grid) under the window, centred on :attr:`_center`, as an image."""
        from PIL import Image, ImageDraw

        width, height = self.box[2] - self.box[0], self.box[3] - self.box[1]
        if self._image is not None:
            column, row = self._floor.pixel(*self._center)
            left = round(float(column) * self._factor + self._margin[0] - width / 2)
            top = round(float(row) * self._factor + self._margin[1] - height / 2)
            return self._image.crop((left, top, left + width, top + height))
        window = Image.new("RGB", (width, height), BOX)
        draw = ImageDraw.Draw(window)
        cx, cy = self._center
        for x in range(math.floor(cx - WINDOW / 2), math.ceil(cx + WINDOW / 2) + 1):
            u = self._to_window(x, cy)[0]
            draw.line((u, 0, u, height), fill=GRID)
        span = height / self.scale
        for y in range(math.floor(cy - span / 2), math.ceil(cy + span / 2) + 1):
            v = self._to_window(cx, y)[1]
            draw.line((0, v, width, v), fill=GRID)
        return window

    def _x(self, t: float, duration: float) -> float:
        x0, x1 = self.lanes_box[0] + LABEL, self.lanes_box[2]
        return x0 + (x1 - x0) * min(max(t / duration, 0.0), 1.0)

    # Drawing.
    def draw(self, trace: dict, t: float, duration: float | None = None) -> np.ndarray:
        """The strip at ``t`` [s] from ``trace``; ``duration`` [s] is the time axis (default: the
        trace's duration)."""
        from PIL import Image, ImageDraw

        self._use(trace)
        duration = max(float(trace["duration"] if duration is None else duration), 1e-3)
        image = Image.new("RGB", (self.width, self.height), BACKGROUND)
        draw = ImageDraw.Draw(image)
        self._header(draw, trace, t)
        k = max(0, int(np.searchsorted(self._times, t, side="right")) - 1) if len(self._poses) else None
        if k is not None:
            self._center = tuple(self._poses[k, :2])
        image.paste(self._map(trace, t, k), self.box[:2])
        draw.rectangle(self.box, outline=OUTLINE, width=1)
        self._lanes(draw, trace, t, duration)
        return np.asarray(image)

    def _header(self, draw, trace: dict, t: float) -> None:
        mono, bold, small = self._fonts
        held = next((item["label"] for item in trace["holding"] if _now(item, t) or t == item["e"]), None)
        force, what = 0.0, ""
        for lane in trace["lanes"]:
            for item in lane["items"]:
                if item.get("peak") and item["s"] <= t <= item["e"] and item["peak"] > force:
                    force, what = item["peak"], item.get("body", "")
        step = " > ".join(item["label"] for item in trace.get("steps", ()) if _now(item, t))
        x = 16
        for text, font, color in (
            (f"sim {t:6.1f} s", bold, TEXT),
            ("holding " + (held or "nothing"), mono, HOLDING if held else MUTED),
            (f"contact {force:.0f} N ({what})" if force else "contact none", mono, CONTACT if force else MUTED),
            (f"step {step}" if step else "", mono, AGENT_NOW),
        ):
            if not text:
                continue
            text = _clip(draw, text, font, self.width - x - 16)
            draw.text((x, 8), text, font=font, fill=color)
            x += draw.textlength(text, font=font) + 28
        x = 16
        title = trace.get("title", "")
        if title:
            title = _clip(draw, title, mono, (self.width - 32) / 2)
            draw.text((x, 30), title, font=mono, fill=TEXT)
            x += draw.textlength(title, font=mono) + 28
        line = next((f"{kind}: {text}" for time, kind, text in reversed(trace.get("narration", ())) if time <= t), "")
        if line:
            draw.text((x, 30), _clip(draw, line, mono, self.width - x - 16), font=mono, fill=MUTED)

    def _map(self, trace: dict, t: float, k: int | None):
        """The window with the route, the target, the path walked and the robot at sample ``k``."""
        from PIL import ImageDraw

        window = self._window()
        draw = ImageDraw.Draw(window)
        path = trace["path"]
        route = None
        for candidate in path["routes"]:
            if candidate["t"] <= t:
                route = candidate
        if route is not None and len(route["path"]) > 1:
            draw.line([self._to_window(x, y) for x, y in route["path"]], fill=ROUTE, width=3)
        if path.get("target") is not None:
            gx, gy = self._to_window(*path["target"])
            draw.ellipse((gx - 7, gy - 7, gx + 7, gy + 7), outline=TARGET, width=3)
        if k is not None:
            walked = [self._to_window(x, y) for x, y, _ in self._poses[: k + 1]]
            if len(walked) > 1:
                draw.line(walked, fill=WALKED, width=4, joint="curve")
            x, y, yaw = self._poses[k]
            rx, ry = self._to_window(x, y)
            c, s = math.cos(-yaw), math.sin(-yaw)  # Image rows grow downward.
            arrow = [
                (rx + 11 * c, ry + 11 * s),
                (rx - 7 * c - 7 * s, ry - 7 * s + 7 * c),
                (rx - 7 * c + 7 * s, ry - 7 * s - 7 * c),
            ]
            draw.polygon(arrow, fill=ROBOT, outline=(23, 20, 10))
        return window

    def _lanes(self, draw, trace: dict, t: float, duration: float) -> None:
        _, _, small = self._fonts
        x0, y0, x1, y1 = self.lanes_box
        lanes = trace["lanes"]
        height = (y1 - y0) / len(lanes)
        for row, lane in enumerate(lanes):
            top = y0 + row * height
            agent = lane["name"] == "Agent"
            draw.text((x0, top + height / 2 - 6), lane["name"].upper(), font=small, fill=MUTED)
            draw.rectangle((x0 + LABEL, top + 3, x1, top + height - 3), fill=BOX)
            labels = []
            for item in lane["items"]:
                end = max(item["e"], t) if item.get("status") in RUNNING else item["e"]
                a = self._x(item["s"], duration)
                b = max(self._x(end, duration) - 1, a + 2)  # A pixel apart from the next item.
                now = _now(item, t)
                color = STATUS_COLORS.get(item.get("status"), LANE_COLORS.get(lane["name"], (60, 70, 82)))
                if agent:
                    color = AGENT_NOW if now else AGENT_PAST
                    if item.get("status") == "failed":
                        color = STATUS_COLORS["failed"]
                draw.rounded_rectangle(
                    (a, top + 5, b, top + height - 5),
                    radius=3,
                    fill=color,
                    outline=TEXT if now else None,
                    width=2 if now else 0,
                )
                ink = (23, 20, 10) if (not agent or now) else (233, 239, 246)
                labels.append((a, b, _short(draw, item["label"], small, b - a - 8), ink))
            drawn = -math.inf  # Where the last label drawn ends: overlapping items label only the first.
            for a, _, label, ink in labels:  # Over every item, so a later item does not cover a label.
                if label and a + 4 >= drawn:
                    draw.text((a + 4, top + height / 2 - 7), label, font=small, fill=ink)
                    drawn = a + 8 + draw.textlength(label, font=small)
        px = self._x(t, duration)
        draw.line((px, y0, px, y1), fill=TEXT, width=2)
        step = _tick_step(duration)
        for second in np.arange(0.0, duration + 1e-6, step):
            draw.text((self._x(second, duration) - 8, y1 + 1), f"{second:.0f} s", font=small, fill=DIM)


def _now(item: dict, t: float) -> bool:
    """Whether a trace item runs at ``t``: begun, and not ended (or still running in a live trace)."""
    return item["s"] <= t and (t < item["e"] or item.get("status") in RUNNING)


def _tick_step(duration: float) -> float:
    """Seconds between the time axis' labels: about eight of them."""
    for step in (1, 2, 5, 10, 20, 30, 60, 120, 300):
        if duration / step <= 8:
            return float(step)
    return 600.0


def _short(draw, label: str, font, width: float) -> str:
    """``label`` if it fits ``width`` [px], else its name before the arguments or the first word, else ''."""
    for text in (label, label.split("(")[0], label.split(" ")[0]):
        if text and draw.textlength(text, font=font) <= width:
            return text
    return ""


def _clip(draw, text: str, font, width: float) -> str:
    """``text`` cut with an ellipsis to fit ``width`` [px]."""
    if draw.textlength(text, font=font) <= width:
        return text
    while text and draw.textlength(text + "…", font=font) > width:
        text = text[:-1]
    return text + "…"
