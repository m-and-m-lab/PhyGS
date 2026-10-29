# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The overview strip under a replay's video, drawn from a run's trace (:mod:`.trace`) alone.

A header line: the simulated time, what the gripper holds, the arm's strongest contact now [N] and
the agent's step. Below it, on the left, the floor plan (:mod:`phygs.scene.plan`) framed on the run:
Nav2's route (blue), the path walked so far (orange), the robot (a yellow arrow) and, when the
agent read it, the target (a green ring); without a plan, the same on a 1 m grid. On the right,
the trace's lanes with a playhead.
"""

from __future__ import annotations

import math

import numpy as np

from phygs.scene.plan import Floor, Plan

BACKGROUND, BOX, OUTLINE = (10, 15, 21), (18, 24, 31), (60, 72, 88)
TEXT, MUTED, DIM = (255, 255, 255), (143, 160, 179), (102, 120, 140)
ROUTE, WALKED, ROBOT, TARGET = (47, 107, 255), (255, 106, 26), (255, 203, 5), (70, 195, 131)
HOLDING, CONTACT = (110, 231, 166), (255, 155, 126)
LANE_COLORS = {"Base": (110, 168, 255), "Arm": (255, 178, 79), "Gripper": (197, 155, 255)}
STATUS_COLORS = {
    "failed": (240, 106, 94),
    "aborted": (240, 106, 94),
    "rejected": (240, 106, 94),
    "cancelled": (90, 100, 112),
    "overridden": (90, 100, 112),
    "grip": (70, 195, 131),
    "touch": (255, 122, 89),
}
AGENT_NOW, AGENT_PAST = (255, 203, 5), (52, 60, 70)
HEADER = 34  # [px]
LABEL = 66  # [px] for the lane names.
FRAME_MARGIN = 1.2  # [m] around the run on the plan.


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
    """Draws the strip at any time of the run: ``draw(t)`` is an RGB image [height, width, 3].

    ``trace`` is :func:`phygs.recording.trace.build`'s dict; ``plan`` the scene's plan (None: a grid).
    """

    def __init__(self, trace: dict, plan: Plan | None, *, width: int = 1280, height: int = 300) -> None:
        from PIL import Image

        self.trace, self.width, self.height = trace, width, height
        self.duration = max(float(trace["duration"]), 1e-3)
        path = trace["path"]
        self._poses = np.asarray([pose[1:] for pose in path["poses"]], dtype=float).reshape(-1, 3)
        self._times = np.asarray([pose[0] for pose in path["poses"]], dtype=float)
        self._routes = path["routes"]
        self._target = path.get("target")
        self._fonts = _font(13), _font(13, bold=True), _font(11)
        side = height - HEADER - 12
        self.box = (16, HEADER, 16 + int(side * 16 / 7.6), HEADER + side)  # The plan.
        self.lanes_box = (self.box[2] + 24, HEADER, width - 16, height - 14)
        self._floor, self._view = self._frame(plan)
        bw, bh = self.box[2] - self.box[0], self.box[3] - self.box[1]
        if plan is not None:
            x0, y0, x1, y1 = self._view
            image = Image.open(plan.image_path(self._floor)).convert("RGB")
            self._plan = image.crop(tuple(round(v) for v in (x0, y0, x1, y1))).resize((bw, bh), Image.LANCZOS)
        else:
            self._plan = self._grid(bw, bh)

    # Framing.
    def _frame(self, plan: Plan | None) -> tuple[Floor, tuple[float, float, float, float]]:
        """The floor and the plan's pixel box (x0, y0, x1, y1) around the run, at the box's aspect."""
        points = [self._poses[:, :2]] if len(self._poses) else []
        points += [np.asarray(route["path"], dtype=float).reshape(-1, 2) for route in self._routes if route["path"]]
        if self._target is not None:
            points.append(np.asarray([self._target], dtype=float))
        xy = np.concatenate(points) if points else np.zeros((1, 2))
        low, high = xy.min(axis=0) - FRAME_MARGIN, xy.max(axis=0) + FRAME_MARGIN
        if plan is not None:
            floor = plan.floor_at(float(self.trace["path"].get("z", 0.0)))  # The floor the robot started on.
        else:
            span = max(high[0] - low[0], high[1] - low[1]) + 2.0
            floor = Floor("grid", "", 0.0, 0.0, 1000, 1000, tuple(((low + high) / 2).tolist()), 1000 / span)
        (px0, px1), (py1, py0) = (np.asarray(v) for v in floor.pixel([low[0], high[0]], [low[1], high[1]]))
        aspect = (self.box[2] - self.box[0]) / (self.box[3] - self.box[1])
        w, h = px1 - px0, py1 - py0
        if w / h < aspect:
            px0, px1 = px0 - (h * aspect - w) / 2, px1 + (h * aspect - w) / 2
        else:
            py0, py1 = py0 - (w / aspect - h) / 2, py1 + (w / aspect - h) / 2
        return floor, (float(px0), float(py0), float(px1), float(py1))

    def _grid(self, width: int, height: int):
        """A dark box with a 1 m grid, for a run without a plan."""
        from PIL import Image, ImageDraw

        image = Image.new("RGB", (width, height), BOX)
        draw = ImageDraw.Draw(image)
        x0, y0, x1, y1 = self._view
        low_x, high_y = self._floor.world(x0, y0)
        high_x, low_y = self._floor.world(x1, y1)
        for x in range(math.floor(low_x), math.ceil(high_x) + 1):
            u = self._to_box(x, 0.0)[0] - self.box[0]
            draw.line((u, 0, u, height), fill=(30, 38, 48))
        for y in range(math.floor(low_y), math.ceil(high_y) + 1):
            v = self._to_box(0.0, y)[1] - self.box[1]
            draw.line((0, v, width, v), fill=(30, 38, 48))
        return image

    def _to_box(self, x: float, y: float) -> tuple[float, float]:
        px, py = self._floor.pixel(x, y)
        x0, y0, x1, y1 = self._view
        return (
            self.box[0] + (float(px) - x0) * (self.box[2] - self.box[0]) / (x1 - x0),
            self.box[1] + (float(py) - y0) * (self.box[3] - self.box[1]) / (y1 - y0),
        )

    def _x(self, t: float) -> float:
        x0, x1 = self.lanes_box[0] + LABEL, self.lanes_box[2]
        return x0 + (x1 - x0) * min(max(t / self.duration, 0.0), 1.0)

    # Drawing.
    def draw(self, t: float) -> np.ndarray:
        from PIL import Image, ImageDraw

        image = Image.new("RGB", (self.width, self.height), BACKGROUND)
        draw = ImageDraw.Draw(image)
        self._header(draw, t)
        image.paste(self._plan, self.box[:2])
        self._map(draw, t)
        self._lanes(draw, t)
        return np.asarray(image)

    def _header(self, draw, t: float) -> None:
        mono, bold, _ = self._fonts
        held = next((item["label"] for item in self.trace["holding"] if item["s"] <= t <= item["e"]), None)
        force, what = 0.0, ""
        for lane in self.trace["lanes"]:
            for item in lane["items"]:
                if item.get("peak") and item["s"] <= t <= item["e"] and item["peak"] > force:
                    force, what = item["peak"], item.get("body", "")
        step = " > ".join(item["label"] for item in self.trace.get("steps", ()) if item["s"] <= t < item["e"])
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
            draw.text((x, 10), text, font=font, fill=color)
            x += draw.textlength(text, font=font) + 28

    def _map(self, draw, t: float) -> None:
        route = None
        for candidate in self._routes:
            if candidate["t"] <= t:
                route = candidate
        if route is not None and len(route["path"]) > 1:
            draw.line([self._to_box(x, y) for x, y in route["path"]], fill=ROUTE, width=3)
        if len(self._poses):
            k = max(0, int(np.searchsorted(self._times, t, side="right")) - 1)
            walked = [self._to_box(x, y) for x, y, _ in self._poses[: k + 1]]
            if len(walked) > 1:
                draw.line(walked, fill=WALKED, width=4, joint="curve")
            if self._target is not None:
                gx, gy = self._to_box(*self._target)
                draw.ellipse((gx - 7, gy - 7, gx + 7, gy + 7), outline=TARGET, width=3)
            x, y, yaw = self._poses[k]
            rx, ry = self._to_box(x, y)
            c, s = math.cos(-yaw), math.sin(-yaw)  # Image rows grow downward.
            arrow = [
                (rx + 11 * c, ry + 11 * s),
                (rx - 7 * c - 7 * s, ry - 7 * s + 7 * c),
                (rx - 7 * c + 7 * s, ry - 7 * s - 7 * c),
            ]
            draw.polygon(arrow, fill=ROBOT, outline=(23, 20, 10))
        draw.rectangle(self.box, outline=OUTLINE, width=1)

    def _lanes(self, draw, t: float) -> None:
        _, _, small = self._fonts
        x0, y0, x1, y1 = self.lanes_box
        lanes = self.trace["lanes"]
        height = (y1 - y0) / len(lanes)
        for row, lane in enumerate(lanes):
            top = y0 + row * height
            agent = lane["name"] == "Agent"
            draw.text((x0, top + height / 2 - 6), lane["name"].upper(), font=small, fill=MUTED)
            draw.rectangle((x0 + LABEL, top + 3, x1, top + height - 3), fill=BOX)
            labels = []
            for item in lane["items"]:
                a = self._x(item["s"])
                b = max(self._x(item["e"]) - 1, a + 2)  # A pixel apart from the next item.
                now = item["s"] <= t < item["e"]
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
            for a, _, label, ink in labels:  # Over every item, so a later item does not cover a label.
                if label:
                    draw.text((a + 4, top + height / 2 - 7), label, font=small, fill=ink)
        px = self._x(t)
        draw.line((px, y0, px, y1), fill=TEXT, width=2)
        step = _tick_step(self.duration)
        for second in np.arange(0.0, self.duration + 1e-6, step):
            draw.text((self._x(second) - 8, y1 + 1), f"{second:.0f} s", font=small, fill=DIM)


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
