# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Panels beside a run's third-person video: what the robot senses and what its planners use.

A grid of labelled panels (:data:`PANELS`) to the right of the main image:

- ``front``, ``front depth``: the two front cameras joined into one forward view
  (:class:`~phygs.observations.front_view.FrontStitcher`), and its distance image; a white
  line marks where the cameras' fields of view end (the blind wedges above and below).
- ``hand``, ``hand depth``: the hand camera and its depth.
- ``navigation``: Nav2's global costmap around the robot, map +x right and +y up, in RViz's
  "costmap" colour scheme on RViz's background (:func:`costmap_colors`), Nav2's route (green),
  the goal (with its heading), the robot's footprint (yellow) and the commanded velocity (cyan).
- ``arm planner``: the last arm plan in the body frame, seen from the robot's right and framed on
  the arm's motion: the arm at the start (grey), along the plan (faint) and at the end (orange,
  with its collision spheres), the jaw's path (magenta), the planner's obstacle voxels coloured
  by their distance to the arm (red: within 5 cm, blue: 30 cm or more), its known boxes
  (outlined) and the floor (a grid).

A panel without data says so. Everything here is plain arrays (:class:`PanelSources`), so the
sources are copied on the simulation thread (:class:`PanelFeed`) and drawn on the video writer's
thread (:class:`Panels`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from phygs.observations.front_view import CameraFrame, FrontStitcher, camera_frame, depth_colors

PANELS = ("front", "hand", "navigation", "front depth", "hand depth", "arm planner")  # Row by row.
COLUMNS = 3
LABEL_H = 18  # [px] above each panel.
DEPTH_RANGE = 6.0  # [m]; colour scale of the depth panels (blue near, red at and beyond).
NAVIGATION_SPAN = 6.0  # [m]; width of the costmap window around the robot.
BACKGROUND, LABEL, MUTED = (24, 26, 30), (240, 240, 240), (130, 134, 140)
RVIZ_BACKGROUND = (48, 48, 48)  # RViz's default background: where the costmap is free (transparent).
ROUTE, GOAL, FOOTPRINT, COMMAND = (0, 230, 0), (255, 255, 255), (240, 200, 0), (0, 190, 210)
ARM_START, ARM_END, TOOL_PATH = (170, 170, 170), (255, 150, 40), (230, 40, 200)


def _rviz_costmap_palette() -> np.ndarray:
    """RGB [256,3] for Nav2 costs 0-255: Nav2's translation to 0-100 (Costmap2DPublisher: 0 free, 1-252 to
    1 + 97 (c - 1) / 251, 253 inscribed 99, 254 lethal 100, 255 unknown -1), then RViz's "costmap" scheme
    (rviz_default_plugins palette_builder.cpp: 1-98 blue to red, 99 cyan, 100 magenta, -1 grey-teal; 0 is
    transparent, here RViz's background)."""
    palette = np.zeros((256, 3), dtype=np.uint8)
    palette[0] = RVIZ_BACKGROUND
    for cost in range(1, 253):
        value = 255 * (1 + 97 * (cost - 1) // 251) // 100
        palette[cost] = (value, 0, 255 - value)
    palette[253], palette[254], palette[255] = (0, 255, 255), (255, 0, 255), (112, 137, 134)
    return palette


RVIZ_COSTMAP = _rviz_costmap_palette()
INSCRIBED, LETHAL, UNKNOWN = (tuple(int(c) for c in RVIZ_COSTMAP[cost]) for cost in (253, 254, 255))


@dataclass(frozen=True)
class NavigationView:
    """Navigation's state for the panel. Poses are world-frame x, y [m], yaw [rad].

    ``costs`` is Nav2's costmap [y,x] (row 0 at the lowest y; 0-252 cost, 253 inscribed, 254
    lethal, 255 unknown) with its ``origin`` [m] and ``resolution`` [m]; None before it serves.
    ``footprint`` is the robot's outline in the body frame [N,2] [m]; ``path`` the route [m];
    ``goal`` (x, y, heading or None); ``command`` the body-frame velocity [m/s, m/s, rad/s]."""

    pose: tuple[float, float, float]
    costs: np.ndarray | None = None
    origin: tuple[float, float] = (0.0, 0.0)
    resolution: float = 0.05
    footprint: np.ndarray | None = None
    path: tuple[tuple[float, float], ...] = ()
    goal: tuple[float, float, float | None] | None = None
    command: tuple[float, float, float] | None = None


@dataclass(frozen=True)
class PlannerView:
    """An arm plan in the body frame [m]: obstacle voxel centres [N,3] of size ``voxel_size``, known
    boxes (centre, size), the arm's link origins shoulder to jaw [K,3] at the start, along the plan
    (``arm_path``) and at the end, the arm's collision spheres [x, y, z, r] at the end, the jaw's
    path [M,3], the floor height (None: unknown) and a one-line ``label`` (time and result)."""

    label: str
    voxels: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    voxel_size: float = 0.02
    boxes: tuple[tuple[tuple[float, float, float], tuple[float, float, float]], ...] = ()
    arm_start: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    arm_path: tuple[np.ndarray, ...] = ()
    arm_end: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    end_spheres: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))
    tool_path: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    floor: float | None = None


@dataclass(frozen=True)
class PanelSources:
    """One video frame's panel data: the front camera frames (one, or a left and a right one to
    stitch), the hand camera's frame, navigation and the last arm plan; any of them may be missing."""

    front: tuple[CameraFrame, ...] = ()
    hand: CameraFrame | None = None
    navigation: NavigationView | None = None
    planner: PlannerView | None = None

    def drawable(self) -> bool:
        """Whether any panel has data: the front camera(s), the hand camera, navigation or a plan."""
        return bool(self.front) or self.hand is not None or self.navigation is not None or self.planner is not None


def _font(size: int):
    from PIL import ImageFont

    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1: one bitmap size.
        return ImageFont.load_default()


def fit(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """The image scaled to fit width x height [px] without distortion, centred on the background."""
    from PIL import Image

    source = Image.fromarray(np.ascontiguousarray(image[..., :3], dtype=np.uint8))
    scale = min(width / source.width, height / source.height)
    size = (max(1, round(source.width * scale)), max(1, round(source.height * scale)))
    canvas = Image.new("RGB", (width, height), BACKGROUND)
    canvas.paste(source.resize(size, Image.BILINEAR), ((width - size[0]) // 2, (height - size[1]) // 2))
    return np.asarray(canvas)


def layout(main: np.ndarray, panels: dict[str, np.ndarray | None]) -> np.ndarray:
    """The main image with the :data:`PANELS` grid to its right (rows share the main image's height);
    each panel under its label, a missing one marked "no data"."""
    from PIL import Image, ImageDraw

    height = main.shape[0]
    rows = math.ceil(len(PANELS) / COLUMNS)
    cell_h = height // rows
    cell_w = 4 * (cell_h - LABEL_H) // 3
    frame = Image.new("RGB", (main.shape[1] + COLUMNS * cell_w, height), BACKGROUND)
    frame.paste(Image.fromarray(np.ascontiguousarray(main[..., :3], dtype=np.uint8)), (0, 0))
    draw, font = ImageDraw.Draw(frame), _font(14)
    for index, name in enumerate(PANELS):
        x, y = main.shape[1] + (index % COLUMNS) * cell_w, (index // COLUMNS) * cell_h
        draw.text((x + 6, y + 2), name.upper(), fill=LABEL, font=font)
        image = panels.get(name)
        if image is None:
            draw.text((x + 6, y + LABEL_H + 6), "no data", fill=MUTED, font=font)
        else:
            frame.paste(Image.fromarray(fit(image, cell_w - 4, cell_h - LABEL_H - 2)), (x + 2, y + LABEL_H))
    width = frame.width - frame.width % 2  # H.264 needs even sizes.
    return np.asarray(frame)[: height - height % 2, :width]


def costmap_colors(costs: np.ndarray) -> np.ndarray:
    """RGB uint8 [y,x,3] of Nav2 costs (0-255) as RViz draws Nav2's published costmap in its "costmap"
    scheme (:data:`RVIZ_COSTMAP`)."""
    return RVIZ_COSTMAP[costs.astype(np.uint8)]


def navigation_image(view: NavigationView, width: int = 400, height: int = 300) -> np.ndarray:
    """The costmap window of :data:`NAVIGATION_SPAN` around the robot, +x right and +y up, with the
    route, goal, footprint and command drawn on it."""
    from PIL import Image, ImageDraw

    x0, y0, yaw = view.pose
    scale = width / NAVIGATION_SPAN  # [px/m]
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    wx, wy = x0 + (cols + 0.5 - width / 2) / scale, y0 - (rows + 0.5 - height / 2) / scale
    rgb = np.full((height, width, 3), RVIZ_BACKGROUND, dtype=np.uint8)  # Outside the costmap.
    if view.costs is not None:
        cx = np.floor((wx - view.origin[0]) / view.resolution).astype(int)
        cy = np.floor((wy - view.origin[1]) / view.resolution).astype(int)
        inside = (cx >= 0) & (cy >= 0) & (cx < view.costs.shape[1]) & (cy < view.costs.shape[0])
        rgb[inside] = costmap_colors(view.costs[cy[inside], cx[inside]])
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)

    def pixel(x: float, y: float) -> tuple[float, float]:
        return width / 2 + (x - x0) * scale, height / 2 - (y - y0) * scale

    if len(view.path) > 1:
        draw.line([pixel(*point) for point in view.path], fill=ROUTE, width=3)
    if view.goal is not None:
        gx, gy, heading = view.goal
        px, py = pixel(gx, gy)
        draw.ellipse((px - 6, py - 6, px + 6, py + 6), outline=GOAL, width=3)
        if heading is not None:
            draw.line((px, py, *pixel(gx + 0.4 * math.cos(heading), gy + 0.4 * math.sin(heading))), fill=GOAL, width=3)
    c, s = math.cos(yaw), math.sin(yaw)
    if view.footprint is not None and len(view.footprint) > 2:
        outline = [pixel(x0 + c * bx - s * by, y0 + s * bx + c * by) for bx, by in view.footprint]
        draw.polygon(outline, outline=FOOTPRINT, width=2)
    draw.line((*pixel(x0, y0), *pixel(x0 + 0.35 * c, y0 + 0.35 * s)), fill=FOOTPRINT, width=3)  # Heading.
    if view.command is not None and math.hypot(view.command[0], view.command[1]) > 1e-3:
        vx, vy = c * view.command[0] - s * view.command[1], s * view.command[0] + c * view.command[1]
        draw.line((*pixel(x0, y0), *pixel(x0 + vx, y0 + vy)), fill=COMMAND, width=3)  # 1 s of travel.
    return np.asarray(image)


# The arm planner view looks from the robot's right, a little behind and above: body +x points right
# on screen, +z up. The arm mostly moves in the x-z plane, which this shows nearly flat.
_FORWARD = np.array([0.35, 1.0, -0.45]) / np.linalg.norm([0.35, 1.0, -0.45])
_RIGHT = np.cross(_FORWARD, [0.0, 0.0, 1.0]) / np.linalg.norm(np.cross(_FORWARD, [0.0, 0.0, 1.0]))
_UP = np.cross(_RIGHT, _FORWARD)
FRAME_MARGIN = 0.25  # [m] around the arm's motion.
NEAR, FAR = 0.05, 0.30  # [m]; voxel colours by distance to the arm: red within NEAR, blue from FAR.


def planner_image(view: PlannerView, width: int = 400, height: int = 300) -> np.ndarray:
    """The plan framed on the arm's motion (see the module docstring); voxels and spheres drawn far to near."""
    from PIL import Image, ImageDraw

    arm = np.concatenate([view.arm_start, *view.arm_path, view.arm_end, view.tool_path]).reshape(-1, 3)
    if not len(arm):
        arm = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.5]])  # No arm known: the body and what is ahead.
    lower, upper = arm.min(axis=0) - FRAME_MARGIN, arm.max(axis=0) + FRAME_MARGIN
    corners = np.array(np.meshgrid(*zip(lower, upper))).reshape(3, -1).T
    sx, sy = corners @ _RIGHT, corners @ _UP
    top = 26  # [px] for the label.
    scale = 0.95 * min(width / np.ptp(sx), (height - top) / np.ptp(sy))  # [px/m]
    ox = width / 2 - scale * (sx.min() + sx.max()) / 2
    oy = top + (height - top) / 2 + scale * (sy.min() + sy.max()) / 2

    def project(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        return ox + scale * points @ _RIGHT, oy - scale * points @ _UP, points @ _FORWARD

    image = Image.new("RGB", (width, height), (36, 39, 45))
    draw = ImageDraw.Draw(image, "RGBA")
    if view.floor is not None:  # The floor as a 0.25 m grid instead of its voxels.
        for x in np.arange(math.floor(lower[0] * 4) / 4, upper[0] + 1e-6, 0.25):
            (u, v, _) = project([[x, lower[1], view.floor], [x, upper[1], view.floor]])
            draw.line((u[0], v[0], u[1], v[1]), fill=(70, 74, 82))
        for y in np.arange(math.floor(lower[1] * 4) / 4, upper[1] + 1e-6, 0.25):
            (u, v, _) = project([[lower[0], y, view.floor], [upper[0], y, view.floor]])
            draw.line((u[0], v[0], u[1], v[1]), fill=(70, 74, 82))
    voxels = view.voxels
    if len(voxels):
        keep = np.all((voxels >= lower) & (voxels <= upper), axis=1)
        if view.floor is not None:
            keep &= voxels[:, 2] > view.floor + 1.5 * view.voxel_size
        voxels = voxels[keep]
    items = []  # (depth, kind, (u, v, radius, colour))
    if len(voxels):
        u, v, d = project(voxels)
        reference = np.concatenate([arm, *(view.end_spheres[:, :3][None] if len(view.end_spheres) else [])])
        near = np.min(np.linalg.norm(voxels[:, None] - reference[None], axis=-1), axis=1)
        shade = np.clip((near - NEAR) / (FAR - NEAR), 0, 1)
        colors = depth_colors((1 - shade).reshape(1, -1) * 0.95, np.ones((1, len(shade)), bool), 1.0)[0]
        half = max(1.0, 0.5 * scale * view.voxel_size * 1.1)
        items += [(d[i], 0, (u[i], v[i], half, tuple(colors[i]))) for i in range(len(d))]
    if len(view.end_spheres):
        u, v, d = project(view.end_spheres[:, :3])
        items += [(d[i], 1, (u[i], v[i], scale * view.end_spheres[i, 3], None)) for i in range(len(d))]
    for _, kind, (u, v, r, color) in sorted(items, key=lambda item: -item[0]):
        if kind == 0:
            draw.rectangle((u - r, v - r, u + r, v + r), fill=color)
        else:
            draw.ellipse((u - r, v - r, u + r, v + r), fill=(*ARM_END, 20), outline=(*ARM_END, 90))  # Behind the links.
    for center, size in view.boxes:
        center, half = np.asarray(center), np.asarray(size) / 2
        u, v, _ = project([center + half * [a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)])
        for a, b in ((0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)):
            draw.line((u[a], v[a], u[b], v[b]), fill=(200, 200, 210), width=1)

    def chain(points: np.ndarray, color: tuple, width: int) -> None:
        if len(points) > 1:
            u, v, _ = project(points)
            draw.line(list(zip(u.tolist(), v.tolist())), fill=color, width=width, joint="curve")
            for x, y in zip(u, v):
                draw.ellipse((x - width / 2, y - width / 2, x + width / 2, y + width / 2), fill=color)

    for ghost in view.arm_path:
        chain(ghost, (*ARM_END, 70), 3)
    chain(view.arm_start, (*ARM_START, 255), 6)
    chain(view.arm_end, (*ARM_END, 255), 6)
    if len(view.tool_path) > 1:
        chain(view.tool_path, (*TOOL_PATH, 255), 3)
    draw.text((6, 3), view.label, fill=LABEL, font=_font(18))
    return np.asarray(image)


class Panels:
    """Draws :class:`PanelSources` beside the main image; keeps the stitcher and the last arm plan's
    drawing (redrawn only when the plan changes)."""

    def __init__(self) -> None:
        self._stitcher = FrontStitcher(width=320)  # About the panel's size: 6x fewer lookups.
        self._planner: tuple[PlannerView, np.ndarray] | None = None

    def compose(self, main: np.ndarray, sources: PanelSources) -> np.ndarray:
        panels: dict[str, np.ndarray | None] = {}
        if len(sources.front) == 2 and sources.front[0].tick == sources.front[1].tick:
            front = self._stitcher.stitch(*sources.front)
            depth = front.depth()
            panels["front"] = front.outlined(front.rgb)
            panels["front depth"] = front.outlined(depth_colors(depth, np.isfinite(depth), DEPTH_RANGE))
        elif len(sources.front) == 1:
            camera = sources.front[0]
            panels["front"], panels["front depth"] = camera.rgb, depth_colors(camera.depth, camera.valid, DEPTH_RANGE)
        hand = sources.hand
        if hand is not None:
            depth, valid = hand.depth[::2, ::2], hand.valid[::2, ::2]  # The panel is under half the image.
            panels["hand"], panels["hand depth"] = hand.rgb[::2, ::2], depth_colors(depth, valid, DEPTH_RANGE)
        if sources.navigation is not None:
            panels["navigation"] = navigation_image(sources.navigation)
        if sources.planner is not None:
            if self._planner is None or self._planner[0] is not sources.planner:
                self._planner = (sources.planner, planner_image(sources.planner))
            panels["arm planner"] = self._planner[1]
        return layout(main, panels)


class PanelFeed:
    """Collects :class:`PanelSources` from a running robot on the simulation thread, once per video frame.

    ``control`` is the robot's control (``RobotControl``): its robot's front and look cameras (camera
    roles ``front`` and ``look``), its navigation backend's ``costmap()`` (read every
    ``costmap_period`` [s sim]; Nav2's is a service call) and ``route()``, and the base controller's
    ``applied_command``. ``footprint`` is the robot's outline [N,2] [m] in the body frame;
    ``planner`` has ``last_view`` (an ``ArmPlanner``; it may be set later). The cameras come from
    the packet.
    """

    @staticmethod
    def cameras(robot) -> tuple[str, ...]:
        """The cameras the panels show: the robot's front cameras and its look camera."""
        return (*robot.cameras_for("front"), *robot.cameras_for("look"))

    def __init__(self, control, *, footprint: np.ndarray | None = None, planner=None, costmap_period: float = 1.0):
        self._control, self._footprint, self.planner = control, footprint, planner
        self._period, self._costmap, self._read = costmap_period, None, -math.inf
        self._front, self._look = control.robot.cameras_for("front"), control.robot.look_camera

    def sources(self, packet) -> PanelSources:
        front = tuple(frame for name in self._front if (frame := camera_frame(packet, name)) is not None)
        hand = camera_frame(packet, self._look) if self._look is not None else None
        front = front if len(front) == len(self._front) else ()  # The Front view needs every front camera.
        return PanelSources(front, hand, self._navigation(packet), getattr(self.planner, "last_view", None))

    def _navigation(self, packet) -> NavigationView | None:
        navigation = getattr(self._control, "navigation", None)
        backend = getattr(navigation, "backend", None)
        if not hasattr(backend, "costmap") or packet.localization is None:
            return None
        now = float(packet.stamp.time_seconds)
        if now - self._read >= self._period or now < self._read:
            self._costmap, self._read = backend.costmap(), now
        matrix = packet.localization.pose.matrix.cpu().numpy()
        pose = (float(matrix[0, 3]), float(matrix[1, 3]), math.atan2(matrix[1, 0], matrix[0, 0]))
        route = backend.route() if hasattr(backend, "route") else None
        goal, path = (None, ()) if route is None else ((route[0].x, route[0].y, route[0].heading), route[1])
        costs, origin, resolution = self._costmap if self._costmap is not None else (None, (0.0, 0.0), 0.05)
        return NavigationView(
            pose=pose,
            costs=costs,
            origin=origin,
            resolution=resolution,
            footprint=self._footprint,
            path=tuple(path),
            goal=goal,
            command=tuple(self._control.locomotion.controller.applied_command),
        )
