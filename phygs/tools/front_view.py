# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The Front view: a robot's two crossed front cameras (camera role ``front``, left first; Spot's
``frontleft`` and ``frontright``) joined into one forward-looking image.

A virtual pinhole camera sits at the midpoint of the two cameras, facing body +x and tilted
down like them (by default the mean of their optical axes' pitch, 21 deg on Spot): image right
is body -y, image up is up the tilted view. Its fields of view (130 x 100 deg by default) keep
90% of the two cameras' pixels, the ground down to the front feet included (a level 110 deg view
kept 55%, 2026-10-25). Each of its pixels is looked up in the front camera that sees it nearer
that camera's image centre (inverse warping): the lookup starts with a point 3 m along the
pixel's ray and is refined with the depth the camera measured there, which corrects the cameras'
3.6 cm offset from the virtual one. Every pixel keeps the body-frame surface point its camera
measured. Pixels no camera saw stay black.

Used by the robot UI, the episode video and anything else that shows what the robot sees ahead:
:func:`camera_frame` turns a camera of an observation packet into a :class:`CameraFrame`,
:meth:`FrontStitcher.stitch` joins two of them, :func:`depth_colors` colours a depth image and
:meth:`FrontView.outlined` draws where the cameras' fields of view end (the blind wedges above
and below, where the two crossed cameras do not overlap), on the image or its depth.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Turbo colour map, polynomial fit (Mikhailov, 2019), on [0, 1].
_TURBO = np.array(
    [
        [0.13572138, 4.61539260, -42.66032258, 132.13108234, -152.94239396, 59.28637943],
        [0.09140261, 2.19418839, 4.84296658, -14.18503333, 4.27729857, 2.82956604],
        [0.10667330, 12.64194608, -60.58204836, 110.36276771, -89.90310912, 27.34824973],
    ]
)


_LEVELS = np.linspace(0.0, 1.0, 256)[:, None]
_TURBO_LUT = np.rint(np.clip(sum(_TURBO[:, k] * _LEVELS**k for k in range(6)), 0.0, 1.0) * 255).astype(np.uint8)


def depth_colors(depth: np.ndarray, valid: np.ndarray, max_depth: float) -> np.ndarray:
    """RGB uint8 [H,W,3] of depth [m]: turbo from 0 (blue) to max_depth (red) in 256 steps; invalid
    pixels black."""
    level = np.rint(np.clip(np.nan_to_num(depth, nan=0.0) * (255.0 / max_depth), 0.0, 255.0)).astype(np.uint8)
    rgb = _TURBO_LUT[level]
    rgb[~(valid & np.isfinite(depth))] = 0
    return rgb


@dataclass(frozen=True)
class CameraFrame:
    """One capture as plain arrays: RGB uint8 [H,W,3], optical-axis depth [m] [H,W], validity [H,W],
    intrinsics [3,3] [px], and the body-from-camera and world-from-camera (optical frame) transforms."""

    name: str
    time_seconds: float
    tick: int
    rgb: np.ndarray
    depth: np.ndarray
    valid: np.ndarray
    intrinsics: np.ndarray
    body_from_camera: np.ndarray
    world_from_camera: np.ndarray | None = None  # [4,4] [m], when localization is permitted.


def camera_frame(packet, name: str) -> CameraFrame | None:
    """The packet's camera ``name`` as a :class:`CameraFrame` (copies), or None when the packet has no
    valid RGB and depth for it. The world transform is filled when localization is permitted."""
    camera = packet.cameras.get(name)
    if camera is None or not camera.valid or camera.rgb is None or camera.depth is None or camera.depth_valid is None:
        return None
    rgb = np.array(camera.rgb.cpu().numpy()[..., :3], dtype=np.uint8)
    world_from_camera = None
    if packet.localization is not None:
        world_from_camera = np.array(packet.transform("world", camera.frame, camera.stamp).cpu().numpy(), dtype=float)
    return CameraFrame(
        name=name,
        time_seconds=float(camera.stamp.time_seconds),
        tick=int(camera.stamp.tick),
        rgb=rgb,
        depth=np.array(camera.depth.cpu().numpy(), dtype=np.float32).reshape(rgb.shape[:2]),
        valid=np.array(camera.depth_valid.cpu().numpy(), dtype=bool).reshape(rgb.shape[:2]),
        intrinsics=np.array(camera.intrinsics.cpu().numpy(), dtype=float),
        body_from_camera=np.array(packet.transform("body", camera.frame, camera.stamp).cpu().numpy(), dtype=float),
        world_from_camera=world_from_camera,
    )


@dataclass(frozen=True)
class FrontView:
    """The joined image (black where no camera saw) and its body-frame point per pixel [m] (NaN: none).

    ``coverage`` [H,W] marks the pixels inside either camera's field of view (for distant
    surfaces; the cameras' 3.6 cm offset moves the edge by under a pixel beyond 1 m)."""

    time_seconds: float
    tick: int
    rgb: np.ndarray
    points: np.ndarray
    origin: np.ndarray | None = None  # The virtual camera's position in the body frame [m].
    coverage: np.ndarray | None = None

    def depth(self) -> np.ndarray:
        """Distance [m] [H,W] from the virtual camera to each pixel's surface point; NaN where none."""
        origin = np.zeros(3) if self.origin is None else self.origin
        return np.linalg.norm(self.points - origin, axis=-1)

    def outlined(self, image: np.ndarray, color: tuple[int, int, int] = (255, 255, 255), width: int = 2) -> np.ndarray:
        """A copy of an image of this view (its RGB, or a colouring of its depth) with the edge of
        the cameras' fields of view drawn ``width`` [px] wide inside the covered side; the frame's
        own border is not drawn."""
        image = np.array(image)
        if self.coverage is None:
            return image
        inside = np.pad(self.coverage, width, mode="edge")  # The frame border is not an edge.
        near_edge = np.zeros_like(self.coverage)
        height, columns = self.coverage.shape
        for dy in range(-width, width + 1):
            for dx in range(-width, width + 1):
                if abs(dy) + abs(dx) <= width:
                    near_edge |= ~inside[width + dy : width + dy + height, width + dx : width + dx + columns]
        image[self.coverage & near_edge] = color
        return image


class FrontStitcher:
    """Join two camera frames with the same capture time into the Front view.

    ``width`` [px] and the horizontal and vertical fields of view [deg] set the image (square
    pixels: the height follows); ``pitch_deg`` tilts it down [deg], None: the cameras' mean tilt,
    taken from the first frames. ``pitch`` [rad] is known once set or taken.
    """

    REFINEMENTS = 1  # Depth lookups after the first; the 3.6 cm parallax converges in one.

    def __init__(
        self,
        width: int = 800,
        hfov_deg: float = 130.0,
        vfov_deg: float = 100.0,
        pitch_deg: float | None = None,
        max_depth: float = 8.0,
    ) -> None:
        if width < 2 or not 0.0 < hfov_deg < 180.0 or not 0.0 < vfov_deg < 180.0 or max_depth <= 0.0:
            raise ValueError("The Front view needs a positive width, fields of view in (0, 180) deg and a depth limit.")
        self.width, self.max_depth = width, max_depth
        self.focal = width / 2 / math.tan(math.radians(hfov_deg) / 2)  # [px]
        self.height = 2 * max(1, round(self.focal * math.tan(math.radians(vfov_deg) / 2)))
        self.pitch: float | None = None
        self._rays = self._axis = None
        if pitch_deg is not None:
            self._tilt(math.radians(pitch_deg))
        self._candidates: dict[tuple, np.ndarray] = {}  # Camera pose -> the Front-view pixels it can see.
        self._fields: dict[tuple, np.ndarray] = {}  # Camera pose -> its field of view [H*W].

    def _tilt(self, pitch: float) -> None:
        """Body-frame rays per pixel of the view tilted down by pitch [rad], scaled to 1 m along its axis."""
        cols, rows = np.meshgrid(np.arange(self.width), np.arange(self.height))
        view = np.stack(  # View frame: x ahead, y left, z up.
            [np.ones_like(cols), -(cols - self.width / 2) / self.focal, -(rows - self.height / 2) / self.focal], axis=-1
        ).reshape(-1, 3)
        c, s = math.cos(pitch), math.sin(pitch)
        body_from_view = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])  # Ahead tilted down by pitch.
        self.pitch, self._axis = pitch, body_from_view[:, 0].astype(np.float32)
        self._rays = (view @ body_from_view.T).astype(np.float32)

    def stitch(self, left: CameraFrame, right: CameraFrame) -> FrontView:
        if left.tick != right.tick:
            raise ValueError("The Front view joins two frames captured at the same time.")
        if self.pitch is None:  # The cameras' mean tilt: their optical axes' pitch below the body's x-y plane.
            axes = [frame.body_from_camera[:3, 2] for frame in (left, right)]
            self._tilt(float(np.mean([math.asin(-axis[2] / np.linalg.norm(axis)) for axis in axes])))
        origin = ((left.body_from_camera[:3, 3] + right.body_from_camera[:3, 3]) / 2).astype(np.float32)
        count = self.width * self.height
        rgb = np.zeros((count, 3), dtype=np.uint8)
        points = np.full((count, 3), np.nan, dtype=np.float32)
        best = np.full(count, np.inf, dtype=np.float32)  # Distance from the chosen camera's image centre.
        coverage = np.zeros(count, dtype=bool)
        for frame in (left, right):
            index, colors, surface, centrality = self._lookup(frame, origin)
            take = centrality < best[index]
            index = index[take]
            rgb[index], points[index], best[index] = colors[take], surface[take], centrality[take]
            coverage |= self._fields[self._key(frame)]
        shape = (self.height, self.width)
        rgb, points, coverage = rgb.reshape(*shape, 3), points.reshape(*shape, 3), coverage.reshape(shape)
        return FrontView(left.time_seconds, left.tick, rgb, points, origin, coverage)

    def _lookup(self, frame: CameraFrame, origin: np.ndarray):
        """The Front-view pixels (flat indices) the frame has a valid depth for, with their colours,
        body-frame surface points [m] and distances from the frame's image centre (normalized)."""
        rotation = frame.body_from_camera[:3, :3].astype(np.float32)
        position = frame.body_from_camera[:3, 3].astype(np.float32)
        (fx, _, cx), (_, fy, cy) = frame.intrinsics[:2]
        height, width = frame.depth.shape
        index = self._candidates_for(frame)
        ahead = np.full(len(index), 3.0, dtype=np.float32)  # Distance along the view's axis [m]: start 3 m out.
        for _ in range(1 + self.REFINEMENTS):
            optical = (origin + self._rays[index] * ahead[:, None] - position) @ rotation  # Camera frame.
            z = optical[:, 2]
            keep = z > 0.05
            z = np.where(keep, z, 1.0)
            u, v = fx * optical[:, 0] / z + cx, fy * optical[:, 1] / z + cy
            col, row = np.rint(u).astype(np.int64), np.rint(v).astype(np.int64)
            keep &= (col >= 0) & (col < width) & (row >= 0) & (row < height)
            index, u, v, col, row = index[keep], u[keep], v[keep], col[keep], row[keep]
            depth = frame.depth[row, col]
            keep = frame.valid[row, col] & np.isfinite(depth) & (depth > 0.0) & (depth <= self.max_depth)
            index, u, v, col, row, depth = index[keep], u[keep], v[keep], col[keep], row[keep], depth[keep]
            camera_points = np.stack([(col - cx) / fx * depth, (row - cy) / fy * depth, depth], axis=-1)
            surface = (camera_points @ rotation.T + position).astype(np.float32)
            ahead = np.maximum((surface - origin) @ self._axis, 0.05)
        centrality = np.hypot((u - cx) / width, (v - cy) / height).astype(np.float32)
        return index, frame.rgb[row, col, :3], surface, centrality

    @staticmethod
    def _key(frame: CameraFrame) -> tuple:
        pose, intrinsics = np.round(frame.body_from_camera[:3], 4), np.round(frame.intrinsics[:2], 2)
        return (frame.name, *pose.ravel(), *intrinsics.ravel())

    def _candidates_for(self, frame: CameraFrame) -> np.ndarray:
        """Front-view pixels whose rays reach the frame's image (50 px margin for parallax); cached per
        pose with the frame's field of view (the pixels whose rays reach its image, no margin).

        The front cameras are fixed to the body, so this is computed once per camera.
        """
        key = self._key(frame)
        if key not in self._candidates:
            (fx, _, cx), (_, fy, cy) = frame.intrinsics[:2]
            height, width = frame.depth.shape
            optical = self._rays @ frame.body_from_camera[:3, :3].astype(np.float32)  # Directions only: far points.
            z = optical[:, 2]
            ahead = z > 1e-3
            z = np.where(ahead, z, 1.0)
            u, v = fx * optical[:, 0] / z + cx, fy * optical[:, 1] / z + cy
            margin = 50.0
            inside = ahead & (u > -margin) & (u < width + margin) & (v > -margin) & (v < height + margin)
            self._candidates[key] = np.nonzero(inside)[0]
            self._fields[key] = ahead & (u >= -0.5) & (u < width - 0.5) & (v >= -0.5) & (v < height - 0.5)
        return self._candidates[key]
