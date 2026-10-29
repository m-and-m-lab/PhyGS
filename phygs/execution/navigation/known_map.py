# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Explicit privileged box geometry adapter for planar known-map experiments."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import numpy as np

from phygs.execution.navigation.map import GridMap, RectangleFootprint
from phygs.scene.assets import has_scene_body, scene_body_pose

if TYPE_CHECKING:
    from isaaclab.scene import InteractiveScene


@dataclass(frozen=True)
class BoxGeometry:
    """Declared collider size/center [m]; scene_key selects a registered movable body.

    A movable body is a rigid-object asset (its scene name) or a scene body (its USD path,
    :func:`~phygs.scene.assets.register_scene_bodies`).
    """

    size: tuple[float, float, float]
    center: tuple[float, float, float]
    scene_key: str | None = None

    def __post_init__(self):
        if len(self.size) != 3 or len(self.center) != 3 or not all(map(math.isfinite, (*self.size, *self.center))):
            raise ValueError("Box geometry must contain finite 3D dimensions and coordinates.")
        if min(self.size) <= 0:
            raise ValueError("Box dimensions must be positive.")


class KnownBoxMap:
    """Conservative rectangle-footprint maps from declared fixed and simulated box colliders.

    Obstacles are the cells a box (grown by ``clearance`` [m]) touches, and the map boundary.

    This privileged provider supports a planar floor and explicitly registered box
    obstacles. It is not a general USD traversability extractor or a sensor mapper.
    """

    def __init__(
        self,
        scene: InteractiveScene,
        boxes: tuple[BoxGeometry, ...],
        *,
        bounds: tuple[float, float, float, float],
        resolution: float,
        footprint: RectangleFootprint,
        clearance: float,
        height: float = 1.2,
    ) -> None:
        values = (*bounds, resolution, clearance, height)
        if not all(map(math.isfinite, values)) or min(resolution, height) <= 0 or clearance < 0:
            raise ValueError("Map dimensions and transit envelope must be finite and valid.")
        xmin, ymin, xmax, ymax = bounds
        if xmax <= xmin or ymax <= ymin:
            raise ValueError("Map bounds must have positive area.")
        shape = ((ymax - ymin) / resolution, (xmax - xmin) / resolution)
        if any(not math.isclose(value, round(value), abs_tol=1e-9) for value in shape):
            raise ValueError("Map spans must be integer multiples of resolution.")
        for box in boxes:
            if box.scene_key is not None and not has_scene_body(scene, box.scene_key):
                raise ValueError(f"Movable map box is not registered: {box.scene_key}")
        self._scene, self._boxes = scene, boxes
        self._bounds, self._resolution = bounds, resolution
        self._footprint, self._clearance, self._height = footprint, clearance, height
        self._shape = tuple(round(value) for value in shape)
        self._map: GridMap | None = None
        self._raw_map: GridMap | None = None

    @property
    def configuration(self) -> dict:
        """Describe privileged map access and physical envelope dimensions [m]."""
        return dict(
            source="privileged_known_box_geometry",
            bounds=self._bounds,
            resolution=self._resolution,
            footprint=asdict(self._footprint),
            clearance=self._clearance,
            height=self._height,
            boxes=[asdict(box) for box in self._boxes],
        )

    def _distance(self, margin: float, reach: float) -> np.ndarray:
        """Signed planar clearance [m] from boxes and map boundaries, less ``margin``; exact where it
        is below ``reach`` [m] (each box updates only the cells within its extent plus reach, so a
        map over a whole house costs per box, not per box and cell)."""
        from isaaclab.utils.math import matrix_from_quat

        xmin, ymin, xmax, ymax = self._bounds
        x = xmin + (np.arange(self._shape[1]) + 0.5) * self._resolution
        y = ymin + (np.arange(self._shape[0]) + 0.5) * self._resolution
        distance = np.minimum(np.minimum(x - xmin, xmax - x)[None, :], np.minimum(y - ymin, ymax - y)[:, None]) - margin
        for box in self._boxes:
            center = np.asarray(box.center)
            half = np.asarray(box.size) / 2
            if box.scene_key is not None:
                pose = scene_body_pose(self._scene, box.scene_key)
                center = pose[:3].cpu().numpy()
                half = np.abs(matrix_from_quat(pose[3:]).cpu().numpy()) @ half
            if center[2] + half[2] < 0.02 or center[2] - half[2] > self._height:
                continue
            extent = half[:2] + margin + reach
            columns = slice(*np.clip(np.searchsorted(x, [center[0] - extent[0], center[0] + extent[0]]), 0, len(x)))
            rows = slice(*np.clip(np.searchsorted(y, [center[1] - extent[1], center[1] + extent[1]]), 0, len(y)))
            if columns.start >= columns.stop or rows.start >= rows.stop:
                continue
            dx = np.maximum(np.abs(x[columns] - center[0]) - half[0], 0)
            dy = np.maximum(np.abs(y[rows] - center[1]) - half[1], 0)
            window = distance[rows, columns]
            np.minimum(window, np.hypot(dx[None, :], dy[:, None]) - margin, out=window)
        return distance

    def snapshot(self) -> GridMap:
        """Return the footprint grid of the box obstacles (grown by ``clearance``)."""
        cell = self._resolution * math.sqrt(2) / 2
        obstacles = self._distance(cell + self._clearance, reach=self._resolution) <= 0
        if self._map is None or not np.array_equal(obstacles, self._map.obstacles):
            version = 0 if self._map is None else self._map.version + 1
            self._map = GridMap.with_footprint(
                obstacles, self._footprint, self._bounds[:2], self._resolution, version,
                "privileged_known_box_geometry",
            )  # fmt: skip
        return self._map

    def raw_snapshot(self) -> GridMap:
        """Return physical obstacle occupancy for external costmaps; no robot inflation."""
        occupied = self._distance(self._resolution * math.sqrt(2) / 2, reach=self._resolution) <= 0
        if self._raw_map is None or not np.array_equal(occupied, self._raw_map.occupied):
            version = 0 if self._raw_map is None else self._raw_map.version + 1
            self._raw_map = GridMap(occupied, self._bounds[:2], self._resolution, version, "privileged_physical_boxes")
        return self._raw_map

    def physical_clearance(self, x: float, y: float, heading: float) -> float:
        """Signed distance [m] from the robot's unpadded footprint rectangle, at position [m] and
        ``heading`` [rad], to the current box bounds and the map bounds (negative: overlap)."""
        from isaaclab.utils.math import matrix_from_quat

        corners = footprint_corners(self._footprint, (x, y), heading)
        xmin, ymin, xmax, ymax = self._bounds
        lower, upper = corners.min(axis=0), corners.max(axis=0)
        distances = [float(min(lower[0] - xmin, xmax - upper[0], lower[1] - ymin, ymax - upper[1]))]
        for box in self._boxes:
            center = np.asarray(box.center)
            half = np.asarray(box.size) / 2
            if box.scene_key is not None:
                pose = scene_body_pose(self._scene, box.scene_key)
                center = pose[:3].cpu().numpy()
                half = np.abs(matrix_from_quat(pose[3:]).cpu().numpy()) @ half
            if center[2] + half[2] < 0.02 or center[2] - half[2] > self._height:
                continue
            distances.append(rectangle_box_distance(corners, center[:2], half[:2]))
        return min(distances)


def footprint_corners(footprint: RectangleFootprint, position: tuple[float, float], heading: float) -> np.ndarray:
    """World corners [m] of the unpadded footprint at a base pose, counterclockwise."""
    corners = np.asarray(footprint.polygon(padded=False))
    c, s = math.cos(heading), math.sin(heading)
    return corners @ np.array([[c, s], [-s, c]]) + np.asarray(position)


def _segment_distances(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Distance [m] from each point to the nearest edge of a closed polygon."""
    a, b = polygon, np.roll(polygon, -1, axis=0)
    d = b - a
    t = np.clip(((points[:, None] - a) * d).sum(-1) / (d * d).sum(-1), 0, 1)
    return np.linalg.norm(points[:, None] - (a + t[..., None] * d), axis=-1).min(axis=1)


def rectangle_box_distance(corners: np.ndarray, center: np.ndarray, half: np.ndarray) -> float:
    """Signed distance [m] between a rectangle (world corners [m]) and an axis-aligned box (centre,
    half sizes [m]): the gap between them, or minus the smallest overlap depth when they overlap."""
    box = center + half * np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
    axes = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    axes += [(b - a) / np.linalg.norm(b - a) for a, b in zip(corners[:2], corners[1:3])]
    depths = []
    for axis in axes:  # Separating axes: the boxes overlap only if they overlap on every axis.
        p, q = corners @ axis, box @ axis
        depths.append(min(p.max(), q.max()) - max(p.min(), q.min()))
    if min(depths) > 0:
        return -float(min(depths))
    return float(min(_segment_distances(corners, box).min(), _segment_distances(box, corners).min()))
