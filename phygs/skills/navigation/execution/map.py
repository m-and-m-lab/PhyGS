# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Conservative occupancy grids for the robot's transit footprint.

A grid either carries configuration space for a circular envelope (``occupied``: cells where the
base may not stand), or the physical obstacles and a :class:`RectangleFootprint`: then it holds a
configuration space per heading, for checking poses, turns and straight moves (the fine approach).
Nav2 plans the routes; the bridge publishes the raw grid as its static map.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

# Heading bins of a footprint grid: 64 headings, each blocked where the rectangle at any heading
# within half a bin (2.8 deg) touches an obstacle. Wider bins swell a long footprint sideways: 13 deg
# adds 0.65 m x sin(13 deg) = 0.15 m to Spot's half width.
_BINS = 64
_BIN_ANGLES = 2 * math.pi * np.arange(_BINS) / _BINS
# The 16 headings of one-cell and knight moves, (column step, row step) counterclockwise from +X:
# a footprint grid's ``occupied`` cells admit none of them.
_MOVES = (
    (1, 0), (2, 1), (1, 1), (1, 2), (0, 1), (-1, 2), (-1, 1), (-2, 1),
    (-1, 0), (-2, -1), (-1, -1), (-1, -2), (0, -1), (1, -2), (1, -1), (2, -1),
)  # fmt: skip
_MOVE_HEADINGS = np.array([math.atan2(dr, dc) for dc, dr in _MOVES])


def _heading_bin(angle: float) -> int:
    """The heading bin nearest to ``angle`` [rad]."""
    return round(angle / (2 * math.pi / _BINS)) % _BINS


_MOVE_BINS = tuple(_heading_bin(angle) for angle in _MOVE_HEADINGS)


@dataclass(frozen=True)
class RectangleFootprint:
    """The robot's outline in the body frame [m] (x forward, y left), grown by ``padding`` [m] on
    every side. Navigation keeps the padded rectangle clear; the padding covers what a standing
    outline leaves out (swinging feet, stopping drift). A base already standing within the padding
    of an obstacle may still turn in place and leave."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    padding: float = 0.0

    def __post_init__(self):
        if not all(map(math.isfinite, (self.x_min, self.x_max, self.y_min, self.y_max, self.padding))):
            raise ValueError("Footprint bounds and padding must be finite.")
        if self.padding < 0 or not (self.x_min < 0 < self.x_max and self.y_min < 0 < self.y_max):
            raise ValueError("The footprint must contain the base origin and its padding be nonnegative.")

    def bounds(self, padded: bool = True) -> tuple[float, float, float, float]:
        """(x_min, x_max, y_min, y_max) [m], with or without the padding."""
        p = self.padding if padded else 0.0
        return (self.x_min - p, self.x_max + p, self.y_min - p, self.y_max + p)

    @property
    def radius(self) -> float:
        """Radius [m] of the circle about the base that encloses the padded rectangle."""
        x0, x1, y0, y1 = self.bounds()
        return math.hypot(max(-x0, x1), max(-y0, y1))

    @property
    def inscribed(self) -> float:
        """Distance [m] from the base to the nearest padded edge."""
        x0, x1, y0, y1 = self.bounds()
        return min(-x0, x1, -y0, y1)

    def polygon(self, padded: bool = True) -> tuple[tuple[float, float], ...]:
        """Corners [m], counterclockwise."""
        x0, x1, y0, y1 = self.bounds(padded)
        return ((x0, y0), (x1, y0), (x1, y1), (x0, y1))

    def offsets(self, resolution: float, first: float, last: float, padded: bool = True) -> np.ndarray:
        """Cells an obstacle may occupy, relative to the base cell, to touch the rectangle at any
        heading from ``first`` to ``last`` [rad]: a square boolean mask [rows=+Y, columns=+X] centred
        on the base, grown by half a cell diagonal (an obstacle cell's extent)."""
        x0, x1, y0, y1 = self.bounds(padded)
        margin = resolution * math.sqrt(2) / 2
        reach = math.ceil((math.hypot(max(-x0, x1), max(-y0, y1)) + margin) / resolution)
        dy, dx = np.mgrid[-reach : reach + 1, -reach : reach + 1] * resolution
        mask = np.zeros(dx.shape, dtype=bool)
        for angle in np.linspace(first, last, max(2, math.ceil(abs(last - first) / math.radians(2)) + 1)):
            c, s = math.cos(angle), math.sin(angle)
            bx, by = c * dx + s * dy, -s * dx + c * dy
            mask |= (bx >= x0 - margin) & (bx <= x1 + margin) & (by >= y0 - margin) & (by <= y1 + margin)
        return mask


def _heading_space(obstacles: np.ndarray, footprint: RectangleFootprint, resolution: float, k: int, padded: bool):
    """Configuration space of heading bin ``k``: True where the base cell would put the rectangle,
    at a heading within half a bin, on an obstacle or outside the map."""
    from scipy.ndimage import binary_dilation

    half = math.pi / _BINS
    mask = footprint.offsets(resolution, _BIN_ANGLES[k] - half, _BIN_ANGLES[k] + half, padded)
    # blocked[c] = any(obstacles[c + d] for d in mask): a dilation by the mirrored mask.
    space = binary_dilation(obstacles, structure=mask[::-1, ::-1], border_value=1)
    space.setflags(write=False)
    return space


@dataclass(frozen=True)
class GridMap:
    """Blocked-cell grid; origin/resolution [m], rows=+Y, columns=+X.

    Provider ``snapshot()`` includes the robot footprint (or transit radius) and clearance.
    Provider ``raw_snapshot()`` excludes robot inflation. Unknown cells are
    blocked in either representation and separately identified by ``unknown``.
    A footprint grid (:meth:`with_footprint`) also holds the physical ``obstacles`` and a
    configuration space per heading bin, computed when first needed; its ``occupied`` cells admit
    none of the 16 move headings.
    The provider must publish a new version whenever occupancy changes. Unknown cells are
    blocked; an optional episode ID prevents reuse after environment reset.
    """

    occupied: np.ndarray
    origin: tuple[float, float]
    resolution: float
    version: int
    source: str
    unknown: np.ndarray | None = None
    episode_id: int | None = None
    frame: str = "world"
    footprint: RectangleFootprint | None = None
    obstacles: np.ndarray | None = None

    @classmethod
    def with_footprint(
        cls,
        obstacles: np.ndarray,
        footprint: RectangleFootprint,
        origin: tuple[float, float],
        resolution: float,
        version: int,
        source: str,
        **kwargs,
    ) -> GridMap:
        """A grid for the rectangle ``footprint`` among physical ``obstacles`` (blocked cells,
        unknown included); the map boundary is an obstacle. Other arguments as the fields."""
        if obstacles.ndim != 2 or obstacles.dtype != np.bool_:
            raise ValueError("Obstacles must be a 2D boolean array.")
        spaces = {(k, True): _heading_space(obstacles, footprint, resolution, k, True) for k in set(_MOVE_BINS)}
        occupied = np.logical_and.reduce(list(spaces.values()))
        grid = cls(occupied, origin, resolution, version, source, **kwargs, footprint=footprint, obstacles=obstacles)
        grid._spaces.update(spaces)
        return grid

    def __post_init__(self):
        if not self.frame or self.frame == "body":
            raise ValueError("Navigation grids require a fixed reference frame.")
        if self.occupied.ndim != 2 or self.occupied.dtype != np.bool_ or not self.occupied.size:
            raise ValueError("Occupancy must be a nonempty 2D boolean array.")
        if not math.isfinite(self.resolution) or self.resolution <= 0 or not all(map(math.isfinite, self.origin)):
            raise ValueError("Map origin and positive resolution must be finite.")
        if self.episode_id is not None and self.episode_id < 0:
            raise ValueError("Map episode ID must be nonnegative.")
        frozen = self.occupied.copy()
        frozen.setflags(write=False)
        object.__setattr__(self, "occupied", frozen)
        if self.unknown is not None:
            if self.unknown.shape != frozen.shape or self.unknown.dtype != np.bool_:
                raise ValueError("Unknown cells must be a boolean array matching occupancy.")
            if np.any(self.unknown & ~frozen):
                raise ValueError("Unknown cells must be blocked in configuration space.")
            unknown = self.unknown.copy()
            unknown.setflags(write=False)
            object.__setattr__(self, "unknown", unknown)
        object.__setattr__(self, "_spaces", {})  # (heading bin, padded) -> configuration space.
        if (self.footprint is None) != (self.obstacles is None):
            raise ValueError("A footprint grid needs its obstacles, and obstacles a footprint.")
        if self.obstacles is not None:
            if self.obstacles.shape != frozen.shape or self.obstacles.dtype != np.bool_:
                raise ValueError("Obstacles must be a boolean array matching occupancy.")
            obstacles = self.obstacles.copy()
            obstacles.setflags(write=False)
            object.__setattr__(self, "obstacles", obstacles)

    def _space(self, k: int, padded: bool = True) -> np.ndarray:
        """Configuration space of heading bin ``k`` (footprint grids)."""
        key = (k, padded)
        if key not in self._spaces:
            self._spaces[key] = _heading_space(self.obstacles, self.footprint, self.resolution, k, padded)
        return self._spaces[key]

    def cell(self, point: tuple[float, float]) -> tuple[int, int]:
        """Convert position [m] to (row, column), without clipping outside positions."""
        return (
            math.floor((point[1] - self.origin[1]) / self.resolution),
            math.floor((point[0] - self.origin[0]) / self.resolution),
        )

    def point(self, cell: tuple[int, int]) -> tuple[float, float]:
        """Return cell-center position [m]."""
        return (self.origin[0] + (cell[1] + 0.5) * self.resolution, self.origin[1] + (cell[0] + 0.5) * self.resolution)

    def free(self, point: tuple[float, float]) -> bool:
        """Whether position [m] is in a free configuration-space cell."""
        if not all(map(math.isfinite, point)):
            return False
        row, col = self.cell(point)
        return 0 <= row < self.occupied.shape[0] and 0 <= col < self.occupied.shape[1] and not self.occupied[row, col]

    def _inside(self, cell: tuple[int, int]) -> bool:
        return 0 <= cell[0] < self.occupied.shape[0] and 0 <= cell[1] < self.occupied.shape[1]

    def pose_free(self, point: tuple[float, float], heading: float, *, padded: bool = True) -> bool:
        """Whether the base at ``point`` [m] facing ``heading`` [rad] keeps the footprint (with its
        padding, or without) clear; a circular grid ignores the heading."""
        if self.footprint is None:
            return self.free(point)
        if not all(map(math.isfinite, (*point, heading))):
            return False
        cell = self.cell(point)
        return self._inside(cell) and not self._space(_heading_bin(heading), padded)[cell]

    def turn_free(self, point: tuple[float, float], heading: float, turn: float, *, padded: bool = True) -> bool:
        """Whether the base at ``point`` [m] may turn in place from ``heading`` by ``turn`` [rad]."""
        count = max(2, math.ceil(abs(turn) / (2 * math.pi / _BINS)) + 1)
        return all(self.pose_free(point, heading + f * turn, padded=padded) for f in np.linspace(0, 1, count))

    def segment_free(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        heading: float | None = None,
        *,
        bare_start: float = 0.0,
    ) -> bool:
        """Check the entire segment [m] against closed occupied cell rectangles.

        A footprint grid checks the base along the segment facing ``heading`` [rad] (default: the
        direction of travel); within ``bare_start`` [m] of ``start`` without the padding, so a base
        standing in the padding may leave. A zero-length segment without a heading checks only that
        some heading fits.
        """
        if self.footprint is not None:
            length = math.dist(start, end)
            if heading is None:
                if length == 0:
                    return self.free(start)
                heading = math.atan2(end[1] - start[1], end[0] - start[0])
            count = max(2, math.ceil(2 * length / self.resolution) + 1)
            for f in np.linspace(0, 1, count):
                point = (start[0] + f * (end[0] - start[0]), start[1] + f * (end[1] - start[1]))
                if not self.pose_free(point, heading, padded=f * length > bare_start):
                    return False
            return True
        if not self.free(start) or not self.free(end):
            return False
        rows, cols = np.where(self.occupied)
        lower = np.column_stack((cols, rows)) * self.resolution + np.asarray(self.origin)
        upper = lower + self.resolution
        near = np.zeros(len(rows))
        far = np.ones(len(rows))
        intersects = np.ones(len(rows), dtype=bool)
        for axis in (0, 1):
            delta = end[axis] - start[axis]
            if delta == 0:
                intersects &= (start[axis] >= lower[:, axis]) & (start[axis] <= upper[:, axis])
            else:
                a = (lower[:, axis] - start[axis]) / delta
                b = (upper[:, axis] - start[axis]) / delta
                near = np.maximum(near, np.minimum(a, b))
                far = np.minimum(far, np.maximum(a, b))
        return not bool(np.any(intersects & (near <= far)))


class MapProvider(Protocol):
    """Supply immutable traversability snapshots under a declared information policy."""

    @property
    def configuration(self) -> dict: ...
    def snapshot(self) -> GridMap: ...


@runtime_checkable
class RawMapProvider(MapProvider, Protocol):
    """Supply the same evidence before and after robot inflation.

    Raw grids retain unknown cells as blocked with a separate unknown mask.
    Versions must change on evidence changes, including changes hidden by
    inflation, and on episode reset. External costmaps inflate the raw view once.
    """

    def raw_snapshot(self) -> GridMap: ...
