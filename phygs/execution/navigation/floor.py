# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Declared flat-floor prior for sensor-owned Nav2 obstacle costmaps."""

from __future__ import annotations

import math
from dataclasses import asdict

import numpy as np

from phygs.execution.navigation.map import GridMap, RectangleFootprint


class DeclaredFloorMap:
    """Only map bounds are declared; physical obstacles come from Nav2 sensors.

    ``explored`` (x [m], y [m], radius [m]) declares the floor unexplored except that disk (the
    validated spawn disk): Nav2's static layer then holds unknown space, which its sensors reveal
    and frontier exploration (``explore``) goes after. The footprint snapshot (goal and fine-move
    checks) still declares the flat floor: Nav2's costmap holds what the cameras have seen.
    """

    def __init__(
        self,
        *,
        bounds: tuple[float, float, float, float],
        resolution: float,
        footprint: RectangleFootprint,
        clearance: float,
        explored: tuple[float, float, float] | None = None,
    ):
        xmin, ymin, xmax, ymax = bounds
        if not all(map(math.isfinite, (*bounds, resolution, clearance))):
            raise ValueError("Flat-floor geometry must be finite")
        if explored is not None and (not all(map(math.isfinite, explored)) or explored[2] <= 0):
            raise ValueError("The explored disk must be finite with a positive radius")
        if resolution <= 0 or clearance < 0 or xmax <= xmin or ymax <= ymin:
            raise ValueError("Invalid flat-floor geometry")
        width, height = (xmax - xmin) / resolution, (ymax - ymin) / resolution
        if any(not math.isclose(value, round(value), abs_tol=1e-9) for value in (width, height)):
            raise ValueError("Flat-floor bounds must align with the map resolution")
        raw = np.zeros((round(height), round(width)), dtype=np.bool_)
        yy, xx = np.indices(raw.shape)
        x, y = xmin + (xx + 0.5) * resolution, ymin + (yy + 0.5) * resolution
        edge = np.minimum.reduce((x - xmin, xmax - x, y - ymin, ymax - y)) <= clearance
        unknown = None
        if explored is not None:
            unknown = np.hypot(x - explored[0], y - explored[1]) > explored[2]
            raw = raw | unknown  # Unknown cells are blocked and marked unknown (RawMapProvider).
        self._raw = GridMap(raw, (xmin, ymin), resolution, 0, "declared_flat_floor", unknown=unknown)
        self._floor = GridMap.with_footprint(edge, footprint, (xmin, ymin), resolution, 0, "declared_flat_floor")
        self._configuration = dict(
            source="declared_flat_floor",
            bounds=bounds,
            resolution=resolution,
            footprint=asdict(footprint),
            clearance=clearance,
            explored=None if explored is None else tuple(explored),
            sensor_obstacles="nav2_obstacle_layer",
        )

    @property
    def configuration(self) -> dict:
        return dict(self._configuration)

    def snapshot(self) -> GridMap:
        return self._floor

    def raw_snapshot(self) -> GridMap:
        return self._raw
