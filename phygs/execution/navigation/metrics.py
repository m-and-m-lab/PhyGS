# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Independent navigation measurements: physical travel, clearance, tracking and stopping."""

from __future__ import annotations

import math

from phygs.execution.navigation.footprint import path_error


class NavigationMeasurements:
    """Track physical travel/clearance and distance to the currently published route [m]."""

    def __init__(self):
        self._previous = None
        self._clearance = math.inf
        self._travel = self._error_square = self._error_max = 0.0
        self._tracking_samples = 0
        self._route_length = None
        self._stop_distance = self._stop_time = 0.0

    def update(self, sample, path, clearance: float, *, stopping: bool = False) -> None:
        """Record one physical state; stopping metrics cover declared controlled-stop intervals."""
        self._clearance = min(self._clearance, clearance)
        if self._previous is not None:
            distance = math.hypot(sample.x - self._previous.x, sample.y - self._previous.y)
            self._travel += distance
            if stopping:
                self._stop_distance += distance
                self._stop_time += sample.time - self._previous.time
        if len(path) >= 2:
            self._route_length = sum(math.dist(a, b) for a, b in zip(path, path[1:]))
            error = path_error((sample.x, sample.y), path)
            self._error_square += error * error
            self._error_max = max(self._error_max, error)
            self._tracking_samples += 1
        self._previous = sample

    def result(self) -> dict:
        return dict(
            travel_distance_m=self._travel,
            latest_planned_path_length_m=self._route_length,
            minimum_envelope_clearance_m=self._clearance if math.isfinite(self._clearance) else None,
            tracking_error_rms_m=math.sqrt(self._error_square / self._tracking_samples)
            if self._tracking_samples
            else None,
            tracking_error_max_m=self._error_max if self._tracking_samples else None,
            stopping_distance_m=self._stop_distance,
            stopping_time_seconds=self._stop_time,
            clearance_definition="distance from the unpadded standing footprint rectangle to physical box bounds",
            tracking_definition="distance to the backend's current published route",
        )
