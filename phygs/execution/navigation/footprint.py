# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Measured planar collision envelopes and geometry-based distance measurements."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from phygs.execution.navigation.map import RectangleFootprint
from phygs.execution.navigation.types import NavigationConfig


def convex_hull(points: np.ndarray) -> np.ndarray:
    """Return the counterclockwise convex hull of finite XY points [m]."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("Footprint points must be finite Nx2 coordinates.")
    ordered = sorted(set(map(tuple, points)))
    if len(ordered) < 3:
        raise ValueError("A footprint needs at least three distinct points.")

    def cross(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    def half(sequence):
        result = []
        for point in sequence:
            while len(result) >= 2 and cross(result[-2], result[-1], point) <= 0:
                result.pop()
            result.append(point)
        return result

    hull = np.asarray(half(ordered)[:-1] + half(reversed(ordered))[:-1])
    if len(hull) < 3:
        raise ValueError("Footprint points are collinear.")
    return hull


class FootprintSweep:
    """Accumulate sampled collision bounds in the level, body-yaw frame [m]."""

    def __init__(self) -> None:
        self.hull: np.ndarray | None = None
        self.samples = 0
        self.radius = 0.0

    def update(self, points: np.ndarray) -> None:
        """Include one physics-tick collision projection [m]."""
        self.hull = convex_hull(points if self.hull is None else np.vstack((self.hull, points)))
        self.radius = max(self.radius, float(np.linalg.norm(self.hull, axis=1).max()))
        self.samples += 1

    def result(self) -> dict:
        """Return measured swept polygon and its conservative orientation-independent radius [m]."""
        if self.hull is None:
            raise RuntimeError("No collision samples were measured.")
        return dict(
            polygon=self.hull.tolist(),
            radius=self.radius,
            samples=self.samples,
            length=float(np.ptp(self.hull[:, 0])),
            width=float(np.ptp(self.hull[:, 1])),
        )


def load_footprint(path: Path, asset_path: Path, base_files: dict[str, Path]) -> dict:
    """Load a measured profile, rejecting mismatched assets/checkpoints and incomplete checks.

    ``base_files`` maps a profile key to a file the base's motion depends on (a walking policy and its
    training configuration: ``policy_sha256``, ``policy_config_sha256``); the profile must hold each
    file's hash, as the robot's asset's (``asset_sha256``).
    """
    profile = json.loads(path.read_text())
    if profile.get("status") != "passed" or profile.get("schema") != 2:
        raise ValueError("Footprint profile must come from a passed automated measurement (schema 2).")
    for name, source in (("asset_sha256", asset_path), *base_files.items()):
        if profile.get(name) != hashlib.sha256(source.read_bytes()).hexdigest():
            raise ValueError(f"Footprint profile does not match {name}.")
    hull = convex_hull(np.asarray(profile["envelope"]["polygon"]))
    radius = profile["envelope"]["radius"]
    if not math.isfinite(radius) or radius <= 0 or np.linalg.norm(hull, axis=1).max() > radius + 1e-9:
        raise ValueError("Measured radius does not enclose the recorded polygon.")
    outline = convex_hull(np.asarray(profile["standing"]["polygon"]))
    x0, x1, y0, y1 = profile["standing"]["bounds"]
    measured = (outline[:, 0].min(), outline[:, 0].max(), outline[:, 1].min(), outline[:, 1].max())
    if not np.allclose(measured, (x0, x1, y0, y1)):
        raise ValueError("Standing bounds do not match the recorded standing outline.")
    transit_footprint(profile)  # Validates the rectangle.
    return profile


def transit_footprint(profile: dict) -> RectangleFootprint:
    """Navigation's footprint from a profile: the measured standing outline, padded [m]."""
    x0, x1, y0, y1 = profile["standing"]["bounds"]
    return RectangleFootprint(x0, x1, y0, y1, profile["standing"]["padding"])


def path_error(position: tuple[float, float], path: tuple[tuple[float, float], ...]) -> float:
    """Shortest point-to-polyline distance [m]."""
    if len(path) < 2:
        raise ValueError("Tracking error requires at least two path points.")
    a, b = np.asarray(path[:-1], dtype=float), np.asarray(path[1:], dtype=float)
    delta = b - a
    squared = (delta * delta).sum(axis=1)
    t = np.divide(
        ((np.asarray(position) - a) * delta).sum(axis=1), squared, out=np.zeros_like(squared), where=squared > 0
    )
    closest = a + np.clip(t, 0, 1)[:, None] * delta
    return float(np.linalg.norm(closest - position, axis=1).min())


def validate_transit(profile: dict, config: NavigationConfig, physics_dt: float) -> None:
    """Reject navigation settings outside the sampled posture, speed [m/s, rad/s] and timestep [s]."""
    if not math.isclose(profile["physics_dt"], physics_dt):
        raise ValueError("Physics timestep differs from the measured footprint experiment")
    if dict(profile["arm_posture"]) != dict(config.arm_posture):
        raise ValueError("Navigation arm posture differs from the measured footprint")
    if config.arm_tolerance > profile["configuration"]["criteria"]["arm_tolerance"]:
        raise ValueError("Navigation arm tolerance exceeds the measured range")
    commands = np.asarray(list(profile["configuration"]["commands"].values()), dtype=float)
    limits = []
    for axis in range(3):
        cardinal = commands[np.all(np.delete(commands, axis, axis=1) == 0, axis=1), axis]
        if len(cardinal) == 0 or cardinal.min() >= 0 or cardinal.max() <= 0:
            raise ValueError("Calibration must include both directions of each motion axis")
        limits.append(min(float(cardinal.max()), -float(cardinal.min())))
    if config.max_speed > min(limits[:2]) or config.max_yaw_speed > limits[2]:
        raise ValueError("Navigation speed exceeds the sampled footprint range")
