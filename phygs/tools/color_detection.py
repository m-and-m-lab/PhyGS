# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reference color detectors for diagnostic targets, not semantic detection."""

import numpy as np

from phygs.observations.types import CameraObservation


def _supported_pixel(camera: CameraObservation, mask: np.ndarray) -> tuple[int, int] | None:
    mask &= camera.depth_valid.cpu().numpy()
    rows, cols = np.where(mask)
    if len(rows) < 9:
        return None
    center = np.array([np.median(cols), np.median(rows)])
    closest = np.argmin((cols - center[0]) ** 2 + (rows - center[1]) ** 2)
    return int(cols[closest]), int(rows[closest])


def green_pixel(camera: CameraObservation) -> tuple[int, int] | None:
    """Find dominant green pixels with valid depth, for the green-cube diagnostic."""
    rgb = camera.rgb.cpu().numpy().astype(np.int16)
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return _supported_pixel(camera, (green > 100) & (green > 1.5 * red) & (green > 1.5 * blue))


def magenta_pixel(camera: CameraObservation) -> tuple[int, int] | None:
    """Find red-and-blue dominant pixels with valid depth for the house diagnostic."""
    rgb = camera.rgb.cpu().numpy().astype(np.int16)
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return _supported_pixel(camera, (red > 100) & (blue > 100) & (red > 1.5 * green) & (blue > 1.5 * green))
