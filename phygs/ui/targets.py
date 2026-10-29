# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""What a pixel the person points at in a camera view is: its depth and 3D point.

Resolved against the exact frame the person saw (docs/ui.md, Running a skill), not the
latest capture. A pixel without depth takes the nearest valid pixel within ``search`` pixels.
"""

from __future__ import annotations

import numpy as np

from phygs.observations.front_view import CameraFrame, FrontView


def _nearest_valid(valid: np.ndarray, x: int, y: int, search: int) -> tuple[int, int] | None:
    height, width = valid.shape
    if not (0 <= x < width and 0 <= y < height):
        return None
    if valid[y, x]:
        return x, y
    top, left = max(0, y - search), max(0, x - search)
    window = valid[top : y + search + 1, left : x + search + 1]
    rows, cols = np.nonzero(window)
    if not len(rows):
        return None
    best = np.argmin((rows + top - y) ** 2 + (cols + left - x) ** 2)
    return int(cols[best] + left), int(rows[best] + top)


def _normal(points: np.ndarray, valid: np.ndarray, col: int, row: int, viewer: np.ndarray, step: int = 3):
    """Unit surface normal at a pixel from its neighbours' 3D points [H,W,3], facing the viewer; or None."""
    height, width = valid.shape
    left, right = max(0, col - step), min(width - 1, col + step)
    top, bottom = max(0, row - step), min(height - 1, row + step)
    if not (valid[row, left] and valid[row, right] and valid[top, col] and valid[bottom, col]):
        return None
    normal = np.cross(points[row, right] - points[row, left], points[bottom, col] - points[top, col])
    length = np.linalg.norm(normal)
    if length < 1e-9:
        return None
    normal /= length
    return normal if np.dot(normal, viewer - points[row, col]) >= 0 else -normal


def resolve_camera_pixel(frame: CameraFrame, x: int, y: int, search: int = 5) -> dict | None:
    """The pixel's optical depth [m], its point in the body frame and, when known, the world frame [m],
    and the surface normal there (unit, facing the camera; None at an edge)."""
    valid = frame.valid & np.isfinite(frame.depth) & (frame.depth > 0.0)
    found = _nearest_valid(valid, x, y, search)
    if found is None:
        return None
    col, row = found
    depth = float(frame.depth[row, col])
    optical = np.linalg.solve(frame.intrinsics, [col, row, 1.0]) * depth
    body = frame.body_from_camera @ [*optical, 1.0]
    world = None if frame.world_from_camera is None else frame.world_from_camera @ [*optical, 1.0]
    rows, cols = np.indices(frame.depth.shape)
    rays = np.einsum("ij,hwj->hwi", np.linalg.inv(frame.intrinsics), np.stack([cols, rows, np.ones_like(rows)], -1))
    points = rays * np.where(valid, frame.depth, 0.0)[..., None]
    normal = _normal(points, valid, col, row, np.zeros(3))
    frame_to = frame.world_from_camera if frame.world_from_camera is not None else frame.body_from_camera
    return dict(
        pixel=[col, row],
        depth=depth,
        distance=float(np.linalg.norm(optical)),
        body=body[:3].tolist(),
        world=None if world is None else world[:3].tolist(),
        normal=None if normal is None else (frame_to[:3, :3] @ normal).tolist(),
    )


def resolve_front_pixel(
    view: FrontView, x: int, y: int, origin: np.ndarray, world_from_body: np.ndarray | None, search: int = 5
) -> dict | None:
    """The Front view's measured point at a pixel: distance from the virtual camera ``origin`` (body
    frame) [m], and the point in the body and, when known, world frames [m]."""
    valid = np.isfinite(view.points[..., 0])
    found = _nearest_valid(valid, x, y, search)
    if found is None:
        return None
    col, row = found
    body = view.points[row, col].astype(float)
    world = None if world_from_body is None else (world_from_body @ [*body, 1.0])[:3].tolist()
    points = view.points.astype(float)
    normal = _normal(points, valid, col, row, np.asarray(origin, dtype=float))
    if world_from_body is not None:
        normal = None if normal is None else world_from_body[:3, :3] @ normal
    return dict(
        pixel=[col, row],
        depth=None,
        distance=float(np.linalg.norm(body - origin)),
        body=body.tolist(),
        world=world,
        normal=None if normal is None else normal.tolist(),
    )
