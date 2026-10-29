# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The robot UI's Front view and image encoders."""

import math

import numpy as np
import pytest
from PIL import Image

from phygs.observations.front_view import CameraFrame, FrontStitcher, depth_colors
from phygs.ui.codecs import (
    FREE_GREY,
    OCCUPIED_GREY,
    TOO_CLOSE,
    UNKNOWN_GREY,
    binary_message,
    jpeg,
    map_png,
    parse_binary_message,
)

WIDTH, HEIGHT = 160, 120
K = np.array([[80.0, 0.0, 80.0], [0.0, 80.0, 60.0], [0.0, 0.0, 1.0]])  # 90 deg wide, as the Gemini 2.
WALL_X, FLOOR_Z = 2.0, -0.5  # Body frame [m].
RED, BLUE, GREEN = (200, 0, 0), (0, 0, 200), (0, 160, 0)


def front_camera(name: str, y: float, yaw_deg: float, pitch_down_deg: float = 21.0, tick: int = 7) -> CameraFrame:
    """A camera at body (0.415, y, 0) [m] looking at yaw [deg], pitched down, seeing a wall at x 2 m
    (red where body y > 0, blue where y < 0) and the floor 0.5 m below (green)."""
    yaw, pitch = math.radians(yaw_deg), math.radians(pitch_down_deg)
    forward = np.array([math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), -math.sin(pitch)])
    right = np.array([math.sin(yaw), -math.cos(yaw), 0.0])
    down = np.cross(forward, right)
    body_from_camera = np.eye(4)
    body_from_camera[:3, :3] = np.stack([right, down, forward], axis=1)  # Optical: x right, y down, z ahead.
    body_from_camera[:3, 3] = [0.415, y, 0.0]
    cols, rows = np.meshgrid(np.arange(WIDTH), np.arange(HEIGHT))
    rays = np.linalg.solve(K, np.stack([cols, rows, np.ones_like(cols)]).reshape(3, -1).astype(float))
    directions = body_from_camera[:3, :3] @ rays  # Optical z = 1: the distance along a ray is the depth.
    origin = body_from_camera[:3, 3:4]
    with np.errstate(divide="ignore", invalid="ignore"):
        to_wall = np.where(directions[0] > 0, (WALL_X - origin[0]) / directions[0], np.inf)
        to_floor = np.where(directions[2] < 0, (FLOOR_Z - origin[2]) / directions[2], np.inf)
    depth = np.minimum(to_wall, to_floor)
    hit = origin + directions * depth
    rgb = np.where((to_wall <= to_floor)[:, None], np.where((hit[1] > 0)[:, None], RED, BLUE), GREEN).astype(np.uint8)
    depth = depth.reshape(HEIGHT, WIDTH)
    return CameraFrame(
        name=name,
        time_seconds=1.5,
        tick=tick,
        rgb=rgb.reshape(HEIGHT, WIDTH, 3),
        depth=depth,
        valid=np.isfinite(depth),
        intrinsics=K,
        body_from_camera=body_from_camera,
    )


def front_view():
    # As on Spot: the left camera looks right and the right camera looks left.
    left, right = front_camera("frontleft", 0.036, -31.0), front_camera("frontright", -0.036, 31.0)
    return FrontStitcher(width=320, hfov_deg=110.0, vfov_deg=77.5, pitch_deg=0.0).stitch(left, right)  # Level.


def test_front_view_looks_along_the_body_x_axis_level_with_body_y_to_the_left():
    view = front_view()
    height, width = view.rgb.shape[:2]
    center = view.points[height // 2, width // 2]
    assert center[0] == pytest.approx(WALL_X, abs=0.03)  # Straight ahead: the wall.
    assert abs(center[1]) < 0.05 and abs(center[2]) < 0.05  # Level, at the cameras' height.
    row = height // 2
    assert tuple(view.rgb[row, width // 4]) == RED  # Image left: body +y.
    assert tuple(view.rgb[row, 3 * width // 4]) == BLUE  # Image right: body -y.
    assert tuple(view.rgb[height - 3, width // 2]) == GREEN  # Down: the floor.


def test_front_view_is_black_where_neither_camera_looks():
    view = front_view()
    top = view.rgb[:10]
    assert not top.any() and np.isnan(view.points[:10]).all()  # Above the cameras' fields of view.
    filled = np.isfinite(view.points[..., 0])
    assert 0.5 < filled.mean() < 0.9
    assert (view.rgb[filled].any(axis=-1)).mean() > 0.99  # Every filled pixel has a colour.


def test_front_view_points_lie_on_the_scene_surfaces():
    view = front_view()
    points = view.points[np.isfinite(view.points[..., 0])]
    on_wall = np.abs(points[:, 0] - WALL_X) < 0.02
    on_floor = np.abs(points[:, 2] - FLOOR_Z) < 0.02
    assert (on_wall | on_floor).all()


def test_front_view_rejects_frames_from_different_captures():
    with pytest.raises(ValueError, match="same time"):
        FrontStitcher().stitch(
            front_camera("frontleft", 0.036, -31.0), front_camera("frontright", -0.036, 31.0, tick=8)
        )


def test_map_png_puts_north_up_and_marks_unknown_free_occupied_and_too_close():
    cells = np.array([[100, 0, -1, 0], [-1, -1, -1, -1]], dtype=np.int8)  # Row 0 is the lowest y.
    blocked = np.array([[True, False, True, True], [True, True, True, True]])  # Robot centre not allowed.
    import io

    image = np.asarray(Image.open(io.BytesIO(map_png(cells, blocked))))
    assert image.shape == (2, 4, 3)
    bottom = [tuple(int(c) for c in pixel) for pixel in image[1]]
    assert bottom == [(OCCUPIED_GREY,) * 3, (FREE_GREY,) * 3, (UNKNOWN_GREY,) * 3, TOO_CLOSE]
    assert (image[0] == UNKNOWN_GREY).all()


def test_depth_colours_run_blue_to_red_and_black_for_invalid():
    depth = np.array([[0.2, 5.8, 1.0]])
    valid = np.array([[True, True, False]])
    colors = depth_colors(depth, valid, 6.0).astype(int)
    near, far, invalid = colors[0]
    assert near[2] > near[0] and far[0] > far[2]
    assert not invalid.any()


def test_binary_messages_round_trip_and_jpeg_decodes():
    payload = jpeg(np.full((4, 6, 3), 128, dtype=np.uint8))
    header, data = parse_binary_message(binary_message({"type": "frame", "view": "hand"}, payload))
    assert header == {"type": "frame", "view": "hand"} and data == payload
    import io

    assert Image.open(io.BytesIO(data)).size == (6, 4)
