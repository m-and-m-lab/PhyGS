# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The panels beside a run's video (recording.panels) and the shared Front view helpers (observations.front_view)."""

import math
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from phygs.observations.front_view import (
    _TURBO,
    CameraFrame,
    FrontStitcher,
    FrontView,
    camera_frame,
    depth_colors,
)
from phygs.recording.panels import (
    BACKGROUND,
    COLUMNS,
    FOOTPRINT,
    INSCRIBED,
    LABEL_H,
    LETHAL,
    PANELS,
    RVIZ_BACKGROUND,
    UNKNOWN,
    NavigationView,
    Panels,
    PanelSources,
    PlannerView,
    costmap_colors,
    layout,
    navigation_image,
    planner_image,
)

K = np.array([[80.0, 0.0, 80.0], [0.0, 80.0, 60.0], [0.0, 0.0, 1.0]])


def wall_camera(name: str, y: float, yaw_deg: float, tick: int = 7) -> CameraFrame:
    """A 160x120 camera at body (0.415, y, 0) [m] looking at yaw [deg], 21 deg down, at a wall 2 m ahead."""
    yaw, pitch = math.radians(yaw_deg), math.radians(21.0)
    forward = np.array([math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), -math.sin(pitch)])
    right = np.array([math.sin(yaw), -math.cos(yaw), 0.0])
    body_from_camera = np.eye(4)
    body_from_camera[:3, :3] = np.stack([right, np.cross(forward, right), forward], axis=1)
    body_from_camera[:3, 3] = [0.415, y, 0.0]
    cols, rows = np.meshgrid(np.arange(160), np.arange(120))
    rays = body_from_camera[:3, :3] @ np.linalg.solve(K, np.stack([cols, rows, np.ones_like(cols)]).reshape(3, -1))
    with np.errstate(divide="ignore"):
        depth = np.where(rays[0] > 0, (2.0 - 0.415) / rays[0], np.inf).reshape(120, 160)
    rgb = np.full((120, 160, 3), 200, dtype=np.uint8)
    return CameraFrame(name, 1.0, tick, rgb, depth, np.isfinite(depth), K, body_from_camera)


def test_depth_colours_follow_the_turbo_curve_in_256_steps_and_black_invalid():
    depth = np.linspace(0.0, 6.0, 50)[None]
    colors = depth_colors(depth, np.ones_like(depth, dtype=bool), 6.0).astype(int)
    x = depth[0] / 6.0
    exact = np.clip(sum(_TURBO[:, k] * x[:, None] ** k for k in range(6)), 0, 1) * 255
    assert np.abs(colors[0] - exact).max() <= 4  # 256 levels: a step moves a channel by up to 4.
    assert not depth_colors(np.array([[1.0, np.nan]]), np.array([[False, True]]), 6.0).any()


def test_front_view_depth_is_the_distance_from_the_virtual_camera():
    level = FrontStitcher(width=160, hfov_deg=110.0, vfov_deg=77.5, pitch_deg=0.0)
    view = level.stitch(wall_camera("frontleft", 0.036, -31.0), wall_camera("frontright", -0.036, 31.0))
    assert view.origin == pytest.approx([0.415, 0.0, 0.0])
    depth = view.depth()
    assert depth[45, 80] == pytest.approx(2.0 - 0.415, abs=0.03)  # Straight ahead to the wall.
    assert np.isnan(depth[0, 0])  # No camera sees the top-left corner.


def test_camera_frame_copies_a_packet_camera_or_returns_none():
    camera = SimpleNamespace(
        valid=True,
        rgb=torch.zeros((2, 3, 4), dtype=torch.uint8),
        depth=torch.ones((2, 3)),
        depth_valid=torch.ones((2, 3), dtype=torch.bool),
        intrinsics=torch.eye(3),
        frame="hand_optical",
        stamp=SimpleNamespace(time_seconds=0.5, tick=3),
    )
    packet = SimpleNamespace(
        cameras={"hand": camera}, localization=None, transform=lambda target, source, stamp: torch.eye(4)
    )
    frame = camera_frame(packet, "hand")
    assert frame.rgb.shape == (2, 3, 3) and frame.depth.dtype == np.float32 and frame.world_from_camera is None
    assert (frame.name, frame.tick, frame.time_seconds) == ("hand", 3, 0.5)
    assert camera_frame(packet, "frontleft") is None
    camera.depth = None
    assert camera_frame(packet, "hand") is None  # Depth not permitted.


def test_layout_puts_the_panels_right_of_the_main_image_and_marks_missing_ones():
    main = np.full((450, 800, 3), 90, dtype=np.uint8)
    frame = layout(main, {"hand": np.full((240, 320, 3), 250, dtype=np.uint8)})
    rows = math.ceil(len(PANELS) / COLUMNS)
    cell_w = 4 * (450 // rows - LABEL_H) // 3
    assert frame.shape == (450, 800 + COLUMNS * cell_w, 3) and frame.shape[1] % 2 == 0
    assert (frame[:, :800] == 90).all()  # The main image unchanged.
    hand = frame[LABEL_H + 20 : 450 // rows - 20, 800 + cell_w + 20 : 800 + 2 * cell_w - 20]
    assert (hand == 250).all()  # "hand" is the second panel of the first row.


def test_costmap_colours_follow_nav2s_translation_and_rvizs_costmap_scheme():
    colors = costmap_colors(np.array([[0, 1, 252, 253, 254, 255]], dtype=np.uint8))[0]
    assert tuple(colors[0]) == RVIZ_BACKGROUND  # Free: transparent in RViz.
    assert tuple(colors[1]) == (2, 0, 253) and tuple(colors[2]) == (249, 0, 6)  # Published 1 and 98.
    assert tuple(colors[3]) == INSCRIBED == (0, 255, 255) and tuple(colors[4]) == LETHAL == (255, 0, 255)
    assert tuple(colors[5]) == UNKNOWN == (112, 137, 134)


def test_navigation_image_is_centred_on_the_robot_with_map_x_right_and_y_up():
    costs = np.zeros((100, 100), dtype=np.uint8)
    costs[50, 70] = 254  # The cell at (3.5, 2.5) m: 1.5 m right of the robot at (2.0, 2.5).
    footprint = np.array([[-0.6, -0.3], [0.6, -0.3], [0.6, 0.3], [-0.6, 0.3]])
    view = NavigationView(pose=(2.0, 2.5, 0.0), costs=costs, origin=(0.0, 0.0), resolution=0.05, footprint=footprint)
    image = navigation_image(view, width=600, height=300)  # 100 px per metre.
    assert tuple(image[147, 300 + 152]) == LETHAL  # The cell 1.5 m right of the centre, 0-5 cm up.
    assert tuple(image[150 - 30, 300]) == FOOTPRINT  # The footprint's left edge, 0.3 m up.
    assert tuple(navigation_image(NavigationView(pose=(0.0, 0.0, 0.0)), 40, 30)[0, 0]) == RVIZ_BACKGROUND  # None yet.


def test_planner_image_frames_the_arm_and_colours_voxels_by_their_distance_to_it():
    arm_start = np.array([[0.3, 0.0, 0.2], [0.3, 0.0, 0.5], [0.6, 0.0, 0.5]])
    arm_end = np.array([[0.3, 0.0, 0.2], [0.5, 0.0, 0.4], [0.8, 0.0, 0.2]])
    floor_voxels = np.array([[x, 0.0, -0.52] for x in np.arange(0.3, 0.9, 0.02)])
    near = np.array([[0.82, 0.0, 0.18]])  # Next to the jaw at the end.
    empty = planner_image(PlannerView(label="", arm_start=arm_start), 200, 150)
    view = PlannerView(label="", voxels=np.vstack([floor_voxels, near]), arm_start=arm_start, arm_end=arm_end,
                       tool_path=arm_end[1:], floor=-0.53)  # fmt: skip
    drawn = planner_image(view, 200, 150)
    assert empty.shape == drawn.shape == (150, 200, 3)
    changed = np.abs(drawn.astype(int) - empty).sum(axis=-1) > 0
    assert changed.sum() > 50
    with_floor = planner_image(replace(view, voxels=floor_voxels), 200, 150)
    without = planner_image(replace(view, voxels=np.zeros((0, 3))), 200, 150)
    assert np.array_equal(with_floor, without)  # Floor voxels are the grid, not drawn.
    reddish = (drawn[..., 0].astype(int) - drawn[..., 2]) > 100
    assert reddish.any()  # The voxel next to the jaw is red.


def test_panels_join_only_front_frames_of_the_same_capture_and_redraw_a_plan_once():
    panels, main = Panels(), np.zeros((450, 800, 3), dtype=np.uint8)
    left, right = wall_camera("frontleft", 0.036, -31.0), wall_camera("frontright", -0.036, 31.0)
    joined = panels.compose(main, PanelSources(front=(left, right)))
    later = wall_camera("frontright", -0.036, 31.0, tick=8)
    late = panels.compose(main, PanelSources(front=(left, later)))
    front = (slice(LABEL_H, 200), slice(800, 1000))
    background = (late[front][30:] == BACKGROUND).all(axis=-1)
    assert (joined[front][30:] != BACKGROUND).any(axis=-1).mean() > 0.5 and background.mean() > 0.95  # "no data".
    plan = PlannerView(label="plan")
    panels.compose(main, PanelSources(planner=plan))
    drawing = panels._planner[1]
    panels.compose(main, PanelSources(planner=plan))
    assert panels._planner[1] is drawing


def test_front_view_without_origin_measures_from_the_body_origin():
    view = FrontView(0.0, 0, np.zeros((1, 1, 3), np.uint8), np.array([[[3.0, 4.0, 0.0]]]))
    assert view.depth()[0, 0] == pytest.approx(5.0)


def test_front_view_outline_marks_the_edge_of_the_fields_of_view_but_not_the_frame_border():
    level = FrontStitcher(width=160, hfov_deg=110.0, vfov_deg=77.5, pitch_deg=0.0)
    view = level.stitch(wall_camera("frontleft", 0.036, -31.0), wall_camera("frontright", -0.036, 31.0))
    assert view.coverage[45, 80] and not view.coverage[0, 0]  # The centre is seen; the top corner is blind.
    white = (view.outlined(np.zeros((90, 160, 3), dtype=np.uint8)) == 255).all(axis=-1)
    assert white.any() and (white <= view.coverage).all()  # Drawn on the seen side of the edge only.
    assert not white[40:60, 75:85].any()  # Not across the middle of the image.
    assert not white[45, [0, 1, -2, -1]].any()  # Not along the frame's own left and right border.


def test_front_view_tilts_with_the_cameras_and_keeps_the_ground_ahead_of_the_feet():
    left, right = wall_camera("frontleft", 0.036, -31.0), wall_camera("frontright", -0.036, 31.0)  # 21 deg down.
    tilted = FrontStitcher(width=160)
    view = tilted.stitch(left, right)
    assert math.degrees(tilted.pitch) == pytest.approx(21.0, abs=0.1) and view.coverage.shape == (tilted.height, 160)
    level = FrontStitcher(width=160, hfov_deg=110.0, vfov_deg=77.5, pitch_deg=0.0)
    level.stitch(left, right)
    ground = np.array([1.0, 0.0, -0.5]) - view.origin  # Floor 0.6 m ahead of the cameras, 0.5 m below: 40 deg down.

    def pixel(stitcher):  # The ground point's pixel in a stitcher's view (row may be off the image).
        c, s = math.cos(stitcher.pitch), math.sin(stitcher.pitch)
        ahead, up = c * ground[0] - s * ground[2], s * ground[0] + c * ground[2]
        return round(stitcher.height / 2 - up * stitcher.focal / ahead), round(80 - ground[1] * stitcher.focal / ahead)

    row, col = pixel(tilted)
    assert 0 <= row < tilted.height and view.coverage[row, col]  # Seen in the tilted view.
    assert pixel(level)[0] >= level.height  # Below the level view's bottom edge.


def test_panel_sources_are_drawable_only_with_data_a_panel_can_show():
    left, right = wall_camera("frontleft", 0.036, -31.0), wall_camera("frontright", -0.036, 31.0)
    assert not PanelSources().drawable()
    assert PanelSources(front=(left,)).drawable()  # A robot with one front camera shows it as the Front view.
    assert PanelSources(front=(left, right)).drawable()
    assert PanelSources(hand=left).drawable() and PanelSources(planner=PlannerView(label="")).drawable()
