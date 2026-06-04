from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from helpers.cam_utils import (
    CameraPipelineConfig,
    DEFAULT_CAMERA_CROPS,
    DepthFilterConfig,
    ImageCrop,
    PointCloudWorldFilterConfig,
    capture_camera_frame,
    capture_pointclouds,
    crop_camera_intrinsics,
    crop_spatial_image,
    depth_to_pointcloud,
    get_depth_image,
    get_rgb_image,
    run_camera_pipeline,
    transform_points_from_world,
    transform_points_to_world,
)


def _fake_camera(
    *,
    name: str = "hand",
    depth: np.ndarray | None = None,
    rgb: np.ndarray | None = None,
    segmentation: np.ndarray | None = None,
    position_w=(0.0, 0.0, 0.0),
    quat_wxyz=(1.0, 0.0, 0.0, 0.0),
):
    if depth is None:
        depth = np.array([[[[1.0], [2.0], [4.0]], [[0.5], [3.0], [6.0]]]], dtype=np.float32)
    if rgb is None:
        rgb = np.arange(1 * 2 * 3 * 3, dtype=np.uint8).reshape(1, 2, 3, 3)
    output = {
        "distance_to_image_plane": depth,
        "rgb": rgb,
    }
    if segmentation is not None:
        output["semantic_segmentation"] = {
            "data": segmentation,
            "info": {"idToLabels": {"7": {"prim_path": "/World/envs/env_0/Drawer"}}},
        }
    return SimpleNamespace(
        name=name,
        data=SimpleNamespace(
            output=output,
            intrinsic_matrices=np.array([[[100.0, 0.0, 1.0], [0.0, 100.0, 0.5], [0.0, 0.0, 1.0]]], dtype=np.float32),
            pos_w=np.asarray([position_w], dtype=np.float32),
            quat_w_ros=np.asarray([quat_wxyz], dtype=np.float32),
        ),
    )


def test_get_rgb_and_depth_from_isaaclab_camera_output() -> None:
    camera = _fake_camera()

    rgb = get_rgb_image(camera)
    depth, valid = get_depth_image(camera, depth_filter=DepthFilterConfig(max_depth_m=3.5))

    assert rgb.shape == (2, 3, 3)
    assert depth.shape == (2, 3)
    assert valid.tolist() == [[True, True, False], [True, True, False]]
    assert depth[0, 2] == 0.0


def test_hand_crop_applies_to_depth_rgb_segmentation_and_intrinsics() -> None:
    segmentation = np.array([[[7, 7, 0], [0, 7, 7]]], dtype=np.int32)
    camera = _fake_camera(segmentation=segmentation)

    frame = capture_camera_frame(
        camera,
        "hand",
        crop=ImageCrop(left=1),
        depth_filter=DepthFilterConfig(max_depth_m=3.5),
        target_prim_path="/World/envs/env_0/Drawer",
    )

    assert frame.depth_image.shape == (2, 2)
    assert frame.rgb_image is not None
    assert frame.rgb_image.shape == (2, 2, 3)
    assert frame.segmentation is not None
    assert frame.segmentation.shape == (2, 2)
    assert frame.intrinsics[0, 2] == pytest.approx(0.0)
    assert frame.segmentation_name == "semantic_segmentation"
    assert 7 in frame.label_metadata


def test_crop_spatial_image_matches_hand_reference_crop() -> None:
    image = np.arange(480 * 640 * 3, dtype=np.uint8).reshape(480, 640, 3)

    cropped = crop_spatial_image(image, DEFAULT_CAMERA_CROPS["hand"])

    assert cropped.shape == (480, 602, 3)
    assert np.array_equal(cropped, image[:, 38:, :])


def test_crop_camera_intrinsics_shifts_principal_point() -> None:
    intrinsics = np.array(
        [
            [400.0, 0.0, 320.0],
            [0.0, 400.0, 240.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    cropped = crop_camera_intrinsics(intrinsics, ImageCrop(left=38, top=5))

    assert cropped[0, 2] == pytest.approx(282.0)
    assert cropped[1, 2] == pytest.approx(235.0)
    assert intrinsics[0, 2] == pytest.approx(320.0)


def test_depth_threshold_is_applied_per_camera_before_pointcloud_projection() -> None:
    near_camera = _fake_camera(depth=np.array([[[[1.0], [4.0]]]], dtype=np.float32))
    far_camera = _fake_camera(depth=np.array([[[[2.0], [5.0]]]], dtype=np.float32))

    clouds = capture_pointclouds(
        {"frontleft": near_camera, "frontright": far_camera},
        ["frontleft", "frontright"],
        depth_filter=DepthFilterConfig(max_depth_m=3.5),
    )

    assert [cloud.points_camera.shape[0] for cloud in clouds] == [1, 1]
    assert clouds[0].valid_mask.tolist() == [[True, False]]
    assert clouds[1].valid_mask.tolist() == [[True, False]]


def test_depth_to_pointcloud_rejects_invalid_intrinsics_and_projects_valid_points() -> None:
    depth = np.array([[1.0, 4.0]], dtype=np.float32)
    intrinsics = np.array([[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)

    points = depth_to_pointcloud(depth, intrinsics, depth_filter=DepthFilterConfig(max_depth_m=3.5))

    assert points.shape == (1, 3)
    assert points[0].tolist() == pytest.approx([0.0, 0.0, 1.0])
    with pytest.raises(ValueError):
        depth_to_pointcloud(depth, np.eye(4, dtype=np.float32))


def test_camera_world_and_target_frame_transforms_are_consistent() -> None:
    points_camera = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    position_w = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    points_world = transform_points_to_world(points_camera, position_w, quat_wxyz)
    points_roundtrip = transform_points_from_world(points_world, position_w, quat_wxyz)

    assert points_world[0].tolist() == pytest.approx([1.0, 2.0, 4.0])
    assert points_roundtrip[0].tolist() == pytest.approx(points_camera[0].tolist())


def test_run_camera_pipeline_returns_fused_cloud() -> None:
    front_camera = _fake_camera(depth=np.ones((1, 2, 2, 1), dtype=np.float32))
    hand_camera = _fake_camera(
        depth=np.array([[[[1.0], [5.0]], [[1.0], [5.0]]]], dtype=np.float32),
        position_w=(0.1, 0.0, 0.0),
    )

    result = run_camera_pipeline(
        {"front": front_camera, "hand": hand_camera},
        ("front", "hand"),
        CameraPipelineConfig(
            depth_filter=DepthFilterConfig(max_depth_m=3.5),
            total_points=6,
            rng_seed=4,
        ),
    )

    assert result.camera_names == ("front", "hand")
    assert len(result.frames) == 2
    assert result.fused_pointcloud is not None
    assert result.fused_pointcloud.fusion_frame_name == "front"
    assert result.fused_pointcloud.points_fusion.shape == (6, 3)
    assert result.fused_pointcloud.per_camera_input_points == {"front": 4, "hand": 2}
    assert sum(result.fused_pointcloud.per_camera_sampled_points.values()) == 6


def test_run_camera_pipeline_applies_world_pointcloud_filter_before_sampling() -> None:
    camera = _fake_camera(
        depth=np.array([[[[0.01], [0.10], [1.50]]]], dtype=np.float32),
        position_w=(0.0, 0.0, 0.0),
    )

    result = run_camera_pipeline(
        {"front": camera},
        ("front",),
        CameraPipelineConfig(
            depth_filter=DepthFilterConfig(max_depth_m=3.5),
            pointcloud_filter=PointCloudWorldFilterConfig(
                min_z_m=0.03,
                bounds_min=(-1.0, -1.0, 0.0),
                bounds_max=(1.0, 1.0, 1.0),
            ),
            total_points=4,
            rng_seed=3,
        ),
    )

    assert result.fused_pointcloud is not None
    assert np.all(result.fused_pointcloud.points_world[:, 2] > 0.03)
    assert np.all(result.fused_pointcloud.points_world[:, 2] <= 1.0)


def test_run_camera_pipeline_can_capture_images_without_pointcloud() -> None:
    camera = _fake_camera()

    result = run_camera_pipeline(
        {"front": camera},
        ("front",),
        CameraPipelineConfig(
            include_rgb=False,
            include_pointcloud=False,
            depth_filter=DepthFilterConfig(max_depth_m=3.5),
        ),
    )

    assert result.camera_names == ("front",)
    assert result.fused_pointcloud is None
    assert len(result.frames) == 1
    assert result.frames[0].rgb_image is None
    assert result.frames[0].depth_image.shape == (2, 3)


def test_run_camera_pipeline_keeps_world_copy_for_debug_and_filters() -> None:
    camera = _fake_camera(depth=np.ones((1, 2, 2, 1), dtype=np.float32), position_w=(0.5, 0.0, 0.0))

    result = run_camera_pipeline(
        {"front": camera},
        ("front",),
        CameraPipelineConfig(
            include_pointcloud=True,
            total_points=4,
        ),
    )

    assert result.fused_pointcloud is not None
    assert result.fused_pointcloud.points_world is not None
    assert result.fused_pointcloud.points_world.shape == (4, 3)
