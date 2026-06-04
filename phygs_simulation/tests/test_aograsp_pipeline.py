from __future__ import annotations

import numpy as np
import pytest

from helpers.cam_utils import CameraFrame, DepthFilterConfig, PointCloudFusionFrameConfig, quat_wxyz_to_matrix
from helpers.pcd_utils import EmptyPointCloudError, fuse_camera_frames


def _frame(
    name: str,
    *,
    position_w=(0.0, 0.0, 0.0),
    position_reference=None,
    depth=None,
) -> CameraFrame:
    if depth is None:
        depth = np.ones((2, 2), dtype=np.float32)
    depth = np.asarray(depth, dtype=np.float32)
    quat = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return CameraFrame(
        name=name,
        depth_image=depth,
        rgb_image=np.full((*depth.shape, 3), 64, dtype=np.uint8),
        intrinsics=np.array(
            [
                [100.0, 0.0, 0.5],
                [0.0, 100.0, 0.5],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        position_w=np.asarray(position_w, dtype=np.float32),
        quat_wxyz=quat,
        depth_valid_mask=np.isfinite(depth) & (depth > 0.0),
        position_reference=None if position_reference is None else np.asarray(position_reference, dtype=np.float32),
        quat_reference=None if position_reference is None else quat,
        reference_position_w=None if position_reference is None else np.zeros(3, dtype=np.float32),
        reference_quat_wxyz=None if position_reference is None else quat,
    )


def test_fuse_camera_frames_samples_exact_count_deterministically() -> None:
    frames = [
        _frame("front", position_w=(1.0, 0.0, 0.0)),
        _frame("hand", position_w=(1.1, 0.0, 0.0)),
    ]

    fused_a = fuse_camera_frames(
        frames,
        camera_names=("front", "hand"),
        total_points=6,
        rng_seed=11,
        depth_filter=DepthFilterConfig(max_depth_m=3.5),
    )
    fused_b = fuse_camera_frames(
        frames,
        camera_names=("front", "hand"),
        total_points=6,
        rng_seed=11,
        depth_filter=DepthFilterConfig(max_depth_m=3.5),
    )

    assert fused_a.fusion_frame_name == "front"
    assert fused_a.points_fusion.shape == (6, 3)
    assert fused_a.points_world is not None
    assert fused_a.points_world.shape == (6, 3)
    assert fused_a.per_camera_input_points == {"front": 4, "hand": 4}
    assert sum(fused_a.per_camera_sampled_points.values()) == 6
    assert np.allclose(fused_a.points_fusion, fused_b.points_fusion)


def test_fuse_camera_frames_can_use_named_camera_fusion_frame() -> None:
    frames = [
        _frame("front", position_w=(1.0, 0.0, 0.0)),
        _frame("hand", position_w=(2.0, 0.0, 0.0)),
    ]

    fused = fuse_camera_frames(
        frames,
        camera_names=("front", "hand"),
        fusion_frame=PointCloudFusionFrameConfig(mode="camera", camera_name="hand"),
        total_points=4,
        rng_seed=0,
    )

    assert fused.fusion_frame_name == "hand"
    assert np.allclose(fused.fusion_frame_position_w, [2.0, 0.0, 0.0])


def test_fuse_camera_frames_can_use_body_aligned_stereo_midpoint_fusion_frame() -> None:
    frames = [
        _frame("frontleft", position_reference=(0.4, -0.1, 0.2)),
        _frame("frontright", position_reference=(0.4, 0.1, 0.2)),
    ]

    fused = fuse_camera_frames(
        frames,
        camera_names=("frontleft", "frontright"),
        fusion_frame=PointCloudFusionFrameConfig(
            mode="body_aligned_stereo_midpoint",
            name="front_center",
            left_camera_name="frontleft",
            right_camera_name="frontright",
        ),
        total_points=4,
        rng_seed=0,
    )

    assert fused.fusion_frame_name == "front_center"
    assert np.allclose(fused.fusion_frame_position_reference, [0.4, 0.0, 0.2])
    assert np.allclose(fused.fusion_frame_position_w, [0.4, 0.0, 0.2])
    expected_rotation = np.column_stack(([0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]))
    assert np.allclose(quat_wxyz_to_matrix(fused.fusion_frame_quat_reference), expected_rotation)


def test_fuse_camera_frames_fails_on_empty_cloud() -> None:
    frames = [_frame("front", depth=np.zeros((2, 2), dtype=np.float32))]

    with pytest.raises(EmptyPointCloudError, match="No valid depth points"):
        fuse_camera_frames(
            frames,
            camera_names=("front",),
            total_points=4,
            depth_filter=DepthFilterConfig(max_depth_m=3.5),
        )
