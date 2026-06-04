"""Point-cloud filtering, sampling, and frame fusion utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import open3d as o3d

from helpers.cam_utils import (
    DepthFilterConfig,
    PointCloudFusionFrameConfig,
    PointCloudWorldFilterConfig,
    depth_to_pointcloud,
    matrix_to_quat_wxyz,
    quat_wxyz_to_matrix,
    to_numpy,
    transform_points_from_world,
    transform_points_to_world,
)


@dataclass(frozen=True)
class FusedPointCloud:
    """Sampled multi-view point cloud expressed in one fusion frame."""

    fusion_frame_name: str
    points_fusion: np.ndarray
    fusion_frame_position_reference: np.ndarray
    fusion_frame_quat_reference: np.ndarray
    fusion_frame_position_w: np.ndarray
    fusion_frame_quat_wxyz: np.ndarray
    per_camera_input_points: Mapping[str, int]
    per_camera_sampled_points: Mapping[str, int]
    points_world: np.ndarray


class PointCloudPreparationError(RuntimeError):
    """Raised when point-cloud fusion cannot build a valid cloud."""


class EmptyPointCloudError(PointCloudPreparationError):
    """Raised when filtering or projection yields no usable points."""


@dataclass(frozen=True)
class _FusionFrame:
    name: str
    position_reference: np.ndarray
    quat_reference: np.ndarray
    position_w: np.ndarray
    quat_wxyz: np.ndarray


def compute_prim_world_bounds(target_prim_path: str) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Return a prim's world-space AABB, or None when the prim is invalid."""

    from isaaclab.sim.utils.stage import get_current_stage
    from pxr import Usd, UsdGeom

    prim = get_current_stage().GetPrimAtPath(str(target_prim_path))
    if prim is None or not prim.IsValid():
        return None

    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
    )
    aligned_box = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
    lower = np.asarray(list(aligned_box.GetMin()), dtype=np.float32)
    upper = np.asarray(list(aligned_box.GetMax()), dtype=np.float32)
    if lower.shape != (3,) or upper.shape != (3,):
        raise ValueError(f"Prim bounds for {target_prim_path!r} did not produce 3-D corners.")
    if not (np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))):
        raise ValueError(f"Prim bounds for {target_prim_path!r} contain non-finite values.")
    if np.any(upper < lower):
        raise ValueError(f"Prim bounds for {target_prim_path!r} are inverted.")
    return lower, upper


def fuse_camera_frames(
    frames: Sequence[Any],
    *,
    camera_names: Sequence[str],
    total_points: int,
    fusion_frame: Optional[PointCloudFusionFrameConfig] = None,
    rng_seed: int = 0,
    depth_filter: Optional[DepthFilterConfig] = None,
    pointcloud_filter: Optional[PointCloudWorldFilterConfig] = None,
) -> FusedPointCloud:
    """Fuse RGB-D frames into an exact-size cloud in one shared frame."""

    names = tuple(str(name) for name in camera_names)
    if not names:
        raise ValueError("camera_names must contain at least one camera name.")
    if int(total_points) <= 0:
        raise ValueError(f"total_points must be positive, got {int(total_points)}.")

    frame_map = {str(getattr(frame, "name")): frame for frame in frames}
    missing = [name for name in names if name not in frame_map]
    if missing:
        raise PointCloudPreparationError(f"Missing camera frames for: {missing}")

    target_frame = _resolve_fusion_frame(frame_map, names, fusion_frame)
    reference_parts: list[np.ndarray] = []
    world_parts: list[np.ndarray] = []
    source_parts: list[np.ndarray] = []
    per_camera_input_points: dict[str, int] = {}

    for name in names:
        points_reference, points_world = _frame_points_reference(
            frame_map[name],
            depth_filter=depth_filter,
            pointcloud_filter=pointcloud_filter,
        )
        per_camera_input_points[name] = int(points_reference.shape[0])
        if points_reference.shape[0] == 0:
            continue
        reference_parts.append(points_reference)
        world_parts.append(points_world)
        source_parts.append(np.full((points_reference.shape[0],), name, dtype=object))

    if not reference_parts:
        raise EmptyPointCloudError("No valid depth points were available from the selected cameras.")

    merged_reference = np.concatenate(reference_parts, axis=0).astype(np.float32, copy=False)
    merged_world = np.concatenate(world_parts, axis=0).astype(np.float32, copy=False)
    merged_sources = np.concatenate(source_parts, axis=0)
    sampled_reference, sampled_world, sampled_sources = _sample_points(
        merged_reference,
        merged_world,
        merged_sources,
        total_points=int(total_points),
        rng=np.random.default_rng(int(rng_seed)),
    )
    sampled_fusion = _transform_points_parent_to_child(
        sampled_reference,
        target_frame.position_reference,
        target_frame.quat_reference,
    )

    return FusedPointCloud(
        fusion_frame_name=target_frame.name,
        points_fusion=sampled_fusion.astype(np.float32, copy=False),
        fusion_frame_position_reference=target_frame.position_reference,
        fusion_frame_quat_reference=target_frame.quat_reference,
        fusion_frame_position_w=target_frame.position_w,
        fusion_frame_quat_wxyz=target_frame.quat_wxyz,
        per_camera_input_points=per_camera_input_points,
        per_camera_sampled_points=_source_counts(sampled_sources, names),
        points_world=sampled_world.astype(np.float32, copy=False),
    )


def _frame_points_reference(
    frame: Any,
    *,
    depth_filter: Optional[DepthFilterConfig],
    pointcloud_filter: Optional[PointCloudWorldFilterConfig],
) -> tuple[np.ndarray, np.ndarray]:
    points_frame = depth_to_pointcloud(
        getattr(frame, "depth_image"),
        getattr(frame, "intrinsics"),
        mask=getattr(frame, "depth_valid_mask", None),
        depth_filter=depth_filter,
    )
    if points_frame.shape[0] == 0:
        empty = np.empty((0, 3), dtype=np.float32)
        return empty, empty

    camera_pose = _frame_reference_pose(frame, str(getattr(frame, "name", "camera")))
    points_reference = transform_points_to_world(
        points_frame,
        camera_pose.position_reference,
        camera_pose.quat_reference,
    )
    if pointcloud_filter is None or pointcloud_filter.is_empty:
        points_reference = points_reference.astype(np.float32, copy=False)
        return points_reference, _reference_to_world(points_reference, camera_pose)

    points_reference = _filter_reference_points(points_reference, pointcloud_filter)
    if pointcloud_filter.voxel_size_m is not None and points_reference.shape[0] > 0:
        points_reference = _voxel_downsample(points_reference, float(pointcloud_filter.voxel_size_m))
    points_reference = points_reference.astype(np.float32, copy=False)
    return points_reference, _reference_to_world(points_reference, camera_pose)


def _filter_reference_points(points_reference: np.ndarray, config: PointCloudWorldFilterConfig) -> np.ndarray:
    if (config.bounds_min is None) != (config.bounds_max is None):
        raise ValueError("pointcloud_filter bounds_min and bounds_max must be configured together.")

    points = _points(points_reference)
    keep = np.ones((points.shape[0],), dtype=bool)
    if config.bounds_min is not None and config.bounds_max is not None:
        lower = _vector3(config.bounds_min, "pointcloud_filter.bounds_min")
        upper = _vector3(config.bounds_max, "pointcloud_filter.bounds_max")
        if np.any(upper < lower):
            raise ValueError("pointcloud_filter bounds_max must be greater than or equal to bounds_min.")
        keep &= np.all(points >= lower[None, :], axis=1)
        keep &= np.all(points <= upper[None, :], axis=1)
    if config.min_z_m is not None:
        keep &= points[:, 2] > float(config.min_z_m)
    return points[keep].astype(np.float32, copy=False)


def _voxel_downsample(points: np.ndarray, voxel_size_m: float) -> np.ndarray:
    if voxel_size_m <= 0.0:
        raise ValueError(f"voxel_size_m must be positive, got {voxel_size_m}.")
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(_points(points).astype(np.float64)))
    return np.asarray(cloud.voxel_down_sample(voxel_size_m).points, dtype=np.float32).reshape(-1, 3)


def _sample_points(
    points_reference: np.ndarray,
    points_world: np.ndarray,
    point_sources: np.ndarray,
    *,
    total_points: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reference = _points(points_reference)
    world = _points(points_world)
    sources = np.asarray(point_sources, dtype=object).reshape(-1)
    if reference.shape[0] == 0:
        raise EmptyPointCloudError("No points were available for sampling.")
    if world.shape[0] != reference.shape[0]:
        raise ValueError("points_world must align with points_reference length.")
    if sources.shape[0] != reference.shape[0]:
        raise ValueError("point_sources must align with points length.")

    indices = rng.choice(reference.shape[0], size=int(total_points), replace=reference.shape[0] < int(total_points))
    return (
        reference[indices].astype(np.float32, copy=False),
        world[indices].astype(np.float32, copy=False),
        sources[indices],
    )


def _source_counts(sources: np.ndarray, camera_names: Sequence[str]) -> dict[str, int]:
    counts = {str(name): 0 for name in camera_names}
    for source in sources.tolist():
        counts[str(source)] = counts.get(str(source), 0) + 1
    return counts


def _resolve_fusion_frame(
    frame_map: Mapping[str, Any],
    camera_names: Sequence[str],
    fusion_frame: Optional[PointCloudFusionFrameConfig],
) -> _FusionFrame:
    if fusion_frame is None:
        return _camera_frame(frame_map, str(camera_names[0]), str(camera_names[0]))

    mode = str(fusion_frame.mode).strip().lower()
    if mode == "camera":
        if not fusion_frame.camera_name:
            raise ValueError("fusion_frame.camera_name is required when fusion_frame.mode is 'camera'.")
        return _camera_frame(
            frame_map,
            str(fusion_frame.camera_name),
            str(fusion_frame.name or fusion_frame.camera_name),
        )
    if mode == "body_aligned_stereo_midpoint":
        return _body_aligned_stereo_midpoint_frame(frame_map, fusion_frame)
    raise ValueError(f"Unsupported fusion_frame.mode {fusion_frame.mode!r}.")


def _camera_frame(frame_map: Mapping[str, Any], camera_name: str, frame_name: str) -> _FusionFrame:
    if camera_name not in frame_map:
        raise ValueError(f"Fusion camera {camera_name!r} is not in the captured camera set.")
    pose = _frame_reference_pose(frame_map[camera_name], camera_name)
    position_w, quat_wxyz = _compose_poses(
        pose.reference_position_w,
        pose.reference_quat_wxyz,
        pose.position_reference,
        pose.quat_reference,
    )
    return _FusionFrame(
        name=str(frame_name),
        position_reference=pose.position_reference,
        quat_reference=pose.quat_reference,
        position_w=position_w,
        quat_wxyz=quat_wxyz,
    )


def _body_aligned_stereo_midpoint_frame(
    frame_map: Mapping[str, Any],
    config: PointCloudFusionFrameConfig,
) -> _FusionFrame:
    if not config.left_camera_name or not config.right_camera_name:
        raise ValueError(
            "fusion_frame left_camera_name and right_camera_name are required for body_aligned_stereo_midpoint."
        )

    left_name = str(config.left_camera_name)
    right_name = str(config.right_camera_name)
    left = _required_reference_pose(_require_frame(frame_map, left_name), left_name)
    right = _required_reference_pose(_require_frame(frame_map, right_name), right_name)
    _require_same_reference(left, right)

    position_reference = ((left.position_reference + right.position_reference) * 0.5).astype(np.float32)
    quat_reference = _body_aligned_fusion_quat()
    position_w, quat_wxyz = _compose_poses(
        left.reference_position_w,
        left.reference_quat_wxyz,
        position_reference,
        quat_reference,
    )
    return _FusionFrame(
        name=str(config.name or f"{config.left_camera_name}_{config.right_camera_name}_midpoint"),
        position_reference=position_reference,
        quat_reference=quat_reference,
        position_w=position_w,
        quat_wxyz=quat_wxyz,
    )


@dataclass(frozen=True)
class _FrameReferencePose:
    position_reference: np.ndarray
    quat_reference: np.ndarray
    reference_position_w: np.ndarray
    reference_quat_wxyz: np.ndarray


def _require_frame(frame_map: Mapping[str, Any], camera_name: str) -> Any:
    if camera_name not in frame_map:
        raise ValueError(f"Fusion camera {camera_name!r} is not in the captured camera set.")
    return frame_map[camera_name]


def _frame_reference_pose(frame: Any, label: str) -> _FrameReferencePose:
    position_reference = getattr(frame, "position_reference", None)
    quat_reference = getattr(frame, "quat_reference", None)
    if position_reference is None and quat_reference is None:
        return _FrameReferencePose(
            position_reference=_vector3(getattr(frame, "position_w"), f"{label}.position_w"),
            quat_reference=_quat4(getattr(frame, "quat_wxyz"), f"{label}.quat_wxyz"),
            reference_position_w=np.zeros(3, dtype=np.float32),
            reference_quat_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        )
    if position_reference is None or quat_reference is None:
        raise ValueError(f"{label} reference pose must include both position_reference and quat_reference.")
    return _FrameReferencePose(
        position_reference=_vector3(position_reference, f"{label}.position_reference"),
        quat_reference=_quat4(quat_reference, f"{label}.quat_reference"),
        reference_position_w=_vector3(getattr(frame, "reference_position_w"), f"{label}.reference_position_w"),
        reference_quat_wxyz=_quat4(getattr(frame, "reference_quat_wxyz"), f"{label}.reference_quat_wxyz"),
    )


def _required_reference_pose(frame: Any, label: str) -> _FrameReferencePose:
    if getattr(frame, "position_reference", None) is None or getattr(frame, "quat_reference", None) is None:
        raise ValueError(f"{label} must include a reference-frame camera pose.")
    return _frame_reference_pose(frame, label)


def _require_same_reference(left: _FrameReferencePose, right: _FrameReferencePose) -> None:
    if not np.allclose(left.reference_position_w, right.reference_position_w, atol=1e-5):
        raise ValueError("Stereo cameras must share the same reference frame position.")
    if not _quat_close(left.reference_quat_wxyz, right.reference_quat_wxyz):
        raise ValueError("Stereo cameras must share the same reference frame orientation.")


def _quat_close(left: np.ndarray, right: np.ndarray) -> bool:
    return bool(np.allclose(left, right, atol=1e-5) or np.allclose(left, -right, atol=1e-5))


def _body_aligned_fusion_quat() -> np.ndarray:
    rotation_reference = np.column_stack(
        (
            np.asarray([0.0, -1.0, 0.0], dtype=np.float64),
            np.asarray([0.0, 0.0, -1.0], dtype=np.float64),
            np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
        )
    )
    return matrix_to_quat_wxyz(rotation_reference)


def _reference_to_world(points_reference: np.ndarray, pose: _FrameReferencePose) -> np.ndarray:
    return transform_points_to_world(points_reference, pose.reference_position_w, pose.reference_quat_wxyz)


def _transform_points_parent_to_child(
    points_parent: np.ndarray,
    child_position_parent: np.ndarray,
    child_quat_parent: np.ndarray,
) -> np.ndarray:
    return transform_points_from_world(points_parent, child_position_parent, child_quat_parent)


def _compose_poses(
    parent_position: Any,
    parent_quat_wxyz: Any,
    child_position: Any,
    child_quat_wxyz: Any,
) -> tuple[np.ndarray, np.ndarray]:
    parent_rot = quat_wxyz_to_matrix(parent_quat_wxyz)
    child_rot = quat_wxyz_to_matrix(child_quat_wxyz)
    parent_pos = _vector3(parent_position, "parent_position")
    child_pos = _vector3(child_position, "child_position")
    position = parent_pos.astype(np.float64) + parent_rot @ child_pos.astype(np.float64)
    rotation = parent_rot @ child_rot
    return position.astype(np.float32), matrix_to_quat_wxyz(rotation)


def _points(value: Any) -> np.ndarray:
    points = to_numpy(value).astype(np.float32, copy=False).reshape(-1, 3)
    if not np.all(np.isfinite(points)):
        raise ValueError("Point cloud contains non-finite values.")
    return points


def _vector3(value: Any, label: str) -> np.ndarray:
    vector = to_numpy(value).astype(np.float32, copy=False).reshape(3)
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{label} contains non-finite values.")
    return vector


def _quat4(value: Any, label: str) -> np.ndarray:
    quat = to_numpy(value).astype(np.float32, copy=False).reshape(4)
    if not np.all(np.isfinite(quat)):
        raise ValueError(f"{label} contains non-finite values.")
    norm = float(np.linalg.norm(quat))
    if norm <= 1e-8:
        raise ValueError(f"{label} has zero length.")
    return (quat / norm).astype(np.float32)


__all__ = [
    "EmptyPointCloudError",
    "FusedPointCloud",
    "PointCloudPreparationError",
    "compute_prim_world_bounds",
    "fuse_camera_frames",
]
