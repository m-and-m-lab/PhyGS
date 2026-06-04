"""AO-Grasp debug visualizations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class AoGraspVizArtifacts:
    heatmap_file: Path
    heatmap_image_file: Path
    heatmap_histogram_file: Path
    proposals_file: Path
    proposals_image_file: Path


def save_aograsp_visualizations(
    *,
    output_root: str | Path,
    request_id: str,
    points_fusion: np.ndarray,
    heatmap: np.ndarray,
    proposals: Sequence[Mapping[str, object]],
    top_k: int,
) -> AoGraspVizArtifacts:
    """Save heatmap and proposal debug artifacts using the current Python env."""

    points = _points(points_fusion)
    labels = _labels(heatmap, expected_points=points.shape[0])
    proposal_list = _proposal_list(proposals)

    root = Path(output_root).expanduser().resolve()
    point_score_dir = root / "point_score"
    heatmap_img_dir = root / "point_score_img"
    proposal_dir = root / "grasp_proposals"
    proposal_img_dir = root / "grasp_proposals_img"
    for directory in (point_score_dir, heatmap_img_dir, proposal_dir, proposal_img_dir):
        directory.mkdir(parents=True, exist_ok=True)

    stem = _safe_stem(request_id)
    artifacts = AoGraspVizArtifacts(
        heatmap_file=point_score_dir / f"{stem}.npz",
        heatmap_image_file=heatmap_img_dir / f"heatmap_{stem}.png",
        heatmap_histogram_file=heatmap_img_dir / f"heatmap_{stem}_hist.png",
        proposals_file=proposal_dir / f"{stem}.npz",
        proposals_image_file=proposal_img_dir / f"{stem}.png",
    )

    np.savez_compressed(artifacts.heatmap_file, data={"pts": points, "labels": labels})
    np.savez_compressed(
        artifacts.proposals_file,
        data={"input": points, "heatmap": labels, "proposals": _jsonable(proposal_list)},
    )
    _save_heatmap_image(artifacts.heatmap_image_file, points, labels)
    _save_histogram_image(artifacts.heatmap_histogram_file, labels)
    _save_proposal_image(artifacts.proposals_image_file, points, labels, proposal_list, top_k=top_k)
    return artifacts


def write_rgb_png(path: str | Path, image_rgb: np.ndarray) -> None:
    image = np.asarray(image_rgb, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected RGB image shape (H, W, 3), got {tuple(image.shape)}.")
    plt = _pyplot()
    plt.imsave(str(path), image)


def show_aograsp_visualization(
    *,
    points_fusion: np.ndarray,
    heatmap: np.ndarray,
    proposals: Sequence[Mapping[str, object]],
    top_k: int,
    body_frame: Mapping[str, object] | None = None,
) -> None:
    """Open an interactive Open3D viewer for AO-Grasp debug artifacts."""

    points = _points(points_fusion)
    labels = _labels(heatmap, expected_points=points.shape[0])
    proposal_list = _proposal_list(proposals)
    geometries = [_pointcloud_geometry(points, labels), _fusion_axis_geometry(points)]
    if body_frame is not None:
        geometries.append(_body_axis_geometry(points, body_frame))
    for proposal in sorted(proposal_list, key=lambda item: -float(item["score"]))[: int(top_k)]:
        geometries.append(_grasp_geometry(proposal))
    _open3d().visualization.draw_geometries(geometries, window_name="AO-Grasp Debug")


def _save_heatmap_image(path: Path, points: np.ndarray, labels: np.ndarray) -> None:
    geometries = [_pointcloud_geometry(points, labels), _fusion_axis_geometry(points)]
    _render_open3d(path, geometries)
    _add_axis_legend(path)


def _save_histogram_image(path: Path, labels: np.ndarray) -> None:
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(8, 4), dpi=140)
    ax.hist(labels, bins=32, log=True, color="#77b255", edgecolor="#303030")
    ax.set_xlabel("AO-Grasp heatmap score")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _save_proposal_image(
    path: Path,
    points: np.ndarray,
    labels: np.ndarray,
    proposals: Sequence[Mapping[str, object]],
    *,
    top_k: int,
) -> None:
    geometries = [_pointcloud_geometry(points, labels), _fusion_axis_geometry(points)]
    for proposal in sorted(proposals, key=lambda item: -float(item["score"]))[: int(top_k)]:
        geometries.append(_grasp_geometry(proposal))
    _render_open3d(path, geometries)
    _add_axis_legend(path)


def _pointcloud_geometry(points: np.ndarray, labels: np.ndarray):
    o3d = _open3d()
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64, copy=False))
    cloud.colors = o3d.utility.Vector3dVector(_heatmap_colors(labels).astype(np.float64, copy=False))
    return cloud


def _grasp_geometry(proposal: Mapping[str, object]):
    o3d = _open3d()
    center = np.asarray(proposal["position_cam"], dtype=np.float32).reshape(3)
    rot = _quat_wxyz_to_matrix(np.asarray(proposal["quaternion_cam"], dtype=np.float32).reshape(4))
    local = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.08],
            [-0.04, 0.0, 0.08],
            [-0.04, 0.0, 0.11],
            [0.04, 0.0, 0.08],
            [0.04, 0.0, 0.11],
        ],
        dtype=np.float32,
    )
    points = center + local @ rot.T
    lines = ((0, 1), (1, 2), (2, 3), (1, 4), (4, 5))
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points.astype(np.float64, copy=False))
    line_set.lines = o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
    line_set.colors = o3d.utility.Vector3dVector(np.tile([[0.1, 0.8, 0.25]], (len(lines), 1)))
    return line_set


def _fusion_axis_geometry(points: np.ndarray):
    return _axis_geometry(
        origin=np.zeros(3, dtype=np.float32),
        axes=np.eye(3, dtype=np.float32),
        scale=_axis_scale(points),
        colors=np.asarray([(1.0, 0.0, 0.0), (0.0, 0.75, 0.0), (0.0, 0.2, 1.0)], dtype=np.float64),
    )


def _body_axis_geometry(points: np.ndarray, body_frame: Mapping[str, object]):
    fusion_position_reference = np.asarray(body_frame["fusion_frame_position_reference"], dtype=np.float32).reshape(3)
    fusion_quat_reference = np.asarray(body_frame["fusion_frame_quat_reference"], dtype=np.float32).reshape(4)
    rotation_fusion_to_reference = _quat_wxyz_to_matrix(fusion_quat_reference)
    body_origin_fusion = -fusion_position_reference @ rotation_fusion_to_reference
    body_axes_fusion = rotation_fusion_to_reference.T
    return _axis_geometry(
        origin=body_origin_fusion,
        axes=body_axes_fusion,
        scale=_axis_scale(points) * 0.8,
        colors=np.asarray([(1.0, 0.0, 1.0), (0.0, 0.9, 0.9), (1.0, 0.85, 0.0)], dtype=np.float64),
    )


def _axis_geometry(*, origin: np.ndarray, axes: np.ndarray, scale: float, colors: np.ndarray):
    o3d = _open3d()
    origin = np.asarray(origin, dtype=np.float64).reshape(3)
    axes = np.asarray(axes, dtype=np.float64).reshape(3, 3)
    if float(np.linalg.det(axes)) <= 0.0:
        raise ValueError("Axis geometry must be right-handed.")
    axis_points = np.asarray(
        [
            origin,
            origin + axes[:, 0] * float(scale),
            origin,
            origin + axes[:, 1] * float(scale),
            origin,
            origin + axes[:, 2] * float(scale),
        ],
        dtype=np.float64,
    )
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(axis_points)
    line_set.lines = o3d.utility.Vector2iVector(np.asarray([(0, 1), (2, 3), (4, 5)], dtype=np.int32))
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def _axis_scale(points: np.ndarray) -> float:
    scale = max(float(np.linalg.norm(points.max(axis=0) - points.min(axis=0))) * 0.18, 0.05)
    return scale


def _render_open3d(path: Path, geometries: Sequence[object]) -> None:
    o3d = _open3d()
    vis = o3d.visualization.Visualizer()
    if not vis.create_window(width=1024, height=1024, visible=False):
        raise RuntimeError("Open3D failed to create an offscreen window for AO-Grasp visualization.")
    try:
        for geometry in geometries:
            vis.add_geometry(geometry)
        vis.poll_events()
        vis.update_renderer()
        if path.exists():
            path.unlink()
        vis.capture_screen_image(str(path), do_render=True)
        if not path.exists() or path.stat().st_size <= 0:
            raise RuntimeError(f"Open3D failed to save AO-Grasp visualization: {path}")
    finally:
        vis.destroy_window()


def _add_axis_legend(path: Path) -> None:
    plt = _pyplot()
    image = plt.imread(str(path))
    fig, ax = plt.subplots(figsize=(8, 8), dpi=128)
    ax.imshow(image)
    ax.axis("off")
    ax.text(18, 34, "fusion frame", color="black", fontsize=12, bbox={"facecolor": "white", "alpha": 0.75})
    ax.text(18, 62, "X red   Y green   Z blue", color="black", fontsize=11, bbox={"facecolor": "white", "alpha": 0.75})
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(path)
    plt.close(fig)


def _heatmap_colors(labels: np.ndarray) -> np.ndarray:
    low = float(labels.min())
    high = float(labels.max())
    scaled = np.full_like(labels, 0.5, dtype=np.float32) if high <= low else (labels - low) / (high - low)
    colors = np.empty((labels.shape[0], 3), dtype=np.float32)
    lower = scaled < 0.5
    colors[lower, 0] = 1.0
    colors[lower, 1] = 0.25 + 1.5 * scaled[lower]
    colors[lower, 2] = 0.1
    colors[~lower, 0] = 1.0 - 1.7 * (scaled[~lower] - 0.5)
    colors[~lower, 1] = 1.0
    colors[~lower, 2] = 0.1
    return np.clip(colors, 0.0, 1.0)


def _open3d():
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError("AO-Grasp 3D debug visualization requires open3d. Install open3d in this env.") from exc
    return o3d


def _pyplot():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("AO-Grasp debug visualization requires matplotlib. Install matplotlib in this env.") from exc
    return plt


def _points(value: np.ndarray) -> np.ndarray:
    points = np.asarray(value, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected points shape (N, 3), got {tuple(points.shape)}.")
    if not np.all(np.isfinite(points)):
        raise ValueError("Point cloud contains non-finite values.")
    return points


def _labels(value: np.ndarray, *, expected_points: int) -> np.ndarray:
    labels = np.asarray(value, dtype=np.float32).reshape(-1)
    if labels.shape != (int(expected_points),):
        raise ValueError(f"Expected heatmap shape {(int(expected_points),)}, got {tuple(labels.shape)}.")
    if not np.all(np.isfinite(labels)):
        raise ValueError("Heatmap contains non-finite values.")
    return labels


def _proposal_list(proposals: Sequence[Mapping[str, object]]) -> list[Mapping[str, object]]:
    parsed = list(proposals)
    if not parsed:
        raise ValueError("Expected at least one AO-Grasp proposal to visualize.")
    for proposal in parsed:
        np.asarray(proposal["position_cam"], dtype=np.float32).reshape(3)
        np.asarray(proposal["quaternion_cam"], dtype=np.float32).reshape(4)
        float(proposal["score"])
    return parsed


def _quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = quat.astype(np.float64, copy=False)
    norm = np.linalg.norm([w, x, y, z])
    if norm <= 0.0:
        raise ValueError("Quaternion norm must be positive.")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _jsonable(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _safe_stem(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(value))
