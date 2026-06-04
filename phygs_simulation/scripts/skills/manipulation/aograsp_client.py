"""Small AO-Grasp client for camera-to-grasp inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence
import uuid
from urllib import error as urllib_error
from urllib import request as urllib_request

import numpy as np
import yaml

from helpers.aograsp_viz import save_aograsp_visualizations, write_rgb_png
from helpers.cam_utils import (
    CameraPipelineConfig,
    CameraPipelineResult,
    DepthFilterConfig,
    ImageCrop,
    PointCloudFusionFrameConfig,
    PointCloudWorldFilterConfig,
    depth_preview_rgb,
    matrix_to_quat_wxyz,
    quat_wxyz_to_matrix,
    run_camera_pipeline,
    squeeze_rgb_image,
    to_numpy,
)
from helpers.pcd_utils import FusedPointCloud


class AoGraspServiceError(RuntimeError):
    """Raised when AO-Grasp sidecars are unavailable or return invalid data."""


@dataclass(frozen=True)
class AoGraspServiceConfig:
    """AO-Grasp service and point-cloud settings."""

    pointscore_url: str = "http://ao-pointscore:8001"
    cgn_url: str = "http://ao-cgn:8002"
    timeout_s: float = 30.0
    debug_root: str = "outputs/ao-grasp"
    total_points: int = 16384
    rng_seed: int = 0
    max_depth_m: float = 3.5
    reference_prim_path: str = "{ENV_REGEX_NS}/Robot/body"
    fusion_frame: Optional[PointCloudFusionFrameConfig] = None
    pointcloud_filter: PointCloudWorldFilterConfig = field(
        default_factory=lambda: PointCloudWorldFilterConfig(min_z_m=0.03)
    )
    proposal_viz_output_root: Optional[str] = "/workspace/isaaclab/outputs/ao-grasp"
    proposal_viz_top_k: int = 10


def load_aograsp_service_config(path: str | Path) -> AoGraspServiceConfig:
    root = _load_yaml_mapping(path, "AO-Grasp config")
    defaults = AoGraspServiceConfig()
    proposal_viz = _optional_mapping(root, "proposal_viz")
    proposal_viz_enabled = _parse_bool(proposal_viz.get("enabled", False), "proposal_viz.enabled")
    proposal_viz_output_root = (
        None
        if not proposal_viz_enabled
        else _optional_str(proposal_viz.get("output_root", defaults.proposal_viz_output_root))
    )
    return AoGraspServiceConfig(
        pointscore_url=os.environ.get("AO_POINTSCORE_URL", str(root.get("pointscore_url", defaults.pointscore_url))),
        cgn_url=os.environ.get("AO_CGN_URL", str(root.get("cgn_url", defaults.cgn_url))),
        timeout_s=_parse_float(root.get("timeout_s", defaults.timeout_s), "timeout_s"),
        debug_root=str(root.get("debug_root", defaults.debug_root)),
        total_points=_parse_int(root.get("total_points", defaults.total_points), "total_points"),
        rng_seed=_parse_int(root.get("rng_seed", defaults.rng_seed), "rng_seed"),
        max_depth_m=_parse_float(root.get("max_depth_m", defaults.max_depth_m), "max_depth_m"),
        reference_prim_path=_required_str(
            root.get("reference_prim_path", defaults.reference_prim_path),
            "reference_prim_path",
        ),
        fusion_frame=_load_fusion_frame_config(root),
        pointcloud_filter=_load_pointcloud_filter_config(root),
        proposal_viz_output_root=proposal_viz_output_root,
        proposal_viz_top_k=_parse_int(
            proposal_viz.get("viz_top_k", defaults.proposal_viz_top_k),
            "proposal_viz.viz_top_k",
        ),
    )


@dataclass(frozen=True)
class GraspPose:
    """Best AO-Grasp proposal expressed in fusion and world frames."""

    request_id: str
    position_world: np.ndarray
    quaternion_world: np.ndarray
    position_fusion: np.ndarray
    quaternion_fusion: np.ndarray
    score: float
    fusion_frame_name: str


class AoGraspDebugWriter:
    """Persist request-level AO-Grasp debug artifacts."""

    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir).expanduser().resolve()

    def request_dir(self, request_id: str) -> Path:
        output_dir = self.root_dir / str(request_id)
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def save_camera_pipeline(self, request_id: str, pipeline: CameraPipelineResult) -> Path:
        cloud = pipeline.fused_pointcloud
        if not isinstance(cloud, FusedPointCloud):
            raise ValueError("Camera pipeline result does not contain a fused point cloud.")

        output_dir = self.request_dir(request_id)
        np.save(output_dir / "points_fusion.npy", cloud.points_fusion.astype(np.float32, copy=False))
        if cloud.points_world is not None:
            np.save(output_dir / "points_world.npy", cloud.points_world.astype(np.float32, copy=False))

        depth_files: dict[str, str] = {}
        rgb_files: dict[str, str] = {}
        for frame in pipeline.frames:
            camera_name = str(frame.name)
            stem = self._safe_stem(camera_name)
            depth_path = self._write_image(output_dir / f"{stem}_depth", depth_preview_rgb(frame.depth_image))
            depth_files[camera_name] = depth_path.name

            if frame.rgb_image is None:
                raise ValueError(f"Debug frame '{camera_name}' has no RGB image.")
            rgb_path = self._write_image(output_dir / f"{stem}_rgb", squeeze_rgb_image(frame.rgb_image))
            rgb_files[camera_name] = rgb_path.name

        metadata = {
            "request_id": str(request_id),
            "camera_names": list(pipeline.camera_names),
            "fusion_frame_name": cloud.fusion_frame_name,
            "points_fusion_file": "points_fusion.npy",
            "points_fusion_shape": list(cloud.points_fusion.shape),
            "points_world_file": "points_world.npy" if cloud.points_world is not None else None,
            "points_world_shape": None if cloud.points_world is None else list(cloud.points_world.shape),
            "fusion_frame_position_w": cloud.fusion_frame_position_w,
            "fusion_frame_quat_wxyz": cloud.fusion_frame_quat_wxyz,
            "fusion_frame_position_reference": cloud.fusion_frame_position_reference,
            "fusion_frame_quat_reference": cloud.fusion_frame_quat_reference,
            "per_camera_input_points": dict(cloud.per_camera_input_points),
            "per_camera_sampled_points": dict(cloud.per_camera_sampled_points),
            "depth_preview_files": depth_files,
            "rgb_image_files": rgb_files,
        }
        self._write_metadata(output_dir, metadata)
        return output_dir

    def save_heatmap(self, request_id: str, heatmap: np.ndarray, *, expected_points: int) -> Path:
        heatmap_np = np.asarray(heatmap, dtype=np.float32)
        if heatmap_np.shape != (int(expected_points),):
            raise ValueError(
                f"Heatmap shape must be {(int(expected_points),)}, got {tuple(heatmap_np.shape)}."
            )
        output_dir = self.request_dir(request_id)
        path = output_dir / "heatmap.npy"
        np.save(path, heatmap_np)
        self._update_metadata(
            output_dir,
            {
                "heatmap_file": path.name,
                "heatmap_shape": list(heatmap_np.shape),
            },
        )
        return path

    def save_proposal(self, pose: GraspPose) -> Path:
        output_dir = self.request_dir(pose.request_id)
        path = output_dir / "proposal.json"
        path.write_text(json.dumps(self._to_jsonable(asdict(pose)), indent=2, sort_keys=True), encoding="utf-8")
        self._update_metadata(output_dir, {"proposal_file": path.name})
        return path

    @classmethod
    def _write_image(cls, path_stem: Path, image_rgb: np.ndarray) -> Path:
        image = np.asarray(image_rgb, dtype=np.uint8)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected an RGB image, got shape={tuple(image.shape)}.")
        path = path_stem.with_suffix(".png")
        write_rgb_png(path, image)
        return path

    @staticmethod
    def _safe_stem(value: str) -> str:
        return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(value))

    @classmethod
    def _to_jsonable(cls, value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Mapping):
            return {str(key): cls._to_jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._to_jsonable(item) for item in value]
        return value

    @classmethod
    def _write_metadata(cls, output_dir: Path, metadata: Mapping[str, Any]) -> None:
        (output_dir / "metadata.json").write_text(
            json.dumps(cls._to_jsonable(metadata), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    @classmethod
    def _update_metadata(cls, output_dir: Path, fields: Mapping[str, Any]) -> None:
        path = output_dir / "metadata.json"
        metadata = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        metadata.update(cls._to_jsonable(fields))
        cls._write_metadata(output_dir, metadata)


class AoGraspClient:
    """Capture cameras, build one fused cloud, and request the best AO-Grasp pose."""

    def __init__(self, config: Optional[AoGraspServiceConfig] = None) -> None:
        self._config = config or AoGraspServiceConfig()

    @property
    def config(self) -> AoGraspServiceConfig:
        return self._config

    def healthcheck(self) -> dict[str, dict]:
        """Return health payloads from both AO-Grasp services."""

        pointscore = self._request_json("GET", f"{self._config.pointscore_url.rstrip('/')}/healthz")
        cgn = self._request_json("GET", f"{self._config.cgn_url.rstrip('/')}/healthz")
        if not isinstance(pointscore, dict) or not isinstance(cgn, dict):
            raise AoGraspServiceError("AO-Grasp healthcheck must return JSON objects.")
        return {"pointscore": pointscore, "cgn": cgn}

    def trigger(
        self,
        camera_map: Mapping[str, object],
        camera_names: Sequence[str],
        *,
        crops_by_camera: Mapping[str, ImageCrop] | None = None,
        camera_prim_paths: Mapping[str, str] | None = None,
        depth_filter: DepthFilterConfig | None = None,
        debug: bool = False,
    ) -> GraspPose:
        """Run the full camera-to-grasp pipeline and return the best world-frame pose."""

        self._require_healthy_point_contract()
        request_id = uuid.uuid4().hex[:12]
        prim_paths = dict(camera_prim_paths or {})
        pipeline = run_camera_pipeline(
            camera_map,
            camera_names,
            CameraPipelineConfig(
                crops_by_camera=dict(crops_by_camera or {}),
                depth_filter=depth_filter or DepthFilterConfig(max_depth_m=self._config.max_depth_m),
                include_rgb=True,
                include_pointcloud=True,
                total_points=self._config.total_points,
                fusion_frame=self._config.fusion_frame,
                camera_prim_paths=prim_paths,
                reference_prim_path=self._config.reference_prim_path if prim_paths else None,
                rng_seed=self._config.rng_seed,
                pointcloud_filter=self._config.pointcloud_filter,
            ),
        )
        cloud = self._require_fused_cloud(pipeline)

        debug_writer = AoGraspDebugWriter(self._config.debug_root) if debug else None
        if debug_writer is not None:
            debug_writer.save_camera_pipeline(request_id, pipeline)

        heatmap = self._request_heatmap(request_id, cloud.points_fusion)
        if debug_writer is not None:
            debug_writer.save_heatmap(request_id, heatmap, expected_points=self._config.total_points)

        raw_proposals = self._request_proposals(request_id, cloud.points_fusion, heatmap)
        if debug_writer is not None and self._config.proposal_viz_output_root:
            proposal_viz_root = Path(self._config.proposal_viz_output_root).expanduser().resolve() / request_id
            artifacts = save_aograsp_visualizations(
                output_root=proposal_viz_root,
                request_id=request_id,
                points_fusion=cloud.points_fusion,
                heatmap=heatmap,
                proposals=raw_proposals,
                top_k=self._config.proposal_viz_top_k,
            )
            debug_writer._update_metadata(
                debug_writer.request_dir(request_id),
                {
                    "aograsp_heatmap_file": str(artifacts.heatmap_file),
                    "aograsp_heatmap_image_file": str(artifacts.heatmap_image_file),
                    "aograsp_heatmap_histogram_file": str(artifacts.heatmap_histogram_file),
                    "aograsp_grasp_proposals_file": str(artifacts.proposals_file),
                    "aograsp_grasp_proposals_image_file": str(artifacts.proposals_image_file),
                },
            )

        pose = self._decode_pose(request_id, cloud, raw_proposals[0])
        if debug_writer is not None:
            debug_writer.save_proposal(pose)
        return pose

    def _request_heatmap(self, request_id: str, points_fusion: np.ndarray) -> np.ndarray:
        points = self._coerce_points(points_fusion)
        payload = self._request_json(
            "POST",
            f"{self._config.pointscore_url.rstrip('/')}/heatmaps",
            payload={
                "request_id": str(request_id),
                "points_cam": points.tolist(),
            },
        )
        labels = np.asarray(payload.get("labels"), dtype=np.float32)
        if labels.shape != (points.shape[0],):
            raise AoGraspServiceError(
                f"Pointscore returned labels shape {tuple(labels.shape)}, expected {(points.shape[0],)}."
            )
        return labels

    def _request_proposals(self, request_id: str, points_fusion: np.ndarray, heatmap: np.ndarray) -> list[dict]:
        points = self._coerce_points(points_fusion)
        heatmap_np = np.asarray(heatmap, dtype=np.float32)
        if heatmap_np.shape != (points.shape[0],):
            raise AoGraspServiceError(
                f"Heatmap shape {tuple(heatmap_np.shape)} does not match points shape {(points.shape[0], 3)}."
            )

        payload = self._request_json(
            "POST",
            f"{self._config.cgn_url.rstrip('/')}/proposals",
            payload={
                "request_id": str(request_id),
                "points_cam": points.tolist(),
                "heatmap": heatmap_np.tolist(),
            },
        )
        proposals = payload.get("proposals")
        if not isinstance(proposals, list) or not proposals:
            raise AoGraspServiceError("AO-Grasp did not return any grasp proposals.")
        if not all(isinstance(item, dict) for item in proposals):
            raise AoGraspServiceError("AO-Grasp proposals must be JSON objects.")
        return proposals

    def _decode_pose(self, request_id: str, cloud: FusedPointCloud, proposal: Mapping[str, Any]) -> GraspPose:
        try:
            position_fusion = np.asarray(proposal["position_cam"], dtype=np.float32).reshape(3)
            quaternion_fusion = np.asarray(proposal["quaternion_cam"], dtype=np.float32).reshape(4)
            score = float(proposal["score"])
        except (KeyError, TypeError, ValueError) as exc:
            raise AoGraspServiceError(f"Invalid AO-Grasp proposal payload: {proposal}") from exc

        position_world, quaternion_world = _transform_pose_to_world(
            position_fusion,
            quaternion_fusion,
            cloud.fusion_frame_position_w,
            cloud.fusion_frame_quat_wxyz,
        )
        return GraspPose(
            request_id=str(request_id),
            position_world=position_world,
            quaternion_world=quaternion_world,
            position_fusion=position_fusion,
            quaternion_fusion=quaternion_fusion,
            score=score,
            fusion_frame_name=cloud.fusion_frame_name,
        )

    def _require_healthy_point_contract(self) -> dict[str, dict]:
        health = self.healthcheck()
        mismatches: list[str] = []
        missing: list[str] = []
        for service_name in ("pointscore", "cgn"):
            payload = health.get(service_name, {})
            expected = payload.get("expected_points")
            if expected is None:
                missing.append(service_name)
                continue
            try:
                expected_points = int(expected)
            except (TypeError, ValueError) as exc:
                raise AoGraspServiceError(
                    f"AO-Grasp {service_name} returned invalid expected_points={expected!r}."
                ) from exc
            if expected_points != int(self._config.total_points):
                mismatches.append(f"{service_name}={expected_points}")

        if missing:
            raise AoGraspServiceError(
                "AO-Grasp healthcheck is missing expected_points for: " + ", ".join(missing)
            )
        if mismatches:
            raise AoGraspServiceError(
                "AO-Grasp point-count contract mismatch: "
                f"client={int(self._config.total_points)}, sidecars {', '.join(mismatches)}."
            )
        return health

    def _require_fused_cloud(self, pipeline: CameraPipelineResult) -> FusedPointCloud:
        cloud = pipeline.fused_pointcloud
        if not isinstance(cloud, FusedPointCloud):
            raise AoGraspServiceError("Camera pipeline did not return a fused point cloud.")
        points = self._coerce_points(cloud.points_fusion)
        expected_shape = (int(self._config.total_points), 3)
        if points.shape != expected_shape:
            raise AoGraspServiceError(f"Fused point cloud shape {tuple(points.shape)} != expected {expected_shape}.")
        return cloud

    def _coerce_points(self, points_fusion: np.ndarray) -> np.ndarray:
        points = to_numpy(points_fusion).astype(np.float32, copy=False)
        expected_shape = (int(self._config.total_points), 3)
        if points.shape != expected_shape:
            raise AoGraspServiceError(f"AO-Grasp input cloud shape {tuple(points.shape)} != expected {expected_shape}.")
        if not np.all(np.isfinite(points)):
            raise AoGraspServiceError("AO-Grasp input cloud contains non-finite values.")
        return points

    def _request_json(self, method: str, url: str, payload: Optional[dict] = None) -> dict:
        body = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        request = urllib_request.Request(url=url, data=body, headers=headers, method=method.upper())
        try:
            with urllib_request.urlopen(request, timeout=float(self._config.timeout_s)) as response:
                raw = response.read().decode("utf-8")
        except urllib_error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise AoGraspServiceError(f"AO-Grasp request failed ({exc.code}) for {url}: {detail}") from exc
        except urllib_error.URLError as exc:
            raise AoGraspServiceError(f"AO-Grasp service is unreachable at {url}: {exc.reason}") from exc

        try:
            decoded = json.loads(raw) if raw else {}
        except json.JSONDecodeError as exc:
            raise AoGraspServiceError(f"AO-Grasp service returned invalid JSON for {url}.") from exc
        if not isinstance(decoded, dict):
            raise AoGraspServiceError(f"AO-Grasp service returned a non-object JSON payload for {url}.")
        return decoded


def _compose_poses(
    parent_position,
    parent_quat_wxyz,
    child_position,
    child_quat_wxyz,
) -> tuple[np.ndarray, np.ndarray]:
    parent_rot = quat_wxyz_to_matrix(parent_quat_wxyz)
    child_rot = quat_wxyz_to_matrix(child_quat_wxyz)
    parent_pos = to_numpy(parent_position).astype(np.float64, copy=False).reshape(3)
    child_pos = to_numpy(child_position).astype(np.float64, copy=False).reshape(3)
    position = parent_pos + parent_rot @ child_pos
    rotation = parent_rot @ child_rot
    return position.astype(np.float32), matrix_to_quat_wxyz(rotation)


def _transform_pose_to_world(
    position,
    quaternion,
    frame_position_w,
    frame_quat_wxyz,
) -> tuple[np.ndarray, np.ndarray]:
    return _compose_poses(frame_position_w, frame_quat_wxyz, position, quaternion)


def _load_pointcloud_filter_config(root: Mapping[str, object]) -> PointCloudWorldFilterConfig:
    filter_data = _optional_mapping(root, "pointcloud_filter")
    return PointCloudWorldFilterConfig(
        min_z_m=None
        if filter_data.get("min_z_m") in (None, "")
        else _parse_float(filter_data.get("min_z_m"), "pointcloud_filter.min_z_m"),
        bounds_min=None
        if filter_data.get("bounds_min") in (None, "")
        else _parse_float_tuple(filter_data.get("bounds_min"), "pointcloud_filter.bounds_min", length=3),
        bounds_max=None
        if filter_data.get("bounds_max") in (None, "")
        else _parse_float_tuple(filter_data.get("bounds_max"), "pointcloud_filter.bounds_max", length=3),
        voxel_size_m=None
        if filter_data.get("voxel_size_m") in (None, "")
        else _parse_float(filter_data.get("voxel_size_m"), "pointcloud_filter.voxel_size_m"),
    )


def _load_fusion_frame_config(root: Mapping[str, object]) -> Optional[PointCloudFusionFrameConfig]:
    data = _optional_mapping(root, "fusion_frame")
    if not data:
        return None

    mode = _required_str(data.get("mode"), "fusion_frame.mode")
    return PointCloudFusionFrameConfig(
        mode=mode,
        name=str(data.get("name", "") or ""),
        camera_name=_optional_str(data.get("camera_name")),
        left_camera_name=_optional_str(data.get("left_camera_name")),
        right_camera_name=_optional_str(data.get("right_camera_name")),
    )


def _load_yaml_mapping(path: str | Path, label: str) -> Mapping[str, object]:
    config_path = Path(path).expanduser().resolve()
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, Mapping):
        raise TypeError(f"Expected {label} to be a mapping, got {type(data).__name__}.")
    return data


def _optional_mapping(value: Mapping[str, object], key: str) -> Mapping[str, object]:
    if key not in value:
        return {}
    item = value[key]
    if not isinstance(item, Mapping):
        raise TypeError(f"Expected {key} to be a mapping, got {type(item).__name__}.")
    return item


def _optional_str(value: object) -> Optional[str]:
    return None if value in (None, "") else str(value)


def _required_str(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"Expected {label} to be a non-empty string.")
    return value.strip()


def _parse_bool(value: object, label: str) -> bool:
    if isinstance(value, bool):
        return value
    raise TypeError(f"Expected {label} to be a boolean, got {value!r}.")


def _parse_float(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"Expected {label} to be numeric, got {value!r}.")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"Expected {label} to be numeric, got {value!r}.") from exc


def _parse_float_tuple(value: object, label: str, *, length: int) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"Expected {label} to be a numeric sequence with {int(length)} values.")
    parsed = tuple(_parse_float(item, f"{label}[{idx}]") for idx, item in enumerate(value))
    if len(parsed) != int(length):
        raise ValueError(f"Expected {label} to contain {int(length)} values, got {len(parsed)}.")
    return parsed


def _parse_int(value: object, label: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"Expected {label} to be an integer, got {value!r}.")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"Expected {label} to be an integer, got {value!r}.") from exc


__all__ = [
    "AoGraspClient",
    "AoGraspDebugWriter",
    "AoGraspServiceConfig",
    "AoGraspServiceError",
    "GraspPose",
    "load_aograsp_service_config",
]
