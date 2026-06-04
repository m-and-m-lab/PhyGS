"""Small CuRobo interface for camera capture, optional ESDF updates, and pose planning."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence
import uuid

import numpy as np
import torch
import yaml

from helpers.cam_utils import (
    CameraPipelineConfig,
    CameraPipelineResult,
    DepthFilterConfig,
    ImageCrop,
    quat_wxyz_to_matrix,
    run_camera_pipeline,
    to_numpy,
)
from helpers.pcd_utils import FusedPointCloud


_DEFAULT_CONTROL_DT_S = 0.0167


class CuroboInterfaceError(RuntimeError):
    """Base error raised by the simple CuRobo interface."""


class CuroboInputError(CuroboInterfaceError, ValueError):
    """Raised when request/config inputs are invalid."""


class CuroboPlanningError(CuroboInterfaceError):
    """Raised when CuRobo does not produce a successful motion plan."""


@dataclass(frozen=True)
class CuroboInterfaceConfig:
    """Configuration for the direct CuRobo planning interface."""

    robot_cfg: str | dict
    command_joint_names: Sequence[str]
    device: str = "cuda:0"
    world_cfg: Optional[Any] = None
    debug_root: str = "outputs/curobo"
    total_points: int = 16384
    rng_seed: int = 0
    interpolation_dt: float = 0.05
    trajopt_dt: Optional[float] = None
    trajopt_tsteps: int = 32
    optimize_dt: bool = True
    max_attempts: int = 4
    enable_finetune_trajopt: bool = True
    num_trajopt_seeds: int = 12
    num_graph_seeds: int = 12
    collision_cache_obb: int = 30
    collision_cache_mesh: int = 100
    use_cuda_graph: bool = True
    warmup: bool = True
    voxel_collision_enabled: bool = True
    voxel_layer_name: str = "world_voxel"
    voxel_bounds_min: Optional[Sequence[float]] = None
    voxel_bounds_max: Optional[Sequence[float]] = None
    voxel_size_m: float = 0.025
    max_depth_m: float = 3.5
    truncation_distance_vox: float = 2.0
    raycast_subsampling: int = 1
    accumulate_depth_frames: bool = False
    import_invert_sign: bool = True
    import_add_half_voxel: bool = True


def load_curobo_interface_config(
    path: str | Path,
    *,
    command_joint_names: Sequence[str],
    device: str,
) -> tuple[CuroboInterfaceConfig, Mapping[str, object]]:
    config_path = Path(path).expanduser().resolve()
    root = _load_yaml_mapping(config_path, "CuRobo runtime config")
    gripper_data = _optional_mapping(root, "gripper")
    robot_config_path = _resolve_child_path(
        root.get("robot_config_path"),
        base_dir=config_path.parent,
        label="robot_config_path",
    )
    close_steps = _parse_int(root.get("close_steps", 30), "close_steps")
    open_timeout_s = _parse_float(gripper_data.get("open_timeout_sec", 3.0), "gripper.open_timeout_sec")
    use_depth_collision = _parse_bool(root.get("use_depth_collision", True), "use_depth_collision")
    voxel_bounds_min = None
    voxel_bounds_max = None
    if use_depth_collision:
        voxel_bounds_min = _parse_float_tuple(root.get("voxel_bounds_min"), "voxel_bounds_min", length=3)
        voxel_bounds_max = _parse_float_tuple(root.get("voxel_bounds_max"), "voxel_bounds_max", length=3)
    return (
        CuroboInterfaceConfig(
            robot_cfg=str(robot_config_path),
            command_joint_names=tuple(str(name) for name in command_joint_names),
            device=str(device),
            total_points=_parse_int(root.get("total_points", 16384), "total_points"),
            max_depth_m=_parse_float(root.get("depth_max_m", 3.5), "depth_max_m"),
            interpolation_dt=_parse_float(root.get("interpolation_dt", 0.05), "interpolation_dt"),
            trajopt_tsteps=_parse_int(root.get("trajopt_tsteps", 32), "trajopt_tsteps"),
            max_attempts=_parse_int(root.get("max_attempts", 4), "max_attempts"),
            num_trajopt_seeds=_parse_int(root.get("num_trajopt_seeds", 12), "num_trajopt_seeds"),
            num_graph_seeds=_parse_int(root.get("num_graph_seeds", 12), "num_graph_seeds"),
            voxel_collision_enabled=use_depth_collision,
            voxel_bounds_min=voxel_bounds_min,
            voxel_bounds_max=voxel_bounds_max,
            voxel_size_m=_parse_float(root.get("depth_voxel_size", 0.025), "depth_voxel_size"),
            truncation_distance_vox=_parse_float(root.get("truncation_distance_vox", 2.0), "truncation_distance_vox"),
            raycast_subsampling=_parse_int(root.get("raycast_subsampling", 1), "raycast_subsampling"),
            accumulate_depth_frames=_parse_bool(root.get("accumulate_frames", False), "accumulate_frames"),
            import_invert_sign=_parse_bool(root.get("import_invert_sign", True), "import_invert_sign"),
            import_add_half_voxel=_parse_bool(root.get("import_add_half_voxel", True), "import_add_half_voxel"),
        ),
        {
            "open_rad": _parse_float(gripper_data.get("open_rad", -1.56), "gripper.open_rad"),
            "closed_rad": _parse_float(gripper_data.get("closed_rad", -0.4), "gripper.closed_rad"),
            "tolerance_rad": _parse_float(gripper_data.get("tolerance_rad", 0.02), "gripper.tolerance_rad"),
            "open_timeout_steps": _steps_from_seconds(open_timeout_s, label="gripper.open_timeout_sec"),
            "close_steps": close_steps,
            "camera_settle_steps": _parse_int(
                gripper_data.get("camera_settle_steps", 10),
                "gripper.camera_settle_steps",
            ),
        },
    )


@dataclass(frozen=True)
class CuroboTriggerRequest:
    """One camera-to-plan request with a caller-provided target pose."""

    camera_map: Mapping[str, object]
    camera_names: Sequence[str]
    joint_positions: Any
    joint_velocities: Any | None
    joint_names: Sequence[str]
    target_position_w: Any
    target_quat_wxyz: Any
    crops_by_camera: Mapping[str, ImageCrop] = field(default_factory=dict)
    depth_filter: DepthFilterConfig = field(default_factory=DepthFilterConfig)
    include_rgb: bool = True
    include_pointcloud: bool = True
    total_points: Optional[int] = None
    rng_seed: Optional[int] = None
    stage: Any | None = None
    stage_only_paths: Sequence[str] = ("/World",)
    stage_ignore_substring: Sequence[str] = ("Robot", "target", "curobo")
    stage_reference_prim_path: Optional[str] = None
    request_id: Optional[str] = None
    debug: bool = False


@dataclass(frozen=True)
class CuroboTriggerResult:
    """Result returned by :meth:`CuroboInterface.trigger`."""

    request_id: str
    target_position_w: np.ndarray
    target_quat_wxyz: np.ndarray
    camera_pipeline: CameraPipelineResult
    plan: Any
    status: str
    debug_dir: Optional[Path] = None
    esdf_grid: Optional[Mapping[str, Any]] = None

    @property
    def fused_pointcloud(self) -> Optional[FusedPointCloud]:
        cloud = self.camera_pipeline.fused_pointcloud
        return cloud if isinstance(cloud, FusedPointCloud) else None


@dataclass(frozen=True)
class _CuroboRuntime:
    motion_gen: Any
    tensor_args: Any
    plan_config: Any
    pose_cls: Any
    joint_state_cls: Any


@dataclass(frozen=True)
class _FixedEsdfGrid:
    origin_m: np.ndarray
    dims_vox: np.ndarray
    voxel_size_m: float
    query_spheres_m: torch.Tensor


class CuroboDebugWriter:
    """Persist request-level camera, point-cloud, ESDF, and trajectory artifacts."""

    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir).expanduser().resolve()

    def request_dir(self, request_id: str) -> Path:
        output_dir = self.root_dir / str(request_id)
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def save(
        self,
        *,
        request_id: str,
        pipeline: CameraPipelineResult,
        target_position_w: np.ndarray,
        target_quat_wxyz: np.ndarray,
        plan: Any,
        status: str,
        esdf_grid: Optional[Mapping[str, Any]] = None,
    ) -> Path:
        output_dir = self.request_dir(request_id)

        frame_files: dict[str, dict[str, str]] = {}
        for frame in pipeline.frames:
            camera_name = str(frame.name)
            stem = self._safe_stem(camera_name)
            files: dict[str, str] = {}
            depth_path = output_dir / f"{stem}_depth.npy"
            np.save(depth_path, to_numpy(frame.depth_image).astype(np.float32, copy=False))
            files["depth"] = depth_path.name
            if frame.rgb_image is not None:
                rgb_path = output_dir / f"{stem}_rgb.npy"
                np.save(rgb_path, to_numpy(frame.rgb_image))
                files["rgb"] = rgb_path.name
            if frame.segmentation is not None:
                segmentation_path = output_dir / f"{stem}_segmentation.npy"
                np.save(segmentation_path, to_numpy(frame.segmentation))
                files["segmentation"] = segmentation_path.name
            frame_files[camera_name] = files

        cloud_metadata: dict[str, Any] = {}
        cloud = pipeline.fused_pointcloud
        if isinstance(cloud, FusedPointCloud):
            points_fusion_path = output_dir / "points_fusion.npy"
            np.save(points_fusion_path, cloud.points_fusion.astype(np.float32, copy=False))
            cloud_metadata["points_fusion_file"] = points_fusion_path.name
            cloud_metadata["points_fusion_shape"] = list(cloud.points_fusion.shape)
            if cloud.points_world is not None:
                points_world_path = output_dir / "points_world.npy"
                np.save(points_world_path, cloud.points_world.astype(np.float32, copy=False))
                cloud_metadata["points_world_file"] = points_world_path.name
                cloud_metadata["points_world_shape"] = list(cloud.points_world.shape)
            cloud_metadata["fusion_frame_name"] = cloud.fusion_frame_name
            cloud_metadata["per_camera_input_points"] = dict(cloud.per_camera_input_points)
            cloud_metadata["per_camera_sampled_points"] = dict(cloud.per_camera_sampled_points)

        plan_position = to_numpy(getattr(plan, "position")).astype(np.float32, copy=False)
        plan_position_path = output_dir / "plan_position.npy"
        np.save(plan_position_path, plan_position)
        plan_velocity_file = None
        plan_velocity = getattr(plan, "velocity", None)
        if plan_velocity is not None:
            plan_velocity_path = output_dir / "plan_velocity.npy"
            np.save(plan_velocity_path, to_numpy(plan_velocity).astype(np.float32, copy=False))
            plan_velocity_file = plan_velocity_path.name

        esdf_metadata = None
        if esdf_grid is not None:
            esdf_path = output_dir / "esdf_grid.npy"
            np.save(esdf_path, to_numpy(esdf_grid["grid"]).astype(np.float32, copy=False))
            esdf_metadata = {
                "grid_file": esdf_path.name,
                "dims": [int(v) for v in esdf_grid["dims"]],
                "voxel_size": float(esdf_grid["voxel_size"]),
                "origin": [float(v) for v in esdf_grid["origin"]],
            }

        metadata = {
            "request_id": str(request_id),
            "status": str(status),
            "camera_names": list(pipeline.camera_names),
            "target_position_w": target_position_w,
            "target_quat_wxyz": target_quat_wxyz,
            "frame_files": frame_files,
            "cloud": cloud_metadata,
            "plan_position_file": plan_position_path.name,
            "plan_position_shape": list(plan_position.shape),
            "plan_velocity_file": plan_velocity_file,
            "plan_joint_names": list(getattr(plan, "joint_names", ())),
            "esdf": esdf_metadata,
        }
        (output_dir / "metadata.json").write_text(
            json.dumps(self._to_jsonable(metadata), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return output_dir

    @staticmethod
    def _safe_stem(value: str) -> str:
        return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(value))

    @classmethod
    def _to_jsonable(cls, value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
        if isinstance(value, Mapping):
            return {str(key): cls._to_jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._to_jsonable(item) for item in value]
        return value


class CuroboInterface:
    """Direct CuRobo camera-to-plan interface with optional nvblox ESDF updates."""

    def __init__(
        self,
        config: CuroboInterfaceConfig,
        *,
        debug_writer: Optional[CuroboDebugWriter] = None,
        runtime: Optional[_CuroboRuntime] = None,
    ) -> None:
        self.config = config
        self._runtime = runtime or self._build_runtime(config)
        self._debug_writer = debug_writer or CuroboDebugWriter(config.debug_root)
        self._fixed_esdf_grid: Optional[_FixedEsdfGrid] = None
        self._mapper: Optional[Any] = None
        self._query_type_esdf: Optional[Any] = None
        self._sensor_cls: Optional[Any] = None
        self._sensor_cache: dict[str, tuple[tuple[float, ...], Any]] = {}

        if config.voxel_collision_enabled:
            self._fixed_esdf_grid = self._make_fixed_esdf_grid(config, self._runtime.tensor_args.device)
            self._initialize_nvblox(config)

    def _update_world_from_stage(
        self,
        stage: Any,
        *,
        only_paths: Sequence[str],
        ignore_substring: Sequence[str],
        reference_prim_path: str,
    ) -> None:
        """Replace CuRobo's collision world with static obstacles from a USD stage."""

        if stage is None:
            raise CuroboInputError("stage is required to update CuRobo's static collision world.")
        if not only_paths:
            raise CuroboInputError("only_paths must contain at least one USD path.")
        if not str(reference_prim_path).strip():
            raise CuroboInputError("reference_prim_path must be a non-empty USD prim path.")

        try:
            from curobo.util.usd_helper import UsdHelper
        except ImportError as exc:
            raise CuroboInterfaceError("Static stage collision updates require curobo.util.usd_helper.") from exc

        usd_helper = UsdHelper()
        usd_helper.load_stage(stage)
        world = usd_helper.get_obstacles_from_stage(
            only_paths=[str(path) for path in only_paths],
            ignore_substring=[str(item) for item in ignore_substring],
            reference_prim_path=str(reference_prim_path),
        ).get_collision_check_world()
        self._runtime.motion_gen.update_world(world)
        if self._runtime.tensor_args.device.type == "cuda":
            torch.cuda.synchronize(device=self._runtime.tensor_args.device)

    def trigger(self, request: CuroboTriggerRequest) -> CuroboTriggerResult:
        """Capture cameras, update optional ESDF collision, and plan to a target pose."""

        request_id = request.request_id or uuid.uuid4().hex[:12]
        target_position = _as_float_vector(request.target_position_w, 3, "target_position_w")
        target_quat = _as_float_vector(request.target_quat_wxyz, 4, "target_quat_wxyz")
        if not self.config.voxel_collision_enabled:
            if request.stage is None:
                raise CuroboInputError("stage is required when voxel_collision_enabled is false.")
            if request.stage_reference_prim_path is None:
                raise CuroboInputError("stage_reference_prim_path is required when voxel_collision_enabled is false.")

        pipeline = run_camera_pipeline(
            request.camera_map,
            request.camera_names,
            CameraPipelineConfig(
                crops_by_camera=dict(request.crops_by_camera),
                depth_filter=request.depth_filter,
                include_rgb=bool(request.include_rgb),
                include_pointcloud=bool(request.include_pointcloud),
                total_points=int(request.total_points or self.config.total_points),
                rng_seed=int(self.config.rng_seed if request.rng_seed is None else request.rng_seed),
            ),
        )

        esdf_grid = None
        if self.config.voxel_collision_enabled:
            esdf_grid = self._update_voxel_world_from_frames(pipeline.frames)
        else:
            self._update_world_from_stage(
                request.stage,
                only_paths=request.stage_only_paths,
                ignore_substring=request.stage_ignore_substring,
                reference_prim_path=request.stage_reference_prim_path,
            )

        plan, status = self._plan_to_pose(
            joint_positions=request.joint_positions,
            joint_velocities=request.joint_velocities,
            joint_names=request.joint_names,
            target_position_w=target_position,
            target_quat_wxyz=target_quat,
        )

        debug_dir = None
        if request.debug:
            debug_dir = self._debug_writer.save(
                request_id=request_id,
                pipeline=pipeline,
                target_position_w=target_position,
                target_quat_wxyz=target_quat,
                plan=plan,
                status=status,
                esdf_grid=esdf_grid,
            )

        return CuroboTriggerResult(
            request_id=request_id,
            target_position_w=target_position,
            target_quat_wxyz=target_quat,
            camera_pipeline=pipeline,
            plan=plan,
            status=status,
            debug_dir=debug_dir,
            esdf_grid=esdf_grid,
        )

    @classmethod
    def _build_runtime(cls, config: CuroboInterfaceConfig) -> _CuroboRuntime:
        (
            CollisionCheckerType,
            TensorDeviceType,
            Pose,
            JointState,
            MotionGen,
            MotionGenConfig,
            MotionGenPlanConfig,
            WorldConfig,
            VoxelGrid,
            setup_curobo_logger,
        ) = _import_curobo_api()

        setup_curobo_logger("warn")
        tensor_args = TensorDeviceType(device=torch.device(config.device))
        world_cfg = config.world_cfg.clone() if config.world_cfg is not None else WorldConfig()
        collision_checker_type = CollisionCheckerType.MESH
        use_cuda_graph = bool(config.use_cuda_graph)

        if config.voxel_collision_enabled:
            collision_checker_type = CollisionCheckerType.VOXEL
            use_cuda_graph = False
            world_cfg = cls._with_placeholder_voxel(world_cfg, config, VoxelGrid, tensor_args.dtype)

        motion_gen_config = MotionGenConfig.load_from_robot_config(
            config.robot_cfg,
            world_cfg,
            tensor_args,
            collision_checker_type=collision_checker_type,
            use_cuda_graph=use_cuda_graph,
            num_trajopt_seeds=int(config.num_trajopt_seeds),
            num_graph_seeds=int(config.num_graph_seeds),
            interpolation_dt=float(config.interpolation_dt),
            collision_cache={"obb": int(config.collision_cache_obb), "mesh": int(config.collision_cache_mesh)},
            optimize_dt=bool(config.optimize_dt),
            trajopt_dt=config.trajopt_dt,
            trajopt_tsteps=int(config.trajopt_tsteps),
        )
        motion_gen = MotionGen(motion_gen_config)
        if config.warmup:
            motion_gen.warmup(enable_graph=True, warmup_js_trajopt=False)

        plan_config = MotionGenPlanConfig(
            enable_graph=False,
            enable_graph_attempt=2,
            max_attempts=int(config.max_attempts),
            enable_finetune_trajopt=bool(config.enable_finetune_trajopt),
            time_dilation_factor=0.5,
        )
        return _CuroboRuntime(
            motion_gen=motion_gen,
            tensor_args=tensor_args,
            plan_config=plan_config,
            pose_cls=Pose,
            joint_state_cls=JointState,
        )

    @staticmethod
    def _with_placeholder_voxel(
        world_cfg: Any,
        config: CuroboInterfaceConfig,
        voxel_grid_cls: Any,
        feature_dtype: Any,
    ) -> Any:
        origin, dims_vox, voxel_size = _fixed_grid_geometry(config)
        if any(voxel.name == config.voxel_layer_name for voxel in getattr(world_cfg, "voxel", [])):
            raise CuroboInputError(f"World config already contains voxel layer '{config.voxel_layer_name}'.")

        dims_span = dims_vox.astype(np.float32) * float(voxel_size)
        dims_m = np.maximum(dims_vox.astype(np.float32) - 1.0, 0.0) * float(voxel_size)
        center = origin + 0.5 * dims_span
        world_cfg.add_obstacle(
            voxel_grid_cls(
                name=str(config.voxel_layer_name),
                pose=[float(center[0]), float(center[1]), float(center[2]), 1.0, 0.0, 0.0, 0.0],
                dims=[float(v) for v in dims_m.tolist()],
                voxel_size=float(voxel_size),
                feature_dtype=feature_dtype,
            )
        )
        return world_cfg

    def _plan_to_pose(
        self,
        *,
        joint_positions: Any,
        joint_velocities: Any | None,
        joint_names: Sequence[str],
        target_position_w: np.ndarray,
        target_quat_wxyz: np.ndarray,
    ) -> tuple[Any, str]:
        runtime = self._runtime
        position = _single_tensor_vector(runtime.tensor_args, joint_positions, "joint_positions")
        velocity = (
            torch.zeros_like(position)
            if joint_velocities is None
            else _single_tensor_vector(runtime.tensor_args, joint_velocities, "joint_velocities")
        )
        if position.shape != velocity.shape:
            raise CuroboInputError(
                "joint_velocities shape "
                f"{tuple(velocity.shape)} does not match joint_positions shape {tuple(position.shape)}."
            )
        names = [str(name) for name in joint_names]
        if len(names) != int(position.numel()):
            raise CuroboInputError(
                f"joint_names length {len(names)} does not match joint_positions length {int(position.numel())}."
            )

        zero = torch.zeros_like(position)
        start_state = runtime.joint_state_cls(
            position=position,
            velocity=velocity,
            acceleration=zero,
            jerk=zero,
            joint_names=names,
        )
        start_state = start_state.get_ordered_joint_state(runtime.motion_gen.kinematics.joint_names)
        self._validate_start_state(start_state)

        goal_pose = runtime.pose_cls(
            position=runtime.tensor_args.to_device(target_position_w),
            quaternion=runtime.tensor_args.to_device(target_quat_wxyz),
        )
        result = runtime.motion_gen.plan_single(start_state.unsqueeze(0), goal_pose, runtime.plan_config)
        if not bool(result.success.item()):
            raise CuroboPlanningError(f"CuRobo planning failed: {result.status}")

        plan = result.get_interpolated_plan()
        plan = runtime.motion_gen.get_full_js(plan)
        command_names = [str(name) for name in self.config.command_joint_names]
        missing = [name for name in command_names if name not in plan.joint_names]
        if missing:
            raise CuroboInputError(f"Planned trajectory is missing command joints: {missing}")
        plan = plan.get_ordered_joint_state(command_names)
        return plan, str(result.status)

    def _validate_start_state(self, start_state: Any) -> None:
        position = getattr(start_state, "position")
        if not torch.isfinite(position).all():
            raise CuroboInputError("joint_positions contains non-finite values.")

        velocity = getattr(start_state, "velocity", None)
        if velocity is not None and not torch.isfinite(velocity).all():
            raise CuroboInputError("joint_velocities contains non-finite values.")

        limits = self._runtime.motion_gen.kinematics.get_joint_limits()
        raw_limits = limits.position
        if raw_limits.shape[0] == 2:
            lower = raw_limits[0].to(device=position.device, dtype=position.dtype)
            upper = raw_limits[1].to(device=position.device, dtype=position.dtype)
        else:
            lower = raw_limits[:, 0].to(device=position.device, dtype=position.dtype)
            upper = raw_limits[:, 1].to(device=position.device, dtype=position.dtype)

        if lower.shape != position.shape or upper.shape != position.shape:
            raise CuroboInputError(
                f"Joint limit shape {tuple(raw_limits.shape)} does not match start state shape {tuple(position.shape)}."
            )

        out_of_bounds = (position < lower) | (position > upper)
        if torch.any(out_of_bounds):
            bad_indices = torch.where(out_of_bounds)[0].detach().cpu().tolist()
            names = list(getattr(limits, "joint_names", getattr(start_state, "joint_names", ())))
            details = [
                f"{names[idx] if idx < len(names) else idx}={float(position[idx].detach().cpu().item()):.6f}"
                " not in "
                f"[{float(lower[idx].detach().cpu().item()):.6f}, "
                f"{float(upper[idx].detach().cpu().item()):.6f}]"
                for idx in bad_indices
            ]
            raise CuroboInputError("Start state violates joint limits: " + "; ".join(details))

    def _initialize_nvblox(self, config: CuroboInterfaceConfig) -> None:
        if self._runtime.tensor_args.device.type != "cuda":
            raise CuroboInputError("voxel_collision_enabled requires a CUDA device for nvblox_torch.")

        try:
            from nvblox_torch.mapper import Mapper, QueryType
            from nvblox_torch.mapper_params import MapperParams, ProjectiveIntegratorParams
            from nvblox_torch.projective_integrator_types import ProjectiveIntegratorType
            from nvblox_torch.sensor import Sensor
        except ImportError as exc:
            raise CuroboInterfaceError("voxel_collision_enabled requires nvblox_torch.") from exc

        projective_params = ProjectiveIntegratorParams()
        projective_params.projective_integrator_max_integration_distance_m = float(config.max_depth_m)
        projective_params.projective_integrator_truncation_distance_vox = float(config.truncation_distance_vox)

        mapper_params = MapperParams()
        mapper_params.set_projective_integrator_params(projective_params)

        esdf_params = mapper_params.get_esdf_integrator_params()
        esdf_params.esdf_integrator_max_distance_m = float(config.max_depth_m)
        mapper_params.set_esdf_integrator_params(esdf_params)

        view_params = mapper_params.get_view_calculator_params()
        view_params.raycast_subsampling_factor = int(max(1, config.raycast_subsampling))
        bounds_min, dims, voxel_size = _fixed_grid_geometry(config)
        bounds_max = bounds_min + dims.astype(np.float32) * float(voxel_size)
        view_params.workspace_bounds_type = "kBoundingBox"
        view_params.workspace_bounds_min_corner_x_m = float(bounds_min[0])
        view_params.workspace_bounds_max_corner_x_m = float(bounds_max[0])
        view_params.workspace_bounds_min_corner_y_m = float(bounds_min[1])
        view_params.workspace_bounds_max_corner_y_m = float(bounds_max[1])
        view_params.workspace_bounds_min_height_m = float(bounds_min[2])
        view_params.workspace_bounds_max_height_m = float(bounds_max[2])
        mapper_params.set_view_calculator_params(view_params)

        self._mapper = Mapper(
            voxel_sizes_m=[float(config.voxel_size_m)],
            integrator_types=[ProjectiveIntegratorType.TSDF],
            mapper_parameters=mapper_params,
        )
        self._query_type_esdf = QueryType.ESDF
        self._sensor_cls = Sensor

    def _update_voxel_world_from_frames(self, frames: Sequence[Any]) -> Mapping[str, Any]:
        if self._mapper is None or self._query_type_esdf is None or self._fixed_esdf_grid is None:
            raise CuroboInterfaceError("Voxel collision is enabled but nvblox was not initialized.")
        if not frames:
            raise CuroboInputError("Voxel collision update requires at least one camera frame.")

        _activate_cuda_device(self._runtime.tensor_args.device)
        if not self.config.accumulate_depth_frames:
            self._mapper.clear(mapper_id=0)

        for frame in frames:
            depth = self._runtime.tensor_args.to_device(frame.depth_image).to(dtype=torch.float32)
            if depth.ndim != 2:
                raise CuroboInputError(f"Frame '{frame.name}' depth image must be 2-D, got {tuple(depth.shape)}.")
            depth = torch.where(torch.isfinite(depth), depth, torch.zeros_like(depth)).contiguous()
            sensor = self._get_or_create_sensor(frame, int(depth.shape[1]), int(depth.shape[0]))
            t_w_c = _pose_to_matrix(frame.position_w, frame.quat_wxyz)
            self._mapper.add_depth_frame(depth, t_w_c, sensor, mapper_id=0)

        self._mapper.update_esdf(mapper_id=0)
        if self._runtime.tensor_args.device.type == "cuda":
            torch.cuda.synchronize(device=self._runtime.tensor_args.device)

        fixed_grid = self._fixed_esdf_grid
        esdf = self._mapper.query_layer(self._query_type_esdf, fixed_grid.query_spheres_m, mapper_id=0)
        if esdf.ndim == 2 and esdf.shape[-1] == 1:
            esdf = esdf.squeeze(-1)
        expected = int(fixed_grid.dims_vox[0] * fixed_grid.dims_vox[1] * fixed_grid.dims_vox[2])
        if int(esdf.numel()) != expected:
            raise CuroboInterfaceError(f"ESDF query size mismatch: expected {expected}, got {int(esdf.numel())}.")

        esdf_grid = {
            "grid": esdf.reshape(int(fixed_grid.dims_vox[0]), int(fixed_grid.dims_vox[1]), int(fixed_grid.dims_vox[2])),
            "dims": [int(v) for v in fixed_grid.dims_vox.tolist()],
            "voxel_size": float(fixed_grid.voxel_size_m),
            "origin": [float(v) for v in fixed_grid.origin_m.tolist()],
        }
        world_collision = self._runtime.motion_gen.world_collision
        update_from_external = getattr(world_collision, "update_voxel_data_from_external_esdf", None)
        if update_from_external is None:
            raise CuroboInterfaceError(
                "CuRobo voxel collision checker must expose update_voxel_data_from_external_esdf()."
            )
        update_from_external(
            esdf_grid,
            name=str(self.config.voxel_layer_name),
            invert_sign=bool(self.config.import_invert_sign),
            add_half_voxel=bool(self.config.import_add_half_voxel),
        )
        return esdf_grid

    def _get_or_create_sensor(self, frame: Any, width: int, height: int) -> Any:
        if self._sensor_cls is None:
            raise CuroboInterfaceError("nvblox Sensor class was not initialized.")
        intrinsics = torch.as_tensor(to_numpy(frame.intrinsics), dtype=torch.float32, device="cpu")
        if intrinsics.shape != (3, 3):
            raise CuroboInputError(f"Frame '{frame.name}' intrinsics must be 3x3, got {tuple(intrinsics.shape)}.")
        key = (
            float(width),
            float(height),
            float(intrinsics[0, 0].item()),
            float(intrinsics[1, 1].item()),
            float(intrinsics[0, 2].item()),
            float(intrinsics[1, 2].item()),
        )
        camera_name = str(frame.name)
        cached = self._sensor_cache.get(camera_name)
        if cached is not None and cached[0] == key:
            return cached[1]
        sensor = self._sensor_cls.from_camera_matrix(intrinsics, int(width), int(height))
        self._sensor_cache[camera_name] = (key, sensor)
        return sensor

    @staticmethod
    def _make_fixed_esdf_grid(config: CuroboInterfaceConfig, device: torch.device) -> _FixedEsdfGrid:
        origin, dims, voxel_size = _fixed_grid_geometry(config)
        x = origin[0] + (torch.arange(int(dims[0]), device=device, dtype=torch.float32) + 0.5) * voxel_size
        y = origin[1] + (torch.arange(int(dims[1]), device=device, dtype=torch.float32) + 0.5) * voxel_size
        z = origin[2] + (torch.arange(int(dims[2]), device=device, dtype=torch.float32) + 0.5) * voxel_size
        x_grid, y_grid, z_grid = torch.meshgrid(x, y, z, indexing="ij")
        query_xyz = torch.stack([x_grid, y_grid, z_grid], dim=-1).reshape(-1, 3)
        query_spheres = torch.zeros((query_xyz.shape[0], 4), device=device, dtype=torch.float32)
        query_spheres[:, :3] = query_xyz
        return _FixedEsdfGrid(
            origin_m=origin.astype(np.float32, copy=False),
            dims_vox=dims.astype(np.int32, copy=False),
            voxel_size_m=float(voxel_size),
            query_spheres_m=query_spheres,
        )


def _import_curobo_api() -> tuple[Any, ...]:
    from curobo.geom.sdf.world import CollisionCheckerType
    from curobo.geom.types import VoxelGrid, WorldConfig
    from curobo.types.base import TensorDeviceType
    from curobo.types.math import Pose
    from curobo.types.state import JointState
    from curobo.util.logger import setup_curobo_logger
    from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig

    return (
        CollisionCheckerType,
        TensorDeviceType,
        Pose,
        JointState,
        MotionGen,
        MotionGenConfig,
        MotionGenPlanConfig,
        WorldConfig,
        VoxelGrid,
        setup_curobo_logger,
    )


def _fixed_grid_geometry(config: CuroboInterfaceConfig) -> tuple[np.ndarray, np.ndarray, float]:
    if config.voxel_bounds_min is None or config.voxel_bounds_max is None:
        raise CuroboInputError("voxel_bounds_min and voxel_bounds_max are required when voxel collision is enabled.")
    bounds_min = _as_float_vector(config.voxel_bounds_min, 3, "voxel_bounds_min")
    bounds_max = _as_float_vector(config.voxel_bounds_max, 3, "voxel_bounds_max")
    if np.any(bounds_max <= bounds_min):
        raise CuroboInputError(
            f"voxel_bounds_max must be greater than voxel_bounds_min, got {bounds_min} -> {bounds_max}."
        )
    voxel_size = float(config.voxel_size_m)
    if voxel_size <= 0.0:
        raise CuroboInputError(f"voxel_size_m must be positive, got {voxel_size}.")
    dims = np.maximum(1, np.ceil((bounds_max - bounds_min) / voxel_size).astype(np.int32))
    return bounds_min.astype(np.float32, copy=False), dims, voxel_size


def _single_tensor_vector(tensor_args: Any, value: Any, label: str) -> torch.Tensor:
    tensor = tensor_args.to_device(value)
    if tensor.ndim == 2 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.ndim != 1:
        raise CuroboInputError(f"{label} must be a 1-D vector or single-row batch, got shape {tuple(tensor.shape)}.")
    if not torch.isfinite(tensor).all():
        raise CuroboInputError(f"{label} contains non-finite values.")
    return tensor


def _as_float_vector(value: Any, size: int, label: str) -> np.ndarray:
    arr = to_numpy(value).astype(np.float32, copy=False).reshape(-1)
    if arr.shape != (int(size),):
        raise CuroboInputError(f"{label} must contain {int(size)} values, got shape {tuple(arr.shape)}.")
    if not np.all(np.isfinite(arr)):
        raise CuroboInputError(f"{label} contains non-finite values.")
    return arr


def _pose_to_matrix(position_w: Any, quat_wxyz: Any) -> torch.Tensor:
    rot = quat_wxyz_to_matrix(quat_wxyz)
    trans = _as_float_vector(position_w, 3, "camera position_w")
    t_w_c = torch.eye(4, dtype=torch.float32, device="cpu")
    t_w_c[:3, :3] = torch.as_tensor(rot, dtype=torch.float32, device="cpu")
    t_w_c[:3, 3] = torch.as_tensor(trans, dtype=torch.float32, device="cpu")
    return t_w_c.contiguous()


def _activate_cuda_device(device: torch.device) -> None:
    if device.type != "cuda":
        return
    device_idx = 0 if device.index is None else int(device.index)
    if torch.cuda.current_device() != device_idx:
        torch.cuda.set_device(device_idx)


def _resolve_child_path(value: object, *, base_dir: Path, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"Expected {label} to be a non-empty path string.")
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = (base_dir / candidate).resolve()
    if not candidate.exists():
        raise FileNotFoundError(f"{label} was not found: {candidate}")
    return candidate


def _load_yaml_mapping(path: Path, label: str) -> Mapping[str, object]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
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


def _parse_bool(value: object, label: str) -> bool:
    if isinstance(value, bool):
        return value
    raise TypeError(f"Expected {label} to be a boolean, got {value!r}.")


def _steps_from_seconds(seconds: float, *, label: str) -> int:
    if seconds <= 0.0:
        raise ValueError(f"Expected {label} to be positive, got {seconds!r}.")
    return max(1, int(float(seconds) / _DEFAULT_CONTROL_DT_S))


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
    "CuroboDebugWriter",
    "CuroboInputError",
    "CuroboInterface",
    "CuroboInterfaceConfig",
    "CuroboInterfaceError",
    "CuroboPlanningError",
    "CuroboTriggerRequest",
    "CuroboTriggerResult",
    "load_curobo_interface_config",
]
