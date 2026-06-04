"""Runtime state machine for camera-driven Spot manipulation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch
import yaml

from helpers.cam_utils import DepthFilterConfig, load_camera_rig_config, to_numpy

from .aograsp_client import AoGraspClient, GraspPose, load_aograsp_service_config
from .api import (
    DEFAULT_SPOT_MANIPULATION_CONFIG,
    ManipulationFeedback,
    ManipulationState,
    SpotManipulationConfig,
    SpotManipulationGraspConfig,
    SpotManipulationGripperConfig,
    SpotManipulationRobotConfig,
    SpotManipulationRobotHandles,
    resolve_spot_manipulation_robot_handles,
)
from .curobo_interfaces import CuroboInterface, CuroboTriggerRequest, load_curobo_interface_config


_CONFIG_DIR = Path(__file__).resolve().parent / "config"


def load_spot_manipulation_config(
    path_or_name: str | Path = DEFAULT_SPOT_MANIPULATION_CONFIG,
    *,
    device: str,
) -> SpotManipulationConfig:
    """Load the YAML profile used by the manipulation runtime."""

    profile_path = _resolve_profile_path(path_or_name)
    root = _load_yaml_mapping(profile_path, "manipulation profile")

    robot_path = _resolve_child_path(
        root.get("robot_config_path"),
        base_dir=profile_path.parent,
        label="robot_config_path",
    )
    camera_path = _resolve_child_path(
        root.get("camera_rig_config_path"),
        base_dir=profile_path.parent,
        label="camera_rig_config_path",
    )
    ao_path = _resolve_child_path(
        root.get("ao_grasp_config_path"),
        base_dir=profile_path.parent,
        label="ao_grasp_config_path",
    )
    curobo_path = _resolve_child_path(
        root.get("curobo_config_path"),
        base_dir=profile_path.parent,
        label="curobo_config_path",
    )

    robot = _load_robot_config(robot_path)
    camera_rig = load_camera_rig_config(camera_path)
    ao_grasp = load_aograsp_service_config(ao_path)
    curobo, gripper_values = load_curobo_interface_config(
        curobo_path,
        command_joint_names=robot.arm_joint_names,
        device=device,
    )

    return SpotManipulationConfig(
        robot=robot,
        camera_names=tuple(camera_rig.grasp_camera_names),
        depth_filter=DepthFilterConfig(max_depth_m=ao_grasp.max_depth_m),
        crops_by_camera=dict(camera_rig.crops_by_camera),
        camera_prim_paths=dict(camera_rig.prim_paths_by_camera),
        ao_grasp=ao_grasp,
        curobo=curobo,
        gripper=SpotManipulationGripperConfig(**dict(gripper_values)),
    )


class _CommandKind(str, Enum):
    GRASP = "grasp"
    RELEASE = "release"
    INTERACT_ARTICULATED = "interact_articulated"


@dataclass
class _PlanCursor:
    positions: torch.Tensor
    joint_names: tuple[str, ...]
    index: int = 0

    @property
    def done(self) -> bool:
        return self.index >= int(self.positions.shape[0])

    def next(self) -> torch.Tensor:
        if self.done:
            raise RuntimeError("No trajectory waypoint remains.")
        waypoint = self.positions[self.index]
        self.index += 1
        return waypoint


@dataclass(frozen=True)
class _ExecutionTargets:
    pregrasp_pos_w: torch.Tensor
    pregrasp_quat_wxyz: torch.Tensor
    grasp_pos_w: torch.Tensor
    grasp_quat_wxyz: torch.Tensor


class SpotManipulationRuntime:
    """Internal state machine for an IsaacLab Spot manipulation client."""

    def __init__(
        self,
        *,
        robot: Any,
        scene: Any,
        cameras: Mapping[str, object],
        config: SpotManipulationConfig,
        handles: SpotManipulationRobotHandles,
        ao_grasp: AoGraspClient,
        curobo: CuroboInterface,
        locomotion: Any | None = None,
        stage_getter: Callable[[], Any] | None = None,
    ) -> None:
        self.robot = robot
        self.scene = scene
        self.cameras = dict(cameras)
        self.config = config
        self.handles = handles
        self.ao_grasp = ao_grasp
        self.curobo = curobo
        self.locomotion = locomotion
        self._stage_getter = stage_getter

        if int(getattr(scene, "num_envs")) != 1:
            raise ValueError("SpotManipulationClient currently supports exactly one IsaacLab environment.")
        self._joint_names = tuple(str(name) for name in getattr(robot, "joint_names"))
        self._require_cameras(config.camera_names)

        self._state = ManipulationState.IDLE
        self._command_kind: _CommandKind | None = None
        self._command_id = 0
        self._message = ""
        self._last_error = ""
        self._debug = False

        self._settle_steps_remaining = 0
        self._gripper_steps_remaining = 0
        self._pull_steps_remaining = 0
        self._plan: _PlanCursor | None = None
        self._targets: _ExecutionTargets | None = None

    @classmethod
    def create(
        cls,
        *,
        robot: Any,
        scene: Any,
        cameras: Mapping[str, object],
        device: str,
        config_path: str | Path = DEFAULT_SPOT_MANIPULATION_CONFIG,
        locomotion: Any | None = None,
        stage: Any | None = None,
        stage_getter: Callable[[], Any] | None = None,
        robot_reference_prim_path: str | None = None,
        log: Callable[[str], None] | None = None,
    ) -> "SpotManipulationRuntime":
        """Create the manipulation API and all required runtime helpers."""

        config = load_spot_manipulation_config(config_path, device=device)
        if robot_reference_prim_path is not None:
            config = replace(
                config,
                world=replace(config.world, robot_reference_prim_path=str(robot_reference_prim_path)),
            )
        resolved_stage_getter = stage_getter or (lambda: stage)
        if not config.curobo.voxel_collision_enabled:
            if resolved_stage_getter() is None:
                raise ValueError("A USD stage or stage_getter is required when use_depth_collision is false.")
            if config.world.robot_reference_prim_path is None:
                raise ValueError("robot_reference_prim_path is required when use_depth_collision is false.")

        handles = resolve_spot_manipulation_robot_handles(robot, config.robot)
        ao_grasp = AoGraspClient(config.ao_grasp)
        curobo = CuroboInterface(config.curobo)
        return cls(
            robot=robot,
            scene=scene,
            cameras=cameras,
            config=config,
            handles=handles,
            ao_grasp=ao_grasp,
            curobo=curobo,
            locomotion=locomotion,
            stage_getter=resolved_stage_getter,
        )

    def grasp(self, *, debug: bool = False) -> int:
        """Start a camera-driven grasp command."""

        return self._start_command(_CommandKind.GRASP, debug=debug)

    def interact_articulated(self, *, debug: bool = False) -> int:
        """Start grasp, pull backward with locomotion, then release."""

        self._require_locomotion()
        return self._start_command(_CommandKind.INTERACT_ARTICULATED, debug=debug)

    def release(self) -> int:
        """Start a gripper release command."""

        return self._start_command(_CommandKind.RELEASE, debug=False)

    def reset(self) -> None:
        """Return the API to idle and clear queued execution state."""

        self._state = ManipulationState.IDLE
        self._command_kind = None
        self._message = ""
        self._last_error = ""
        self._debug = False
        self._settle_steps_remaining = 0
        self._gripper_steps_remaining = 0
        self._pull_steps_remaining = 0
        self._plan = None
        self._targets = None
        self._hold_arm_position()

    def stop(self) -> None:
        """Stop active manipulation and hold the arm at its current position."""

        self.reset()
        if self.locomotion is not None:
            self.locomotion.stop()

    def step(self) -> ManipulationFeedback:
        """Advance the manipulation state machine by one control tick."""

        try:
            self._step_impl()
        except Exception as exc:
            self._state = ManipulationState.FAILED
            self._last_error = str(exc)
            self._message = str(exc)
            raise
        return self.feedback

    @property
    def feedback(self) -> ManipulationFeedback:
        return ManipulationFeedback(command_id=self._command_id, state=self._state, message=self._message)

    @property
    def state(self) -> ManipulationState:
        return self._state

    @property
    def is_active(self) -> bool:
        return self._state not in {ManipulationState.IDLE, ManipulationState.DONE, ManipulationState.FAILED}

    @property
    def last_error(self) -> str:
        return self._last_error

    @property
    def current_command_id(self) -> int:
        return self._command_id

    def _start_command(self, kind: _CommandKind, *, debug: bool) -> int:
        if self.is_active:
            raise RuntimeError(f"Cannot start {kind.value}; command {self._command_id} is still {self._state.value}.")

        self._command_id += 1
        self._command_kind = kind
        self._debug = bool(debug)
        self._last_error = ""
        self._plan = None
        self._targets = None
        self._pull_steps_remaining = 0

        if kind == _CommandKind.RELEASE:
            self._state = ManipulationState.RELEASING
            self._message = "Opening gripper."
            self._gripper_steps_remaining = self.config.gripper.open_timeout_steps
        else:
            self._state = ManipulationState.OPENING_GRIPPER
            self._message = "Opening gripper before camera capture."
            self._gripper_steps_remaining = self.config.gripper.open_timeout_steps
        return self._command_id

    def _step_impl(self) -> None:
        if self._state == ManipulationState.IDLE or self._state.is_terminal:
            self._hold_arm_position()
            return

        if self._state == ManipulationState.OPENING_GRIPPER:
            self._step_opening_gripper()
        elif self._state == ManipulationState.SETTLING_CAMERA:
            self._step_settling_camera()
        elif self._state == ManipulationState.PERCEIVING:
            self._run_perception()
        elif self._state == ManipulationState.PLANNING_PREGRASP:
            self._plan_pregrasp()
        elif self._state == ManipulationState.EXECUTING_PREGRASP:
            self._step_plan(ManipulationState.PLANNING_GRASP, "Pregrasp reached.")
        elif self._state == ManipulationState.PLANNING_GRASP:
            self._plan_grasp()
        elif self._state == ManipulationState.EXECUTING_GRASP:
            self._step_plan(ManipulationState.CLOSING_GRIPPER, "Grasp pose reached; closing gripper.")
        elif self._state == ManipulationState.CLOSING_GRIPPER:
            self._step_closing_gripper()
        elif self._state == ManipulationState.PULLING_BACK:
            self._step_pull_back()
        elif self._state == ManipulationState.RELEASING:
            self._step_release()
        else:
            raise RuntimeError(f"Unhandled manipulation state: {self._state.value}.")

    def _step_opening_gripper(self) -> None:
        self._set_gripper_target(self.config.gripper.open_rad)
        self._hold_arm_position()
        if self._is_gripper_at(self.config.gripper.open_rad):
            self._state = ManipulationState.SETTLING_CAMERA
            self._settle_steps_remaining = self.config.gripper.camera_settle_steps
            self._message = "Gripper open; settling cameras."
            return
        self._gripper_steps_remaining -= 1
        if self._gripper_steps_remaining < 0:
            raise TimeoutError("Gripper did not reach the open target before camera capture.")

    def _step_settling_camera(self) -> None:
        self._hold_arm_position()
        if self._settle_steps_remaining > 0:
            self._settle_steps_remaining -= 1
            return
        self._state = ManipulationState.PERCEIVING
        self._message = "Running AO-Grasp."

    def _run_perception(self) -> None:
        self._require_cameras(self.config.camera_names)
        pose = self.ao_grasp.trigger(
            self.cameras,
            self.config.camera_names,
            crops_by_camera=self.config.crops_by_camera,
            camera_prim_paths=self.config.camera_prim_paths,
            depth_filter=self.config.depth_filter,
            debug=self._debug,
        )
        self._targets = self._build_execution_targets(pose)
        self._state = ManipulationState.PLANNING_PREGRASP
        self._message = f"AO-Grasp proposal accepted with score={pose.score:.4f}."

    def _plan_pregrasp(self) -> None:
        targets = self._require_targets()
        self._plan = self._plan_to_world_pose(targets.pregrasp_pos_w, targets.pregrasp_quat_wxyz)
        self._state = ManipulationState.EXECUTING_PREGRASP
        self._message = "Executing pregrasp trajectory."

    def _plan_grasp(self) -> None:
        targets = self._require_targets()
        self._plan = self._plan_to_world_pose(targets.grasp_pos_w, targets.grasp_quat_wxyz)
        self._state = ManipulationState.EXECUTING_GRASP
        self._message = "Executing grasp trajectory."

    def _require_targets(self) -> _ExecutionTargets:
        if self._targets is None:
            raise RuntimeError("Cannot plan before AO-Grasp targets are available.")
        return self._targets

    def _step_plan(self, next_state: ManipulationState, next_message: str) -> None:
        if self._plan is None:
            raise RuntimeError(f"No CuRobo plan is active for state {self._state.value}.")
        if self._plan.done:
            self._plan = None
            self._state = next_state
            if next_state == ManipulationState.CLOSING_GRIPPER:
                self._gripper_steps_remaining = self.config.gripper.close_steps
            self._message = next_message
            return
        self._apply_arm_waypoint(self._plan.next(), self._plan.joint_names)

    def _step_closing_gripper(self) -> None:
        self._hold_arm_position()
        self._set_gripper_target(self.config.gripper.closed_rad)
        self._gripper_steps_remaining -= 1
        if self._gripper_steps_remaining >= 0:
            return
        if self._command_kind == _CommandKind.INTERACT_ARTICULATED:
            self._state = ManipulationState.PULLING_BACK
            self._pull_steps_remaining = self.config.pull.steps
            self._message = "Pulling backward with locomotion."
        else:
            self._state = ManipulationState.DONE
            self._message = "Grasp command completed."

    def _step_pull_back(self) -> None:
        self._require_locomotion()
        self._hold_arm_position()
        if self._pull_steps_remaining > 0:
            self.locomotion.command_velocity(-abs(self.config.pull.backward_velocity_mps), 0.0, 0.0)
            self.locomotion.step()
            self._pull_steps_remaining -= 1
            return
        self.locomotion.stop()
        self.locomotion.step()
        self._state = ManipulationState.RELEASING
        self._gripper_steps_remaining = self.config.gripper.open_timeout_steps
        self._message = "Pull complete; releasing gripper."

    def _step_release(self) -> None:
        self._hold_arm_position()
        self._set_gripper_target(self.config.gripper.open_rad)
        if self._is_gripper_at(self.config.gripper.open_rad):
            self._state = ManipulationState.DONE
            self._message = "Release command completed."
            return
        self._gripper_steps_remaining -= 1
        if self._gripper_steps_remaining < 0:
            raise TimeoutError("Gripper did not reach the open target during release.")

    def _plan_to_world_pose(self, pos_w: torch.Tensor, quat_wxyz: torch.Tensor) -> _PlanCursor:
        self._require_finite_robot_state()
        root_pose = self.robot.data.root_pose_w[0:1]
        pos_b, quat_b = _subtract_frame_transforms(
            root_pose[:, 0:3],
            root_pose[:, 3:7],
            _ensure_batched(pos_w).to(root_pose.device, dtype=root_pose.dtype),
            _ensure_batched(quat_wxyz).to(root_pose.device, dtype=root_pose.dtype),
        )
        stage = None
        stage_reference_prim_path = None
        if not self.config.curobo.voxel_collision_enabled:
            if self._stage_getter is None:
                raise RuntimeError("stage_getter is required when use_depth_collision is false.")
            stage = self._stage_getter()
            stage_reference_prim_path = self.config.world.robot_reference_prim_path
        result = self.curobo.trigger(
            CuroboTriggerRequest(
                camera_map=self.cameras,
                camera_names=self.config.camera_names,
                joint_positions=self.robot.data.joint_pos[0],
                joint_velocities=self.robot.data.joint_vel[0],
                joint_names=self._joint_names,
                target_position_w=pos_b[0],
                target_quat_wxyz=quat_b[0],
                crops_by_camera=self.config.crops_by_camera,
                depth_filter=self.config.depth_filter,
                include_rgb=False,
                include_pointcloud=False,
                stage=stage,
                stage_only_paths=self.config.world.only_paths,
                stage_ignore_substring=self.config.world.ignore_substring,
                stage_reference_prim_path=stage_reference_prim_path,
                debug=self._debug,
            )
        )
        return _plan_cursor_from_curobo_plan(result.plan, device=self.robot.data.joint_pos.device)

    def _build_execution_targets(self, grasp: GraspPose) -> _ExecutionTargets:
        grasp_pos = _tensor_vector(grasp.position_world, 3, self.robot.data.joint_pos.device)
        grasp_quat = _tensor_vector(grasp.quaternion_world, 4, self.robot.data.joint_pos.device)
        pregrasp_offset = _pregrasp_offset(self.config.grasp, device=grasp_pos.device)

        pregrasp_pos, pregrasp_quat = _combine_frame_transforms(
            grasp_pos.unsqueeze(0),
            grasp_quat.unsqueeze(0),
            pregrasp_offset.unsqueeze(0),
            None,
        )
        grasp_to_ee_pos = _tensor_vector(self.config.grasp.grasp_to_ee_position, 3, grasp_pos.device)
        grasp_to_ee_quat = _tensor_vector(self.config.grasp.grasp_to_ee_quat_wxyz, 4, grasp_pos.device)
        ee_grasp_pos, ee_grasp_quat = _combine_frame_transforms(
            grasp_pos.unsqueeze(0),
            grasp_quat.unsqueeze(0),
            grasp_to_ee_pos.unsqueeze(0),
            grasp_to_ee_quat.unsqueeze(0),
        )
        ee_pregrasp_pos, ee_pregrasp_quat = _combine_frame_transforms(
            pregrasp_pos,
            pregrasp_quat,
            grasp_to_ee_pos.unsqueeze(0),
            grasp_to_ee_quat.unsqueeze(0),
        )
        return _ExecutionTargets(
            pregrasp_pos_w=ee_pregrasp_pos[0],
            pregrasp_quat_wxyz=ee_pregrasp_quat[0],
            grasp_pos_w=ee_grasp_pos[0],
            grasp_quat_wxyz=ee_grasp_quat[0],
        )

    def _apply_arm_waypoint(self, waypoint: torch.Tensor, joint_names: Sequence[str]) -> None:
        name_to_idx = {name: idx for idx, name in enumerate(joint_names)}
        missing = [name for name in self.config.robot.arm_joint_names if name not in name_to_idx]
        if missing:
            raise RuntimeError(f"CuRobo waypoint is missing arm joints: {missing}.")
        ordered = waypoint[[name_to_idx[name] for name in self.config.robot.arm_joint_names]]
        self.robot.set_joint_position_target(ordered.unsqueeze(0), joint_ids=self.handles.arm_joint_ids)

    def _hold_arm_position(self) -> None:
        current = self.robot.data.joint_pos[:, list(self.handles.arm_joint_ids)]
        self.robot.set_joint_position_target(current, joint_ids=self.handles.arm_joint_ids)

    def _set_gripper_target(self, target: float) -> None:
        if not self.handles.gripper_joint_ids:
            raise RuntimeError("At least one gripper joint is required for pure-claw manipulation.")
        target_tensor = torch.full(
            (1, len(self.handles.gripper_joint_ids)),
            float(target),
            dtype=self.robot.data.joint_pos.dtype,
            device=self.robot.data.joint_pos.device,
        )
        self.robot.set_joint_position_target(target_tensor, joint_ids=self.handles.gripper_joint_ids)

    def _is_gripper_at(self, target: float) -> bool:
        current = self.robot.data.joint_pos[:, list(self.handles.gripper_joint_ids)]
        err = torch.abs(current - float(target))
        return bool(torch.all(err <= self.config.gripper.tolerance_rad).item())

    def _require_cameras(self, camera_names: Sequence[str]) -> None:
        missing = [name for name in camera_names if name not in self.cameras]
        if missing:
            raise RuntimeError(f"Missing configured manipulation cameras: {missing}.")

    def _require_locomotion(self) -> None:
        if self.locomotion is None:
            raise RuntimeError("interact_articulated requires a locomotion client.")
        for method_name in ("command_velocity", "step", "stop"):
            if not callable(getattr(self.locomotion, method_name, None)):
                raise TypeError(f"locomotion must provide a callable {method_name}().")

    def _require_finite_robot_state(self) -> None:
        checks = {
            "arm joint positions": self.robot.data.joint_pos[:, list(self.handles.arm_joint_ids)],
            "gripper joint positions": self.robot.data.joint_pos[:, list(self.handles.gripper_joint_ids)],
            "root pose": self.robot.data.root_pose_w,
            "end-effector body pose": self.robot.data.body_pose_w[:, self.handles.ee_body_id, :],
        }
        for label, value in checks.items():
            if not torch.isfinite(value).all():
                raise RuntimeError(f"Robot {label} contains non-finite values.")


def _plan_cursor_from_curobo_plan(plan: Any, *, device: torch.device) -> _PlanCursor:
    positions = torch.as_tensor(getattr(plan, "position"), dtype=torch.float32, device=device)
    if positions.ndim == 3 and positions.shape[0] == 1:
        positions = positions[0]
    if positions.ndim == 1:
        positions = positions.unsqueeze(0)
    if positions.ndim != 2:
        raise RuntimeError(f"CuRobo plan positions must be 2-D, got shape {tuple(positions.shape)}.")
    joint_names = tuple(str(name) for name in getattr(plan, "joint_names"))
    if not joint_names:
        raise RuntimeError("CuRobo plan did not include joint_names.")
    if len(joint_names) != int(positions.shape[1]):
        raise RuntimeError(
            "CuRobo plan joint_names length "
            f"{len(joint_names)} does not match positions width {int(positions.shape[1])}."
        )
    return _PlanCursor(positions=positions, joint_names=joint_names)


def _pregrasp_offset(config: SpotManipulationGraspConfig, *, device: torch.device) -> torch.Tensor:
    axis_to_index = {"x": 0, "y": 1, "z": 2}
    axis = config.pregrasp_axis.lower()
    if axis not in axis_to_index:
        raise ValueError(f"pregrasp_axis must be one of {tuple(axis_to_index)}, got {config.pregrasp_axis!r}.")
    offset = torch.zeros(3, dtype=torch.float32, device=device)
    offset[axis_to_index[axis]] = float(config.pregrasp_sign) * float(config.pregrasp_offset_m)
    return offset


def _tensor_vector(value: Any, size: int, device: torch.device) -> torch.Tensor:
    tensor = torch.as_tensor(to_numpy(value), dtype=torch.float32, device=device).reshape(-1)
    if tensor.shape != (int(size),):
        raise ValueError(f"Expected vector of length {int(size)}, got shape {tuple(tensor.shape)}.")
    if not torch.isfinite(tensor).all():
        raise ValueError("Vector contains non-finite values.")
    return tensor


def _ensure_batched(value: torch.Tensor) -> torch.Tensor:
    return value.unsqueeze(0) if value.ndim == 1 else value


def _combine_frame_transforms(*args, **kwargs):
    from isaaclab.utils.math import combine_frame_transforms

    return combine_frame_transforms(*args, **kwargs)


def _subtract_frame_transforms(*args, **kwargs):
    from isaaclab.utils.math import subtract_frame_transforms

    return subtract_frame_transforms(*args, **kwargs)


def _load_robot_config(path: Path) -> SpotManipulationRobotConfig:
    root = _load_yaml_mapping(path, "Spot robot manipulation config")
    return SpotManipulationRobotConfig(
        arm_joint_names=_parse_string_tuple(root.get("arm_joint_names"), "arm_joint_names"),
        gripper_joint_names=_parse_string_tuple(root.get("gripper_joint_names"), "gripper_joint_names"),
        ee_body_name=_parse_required_str(root.get("ee_body_name"), "ee_body_name"),
    )


def _resolve_profile_path(path_or_name: str | Path) -> Path:
    raw = Path(path_or_name).expanduser()
    if raw.is_absolute() or raw.exists() or raw.suffix:
        return _resolve_existing_path(raw, base_dir=Path.cwd(), label="manipulation config")
    return _resolve_existing_path(_CONFIG_DIR / f"{raw}.yaml", base_dir=Path.cwd(), label="manipulation config")


def _resolve_child_path(value: object, *, base_dir: Path, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"Expected {label} to be a non-empty path string.")
    return _resolve_existing_path(value, base_dir=base_dir, label=label)


def _resolve_existing_path(path: str | Path, *, base_dir: Path, label: str) -> Path:
    candidate = Path(path).expanduser()
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


def _parse_required_str(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"Expected {label} to be a non-empty string.")
    return value.strip()


def _parse_string_tuple(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise TypeError(f"Expected {label} to be a non-empty string sequence.")
    parsed = tuple(_parse_required_str(item, label) for item in value)
    if len(set(parsed)) != len(parsed):
        raise ValueError(f"Expected {label} to contain unique names, got {parsed}.")
    return parsed


__all__ = ["SpotManipulationRuntime", "load_spot_manipulation_config"]
