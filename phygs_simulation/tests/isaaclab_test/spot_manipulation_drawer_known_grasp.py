#!/usr/bin/env python3
"""Standalone IsaacLab test for pulling a drawer from a known Spot grasp pose."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import math
from pathlib import Path
import sys
import time
import traceback
from typing import Any

import yaml


PHYGS_SIMULATION_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = PHYGS_SIMULATION_ROOT / "scripts"
MANIPULATION_MODULE_ROOT = SCRIPTS_ROOT / "skills" / "manipulation"
if not SCRIPTS_ROOT.is_dir():
    raise FileNotFoundError(f"Expected phygs_simulation scripts directory at {SCRIPTS_ROOT}")
if not MANIPULATION_MODULE_ROOT.is_dir():
    raise FileNotFoundError(f"Expected manipulation module directory at {MANIPULATION_MODULE_ROOT}")
for path in (SCRIPTS_ROOT, MANIPULATION_MODULE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from isaaclab.app import AppLauncher


DEFAULT_APP_CONFIG_PATH = Path(__file__).with_name("spot_manipulation_drawer.yaml")
MANIPULATION_PROFILE_PATH = SCRIPTS_ROOT / "skills" / "manipulation" / "config" / "spot_arm_default.yaml"
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
REFERENCE_DRAWER_POSITION_W = (1.05, 0.0, 0.0)
REFERENCE_HANDLE_CENTER_W = (0.67788, 0.0, 0.4344)
HANDLE_OFFSET_FROM_DRAWER_W = tuple(
    handle - drawer for handle, drawer in zip(REFERENCE_HANDLE_CENTER_W, REFERENCE_DRAWER_POSITION_W)
)
PLANNING_EE_BODY_NAME = "arm_link_wr1"
WR1_TO_GRASP_FRAME_OFFSET = (0.218, 0.0, -0.01)
GRASP_QUAT_WXYZ = (1.0, 0.0, 0.0, 0.0)
GRASP_EE_OFFSET_X_M = 0.0
PREGRASP_OFFSET_X_M = -0.08
APPROACH_SEGMENT_LENGTH_M = 0.08
PULL_OFFSET_X_M = -0.33
PULL_SEGMENT_LENGTH_M = 0.05
MIN_DRAWER_OPEN_DELTA_M = 0.33
EE_POSITION_TOLERANCE_M = 0.02
EE_QUAT_TOLERANCE_DEG = 5.0


def _normalize_quat(value: Sequence[float]) -> tuple[float, float, float, float]:
    if len(value) != 4:
        raise ValueError(f"Expected quaternion length 4, got {value!r}.")
    norm = math.sqrt(sum(float(item) * float(item) for item in value))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError(f"Quaternion must have a positive finite norm, got {value!r}.")
    quat = tuple(float(item) / norm for item in value)
    return quat if quat[0] >= 0.0 else tuple(-item for item in quat)


parser = argparse.ArgumentParser(description="Run one Spot known-grasp drawer pull test.")
parser.add_argument(
    "--config",
    type=str,
    default=str(DEFAULT_APP_CONFIG_PATH),
    help="Path to the drawer pull test YAML config.",
)
parser.add_argument(
    "--grasp-quat-wxyz",
    type=float,
    nargs=4,
    default=GRASP_QUAT_WXYZ,
    metavar=("W", "X", "Y", "Z"),
    help="World-frame EE quaternion to use for pregrasp, grasp, and pull.",
)
parser.add_argument(
    "--grasp-ee-offset-x",
    type=float,
    default=GRASP_EE_OFFSET_X_M,
    help="X offset from the handle center to the grasp-frame target.",
)
parser.add_argument(
    "--pregrasp-offset-x",
    type=float,
    default=PREGRASP_OFFSET_X_M,
    help="Additional X offset from the EE grasp target to the pregrasp target.",
)
parser.add_argument(
    "--pull-offset-x",
    type=float,
    default=PULL_OFFSET_X_M,
    help="Additional X offset from the EE grasp target to the pull target.",
)
parser.add_argument(
    "--pose-sweep",
    action="store_true",
    help="Try canonical 90-degree EE orientations at the pregrasp pose, print the first feasible quaternion, and exit.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import torch

import isaaclab.sim as isaaclab_sim
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz, subtract_frame_transforms

from curobo_interfaces import CuroboInterface, CuroboPlanningError, CuroboTriggerRequest, load_curobo_interface_config
from helpers.cam_utils import CameraRigConfig, DepthFilterConfig, load_camera_rig_config


@dataclass(frozen=True, slots=True)
class RobotSpawnConfig:
    position: tuple[float, float, float]
    rotation_wxyz: tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class DrawerSpawnConfig:
    usd_path: str
    position: tuple[float, float, float]
    rotation_wxyz: tuple[float, float, float, float]
    joint_name: str
    prim_name: str


@dataclass(frozen=True, slots=True)
class RequestConfig:
    debug: bool


@dataclass(frozen=True, slots=True)
class SceneConfig:
    env_spacing: float
    physics_dt: float
    render_interval: int
    dome_light_intensity: float
    dome_light_color: tuple[float, float, float]
    viewer_eye: tuple[float, float, float]
    viewer_target: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class TimeoutConfig:
    manipulation_sec: float
    pre_manipulation_delay_sec: float
    sensor_warmup_steps: int


@dataclass(frozen=True, slots=True)
class KnownGraspDrawerConfig:
    robot: RobotSpawnConfig
    drawer: DrawerSpawnConfig
    request: RequestConfig
    scene: SceneConfig
    timeouts: TimeoutConfig


@dataclass(frozen=True, slots=True)
class ProfileFiles:
    robot_config_path: Path
    camera_rig_config_path: Path
    curobo_config_path: Path


@dataclass(frozen=True, slots=True)
class RobotConfig:
    usd_path: str
    arm_joint_names: tuple[str, ...]
    gripper_joint_names: tuple[str, ...]
    ee_body_name: str
    disable_gravity: bool
    fix_root_link: bool
    retract_joint_pos: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class GripperConfig:
    open_rad: float
    closed_rad: float
    tolerance_rad: float
    open_timeout_steps: int
    close_steps: int
    camera_settle_steps: int


@dataclass(frozen=True, slots=True)
class RobotHandles:
    arm_joint_ids: tuple[int, ...]
    gripper_joint_ids: tuple[int, ...]
    ee_body_id: int


@dataclass(frozen=True, slots=True)
class CameraSensorSpec:
    name: str
    scene_key: str
    prim_path: str
    data_types: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ArmPlan:
    label: str
    positions: torch.Tensor
    joint_names: tuple[str, ...]


def load_known_grasp_drawer_config(path: str | Path) -> KnownGraspDrawerConfig:
    config_path = _existing_path(path, base_dir=Path.cwd(), label="drawer pull config")
    root = _read_mapping(config_path, "drawer pull config")
    base_dir = config_path.parent

    robot = _required_mapping(root, "robot", "root")
    drawer = _required_mapping(root, "drawer", "root")
    request = _required_mapping(root, "request", "root")
    scene = _required_mapping(root, "scene", "root")
    timeouts = _required_mapping(root, "timeouts", "root")

    return KnownGraspDrawerConfig(
        robot=RobotSpawnConfig(
            position=_required_float_tuple(robot, "position", 3, "robot"),
            rotation_wxyz=_required_float_tuple(robot, "rotation_wxyz", 4, "robot"),
        ),
        drawer=DrawerSpawnConfig(
            usd_path=str(_required_existing_path(drawer, "usd_path", base_dir, "drawer")),
            position=_required_float_tuple(drawer, "position", 3, "drawer"),
            rotation_wxyz=_required_float_tuple(drawer, "rotation_wxyz", 4, "drawer"),
            joint_name=_required_str(drawer, "joint_name", "drawer"),
            prim_name=_required_str(drawer, "prim_name", "drawer"),
        ),
        request=RequestConfig(debug=_required_bool(request, "debug", "request")),
        scene=SceneConfig(
            env_spacing=_required_float(scene, "env_spacing", "scene"),
            physics_dt=_required_float(scene, "physics_dt", "scene"),
            render_interval=_required_int(scene, "render_interval", "scene"),
            dome_light_intensity=_required_float(scene, "dome_light_intensity", "scene"),
            dome_light_color=_required_float_tuple(scene, "dome_light_color", 3, "scene"),
            viewer_eye=_required_float_tuple(scene, "viewer_eye", 3, "scene"),
            viewer_target=_required_float_tuple(scene, "viewer_target", 3, "scene"),
        ),
        timeouts=TimeoutConfig(
            manipulation_sec=_required_float(timeouts, "manipulation_sec", "timeouts"),
            pre_manipulation_delay_sec=_required_float(timeouts, "pre_manipulation_delay_sec", "timeouts"),
            sensor_warmup_steps=_required_int(timeouts, "sensor_warmup_steps", "timeouts"),
        ),
    )


def load_profile_files(path: str | Path) -> ProfileFiles:
    profile_path = _existing_path(path, base_dir=Path.cwd(), label="manipulation profile")
    root = _read_mapping(profile_path, "manipulation profile")
    return ProfileFiles(
        robot_config_path=_required_existing_path(root, "robot_config_path", profile_path.parent, "manipulation profile"),
        camera_rig_config_path=_required_existing_path(
            root,
            "camera_rig_config_path",
            profile_path.parent,
            "manipulation profile",
        ),
        curobo_config_path=_required_existing_path(root, "curobo_config_path", profile_path.parent, "manipulation profile"),
    )


def load_robot_config(path: str | Path) -> RobotConfig:
    robot_path = _existing_path(path, base_dir=Path.cwd(), label="Spot robot config")
    root = _read_mapping(robot_path, "Spot robot config")
    retract_joint_pos = _required_mapping(root, "retract_joint_pos", "Spot robot config")
    return RobotConfig(
        usd_path=str(_required_existing_path(root, "usd_path", robot_path.parent, "Spot robot config")),
        arm_joint_names=_required_str_tuple(root, "arm_joint_names", "Spot robot config"),
        gripper_joint_names=_required_str_tuple(root, "gripper_joint_names", "Spot robot config"),
        ee_body_name=_required_str(root, "ee_body_name", "Spot robot config"),
        disable_gravity=_required_bool(root, "disable_gravity", "Spot robot config"),
        fix_root_link=_required_bool(root, "fix_root_link", "Spot robot config"),
        retract_joint_pos={
            _non_empty_str(name, "Spot robot config.retract_joint_pos key"): _as_float(
                value,
                f"Spot robot config.retract_joint_pos.{name}",
            )
            for name, value in retract_joint_pos.items()
        },
    )


def load_curobo_depth_filter(path: str | Path) -> DepthFilterConfig:
    curobo_path = _existing_path(path, base_dir=Path.cwd(), label="CuRobo runtime config")
    root = _read_mapping(curobo_path, "CuRobo runtime config")
    return DepthFilterConfig(
        min_depth_m=_required_float(root, "depth_min_m", "CuRobo runtime config"),
        max_depth_m=_required_float(root, "depth_max_m", "CuRobo runtime config"),
    )


def resolve_robot_handles(robot: Any, config: RobotConfig) -> RobotHandles:
    arm_ids, arm_names = robot.find_joints(list(config.arm_joint_names), preserve_order=True)
    gripper_ids, gripper_names = robot.find_joints(list(config.gripper_joint_names), preserve_order=True)
    ee_body_ids, ee_body_names = robot.find_bodies([config.ee_body_name], preserve_order=True)

    if tuple(arm_names) != config.arm_joint_names:
        raise RuntimeError(f"Resolved arm joints {tuple(arm_names)} do not match {config.arm_joint_names}.")
    if tuple(gripper_names) != config.gripper_joint_names:
        raise RuntimeError(f"Resolved gripper joints {tuple(gripper_names)} do not match {config.gripper_joint_names}.")
    if len(ee_body_ids) != 1 or tuple(ee_body_names) != (config.ee_body_name,):
        raise RuntimeError(f"Expected one EE body named {config.ee_body_name!r}, got {tuple(ee_body_names)}.")

    return RobotHandles(
        arm_joint_ids=tuple(int(idx) for idx in arm_ids),
        gripper_joint_ids=tuple(int(idx) for idx in gripper_ids),
        ee_body_id=int(ee_body_ids[0]),
    )


def camera_sensor_specs(camera_rig: CameraRigConfig) -> tuple[CameraSensorSpec, ...]:
    return tuple(
        CameraSensorSpec(
            name=camera.name,
            scene_key=f"camera_sensor_{index}",
            prim_path=camera.prim_path,
            data_types=camera.data_types,
        )
        for index, camera in enumerate(camera_rig.cameras)
    )


def add_camera_sensors(namespace: dict[str, object], specs: Sequence[CameraSensorSpec]) -> None:
    for spec in specs:
        namespace[spec.scene_key] = CameraCfg(
            prim_path=spec.prim_path,
            height=CAMERA_HEIGHT,
            width=CAMERA_WIDTH,
            data_types=list(spec.data_types),
            colorize_semantic_segmentation=False,
            colorize_instance_segmentation=False,
            colorize_instance_id_segmentation=False,
            update_latest_camera_pose=True,
            spawn=None,
        )


def resolve_scene_cameras(scene: InteractiveScene, specs: Sequence[CameraSensorSpec]) -> dict[str, object]:
    return {spec.name: scene[spec.scene_key] for spec in specs}


def _curobo_ee_link(robot_cfg: str | Mapping[str, object]) -> str:
    if isinstance(robot_cfg, str):
        root = _read_mapping(Path(robot_cfg).expanduser().resolve(), "CuRobo robot config")
        if "robot_cfg" not in root:
            raise KeyError(f"CuRobo robot config at {robot_cfg} is missing robot_cfg.")
        parsed = root["robot_cfg"]
    elif isinstance(robot_cfg, Mapping):
        parsed = robot_cfg["robot_cfg"] if "robot_cfg" in robot_cfg else robot_cfg
    else:
        raise TypeError(f"Expected CuRobo robot_cfg path or mapping, got {type(robot_cfg).__name__}.")

    if not isinstance(parsed, Mapping):
        raise TypeError("CuRobo robot_cfg must be a mapping.")
    kinematics = parsed.get("kinematics")
    if not isinstance(kinematics, Mapping):
        raise TypeError("CuRobo robot_cfg.kinematics must be a mapping.")
    return _required_str(kinematics, "ee_link", "CuRobo robot config.robot_cfg.kinematics")


class KnownGraspDrawerPullTest:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.config = load_known_grasp_drawer_config(args.config)
        self.profile_files = load_profile_files(MANIPULATION_PROFILE_PATH)
        self.robot_config = load_robot_config(self.profile_files.robot_config_path)
        self.camera_rig = load_camera_rig_config(self.profile_files.camera_rig_config_path)
        self.camera_specs = camera_sensor_specs(self.camera_rig)
        self.camera_names = tuple(self.camera_rig.depth_collision_camera_names)
        self.curobo_config, gripper_values = load_curobo_interface_config(
            self.profile_files.curobo_config_path,
            command_joint_names=self.robot_config.arm_joint_names,
            device=args.device,
        )
        planning_ee_body = _curobo_ee_link(self.curobo_config.robot_cfg)
        if planning_ee_body != PLANNING_EE_BODY_NAME:
            raise RuntimeError(
                f"This test expects CuRobo ee_link={PLANNING_EE_BODY_NAME!r}, got {planning_ee_body!r}."
            )
        self.robot_config = replace(self.robot_config, ee_body_name=planning_ee_body)
        self.gripper = GripperConfig(**dict(gripper_values))
        self.curobo_config = replace(self.curobo_config, voxel_collision_enabled=False)
        self.depth_filter = load_curobo_depth_filter(self.profile_files.curobo_config_path)
        self.grasp_quat_wxyz = _normalize_quat(args.grasp_quat_wxyz)
        self.grasp_ee_offset_x = float(args.grasp_ee_offset_x)
        self.pregrasp_offset_x = float(args.pregrasp_offset_x)
        self.pull_offset_x = float(args.pull_offset_x)
        self.handle_center_w = _add_positions(self.config.drawer.position, HANDLE_OFFSET_FROM_DRAWER_W)
        self.stage_reference_prim_path: str | None = None
        self.transit_stage_ignore_substring = ("Robot", "target", "curobo")
        self.contact_stage_ignore_substring = (
            "Robot",
            "target",
            "curobo",
            self.config.drawer.prim_name,
            "Drawer",
            "drawer",
        )
        self._validate_camera_contract()

    def run(self) -> None:
        self._log_startup()

        sim = self._build_sim()
        scene = self._build_scene()
        sim.reset()

        robot = scene["robot"]
        drawer = scene["drawer"]
        handles = resolve_robot_handles(robot, self.robot_config)
        cameras = resolve_scene_cameras(scene, self.camera_specs)
        curobo = CuroboInterface(self.curobo_config)
        self.stage_reference_prim_path = self._env_prim_path(scene, "Robot")

        self._sync_targets_to_spawn(robot)
        self._open_gripper(sim, scene, robot, handles)
        self._step_scene(sim, scene, self.config.timeouts.sensor_warmup_steps, "sensor warm-up")
        self._step_scene_for_seconds(
            sim,
            scene,
            self.config.timeouts.pre_manipulation_delay_sec,
            "pre-grasp settle",
        )

        start_opening = self._drawer_joint_position(drawer)
        self._log(f"[Drawer] start {self.config.drawer.joint_name}={start_opening:.4f} m")
        self._log_robot_root_pose(robot)
        if self.args.pose_sweep:
            self._run_pose_sweep(sim, scene, robot, cameras, curobo, handles)
            return

        pregrasp_pos_w, grasp_pos_w = self._target_positions_w()
        pregrasp_plan = self._plan_to_pose(
            robot,
            cameras,
            curobo,
            target_pos_w=pregrasp_pos_w,
            target_quat_wxyz=self.grasp_quat_wxyz,
            label="pregrasp",
            stage_ignore_substring=self.transit_stage_ignore_substring,
        )
        approach_plans: list[ArmPlan] = []
        approach_targets_w = _line_waypoints_w(pregrasp_pos_w, grasp_pos_w, APPROACH_SEGMENT_LENGTH_M)
        approach_joint_pos = self._joint_positions_after_plan(robot, handles, pregrasp_plan)
        for index, approach_target_w in enumerate(approach_targets_w, start=1):
            label = "grasp" if index == len(approach_targets_w) else f"approach_{index:02d}"
            approach_plan = self._plan_to_pose(
                robot,
                cameras,
                curobo,
                target_pos_w=approach_target_w,
                target_quat_wxyz=self.grasp_quat_wxyz,
                label=label,
                joint_positions=approach_joint_pos,
                stage_ignore_substring=self.contact_stage_ignore_substring,
            )
            approach_plans.append(approach_plan)
            approach_joint_pos = self._joint_positions_after_plan(robot, handles, approach_plan)
        if not approach_plans:
            approach_plans.append(self._hold_plan_from_joint_positions(robot, handles, approach_joint_pos, "grasp"))
            approach_targets_w = (grasp_pos_w,)
            self._log("[CuRobo] grasp: using reached pregrasp pose as the grasp pose")

        pregrasp_deadline = time.monotonic() + self.config.timeouts.manipulation_sec
        self._execute_plan(
            sim,
            scene,
            robot,
            handles,
            pregrasp_plan,
            self.gripper.open_rad,
            pregrasp_deadline,
        )
        self._settle_to_pose(
            sim,
            scene,
            robot,
            handles,
            label="pregrasp",
            target_pos_w=pregrasp_pos_w,
            target_quat_wxyz=self.grasp_quat_wxyz,
            arm_target=pregrasp_plan.positions[-1],
            arm_target_joint_names=pregrasp_plan.joint_names,
            gripper_target=self.gripper.open_rad,
            deadline=pregrasp_deadline,
        )
        self._log_ee_pose_error(robot, handles, "pregrasp", pregrasp_pos_w, self.grasp_quat_wxyz)
        approach_deadline = time.monotonic() + self.config.timeouts.manipulation_sec
        for approach_plan, approach_target_w in zip(approach_plans, approach_targets_w):
            self._execute_plan(
                sim,
                scene,
                robot,
                handles,
                approach_plan,
                self.gripper.open_rad,
                approach_deadline,
            )
            self._settle_to_pose(
                sim,
                scene,
                robot,
                handles,
                label=approach_plan.label,
                target_pos_w=approach_target_w,
                target_quat_wxyz=self.grasp_quat_wxyz,
                arm_target=approach_plan.positions[-1],
                arm_target_joint_names=approach_plan.joint_names,
                gripper_target=self.gripper.open_rad,
                deadline=approach_deadline,
            )
            self._log_ee_pose_error(robot, handles, approach_plan.label, approach_target_w, self.grasp_quat_wxyz)
        self._close_gripper(sim, scene, robot, handles)
        self._log_ee_pose_error(robot, handles, "after_close", grasp_pos_w, self.grasp_quat_wxyz)
        self._log_grasp_frame_position_error(robot, handles, "after_close", self._grasp_frame_target_w())

        closed_pos_w, closed_quat_wxyz = self._ee_pose_w(robot, handles)
        closed_grasp_frame_pos_w = self._grasp_frame_position_from_ee_pose_w(closed_pos_w, closed_quat_wxyz)
        pull_grasp_start_w = closed_grasp_frame_pos_w
        pull_quat_wxyz = self.grasp_quat_wxyz
        pull_grasp_end_w = _offset_x(pull_grasp_start_w, self.pull_offset_x)
        pull_grasp_targets_w = _pull_waypoints_w(pull_grasp_start_w, pull_grasp_end_w)
        pull_targets_w = tuple(
            self._ee_position_from_grasp_frame_w(grasp_frame_pos_w, pull_quat_wxyz)
            for grasp_frame_pos_w in pull_grasp_targets_w
        )
        self._log(
            "[CuRobo] pull in grasp frame with nominal grasp orientation: "
            f"closed_ee_w={_format_tuple(closed_pos_w)} "
            f"closed_grasp_frame_w={_format_tuple(closed_grasp_frame_pos_w)} "
            f"grasp_start_w={_format_tuple(pull_grasp_start_w)} "
            f"grasp_end_w={_format_tuple(pull_grasp_end_w)} "
            f"closed_quat_w={_format_tuple(closed_quat_wxyz)} "
            f"pull_quat_w={_format_tuple(pull_quat_wxyz)}"
        )
        pull_plans: list[ArmPlan] = []
        pull_joint_pos = robot.data.joint_pos[0].clone()
        for index, pull_waypoint_w in enumerate(pull_targets_w, start=1):
            pull_plan = self._plan_to_pose(
                robot,
                cameras,
                curobo,
                target_pos_w=pull_waypoint_w,
                target_quat_wxyz=pull_quat_wxyz,
                label=f"pull_{index:02d}",
                joint_positions=pull_joint_pos,
                stage_ignore_substring=self.contact_stage_ignore_substring,
            )
            pull_plans.append(pull_plan)
            pull_joint_pos = self._joint_positions_after_plan(robot, handles, pull_plan)

        pull_deadline = time.monotonic() + self.config.timeouts.manipulation_sec
        for pull_plan, pull_target_w, pull_grasp_target_w in zip(pull_plans, pull_targets_w, pull_grasp_targets_w):
            self._execute_plan(
                sim,
                scene,
                robot,
                handles,
                pull_plan,
                self.gripper.closed_rad,
                pull_deadline,
            )
            self._settle_to_pose(
                sim,
                scene,
                robot,
                handles,
                label=pull_plan.label,
                target_pos_w=pull_target_w,
                target_quat_wxyz=pull_quat_wxyz,
                arm_target=pull_plan.positions[-1],
                arm_target_joint_names=pull_plan.joint_names,
                gripper_target=self.gripper.closed_rad,
                deadline=pull_deadline,
            )
            self._log_ee_pose_error(robot, handles, pull_plan.label, pull_target_w, pull_quat_wxyz)
            self._log_grasp_frame_position_error(robot, handles, pull_plan.label, pull_grasp_target_w)

        final_opening = self._drawer_joint_position(drawer)
        opening_delta = final_opening - start_opening
        self._log(f"[Drawer] final {self.config.drawer.joint_name}={final_opening:.4f} m delta={opening_delta:.4f} m")
        if opening_delta < MIN_DRAWER_OPEN_DELTA_M:
            raise RuntimeError(
                f"Drawer opened only {opening_delta:.4f} m; expected at least {MIN_DRAWER_OPEN_DELTA_M:.4f} m."
            )

        self._log("[RESULT] Spot known-grasp drawer pull test completed successfully.")

    def _build_sim(self) -> isaaclab_sim.SimulationContext:
        cfg = self.config.scene
        sim_cfg = isaaclab_sim.SimulationCfg(
            dt=cfg.physics_dt,
            render_interval=cfg.render_interval,
            device=self.args.device,
        )
        sim_cfg.use_fabric = True
        sim = isaaclab_sim.SimulationContext(sim_cfg)
        sim.set_camera_view(eye=cfg.viewer_eye, target=cfg.viewer_target)
        return sim

    def _build_scene(self) -> InteractiveScene:
        config = self.config
        robot_config = self.robot_config
        camera_specs = self.camera_specs

        @configclass
        class DrawerSceneCfg(InteractiveSceneCfg):
            ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=isaaclab_sim.GroundPlaneCfg())
            light = AssetBaseCfg(
                prim_path="/World/Light",
                spawn=isaaclab_sim.DomeLightCfg(
                    intensity=config.scene.dome_light_intensity,
                    color=config.scene.dome_light_color,
                ),
            )
            robot = ArticulationCfg(
                prim_path="{ENV_REGEX_NS}/Robot",
                spawn=isaaclab_sim.UsdFileCfg(
                    usd_path=robot_config.usd_path,
                    activate_contact_sensors=True,
                    rigid_props=isaaclab_sim.RigidBodyPropertiesCfg(disable_gravity=robot_config.disable_gravity),
                    articulation_props=isaaclab_sim.ArticulationRootPropertiesCfg(
                        fix_root_link=robot_config.fix_root_link
                    ),
                ),
                init_state=ArticulationCfg.InitialStateCfg(
                    pos=config.robot.position,
                    rot=config.robot.rotation_wxyz,
                    joint_pos={".*_knee": -1.5, **dict(robot_config.retract_joint_pos)},
                ),
                actuators={
                    "arm": ImplicitActuatorCfg(
                        joint_names_expr=list(robot_config.arm_joint_names),
                        effort_limit_sim=200.0,
                        stiffness=400.0,
                        damping=80.0,
                    ),
                    "gripper": ImplicitActuatorCfg(
                        joint_names_expr=list(robot_config.gripper_joint_names),
                        effort_limit_sim=200.0,
                        stiffness=2e3,
                        damping=1e2,
                    ),
                },
            )
            drawer = ArticulationCfg(
                prim_path=f"{{ENV_REGEX_NS}}/{config.drawer.prim_name}",
                spawn=isaaclab_sim.UsdFileCfg(
                    usd_path=config.drawer.usd_path,
                    activate_contact_sensors=True,
                    rigid_props=isaaclab_sim.RigidBodyPropertiesCfg(disable_gravity=False),
                    articulation_props=isaaclab_sim.ArticulationRootPropertiesCfg(fix_root_link=None),
                ),
                init_state=ArticulationCfg.InitialStateCfg(
                    pos=config.drawer.position,
                    rot=config.drawer.rotation_wxyz,
                    joint_pos={config.drawer.joint_name: 0.0},
                ),
                actuators={
                    "drawer": ImplicitActuatorCfg(
                        joint_names_expr=[config.drawer.joint_name],
                        effort_limit_sim=20.0,
                        stiffness=0.0,
                        damping=10.0,
                    )
                },
            )

            add_camera_sensors(locals(), camera_specs)

        return InteractiveScene(
            DrawerSceneCfg(
                num_envs=1,
                env_spacing=config.scene.env_spacing,
                replicate_physics=False,
            )
        )

    def _plan_to_pose(
        self,
        robot: Any,
        cameras: Mapping[str, object],
        curobo: CuroboInterface,
        *,
        target_pos_w: tuple[float, float, float],
        target_quat_wxyz: tuple[float, float, float, float],
        label: str,
        joint_positions: torch.Tensor | None = None,
        stage_ignore_substring: Sequence[str] | None = None,
    ) -> ArmPlan:
        handles = resolve_robot_handles(robot, self.robot_config)
        target_pos_b, target_quat_b = self._target_in_body_frame(robot, handles, target_pos_w, target_quat_wxyz)
        self._log(
            f"[CuRobo] planning {label}: "
            f"target_w={_format_tuple(target_pos_w)} "
            f"target_body={_format_tensor(target_pos_b)} "
            f"quat_w={_format_tuple(target_quat_wxyz)} "
            f"quat_body={_format_tensor(target_quat_b)}"
        )
        start_joint_pos = robot.data.joint_pos[0] if joint_positions is None else joint_positions
        result = curobo.trigger(
            CuroboTriggerRequest(
                camera_map=cameras,
                camera_names=self.camera_names,
                joint_positions=start_joint_pos,
                joint_velocities=torch.zeros_like(start_joint_pos),
                joint_names=tuple(str(name) for name in robot.joint_names),
                target_position_w=target_pos_b,
                target_quat_wxyz=target_quat_b,
                crops_by_camera=self.camera_rig.crops_by_camera,
                depth_filter=self.depth_filter,
                include_rgb=False,
                include_pointcloud=False,
                stage=get_current_stage(),
                stage_only_paths=("/World",),
                stage_ignore_substring=stage_ignore_substring or self.contact_stage_ignore_substring,
                stage_reference_prim_path=self._require_stage_reference_prim_path(),
                request_id=f"known_grasp_{label}",
                debug=self.config.request.debug,
            )
        )
        positions, joint_names = self._plan_positions(result.plan, robot)
        self._log(f"[CuRobo] {label}: status={result.status} waypoints={int(positions.shape[0])}")
        return ArmPlan(label=label, positions=positions, joint_names=joint_names)

    def _run_pose_sweep(
        self,
        sim: isaaclab_sim.SimulationContext,
        scene: InteractiveScene,
        robot: Any,
        cameras: Mapping[str, object],
        curobo: CuroboInterface,
        handles: RobotHandles,
    ) -> None:
        del sim, scene
        target_pos_w = self._target_positions_w()[0]
        self._log(f"[Sweep] pregrasp target_w={_format_tuple(target_pos_w)}")
        for index, candidate in enumerate(canonical_quat_candidates(), start=1):
            label = f"pose_sweep_{index:02d}"
            target_pos_b, target_quat_b = self._target_in_body_frame(robot, handles, target_pos_w, candidate)
            try:
                result = curobo.trigger(
                    CuroboTriggerRequest(
                        camera_map=cameras,
                        camera_names=self.camera_names,
                        joint_positions=robot.data.joint_pos[0],
                        joint_velocities=robot.data.joint_vel[0],
                        joint_names=tuple(str(name) for name in robot.joint_names),
                        target_position_w=target_pos_b,
                        target_quat_wxyz=target_quat_b,
                        crops_by_camera=self.camera_rig.crops_by_camera,
                        depth_filter=self.depth_filter,
                        include_rgb=False,
                        include_pointcloud=False,
                        stage=get_current_stage(),
                        stage_only_paths=("/World",),
                        stage_ignore_substring=self.contact_stage_ignore_substring,
                        stage_reference_prim_path=self._require_stage_reference_prim_path(),
                        request_id=label,
                        debug=False,
                    )
                )
            except CuroboPlanningError as exc:
                self._log(f"[Sweep] {index:02d} failed quat={_format_tuple(candidate)} error={exc}")
                continue
            positions, _joint_names = self._plan_positions(result.plan, robot)
            self._log(
                "[Sweep] success "
                f"quat={_format_tuple(candidate)} "
                f"body_quat={_format_tensor(target_quat_b)} "
                f"waypoints={int(positions.shape[0])}"
            )
            self._log(
                "[Sweep] rerun with "
                "--grasp-quat-wxyz "
                + " ".join(f"{value:.8f}" for value in candidate)
            )
            return
        raise RuntimeError("No canonical EE orientation reached the known pregrasp target.")

    def _target_positions_w(
        self,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        grasp_frame_pos_w = self._grasp_frame_target_w()
        pregrasp_frame_pos_w = _offset_x(grasp_frame_pos_w, self.pregrasp_offset_x)
        grasp_pos_w = self._ee_position_from_grasp_frame_w(grasp_frame_pos_w, self.grasp_quat_wxyz)
        pregrasp_pos_w = self._ee_position_from_grasp_frame_w(pregrasp_frame_pos_w, self.grasp_quat_wxyz)
        return pregrasp_pos_w, grasp_pos_w

    def _grasp_frame_target_w(self) -> tuple[float, float, float]:
        return _offset_x(self.handle_center_w, self.grasp_ee_offset_x)

    def _ee_position_from_grasp_frame_w(
        self,
        grasp_frame_pos_w: Sequence[float],
        grasp_quat_wxyz: Sequence[float],
    ) -> tuple[float, float, float]:
        return _subtract_positions(
            grasp_frame_pos_w,
            _rotate_vector_by_quat(WR1_TO_GRASP_FRAME_OFFSET, grasp_quat_wxyz),
        )

    def _grasp_frame_position_from_ee_pose_w(
        self,
        ee_pos_w: Sequence[float],
        ee_quat_wxyz: Sequence[float],
    ) -> tuple[float, float, float]:
        return _add_positions(
            ee_pos_w,
            _rotate_vector_by_quat(WR1_TO_GRASP_FRAME_OFFSET, ee_quat_wxyz),
        )

    def _target_in_body_frame(
        self,
        robot: Any,
        handles: RobotHandles,
        pos_w: Sequence[float],
        quat_wxyz: Sequence[float],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._require_finite_robot_state(robot, handles)
        root_pose = robot.data.root_pose_w[0:1]
        pos_w_tensor = torch.tensor([tuple(pos_w)], dtype=root_pose.dtype, device=root_pose.device)
        quat_w_tensor = torch.tensor([tuple(quat_wxyz)], dtype=root_pose.dtype, device=root_pose.device)
        pos_b, quat_b = subtract_frame_transforms(root_pose[:, 0:3], root_pose[:, 3:7], pos_w_tensor, quat_w_tensor)
        return pos_b[0], quat_b[0]

    def _log_robot_root_pose(self, robot: Any) -> None:
        root_pose = robot.data.root_pose_w[0]
        self._log(
            "[DebugPose] root "
            f"pos_w={_format_tensor(root_pose[0:3])} "
            f"quat_wxyz={_format_tensor(root_pose[3:7])}"
        )

    def _ee_pose_w(
        self,
        robot: Any,
        handles: RobotHandles,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
        self._require_finite_robot_state(robot, handles)
        ee_pose_w = robot.data.body_pose_w[0, handles.ee_body_id, :].detach().cpu().tolist()
        return (
            tuple(float(value) for value in ee_pose_w[0:3]),
            _normalize_quat(tuple(float(value) for value in ee_pose_w[3:7])),
        )

    def _log_ee_pose_error(
        self,
        robot: Any,
        handles: RobotHandles,
        label: str,
        target_pos_w: Sequence[float],
        target_quat_wxyz: Sequence[float],
    ) -> None:
        self._require_finite_robot_state(robot, handles)
        ee_pose_w = robot.data.body_pose_w[0, handles.ee_body_id, :]
        actual_pos_w = ee_pose_w[0:3]
        actual_quat_w = ee_pose_w[3:7]
        pos_error_m, quat_error_deg = self._ee_pose_error(robot, handles, target_pos_w, target_quat_wxyz)
        target_pos_b, target_quat_b = self._target_in_body_frame(robot, handles, target_pos_w, target_quat_wxyz)
        actual_pos_b, actual_quat_b = self._target_in_body_frame(
            robot,
            handles,
            actual_pos_w.detach().cpu().tolist(),
            actual_quat_w.detach().cpu().tolist(),
        )
        self._log(
            f"[DebugPose] {label} "
            f"target_pos_w={_format_tuple(target_pos_w)} "
            f"actual_pos_w={_format_tensor(actual_pos_w)} "
            f"pos_err_m={pos_error_m:.5f} "
            f"target_quat_w={_format_tuple(target_quat_wxyz)} "
            f"actual_quat_w={_format_tensor(actual_quat_w)} "
            f"quat_err_deg={quat_error_deg:.3f}"
        )
        self._log(
            f"[DebugPose] {label} body "
            f"target_pos_b={_format_tensor(target_pos_b)} "
            f"actual_pos_b={_format_tensor(actual_pos_b)} "
            f"target_quat_b={_format_tensor(target_quat_b)} "
            f"actual_quat_b={_format_tensor(actual_quat_b)}"
        )

    def _log_grasp_frame_position_error(
        self,
        robot: Any,
        handles: RobotHandles,
        label: str,
        target_pos_w: Sequence[float],
    ) -> None:
        ee_pos_w, ee_quat_wxyz = self._ee_pose_w(robot, handles)
        actual_pos_w = self._grasp_frame_position_from_ee_pose_w(ee_pos_w, ee_quat_wxyz)
        error_m = _distance(actual_pos_w, target_pos_w)
        self._log(
            f"[DebugPose] {label} grasp_frame "
            f"target_pos_w={_format_tuple(target_pos_w)} "
            f"actual_pos_w={_format_tuple(actual_pos_w)} "
            f"pos_err_m={error_m:.5f}"
        )

    def _ee_pose_error(
        self,
        robot: Any,
        handles: RobotHandles,
        target_pos_w: Sequence[float],
        target_quat_wxyz: Sequence[float],
    ) -> tuple[float, float]:
        self._require_finite_robot_state(robot, handles)
        ee_pose_w = robot.data.body_pose_w[0, handles.ee_body_id, :]
        actual_pos_w = ee_pose_w[0:3]
        actual_quat_w = ee_pose_w[3:7]
        target_pos_w_tensor = torch.tensor(
            tuple(target_pos_w),
            dtype=actual_pos_w.dtype,
            device=actual_pos_w.device,
        )
        target_quat_w_tensor = torch.tensor(
            tuple(target_quat_wxyz),
            dtype=actual_quat_w.dtype,
            device=actual_quat_w.device,
        )
        pos_error_m = float(torch.linalg.norm(actual_pos_w - target_pos_w_tensor).detach().cpu().item())
        quat_error_deg = _quat_angle_error_deg(actual_quat_w, target_quat_w_tensor)
        return pos_error_m, quat_error_deg

    def _plan_positions(self, plan: Any, robot: Any) -> tuple[torch.Tensor, tuple[str, ...]]:
        positions = torch.as_tensor(
            getattr(plan, "position"),
            dtype=robot.data.joint_pos.dtype,
            device=robot.data.joint_pos.device,
        )
        if positions.ndim == 3 and positions.shape[0] == 1:
            positions = positions[0]
        if positions.ndim == 1:
            positions = positions.unsqueeze(0)
        if positions.ndim != 2:
            raise RuntimeError(f"CuRobo plan positions must be 2-D, got shape {tuple(positions.shape)}.")
        joint_names = tuple(str(name) for name in getattr(plan, "joint_names"))
        if len(joint_names) != int(positions.shape[1]):
            raise RuntimeError(
                f"CuRobo plan joint_names length {len(joint_names)} does not match positions width "
                f"{int(positions.shape[1])}."
            )
        return positions, joint_names

    def _joint_positions_after_plan(self, robot: Any, handles: RobotHandles, plan: ArmPlan) -> torch.Tensor:
        name_to_idx = {name: idx for idx, name in enumerate(plan.joint_names)}
        missing = [name for name in self.robot_config.arm_joint_names if name not in name_to_idx]
        if missing:
            raise RuntimeError(f"CuRobo {plan.label} plan is missing arm joints: {missing}.")
        joint_positions = robot.data.joint_pos[0].clone()
        final_arm = plan.positions[-1, [name_to_idx[name] for name in self.robot_config.arm_joint_names]]
        joint_positions[list(handles.arm_joint_ids)] = final_arm
        return joint_positions

    def _hold_plan_from_joint_positions(
        self,
        robot: Any,
        handles: RobotHandles,
        joint_positions: torch.Tensor,
        label: str,
    ) -> ArmPlan:
        arm_positions = joint_positions[list(handles.arm_joint_ids)].to(
            dtype=robot.data.joint_pos.dtype,
            device=robot.data.joint_pos.device,
        )
        return ArmPlan(
            label=label,
            positions=arm_positions.unsqueeze(0),
            joint_names=self.robot_config.arm_joint_names,
        )

    def _execute_plan(
        self,
        sim: isaaclab_sim.SimulationContext,
        scene: InteractiveScene,
        robot: Any,
        handles: RobotHandles,
        plan: ArmPlan,
        gripper_target: float,
        deadline: float,
    ) -> None:
        name_to_idx = {name: idx for idx, name in enumerate(plan.joint_names)}
        missing = [name for name in self.robot_config.arm_joint_names if name not in name_to_idx]
        if missing:
            raise RuntimeError(f"CuRobo {plan.label} plan is missing arm joints: {missing}.")

        indices = [name_to_idx[name] for name in self.robot_config.arm_joint_names]
        hold_steps = self._waypoint_hold_steps(sim)
        for waypoint_index, waypoint in enumerate(plan.positions):
            ordered = waypoint[indices]
            for hold_index in range(hold_steps):
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"{plan.label} execution exceeded {self.config.timeouts.manipulation_sec:.2f}s.")
                self._command_arm_and_gripper(robot, handles, ordered, gripper_target)
                self._step_once(
                    sim,
                    scene,
                    f"{plan.label} waypoint {waypoint_index + 1}/{int(plan.positions.shape[0])} "
                    f"hold {hold_index + 1}/{hold_steps}",
                )

    def _settle_to_pose(
        self,
        sim: isaaclab_sim.SimulationContext,
        scene: InteractiveScene,
        robot: Any,
        handles: RobotHandles,
        *,
        label: str,
        target_pos_w: Sequence[float],
        target_quat_wxyz: Sequence[float],
        arm_target: torch.Tensor,
        arm_target_joint_names: Sequence[str],
        gripper_target: float,
        deadline: float,
    ) -> None:
        name_to_idx = {name: idx for idx, name in enumerate(arm_target_joint_names)}
        missing = [name for name in self.robot_config.arm_joint_names if name not in name_to_idx]
        if missing:
            raise RuntimeError(f"CuRobo {label} final waypoint is missing arm joints: {missing}.")

        indices = [name_to_idx[name] for name in self.robot_config.arm_joint_names]
        ordered = arm_target[indices]
        while time.monotonic() < deadline:
            pos_error_m, quat_error_deg = self._ee_pose_error(robot, handles, target_pos_w, target_quat_wxyz)
            if pos_error_m <= EE_POSITION_TOLERANCE_M and quat_error_deg <= EE_QUAT_TOLERANCE_DEG:
                self._log(
                    f"[Sim] {label}: settled pos_err={pos_error_m:.4f} m "
                    f"quat_err={quat_error_deg:.2f} deg"
                )
                return
            self._command_arm_and_gripper(robot, handles, ordered, gripper_target)
            self._step_once(sim, scene, f"{label} final-pose settle")

        pos_error_m, quat_error_deg = self._ee_pose_error(robot, handles, target_pos_w, target_quat_wxyz)
        raise TimeoutError(
            f"{label} did not settle to the planned EE pose: pos_err={pos_error_m:.4f} m "
            f"(tol {EE_POSITION_TOLERANCE_M:.4f}), quat_err={quat_error_deg:.2f} deg "
            f"(tol {EE_QUAT_TOLERANCE_DEG:.2f})."
        )

    def _waypoint_hold_steps(self, sim: isaaclab_sim.SimulationContext) -> int:
        physics_dt = sim.get_physics_dt()
        if physics_dt <= 0.0:
            raise RuntimeError(f"Simulation physics dt must be positive, got {physics_dt}.")
        return max(1, math.ceil(float(self.curobo_config.interpolation_dt) / physics_dt))

    def _command_arm_and_gripper(
        self,
        robot: Any,
        handles: RobotHandles,
        ordered_arm_target: torch.Tensor,
        gripper_target: float,
    ) -> None:
        robot.set_joint_position_target(ordered_arm_target.unsqueeze(0), joint_ids=handles.arm_joint_ids)
        self._set_gripper_target(robot, handles, gripper_target)

    def _open_gripper(
        self,
        sim: isaaclab_sim.SimulationContext,
        scene: InteractiveScene,
        robot: Any,
        handles: RobotHandles,
    ) -> None:
        self._log("[Gripper] opening")
        for step_idx in range(self.gripper.open_timeout_steps):
            self._hold_arm_position(robot, handles)
            self._set_gripper_target(robot, handles, self.gripper.open_rad)
            self._step_once(sim, scene, f"open gripper {step_idx + 1}/{self.gripper.open_timeout_steps}")
            if self._is_gripper_at(robot, handles, self.gripper.open_rad):
                self._log("[Gripper] open")
                return
        raise TimeoutError("Gripper did not reach the open target.")

    def _close_gripper(
        self,
        sim: isaaclab_sim.SimulationContext,
        scene: InteractiveScene,
        robot: Any,
        handles: RobotHandles,
    ) -> None:
        self._log(f"[Gripper] closing for {self.gripper.close_steps} steps")
        for step_idx in range(self.gripper.close_steps):
            self._hold_arm_position(robot, handles)
            self._set_gripper_target(robot, handles, self.gripper.closed_rad)
            self._step_once(sim, scene, f"close gripper {step_idx + 1}/{self.gripper.close_steps}")

    def _sync_targets_to_spawn(self, robot: Any) -> None:
        joint_names = (*self.robot_config.arm_joint_names, *self.robot_config.gripper_joint_names)
        joint_ids, resolved_names = robot.find_joints(list(joint_names), preserve_order=True)
        if tuple(resolved_names) != joint_names:
            raise RuntimeError(f"Expected joints {joint_names}, resolved {tuple(resolved_names)}.")
        spawn_joint_pos = robot.data.joint_pos[:, joint_ids].clone()
        robot.set_joint_position_target(spawn_joint_pos, joint_ids=joint_ids)
        self._log(f"[Sim] synced targets for joints={tuple(resolved_names)}")

    def _hold_arm_position(self, robot: Any, handles: RobotHandles) -> None:
        current = robot.data.joint_pos[:, list(handles.arm_joint_ids)]
        robot.set_joint_position_target(current, joint_ids=handles.arm_joint_ids)

    def _set_gripper_target(self, robot: Any, handles: RobotHandles, target: float) -> None:
        if not handles.gripper_joint_ids:
            raise RuntimeError("At least one gripper joint is required.")
        target_tensor = torch.full(
            (1, len(handles.gripper_joint_ids)),
            float(target),
            dtype=robot.data.joint_pos.dtype,
            device=robot.data.joint_pos.device,
        )
        robot.set_joint_position_target(target_tensor, joint_ids=handles.gripper_joint_ids)

    def _is_gripper_at(self, robot: Any, handles: RobotHandles, target: float) -> bool:
        current = robot.data.joint_pos[:, list(handles.gripper_joint_ids)]
        return bool(torch.all(torch.abs(current - float(target)) <= self.gripper.tolerance_rad).item())

    def _drawer_joint_position(self, drawer: Any) -> float:
        joint_ids, joint_names = drawer.find_joints([self.config.drawer.joint_name], preserve_order=True)
        if tuple(joint_names) != (self.config.drawer.joint_name,):
            raise RuntimeError(f"Expected drawer joint {self.config.drawer.joint_name!r}, got {tuple(joint_names)}.")
        return float(drawer.data.joint_pos[0, int(joint_ids[0])].detach().cpu().item())

    def _step_scene(
        self,
        sim: isaaclab_sim.SimulationContext,
        scene: InteractiveScene,
        steps: int,
        label: str,
    ) -> None:
        if steps < 0:
            raise ValueError(f"{label} steps must be non-negative, got {steps}.")
        if steps == 0:
            return
        self._log(f"[Sim] stepping {steps} frames for {label}")
        for step_idx in range(steps):
            self._step_once(sim, scene, f"{label} {step_idx + 1}/{steps}")

    def _step_scene_for_seconds(
        self,
        sim: isaaclab_sim.SimulationContext,
        scene: InteractiveScene,
        seconds: float,
        label: str,
    ) -> None:
        if seconds < 0.0:
            raise ValueError(f"{label} duration must be non-negative, got {seconds}.")
        if seconds == 0.0:
            return
        physics_dt = sim.get_physics_dt()
        if physics_dt <= 0.0:
            raise RuntimeError(f"Simulation physics dt must be positive, got {physics_dt}.")
        self._step_scene(sim, scene, math.ceil(seconds / physics_dt), f"{label} ({seconds:.2f}s)")

    def _step_once(self, sim: isaaclab_sim.SimulationContext, scene: InteractiveScene, label: str) -> None:
        if not simulation_app.is_running():
            raise RuntimeError(f"Simulation app stopped during {label}.")
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())

    def _validate_camera_contract(self) -> None:
        rig_camera_names = set(self.camera_rig.camera_names)
        missing = [name for name in self.camera_names if name not in rig_camera_names]
        if missing:
            raise RuntimeError(f"Depth-collision cameras missing from the rig: {missing}.")

    def _require_finite_robot_state(self, robot: Any, handles: RobotHandles) -> None:
        checks = {
            "joint positions": robot.data.joint_pos,
            "joint velocities": robot.data.joint_vel,
            "root pose": robot.data.root_pose_w,
            "end-effector pose": robot.data.body_pose_w[:, handles.ee_body_id, :],
        }
        for label, value in checks.items():
            if not torch.isfinite(value).all():
                raise RuntimeError(f"Robot {label} contains non-finite values.")

    def _require_stage_reference_prim_path(self) -> str:
        if self.stage_reference_prim_path is None:
            raise RuntimeError("Stage reference prim path has not been initialized.")
        return self.stage_reference_prim_path

    def _log_startup(self) -> None:
        robot_relative_handle = (
            self.handle_center_w[0] - self.config.robot.position[0],
            self.handle_center_w[1] - self.config.robot.position[1],
            self.handle_center_w[2] - self.config.robot.position[2],
        )
        self._log("[Config] Spot known-grasp drawer pull test")
        self._log(f"[Config] robot_usd={self.robot_config.usd_path}")
        self._log(f"[Config] drawer_usd={self.config.drawer.usd_path}")
        self._log(f"[Config] drawer_position={self.config.drawer.position}")
        self._log(f"[Config] cameras={self.camera_names}")
        self._log(f"[Config] planning_ee_body={self.robot_config.ee_body_name}")
        self._log(f"[Config] handle_center_w={self.handle_center_w}")
        self._log(f"[Config] handle_offset_from_drawer_w={HANDLE_OFFSET_FROM_DRAWER_W}")
        self._log(f"[Config] wr1_to_grasp_frame_offset={WR1_TO_GRASP_FRAME_OFFSET}")
        self._log(f"[Config] handle_minus_robot_spawn={robot_relative_handle}")
        self._log(f"[Config] grasp_quat_wxyz={_format_tuple(self.grasp_quat_wxyz)}")
        self._log(f"[Config] grasp_ee_offset_x={self.grasp_ee_offset_x:.4f} m")
        self._log(f"[Config] pregrasp_offset_x={self.pregrasp_offset_x:.4f} m")
        self._log(f"[Config] approach_segment_length={APPROACH_SEGMENT_LENGTH_M:.4f} m")
        self._log(f"[Config] pull_offset_x={self.pull_offset_x:.4f} m")
        self._log(f"[Config] pull_segment_length={PULL_SEGMENT_LENGTH_M:.4f} m")
        self._log(f"[Config] curobo_robot_cfg={self.curobo_config.robot_cfg}")
        self._log("[Config] curobo_collision=transit_uses_drawer_contact_ignores_drawer")
        self._log(f"[Config] transit_stage_ignore_substring={self.transit_stage_ignore_substring}")
        self._log(f"[Config] contact_stage_ignore_substring={self.contact_stage_ignore_substring}")

    @staticmethod
    def _env_prim_path(scene: InteractiveScene, prim_name: str) -> str:
        return f"{scene.env_prim_paths[0]}/{prim_name.lstrip('/')}"

    @staticmethod
    def _log(message: str) -> None:
        print(message)


def _offset_x(position: Sequence[float], offset: float) -> tuple[float, float, float]:
    if len(position) != 3:
        raise ValueError(f"Expected position length 3, got {position!r}.")
    return (float(position[0]) + float(offset), float(position[1]), float(position[2]))


def _add_positions(left: Sequence[float], right: Sequence[float]) -> tuple[float, float, float]:
    if len(left) != 3 or len(right) != 3:
        raise ValueError(f"Expected two position triples, got {left!r} and {right!r}.")
    return tuple(float(a) + float(b) for a, b in zip(left, right))


def _subtract_positions(left: Sequence[float], right: Sequence[float]) -> tuple[float, float, float]:
    if len(left) != 3 or len(right) != 3:
        raise ValueError(f"Expected two position triples, got {left!r} and {right!r}.")
    return tuple(float(a) - float(b) for a, b in zip(left, right))


def _rotate_vector_by_quat(vector: Sequence[float], quat_wxyz: Sequence[float]) -> tuple[float, float, float]:
    if len(vector) != 3:
        raise ValueError(f"Expected vector length 3, got {vector!r}.")
    w, x, y, z = _normalize_quat(quat_wxyz)
    vx, vy, vz = (float(value) for value in vector)
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (
        vx + w * tx + y * tz - z * ty,
        vy + w * ty + z * tx - x * tz,
        vz + w * tz + x * ty - y * tx,
    )


def _pull_waypoints_w(
    start_w: Sequence[float],
    end_w: Sequence[float],
) -> tuple[tuple[float, float, float], ...]:
    return _line_waypoints_w(start_w, end_w, PULL_SEGMENT_LENGTH_M)


def _line_waypoints_w(
    start_w: Sequence[float],
    end_w: Sequence[float],
    segment_length_m: float,
) -> tuple[tuple[float, float, float], ...]:
    if len(start_w) != 3 or len(end_w) != 3:
        raise ValueError(f"Expected two position triples, got {start_w!r} and {end_w!r}.")
    if segment_length_m <= 0.0:
        raise ValueError(f"Segment length must be positive, got {segment_length_m}.")
    delta = tuple(float(end_w[index]) - float(start_w[index]) for index in range(3))
    distance = math.sqrt(sum(value * value for value in delta))
    if distance <= 0.0:
        return ()
    step_count = max(1, math.ceil(distance / segment_length_m))
    return tuple(
        tuple(float(start_w[axis]) + delta[axis] * (step_index / step_count) for axis in range(3))
        for step_index in range(1, step_count + 1)
    )


def _distance(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != 3 or len(right) != 3:
        raise ValueError(f"Expected two position triples, got {left!r} and {right!r}.")
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(left, right)))


def _quat_angle_error_deg(actual_wxyz: torch.Tensor, target_wxyz: torch.Tensor) -> float:
    actual = actual_wxyz / torch.linalg.norm(actual_wxyz)
    target = target_wxyz / torch.linalg.norm(target_wxyz)
    dot = torch.clamp(torch.abs(torch.dot(actual, target)), 0.0, 1.0)
    return float((2.0 * torch.acos(dot) * 180.0 / math.pi).detach().cpu().item())


def canonical_quat_candidates() -> tuple[tuple[float, float, float, float], ...]:
    angles = (0.0, math.pi / 2.0, math.pi, -math.pi / 2.0)
    candidates: list[tuple[float, float, float, float]] = []
    seen: set[tuple[float, float, float, float]] = set()
    for roll in angles:
        for pitch in angles:
            for yaw in angles:
                quat = quat_from_euler_xyz(
                    torch.tensor([roll], dtype=torch.float32),
                    torch.tensor([pitch], dtype=torch.float32),
                    torch.tensor([yaw], dtype=torch.float32),
                )[0]
                parsed = _normalize_quat(tuple(float(value) for value in quat.tolist()))
                key = tuple(round(value, 6) for value in parsed)
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(parsed)
    return tuple(candidates)


def _format_tuple(value: Sequence[float]) -> str:
    return "(" + ", ".join(f"{float(item):.5f}" for item in value) + ")"


def _format_tensor(value: torch.Tensor) -> str:
    return _format_tuple(value.detach().cpu().tolist())


def _read_mapping(path: Path, label: str) -> Mapping[str, object]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise TypeError(f"Expected {label} at {path} to be a YAML mapping.")
    return data


def _existing_path(path: str | Path, *, base_dir: Path, label: str) -> Path:
    if not isinstance(path, (str, Path)):
        raise TypeError(f"Expected {label} to be a path string.")
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = (base_dir / resolved).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    return resolved


def _required_existing_path(root: Mapping[str, object], key: str, base_dir: Path, label: str) -> Path:
    return _existing_path(_required(root, key, label), base_dir=base_dir, label=f"{label}.{key}")


def _required_mapping(root: Mapping[str, object], key: str, label: str) -> Mapping[str, object]:
    value = _required(root, key, label)
    if not isinstance(value, Mapping):
        raise TypeError(f"Expected {label}.{key} to be a mapping, got {type(value).__name__}.")
    return value


def _required_str(root: Mapping[str, object], key: str, label: str) -> str:
    return _non_empty_str(_required(root, key, label), f"{label}.{key}")


def _required_str_tuple(root: Mapping[str, object], key: str, label: str) -> tuple[str, ...]:
    value = _required(root, key, label)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"Expected {label}.{key} to be a string sequence, got {value!r}.")
    parsed = tuple(_non_empty_str(item, f"{label}.{key}") for item in value)
    if not parsed:
        raise ValueError(f"Expected {label}.{key} to contain at least one item.")
    return parsed


def _required_bool(root: Mapping[str, object], key: str, label: str) -> bool:
    value = _required(root, key, label)
    if not isinstance(value, bool):
        raise TypeError(f"Expected {label}.{key} to be a boolean, got {value!r}.")
    return value


def _required_float(root: Mapping[str, object], key: str, label: str) -> float:
    return _as_float(_required(root, key, label), f"{label}.{key}")


def _required_int(root: Mapping[str, object], key: str, label: str) -> int:
    value = _required(root, key, label)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"Expected {label}.{key} to be an integer, got {value!r}.")
    return value


def _required_float_tuple(root: Mapping[str, object], key: str, size: int, label: str) -> tuple[float, ...]:
    value = _required(root, key, label)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != size:
        raise TypeError(f"Expected {label}.{key} to be a sequence of length {size}, got {value!r}.")
    return tuple(_as_float(item, f"{label}.{key}") for item in value)


def _required(root: Mapping[str, object], key: str, label: str) -> object:
    if key not in root:
        raise KeyError(f"Missing required key {label}.{key}.")
    return root[key]


def _non_empty_str(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"Expected {label} to be a non-empty string, got {value!r}.")
    return value.strip()


def _as_float(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"Expected {label} to be numeric, got {value!r}.")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"Expected {label} to be numeric, got {value!r}.") from exc


def main() -> None:
    KnownGraspDrawerPullTest(args_cli).run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("[ABORTED] Spot known-grasp drawer pull test interrupted by user.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"[FATAL] Spot known-grasp drawer pull test failed: {exc}", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)
    finally:
        simulation_app.close()
