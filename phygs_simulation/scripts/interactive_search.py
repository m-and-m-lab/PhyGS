#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Interactive Search (Standalone)

Loads a USD scene, spawns a robot, and runs a CuRobo-based manipulation skill.

Controls:
  - Press 'M' to focus the docked AO-Grasp Target panel for target entry and submission (GUI mode).
  - Press 'R' to release the current grasp; if the gripper is already open, reset the robot (GUI mode).
  - Press 'T' to plan a random CuRobo target (GUI mode).

.. code-block:: bash

  ./isaaclab.sh -p scripts/interactive-search/scripts/interactive_search.py \\
    --enable_cameras \\
    --scene_usd scripts/bedroom1/export_scene.blend/export_scene.usdc \\
    --robot_usd scripts/interactive-search/spot_model/spot_arm_w_cam.usd
"""

"""Launch Isaac Sim Simulator first."""

import argparse
from dataclasses import replace
from pathlib import Path
import sys

from isaaclab.app import AppLauncher


_DEFAULT_ROBOT_CAMERA_PATHS = (
    "{ENV_REGEX_NS}/Robot/body/Gemini2_front_left/Orbbec_Gemini2/camera_ir_left/camera_left/Stream_depth",
    "{ENV_REGEX_NS}/Robot/body/Gemini2_front_right/Orbbec_Gemini2/camera_ir_left/camera_left/Stream_depth",
    "{ENV_REGEX_NS}/Robot/arm_link_wr1/Gemini2_arm/Orbbec_Gemini2/camera_ir_left/camera_left/Stream_depth",
)
_DEFAULT_ROBOT_CAMERA_NAMES = ("frontleft", "frontright", "hand")
_DEFAULT_AOGRASP_CONFIG_PATH = Path(__file__).resolve().parent / "skills" / "manipulation" / "config" / "aograsp.yaml"

_NO_PANEL_RESULT = object()


def _resolve_usd_path(path_str: str) -> str:
    """Resolve local file paths; keep non-local (e.g., Nucleus) paths unchanged."""
    if not path_str:
        return path_str
    candidate = Path(path_str)
    if candidate.exists():
        return str(candidate.resolve())
    return path_str


def _sanitize_camera_path(path: str) -> str:
    path = path.strip().strip('"').strip("'")
    if path.startswith("/{ENV_REGEX_NS}"):
        path = path[1:]
    while "//" in path:
        path = path.replace("//", "/")
    return path


def _build_camera_sensor_specs() -> list[dict[str, object]]:
    sensor_specs_by_path: dict[str, dict[str, object]] = {}

    def register_camera(path: str, *, include_rgb: bool = False, include_segmentation: bool = False) -> None:
        normalized_path = _sanitize_camera_path(path)
        if not normalized_path:
            return
        spec = sensor_specs_by_path.get(normalized_path)
        if spec is None:
            spec = {
                "scene_key": f"camera_sensor_{len(sensor_specs_by_path)}",
                "prim_path": normalized_path,
                "data_types": {"distance_to_image_plane"},
            }
            sensor_specs_by_path[normalized_path] = spec
        if include_rgb:
            spec["data_types"].add("rgb")
        if include_segmentation:
            spec["data_types"].add("instance_id_segmentation_fast")

    if args_cli.curobo_use_depth_collision:
        for depth_path in args_cli.curobo_depth_camera_paths:
            register_camera(depth_path)

    if args_cli.grasp_backend == "ao-grasp":
        for grasp_path in args_cli.ao_grasp_camera_paths:
            register_camera(grasp_path, include_rgb=True, include_segmentation=False)

    return [
        {
            "scene_key": spec["scene_key"],
            "prim_path": spec["prim_path"],
            "data_types": tuple(sorted(spec["data_types"])),
        }
        for spec in sensor_specs_by_path.values()
    ]


parser = argparse.ArgumentParser(
    description="Load a USD scene, spawn a robot, and trigger a manipulation skill."
)

parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--env_spacing", type=float, default=20.0)

# Scene/robot inputs
parser.add_argument(
    "--scene_usd",
    type=str,
    default="scripts/bedroom1/export_scene.blend/export_scene.usdc",
    help="USD/USDC file to reference as the environment scene (local path or Nucleus path).",
)
parser.add_argument(
    "--robot_usd",
    type=str,
    default="scripts/interactive-search/spot_model/spot_arm_w_cam.usd",
    help="Robot USD file to reference (local path or Nucleus path).",
)
parser.add_argument(
    "--robot_prim_path",
    type=str,
    default="{ENV_REGEX_NS}/Robot",
    help="Prim path for the robot articulation. Use {ENV_REGEX_NS} for per-env expansion.",
)
parser.add_argument(
    "--robot_pos",
    type=float,
    nargs=3,
    # default=(4.41245, 4.75, 0.75),
    # default=(11.6, 3.6, 0.75),
    default=(3.1, 1.85, 0.75),
    metavar=("X", "Y", "Z"),
    help="Robot base position in world frame (meters).",
)
parser.add_argument(
    "--robot_rot",
    type=float,
    nargs=4,
    # default=(0.7071, 0.0, 0.0, 0.7071),
    default=(0.0, 0.0, 0.0, 1.0),
    metavar=("W", "X", "Y", "Z"),
    help="Robot base orientation quaternion in world frame (w, x, y, z).",
)

# Simulation toggles
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable Fabric (use USD I/O for state reads)."
)
parser.add_argument(
    "--fix_root",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Fix the robot root link to the world (recommended for arm-only manipulation).",
)
parser.add_argument(
    "--robot_disable_gravity",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Disable gravity on the robot rigid bodies (recommended if legs are not actively controlled).",
)

# Manipulation controller config (defaults match the Spot arm USD in this repo)
parser.add_argument(
    "--arm_joints",
    type=str,
    nargs="+",
    default=["arm_sh0", "arm_sh1", "arm_el0", "arm_el1", "arm_wr0", "arm_wr1"],
    help="Arm joint names (order matters) to be controlled by the CuRobo controller.",
)
parser.add_argument("--ee_body", type=str, default="arm_link_fngr", help="End-effector body name for planning.")
parser.add_argument(
    "--gripper_joints",
    type=str,
    nargs="+",
    default=["arm_f1x"],
    help="Gripper joint names to command (e.g., ['arm_f1x']).",
)
parser.add_argument("--gripper_open", type=float, default=-1.2, help="Gripper joint position for open.")
parser.add_argument("--gripper_closed", type=float, default=-0.4, help="Gripper joint position for closed.")
parser.add_argument(
    "--curobo_robot_cfg",
    type=str,
    default="scripts/interactive-search/spot_model/configuration/spot_arm_curobo.yaml",
    help="CuRobo robot configuration YAML for motion generation.",
)
parser.add_argument(
    "--curobo_update_world",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Update CuRobo collision world from the current USD stage when planning.",
)
parser.add_argument(
    "--curobo_use_depth_collision",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Build CuRobo VOXEL world from onboard depth cameras and plan around sensed obstacles.",
)
parser.add_argument(
    "--curobo_depth_camera_paths",
    type=str,
    nargs="+",
    default=list(_DEFAULT_ROBOT_CAMERA_PATHS),
    help=(
        "Existing depth camera prim path(s) on the robot used for CuRobo VOXEL updates. "
        "Use {ENV_REGEX_NS} for per-env expansion."
    ),
)
parser.add_argument(
    "--curobo_depth_voxel_size",
    type=float,
    default=0.025,
    help="Depth voxel size (m) for nvblox and cuRobo VOXEL import.",
)
parser.add_argument(
    "--curobo_depth_min_m",
    type=float,
    default=0.18,
    help="Minimum depth (m) kept from each camera frame.",
)
parser.add_argument(
    "--curobo_depth_max_m",
    type=float,
    default=2.2,
    help="Maximum depth (m) kept/integrated from each camera frame.",
)
parser.add_argument(
    "--curobo_truncation_distance_vox",
    type=float,
    default=2.0,
    help="TSDF truncation distance in voxel units.",
)
parser.add_argument(
    "--curobo_raycast_subsampling",
    type=int,
    default=1,
    help="Raycast subsampling factor used by nvblox view calculator.",
)
parser.add_argument(
    "--curobo_accumulate_frames",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Accumulate depth frames across triggers instead of clearing each trigger.",
)
parser.add_argument(
    "--curobo_import_invert_sign",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Invert ESDF sign when importing external grid into cuRobo voxel layer.",
)
parser.add_argument(
    "--curobo_import_add_half_voxel",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Offset imported voxel centers by half a voxel during cuRobo import.",
)
parser.add_argument(
    "--curobo_frustum_padding_xyz",
    type=float,
    nargs=3,
    default=(0.10, 0.10, 0.08),
    metavar=("PX", "PY", "PZ"),
    help="XYZ padding (m) added around merged camera frustum bounds before ESDF query.",
)

# AO-Grasp backend config
parser.add_argument(
    "--grasp_backend",
    type=str,
    choices=("none", "ao-grasp"),
    default="none",
    help="Manipulation backend used by the manual M trigger.",
)
parser.add_argument(
    "--manip_target_mode",
    type=str,
    choices=("center_b", "prim_path"),
    default="prim_path",
    help="AO-Grasp benchmark target source: direct center in robot base frame or ground-truth prim path adapter.",
)
parser.add_argument(
    "--manip_target_prim_path",
    type=str,
    default="",
    help="Target prim path used when --grasp_backend=ao-grasp and --manip_target_mode=prim_path.",
)
parser.add_argument(
    "--manip_target_center_b",
    type=float,
    nargs=3,
    default=None,
    metavar=("X", "Y", "Z"),
    help="Target center in the robot base frame used when --manip_target_mode=center_b.",
)
parser.add_argument(
    "--manip_camera_names",
    type=str,
    nargs="+",
    default=None,
    help="Subset of AO-Grasp camera names to use for fusion. Defaults to all configured AO-Grasp cameras.",
)
parser.add_argument(
    "--manip_debug",
    action="store_true",
    default=False,
    help="Save AO-Grasp debug artifacts under outputs/ao-grasp/<request_id>/.",
)
parser.add_argument(
    "--ao_grasp_camera_paths",
    type=str,
    nargs="+",
    default=list(_DEFAULT_ROBOT_CAMERA_PATHS),
    help="Existing camera prim path(s) on the robot used to build AO-Grasp point clouds.",
)
parser.add_argument(
    "--ao_grasp_camera_names",
    type=str,
    nargs="+",
    default=list(_DEFAULT_ROBOT_CAMERA_NAMES),
    help="Logical names matching --ao_grasp_camera_paths in order.",
)
parser.add_argument(
    "--ao_config",
    type=str,
    default=str(_DEFAULT_AOGRASP_CONFIG_PATH),
    help="AO-Grasp YAML config path.",
)
parser.add_argument(
    "--ao_pointscore_url",
    type=str,
    default="http://127.0.0.1:18081",
    help="HTTP base URL for the AO-Grasp pointscore sidecar.",
)
parser.add_argument(
    "--ao_cgn_url",
    type=str,
    default="http://127.0.0.1:18082",
    help="HTTP base URL for the AO-Grasp Contact-GraspNet sidecar.",
)
parser.add_argument(
    "--ao_timeout_s",
    type=float,
    default=30.0,
    help="Request timeout in seconds for AO-Grasp sidecar calls.",
)
parser.add_argument(
    "--ao_debug_root",
    type=str,
    default="outputs/ao-grasp",
    help="Root directory used for AO-Grasp debug artifacts.",
)
parser.add_argument(
    "--ao_proposal_viz_output_root",
    type=str,
    default=None,
    help="Optional output root for AO-Grasp heatmap and proposal images.",
)
parser.add_argument(
    "--ao_proposal_viz_top_k",
    type=int,
    default=10,
    help="Top-k proposals highlighted in AO-Grasp debug images.",
)
parser.add_argument(
    "--ao_total_points",
    type=int,
    default=16384,
    help="Exact number of points sent to AO-Grasp after fusion and downsampling.",
)
parser.add_argument(
    "--ao_rng_seed",
    type=int,
    default=0,
    help="Deterministic RNG seed used when sampling AO-Grasp point clouds.",
)
parser.add_argument(
    "--ao_grasp_to_ee_position",
    type=float,
    nargs=3,
    default=(0.0, 0.0, 0.0),
    metavar=("X", "Y", "Z"),
    help="Static translation from AO-Grasp grasp frame to the robot end-effector frame.",
)
parser.add_argument(
    "--ao_grasp_to_ee_quat_wxyz",
    type=float,
    nargs=4,
    default=(1.0, 0.0, 0.0, 0.0),
    metavar=("W", "X", "Y", "Z"),
    help="Static quaternion from AO-Grasp grasp frame to the robot end-effector frame.",
)
parser.add_argument(
    "--ao_pregrasp_offset_m",
    type=float,
    default=0.10,
    help="Distance to offset the pre-grasp target from the grasp target along the configured approach axis.",
)
parser.add_argument(
    "--ao_pregrasp_approach_axis",
    type=str,
    choices=("x", "y", "z"),
    default="z",
    help="Local grasp-frame axis used to compute the pre-grasp offset.",
)
parser.add_argument(
    "--ao_pregrasp_approach_sign",
    type=float,
    default=-1.0,
    help="Signed direction multiplier applied to the AO-Grasp pre-grasp approach axis.",
)
parser.add_argument(
    "--ao_target_crop_radius_m",
    type=float,
    default=0.30,
    help="Target-local crop radius (m) around the requested center before AO-Grasp component selection.",
)
parser.add_argument(
    "--ao_guarded_approach_steps",
    type=int,
    default=6,
    help="Number of straight-line guarded approach waypoints from pregrasp to grasp.",
)
parser.add_argument(
    "--curobo_local_esdf_voxel_size",
    type=float,
    default=0.025,
    help="Voxel size (m) for the local pointcloud-sourced ESDF imported into CuRobo.",
)
parser.add_argument(
    "--curobo_local_esdf_margin_xyz",
    type=float,
    nargs=3,
    default=(0.20, 0.20, 0.20),
    metavar=("MX", "MY", "MZ"),
    help="XYZ margin (m) added around the EE/target/pregrasp/grasp corridor when building the local ESDF ROI.",
)
parser.add_argument(
    "--curobo_local_esdf_max_voxels",
    type=int,
    default=150000,
    help="Hard cap on local ESDF voxels imported into CuRobo for benchmark AO-Grasp planning.",
)
parser.add_argument(
    "--debug_robot_truth",
    action="store_true",
    default=False,
    help=(
        "Print env_0 joint state, body poses, and the matching stage prim transform after startup and each manual "
        "reset to compare articulation truth against the Scene panel."
    ),
)

# Skills config
parser.add_argument("--auto_start", action="store_true", default=False, help="Auto-start the manipulation skill.")
parser.add_argument(
    "--curobo_offset_min",
    type=float,
    nargs=3,
    default=(-1.0, -1.0, -0.3),
    metavar=("X", "Y", "Z"),
    help="Min XYZ offset (m) from current EE pose when sampling random CuRobo targets.",
)
parser.add_argument(
    "--curobo_offset_max",
    type=float,
    nargs=3,
    default=(1.0, 1.0, 0.3),
    metavar=("X", "Y", "Z"),
    help="Max XYZ offset (m) from current EE pose when sampling random CuRobo targets.",
)
parser.add_argument("--curobo_retries", type=int, default=10, help="Retries when sampling CuRobo targets.")
parser.add_argument("--curobo_close_steps", type=int, default=30, help="Steps to keep closing the gripper.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Isaac Sim at a fixed 1080p viewport/window size for clean recordings.
args_cli.width = 1920
args_cli.height = 1080
args_cli.window_width = 1920
args_cli.window_height = 1080

args_cli.curobo_depth_camera_paths = [
    _sanitize_camera_path(path) for path in args_cli.curobo_depth_camera_paths
]
args_cli.ao_grasp_camera_paths = [
    _sanitize_camera_path(path) for path in args_cli.ao_grasp_camera_paths
]
if args_cli.manip_camera_names is None:
    args_cli.manip_camera_names = list(args_cli.ao_grasp_camera_names)

if args_cli.curobo_use_depth_collision and not args_cli.enable_cameras:
    parser.error("--curobo_use_depth_collision requires --enable_cameras.")
if args_cli.curobo_use_depth_collision and len(args_cli.curobo_depth_camera_paths) == 0:
    parser.error("--curobo_use_depth_collision requires at least one --curobo_depth_camera_paths entry.")
if len(args_cli.curobo_frustum_padding_xyz) != 3:
    parser.error("--curobo_frustum_padding_xyz requires exactly 3 values.")
if len(args_cli.curobo_local_esdf_margin_xyz) != 3:
    parser.error("--curobo_local_esdf_margin_xyz requires exactly 3 values.")
if len(args_cli.ao_grasp_camera_names) != len(args_cli.ao_grasp_camera_paths):
    parser.error("--ao_grasp_camera_names must match --ao_grasp_camera_paths in length.")
if len(set(args_cli.ao_grasp_camera_names)) != len(args_cli.ao_grasp_camera_names):
    parser.error("--ao_grasp_camera_names must be unique.")
if args_cli.grasp_backend == "ao-grasp":
    if not args_cli.enable_cameras:
        parser.error("--grasp_backend=ao-grasp requires --enable_cameras.")
    if len(args_cli.ao_grasp_camera_paths) == 0:
        parser.error("--grasp_backend=ao-grasp requires at least one --ao_grasp_camera_paths entry.")
    if args_cli.manip_target_mode == "center_b":
        if args_cli.manip_target_center_b is None:
            parser.error("--grasp_backend=ao-grasp with --manip_target_mode=center_b requires --manip_target_center_b.")
    elif args_cli.auto_start and not str(args_cli.manip_target_prim_path).strip():
        parser.error(
            "--auto_start with --grasp_backend=ao-grasp and --manip_target_mode=prim_path "
            "requires --manip_target_prim_path."
        )
    if len(args_cli.manip_camera_names) == 0:
        parser.error("--grasp_backend=ao-grasp requires at least one --manip_camera_names entry.")
    unknown_manip_cameras = [
        name for name in args_cli.manip_camera_names if name not in set(args_cli.ao_grasp_camera_names)
    ]
    if unknown_manip_cameras:
        parser.error(
            "--manip_camera_names must be drawn from --ao_grasp_camera_names; got unknown entries: "
            f"{unknown_manip_cameras}."
        )

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import carb

import isaaclab.sim as isaaclab_sim
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms
from pxr import PhysxSchema, UsdGeom, UsdPhysics
import isaaclab.sim.utils as sim_utils

from skills.manipulation import (
    AoGraspClient,
    AoGraspServiceError,
    IsaacLabManipulationRunner,
    ManipulationRequest,
    build_manipulation_skill,
)
from skills.manipulation.aograsp_client import load_aograsp_service_config
from helpers.pcd_utils import compute_prim_world_bounds

_DEPTH_SENSOR_WIDTH = 640
_DEPTH_SENSOR_HEIGHT = 480
_DEFAULT_CUROBO_RETRACT_JOINT_POS = {
    # Hard-code the current CuRobo retract pose into the articulation default state
    # so the arm spawns safely instead of snapping forward after sim reset.
    "arm_sh0": 0.0,
    "arm_sh1": -3.14,
    "arm_el0": 3.14,
    "arm_el1": 0.0,
    "arm_wr0": 0.0,
    "arm_wr1": 0.0,
    "arm_f1x": -1.56,
}

class _KeyInput:
    """Minimal keyboard handler for triggering manipulation, release/reset, and replans in GUI mode."""

    def __init__(self) -> None:
        import omni.appwindow

        self.trigger_skill = False
        self.release_or_reset_requested = False
        self.trigger_random_target = False

        self._input = carb.input.acquire_input_interface()
        self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        self._sub = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_event)

    def _on_event(self, event) -> None:
        if event.type != carb.input.KeyboardEventType.KEY_PRESS:
            return
        if event.input.name == "M":
            self.trigger_skill = True
        elif event.input.name == "R":
            self.release_or_reset_requested = True
        elif event.input.name == "T":
            self.trigger_random_target = True

    def consume_trigger(self) -> bool:
        if self.trigger_skill:
            self.trigger_skill = False
            return True
        return False

    def consume_release_or_reset(self) -> bool:
        if self.release_or_reset_requested:
            self.release_or_reset_requested = False
            return True
        return False

    def consume_random_target(self) -> bool:
        if self.trigger_random_target:
            self.trigger_random_target = False
            return True
        return False


class _DockedPrimPathPanel:
    """Docked Omniverse side panel for entering a target prim path."""

    def __init__(self, default_target_prim_path: str = "") -> None:
        import asyncio
        import omni.kit.app
        import omni.ui as ui

        self._asyncio = asyncio
        self._app = omni.kit.app.get_app()
        self._ui = ui
        self._pending_result = _NO_PANEL_RESULT
        self._text_model = ui.SimpleStringModel(str(default_target_prim_path))
        self._status_label = None
        self._window = ui.Window(
            "AO-Grasp Target",
            width=420,
            height=180,
            visible=True,
            dock_preference=ui.DockPreference.RIGHT_TOP,
        )
        with self._window.frame:
            with ui.VStack(spacing=8, height=0):
                ui.Label(
                    "Paste the target prim path here, then click Submit.",
                    alignment=ui.Alignment.LEFT_TOP,
                    word_wrap=True,
                )
                ui.StringField(model=self._text_model, width=ui.Fraction(1))
                self._status_label = ui.Label(
                    "Press 'M' to focus this panel for the next manual grasp.",
                    alignment=ui.Alignment.LEFT_TOP,
                    word_wrap=True,
                )
                with ui.HStack(height=0, spacing=8):
                    ui.Button("Submit", clicked_fn=self._on_submit)
                    ui.Button("Reset", clicked_fn=lambda: self._set_model_string(default_target_prim_path))

        self._asyncio.ensure_future(self._dock_window())

    def destroy(self) -> None:
        if self._window is not None:
            self._window.visible = False
            self._window.destroy()
            self._window = None

    def request_input(self, default_target_prim_path: str) -> None:
        current_value = self._read_model_string().strip()
        if not current_value and default_target_prim_path:
            self._set_model_string(default_target_prim_path)
        if self._status_label is not None:
            self._status_label.text = "Paste a target prim path, then click Submit."
        self._focus_window()

    def consume_result(self):
        result = self._pending_result
        self._pending_result = _NO_PANEL_RESULT
        return result

    async def _dock_window(self) -> None:
        for _ in range(5):
            if self._ui.Workspace.get_window(self._window.title):
                break
            await self._app.next_update_async()

        custom_window = self._ui.Workspace.get_window(self._window.title)
        property_window = self._ui.Workspace.get_window("Property")
        if custom_window and property_window:
            custom_window.dock_in(property_window, self._ui.DockPosition.SAME, 1.0)

    def _focus_window(self) -> None:
        try:
            workspace_window = self._ui.Workspace.get_window(self._window.title)
            if workspace_window is not None:
                workspace_window.focus()
        except Exception:
            return

    def _read_model_string(self) -> str:
        if hasattr(self._text_model, "get_value_as_string"):
            return str(self._text_model.get_value_as_string())
        if hasattr(self._text_model, "as_string"):
            return str(self._text_model.as_string)
        return ""

    def _set_model_string(self, value: str) -> None:
        self._text_model.set_value(str(value))

    def _on_submit(self) -> None:
        value = self._read_model_string().strip()
        if not value:
            if self._status_label is not None:
                self._status_label.text = "Enter a prim path before submitting."
            self._focus_window()
            return
        self._pending_result = value
        if self._status_label is not None:
            self._status_label.text = f"Submitted: {value}"


def _apply_solver_iteration_clamps(pos_iters: int = 16, vel_iters: int = 1, scene_path: str = "/World") -> None:
    """Clamp global PhysX solver iteration counts for all actors in the scene."""
    stage = get_current_stage()

    # Ensure the physics scene prim exists
    scene = UsdPhysics.Scene.Get(stage, scene_path)
    if not scene:
        scene = UsdPhysics.Scene.Define(stage, scene_path)

    physx_scene_api = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())

    physx_scene_api.CreateEnableGPUDynamicsAttr().Set(True)

    # Broadphase type:
    # - "GPU" : GPU broadphase
    # - "MBP" : CPU multi-box pruning
    # - "SAP" : CPU sweep-and-prune
    physx_scene_api.CreateBroadphaseTypeAttr().Set("GPU")


    # Force fixed iteration counts (global clamps)
    physx_scene_api.CreateMaxPositionIterationCountAttr().Set(int(pos_iters))
    physx_scene_api.CreateMinPositionIterationCountAttr().Set(int(pos_iters))
    physx_scene_api.CreateMaxVelocityIterationCountAttr().Set(int(vel_iters))
    physx_scene_api.CreateMinVelocityIterationCountAttr().Set(int(vel_iters))


def _make_scene_cfg() -> type[InteractiveSceneCfg]:
    scene_usd = _resolve_usd_path(args_cli.scene_usd)
    robot_usd = _resolve_usd_path(args_cli.robot_usd)

    @configclass
    class InteractiveSearchSceneCfg(InteractiveSceneCfg):
        # ground plane
        ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=isaaclab_sim.GroundPlaneCfg())

        light = AssetBaseCfg(
            prim_path="/World/Light",
            spawn=isaaclab_sim.DomeLightCfg(intensity=2500.0, color=(0.75, 0.75, 0.75)),
        )

        robot = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=isaaclab_sim.UsdFileCfg(
                usd_path=robot_usd,
                activate_contact_sensors=True,
                rigid_props=isaaclab_sim.RigidBodyPropertiesCfg(disable_gravity=args_cli.robot_disable_gravity),
                articulation_props=isaaclab_sim.ArticulationRootPropertiesCfg(fix_root_link=args_cli.fix_root),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=tuple(args_cli.robot_pos),
                rot=tuple(args_cli.robot_rot),
                # The Spot base joints in this USD ship with knee joint defaults at 0.0, but their limits
                # are strictly negative (e.g. [-2.79, -0.25]). Set a valid crouched pose to avoid
                # Articulation validation failures on startup.
                joint_pos={
                    ".*_knee": -1.5,
                    **_DEFAULT_CUROBO_RETRACT_JOINT_POS,
                },
            ),
            actuators={
                "arm": ImplicitActuatorCfg(
                    joint_names_expr=args_cli.arm_joints,
                    effort_limit_sim=200.0,
                    stiffness=400.0,
                    damping=80.0,
                ),
                "gripper": ImplicitActuatorCfg(
                    joint_names_expr=args_cli.gripper_joints,
                    effort_limit_sim=200.0,
                    stiffness=2e3,
                    damping=1e2,
                ),
            },
        )

        if scene_usd:
            environment = AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/Scene",
                spawn=isaaclab_sim.UsdFileCfg(
                    usd_path=scene_usd
                ),
            )
        _camera_sensor_specs = tuple(_build_camera_sensor_specs())
        for _camera_spec in _camera_sensor_specs:
            # Attach sensor wrappers to existing robot-mounted camera prims.
            # We do not spawn/initialize camera assets here.
            locals()[_camera_spec["scene_key"]] = CameraCfg(
                prim_path=_camera_spec["prim_path"],
                height=_DEPTH_SENSOR_HEIGHT,
                width=_DEPTH_SENSOR_WIDTH,
                data_types=list(_camera_spec["data_types"]),
                colorize_semantic_segmentation=False,
                colorize_instance_segmentation=False,
                colorize_instance_id_segmentation=False,
                update_latest_camera_pose=True,
                spawn=None,
            )
        if _camera_sensor_specs:
            del _camera_spec
        del _camera_sensor_specs

    return InteractiveSearchSceneCfg


def _reset_robot(scene: InteractiveScene) -> None:
    robot = scene["robot"]
    root_state = robot.data.default_root_state.clone()
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.reset()

    actuated_joint_ids, _ = robot.find_joints([*args_cli.arm_joints, *args_cli.gripper_joints], preserve_order=True)
    if actuated_joint_ids:
        robot.set_joint_position_target(joint_pos[:, actuated_joint_ids], joint_ids=actuated_joint_ids)


def _format_debug_vector(values: torch.Tensor) -> str:
    return "[" + ", ".join(f"{float(value): .4f}" for value in values.tolist()) + "]"


def _dump_robot_truth(
    scene: InteractiveScene,
    robot,
    *,
    reason: str,
    joint_names: list[str] | tuple[str, ...],
    body_names: list[str] | tuple[str, ...],
    stage_link_name: str = "arm_link_el0",
) -> None:
    env_prim_path = scene.env_prim_paths[0]
    joint_name_to_idx = {name: idx for idx, name in enumerate(robot.joint_names)}
    body_name_to_idx = {name: idx for idx, name in enumerate(robot.body_names)}
    default_joint_pos = robot.data.default_joint_pos[0].detach().cpu()
    target_joint_pos = robot.data.joint_pos_target[0].detach().cpu()
    live_joint_pos = robot.data.joint_pos[0].detach().cpu()

    print(
        f"[DEBUG] Robot truth dump ({reason}) env_0={env_prim_path} fabric_enabled={not args_cli.disable_fabric}"
    )
    print("[DEBUG] Joint positions (default vs target vs live):")
    for joint_name in joint_names:
        joint_idx = joint_name_to_idx.get(joint_name)
        if joint_idx is None:
            print(f"[DEBUG]   {joint_name:<12} missing from articulation")
            continue
        print(
            f"[DEBUG]   {joint_name:<12} default={float(default_joint_pos[joint_idx]): .4f} "
            f"target={float(target_joint_pos[joint_idx]): .4f} "
            f"live={float(live_joint_pos[joint_idx]): .4f}"
        )

    print("[DEBUG] Body world poses from articulation:")
    for body_name in body_names:
        body_idx = body_name_to_idx.get(body_name)
        if body_idx is None:
            print(f"[DEBUG]   {body_name:<12} missing from articulation")
            continue
        body_pose = robot.data.body_pose_w[0, body_idx].detach().cpu()
        body_pos = body_pose[:3]
        body_quat = body_pose[3:7]
        print(
            f"[DEBUG]   {body_name:<12} pos={_format_debug_vector(body_pos)} "
            f"quat_wxyz={_format_debug_vector(body_quat)}"
        )

    stage = get_current_stage()
    stage_link_path = f"{env_prim_path}/Robot/{stage_link_name}"
    stage_link_prim = stage.GetPrimAtPath(stage_link_path)
    if not stage_link_prim or not stage_link_prim.IsValid():
        print(f"[DEBUG] Stage prim '{stage_link_path}' was not found.")
        return

    stage_transform = UsdGeom.Xformable(stage_link_prim).ComputeLocalToWorldTransform(0.0)
    stage_translation = stage_transform.ExtractTranslation()
    stage_position = torch.tensor(
        [stage_translation[0], stage_translation[1], stage_translation[2]], dtype=torch.float32
    )
    print(f"[DEBUG] Stage prim '{stage_link_name}' world pos={_format_debug_vector(stage_position)}")


def run_simulator(sim: isaaclab_sim.SimulationContext, scene: InteractiveScene) -> None:
    robot = scene["robot"]

    robot_entity_cfg = SceneEntityCfg(
        "robot",
        joint_names=args_cli.arm_joints,
        body_names=[args_cli.ee_body],
        preserve_order=True,
    )
    robot_entity_cfg.resolve(scene)
    arm_joint_ids = robot_entity_cfg.joint_ids
    arm_joint_names = [robot.joint_names[idx] for idx in arm_joint_ids]
    ee_body_id = robot_entity_cfg.body_ids[0]

    gripper_joint_ids, gripper_joint_names = robot.find_joints(args_cli.gripper_joints, preserve_order=True)
    if len(gripper_joint_ids) == 0:
        print("[WARN] No gripper joints matched. Gripper commands will be ignored.")
    else:
        print(f"[INFO] Gripper joints: {gripper_joint_names}")

    key_input = None if args_cli.headless else _KeyInput()
    prim_path_panel = None
    if not args_cli.headless and args_cli.grasp_backend == "ao-grasp" and args_cli.manip_target_mode == "prim_path":
        prim_path_panel = _DockedPrimPathPanel(str(args_cli.manip_target_prim_path or "").strip())
    depth_camera_handles: list[object] = []
    grasp_camera_handles: dict[str, object] = {}
    use_depth_collision = bool(args_cli.curobo_use_depth_collision)
    use_voxel_collision = bool(use_depth_collision or args_cli.grasp_backend == "ao-grasp")
    camera_sensor_specs = _build_camera_sensor_specs()
    camera_spec_by_path = {spec["prim_path"]: spec for spec in camera_sensor_specs}

    if use_depth_collision:
        for cam_idx, cam_path in enumerate(args_cli.curobo_depth_camera_paths):
            spec = camera_spec_by_path.get(cam_path)
            if spec is None:
                raise RuntimeError(f"Depth collision camera path '{cam_path}' was not registered in the scene config.")
            try:
                depth_camera_handles.append(scene[spec["scene_key"]])
                print(f"[INFO] Depth collision camera[{cam_idx}]: {cam_path}")
            except KeyError as exc:
                raise RuntimeError(
                    "Depth collision is enabled, but camera sensor "
                    f"'{spec['scene_key']}' could not be resolved for prim path '{cam_path}'."
                ) from exc

    if args_cli.grasp_backend == "ao-grasp":
        for camera_name, camera_path in zip(args_cli.ao_grasp_camera_names, args_cli.ao_grasp_camera_paths):
            spec = camera_spec_by_path.get(camera_path)
            if spec is None:
                raise RuntimeError(f"AO-Grasp camera path '{camera_path}' was not registered in the scene config.")
            try:
                grasp_camera_handles[camera_name] = scene[spec["scene_key"]]
                print(f"[INFO] AO-Grasp camera '{camera_name}': {camera_path}")
            except KeyError as exc:
                raise RuntimeError(
                    "AO-Grasp is enabled, but camera sensor "
                    f"'{spec['scene_key']}' could not be resolved for prim path '{camera_path}'."
                ) from exc

    grasp_client = None
    if args_cli.grasp_backend == "ao-grasp":
        ao_config = load_aograsp_service_config(args_cli.ao_config)
        ao_overrides = {
            "pointscore_url": args_cli.ao_pointscore_url,
            "cgn_url": args_cli.ao_cgn_url,
            "timeout_s": args_cli.ao_timeout_s,
            "debug_root": args_cli.ao_debug_root,
            "total_points": args_cli.ao_total_points,
            "rng_seed": args_cli.ao_rng_seed,
            "proposal_viz_top_k": args_cli.ao_proposal_viz_top_k,
        }
        if args_cli.ao_proposal_viz_output_root is not None:
            ao_overrides["proposal_viz_output_root"] = args_cli.ao_proposal_viz_output_root
        grasp_client = AoGraspClient(
            replace(
                ao_config,
                **ao_overrides,
            )
        )
        try:
            health_payload = grasp_client.healthcheck()
        except AoGraspServiceError as exc:
            raise RuntimeError(f"AO-Grasp backend is enabled but failed healthcheck: {exc}") from exc
        print(f"[INFO] AO-Grasp health: {health_payload}")

    curobo_robot_cfg = _resolve_usd_path(args_cli.curobo_robot_cfg)
    manipulation_skill = build_manipulation_skill(
        curobo_robot_cfg,
        args_cli.arm_joints,
        sim.device,
        gripper_open=args_cli.gripper_open,
        gripper_closed=args_cli.gripper_closed,
        offset_min=args_cli.curobo_offset_min,
        offset_max=args_cli.curobo_offset_max,
        retries=args_cli.curobo_retries,
        close_steps=args_cli.curobo_close_steps,
        use_depth_collision=use_depth_collision,
        use_voxel_collision=use_voxel_collision,
        depth_voxel_size=args_cli.curobo_depth_voxel_size,
        depth_min_m=args_cli.curobo_depth_min_m,
        depth_max_m=args_cli.curobo_depth_max_m,
        depth_truncation_distance_vox=args_cli.curobo_truncation_distance_vox,
        depth_raycast_subsampling=args_cli.curobo_raycast_subsampling,
        depth_accumulate_frames=args_cli.curobo_accumulate_frames,
        depth_import_invert_sign=args_cli.curobo_import_invert_sign,
        depth_import_add_half_voxel=args_cli.curobo_import_add_half_voxel,
        depth_frustum_padding_xyz=args_cli.curobo_frustum_padding_xyz,
        log=print,
    )
    runner = IsaacLabManipulationRunner(
        manipulation_skill,
        robot=robot,
        scene=scene,
        ee_body_id=ee_body_id,
        arm_joint_ids=arm_joint_ids,
        arm_joint_names=arm_joint_names,
        gripper_joint_ids=gripper_joint_ids,
        gripper_open=args_cli.gripper_open,
        depth_cameras=depth_camera_handles,
        grasp_client=grasp_client,
        grasp_cameras=grasp_camera_handles,
        default_grasp_camera_names=args_cli.manip_camera_names,
        grasp_to_ee_position=args_cli.ao_grasp_to_ee_position,
        grasp_to_ee_quat_wxyz=args_cli.ao_grasp_to_ee_quat_wxyz,
        pregrasp_offset_m=args_cli.ao_pregrasp_offset_m,
        pregrasp_approach_axis=args_cli.ao_pregrasp_approach_axis,
        pregrasp_approach_sign=args_cli.ao_pregrasp_approach_sign,
        target_crop_radius_m=args_cli.ao_target_crop_radius_m,
        local_esdf_voxel_size=args_cli.curobo_local_esdf_voxel_size,
        local_esdf_margin_xyz=args_cli.curobo_local_esdf_margin_xyz,
        local_esdf_max_voxels=args_cli.curobo_local_esdf_max_voxels,
        guarded_approach_steps=args_cli.ao_guarded_approach_steps,
        log=print,
    )
    runner.open_gripper()

    sim_dt = sim.get_physics_dt()
    debug_truth_joint_names = list(dict.fromkeys([*args_cli.arm_joints, *args_cli.gripper_joints]))
    debug_truth_body_names = ("body", "arm_link_sh0", "arm_link_el0", args_cli.ee_body)

    def _dump_robot_truth_snapshot(reason: str) -> None:
        if not args_cli.debug_robot_truth:
            return
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        _dump_robot_truth(
            scene,
            robot,
            reason=reason,
            joint_names=debug_truth_joint_names,
            body_names=debug_truth_body_names,
            stage_link_name="arm_link_el0",
        )

    def _is_gripper_mostly_open(tolerance: float = 0.05) -> bool:
        if len(gripper_joint_ids) == 0:
            return True
        joint_pos = robot.data.joint_pos[0, gripper_joint_ids].detach().cpu()
        open_target = torch.full_like(joint_pos, float(args_cli.gripper_open))
        return bool(torch.all(torch.abs(joint_pos - open_target) <= float(tolerance)))

    def _prompt_target_prim_path_terminal() -> str | None:
        default_target_prim_path = str(args_cli.manip_target_prim_path or "").strip()
        print("[INFO] Focus the launching terminal, paste the target prim path, and press Enter.")
        prompt = "[INPUT] Enter target object prim path"
        if default_target_prim_path:
            prompt += f" [{default_target_prim_path}]"
        prompt += ": "
        try:
            raw_target_prim_path = input(prompt)
        except EOFError:
            print("[WARN] Terminal input is unavailable; skipping manual AO-Grasp trigger.")
            return None
        except KeyboardInterrupt:
            print("\n[INFO] Manual AO-Grasp trigger canceled.")
            return None

        target_prim_path = str(raw_target_prim_path).strip() or default_target_prim_path
        if not target_prim_path:
            print("[WARN] No target prim path was provided; skipping manual AO-Grasp trigger.")
            return None
        return target_prim_path

    def _resolve_target_center_request(target_prim_path_override: str | None = None) -> tuple[object, str, str, str]:
        if args_cli.manip_target_mode == "center_b":
            target_center_b = torch.tensor(args_cli.manip_target_center_b, dtype=torch.float32)
            return target_center_b.detach().cpu().numpy(), "", "cli_center_b", "center_b"

        target_prim_path_raw = target_prim_path_override
        if target_prim_path_raw is None:
            target_prim_path_raw = args_cli.manip_target_prim_path
        if not str(target_prim_path_raw).strip():
            raise RuntimeError(
                "No target prim path is configured. Press 'M' again and enter a target prim path, "
                "or pass --manip_target_prim_path."
            )

        target_prim_path = _resolve_scene_prim_path(target_prim_path_raw)
        bounds = compute_prim_world_bounds(target_prim_path)
        if bounds is None:
            raise RuntimeError(f"Failed to resolve a world-space AABB for target prim '{target_prim_path}'.")
        bounds_min, bounds_max = bounds
        target_center_w = torch.as_tensor((bounds_min + bounds_max) * 0.5, dtype=robot.data.root_pose_w.dtype)
        target_center_w = target_center_w.view(1, 3).to(robot.data.root_pose_w.device)
        identity_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=robot.data.root_pose_w.dtype, device=robot.data.root_pose_w.device)
        base_pos = robot.data.root_pose_w[0:1, 0:3]
        base_quat = robot.data.root_pose_w[0:1, 3:7]
        target_center_b, _ = subtract_frame_transforms(base_pos, base_quat, target_center_w, identity_quat)
        target_label = target_prim_path.rsplit("/", 1)[-1]
        target_source = "manual_prompt" if target_prim_path_override is not None else args_cli.manip_target_mode
        return target_center_b[0].detach().cpu().numpy(), target_prim_path, target_label, target_source

    def _build_manipulation_request(
        reason: str,
        *,
        target_prim_path_override: str | None = None,
        prompt_for_target_prim: bool = False,
    ) -> ManipulationRequest | None:
        if args_cli.grasp_backend != "ao-grasp":
            return None
        resolved_target_prim_path_override = target_prim_path_override
        if (
            args_cli.manip_target_mode == "prim_path"
            and prompt_for_target_prim
            and resolved_target_prim_path_override is None
        ):
            resolved_target_prim_path_override = _prompt_target_prim_path_terminal()
            if resolved_target_prim_path_override is None:
                return None
        target_center_b, target_prim_path, target_label, target_source = _resolve_target_center_request(
            resolved_target_prim_path_override
        )
        return ManipulationRequest(
            target_prim_path=target_prim_path,
            trigger_reason=reason,
            camera_names=tuple(args_cli.manip_camera_names),
            debug=bool(args_cli.manip_debug),
            target_center_b=target_center_b,
            target_source=target_source,
            target_label=target_label,
        )

    def _build_manual_manipulation_request(
        target_prim_path_override: str | None = None,
    ) -> tuple[bool, ManipulationRequest | None]:
        if args_cli.grasp_backend != "ao-grasp":
            return True, None
        try:
            request = _build_manipulation_request(
                "manual",
                target_prim_path_override=target_prim_path_override,
                prompt_for_target_prim=(
                    args_cli.manip_target_mode == "prim_path"
                    and target_prim_path_override is None
                    and prim_path_panel is None
                ),
            )
        except RuntimeError as exc:
            print(f"[WARN] {exc}")
            return False, None
        if request is None:
            return False, None
        return True, request

    def _resolve_scene_prim_path(path: str) -> str:
        normalized = str(path or "").strip()
        if "{ENV_REGEX_NS}" in normalized:
            return normalized.replace("{ENV_REGEX_NS}", scene.env_prim_paths[0])
        if normalized == "/World/Robot":
            return f"{scene.env_prim_paths[0]}/Robot"
        return normalized

    robot_reference_prim_path = _resolve_scene_prim_path(args_cli.robot_prim_path)

    def _trigger_skill(reason: str, manipulation_request: ManipulationRequest | None = None) -> None:
        runner.trigger(
            sim_dt=sim_dt,
            update_world=args_cli.curobo_update_world,
            stage=get_current_stage(),
            ignore_substring=[robot_reference_prim_path],
            reference_prim_path=robot_reference_prim_path,
            reason=reason,
            manipulation_request=manipulation_request,
        )

    _dump_robot_truth_snapshot("startup_after_reset")

    if args_cli.auto_start:
        scene.update(sim_dt)
        _trigger_skill("auto_start", _build_manipulation_request("auto_start"))

    if prim_path_panel is not None:
        manual_prompt_message = "focus the docked AO-Grasp Target panel, paste a prim path, and click Submit"
    elif args_cli.grasp_backend == "ao-grasp" and args_cli.manip_target_mode == "prim_path":
        manual_prompt_message = "prompt for a target prim in the terminal and trigger manipulation"
    else:
        manual_prompt_message = "trigger manipulation"

    print(
        f"[INFO] Running. Press 'M' to {manual_prompt_message}, "
        "'R' to release the current grasp (or reset if already open), 'T' to replan a random target."
    )
    while simulation_app.is_running():
        if prim_path_panel is not None:
            prompt_result = prim_path_panel.consume_result()
            if prompt_result not in (_NO_PANEL_RESULT, None):
                should_trigger, manipulation_request = _build_manual_manipulation_request(
                    target_prim_path_override=str(prompt_result)
                )
                if should_trigger:
                    _trigger_skill("manual", manipulation_request)

        if key_input is not None and key_input.consume_release_or_reset():
            if _is_gripper_mostly_open():
                runner.reset_skill()
                _reset_robot(scene)
                runner.open_gripper()
                _dump_robot_truth_snapshot("manual_reset")
            else:
                runner.reset_skill()
                runner.open_gripper()
                _dump_robot_truth_snapshot("manual_release")

        if key_input is not None and key_input.consume_trigger():
            if prim_path_panel is not None:
                prim_path_panel.request_input(str(args_cli.manip_target_prim_path or "").strip())
            else:
                should_trigger, manipulation_request = _build_manual_manipulation_request()
                if should_trigger:
                    _trigger_skill("manual", manipulation_request)

        if key_input is not None and key_input.consume_random_target():
            _trigger_skill("replan")

        runner.step()

        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

    if prim_path_panel is not None:
        prim_path_panel.destroy()


# Set up rigid body and collision APIs; SDF provides better collision handling for complex meshes and physics stability in simulation
def _set_rigid_body_and_colliders(prim_path, objects_collision_approximation="sdf", structure_collision_approximation="none") -> None:

    stage = get_current_stage()
    objects = stage.GetPrimAtPath(f"{prim_path}/objects")
    # objects = prim_utils.get_prim_at_path()
    if objects and objects.IsValid():
        for child_prim in objects.GetChildren():
            if child_prim.IsA(UsdGeom.Xform):
                child_path = child_prim.GetPath().pathString
                print(f"Defining rigid body for child xform prim: {child_path}")
                isaaclab_sim.schemas.define_rigid_body_properties(
                    child_path,
                    isaaclab_sim.schemas.RigidBodyPropertiesCfg(rigid_body_enabled=True, disable_gravity=False),
                )
                isaaclab_sim.schemas.define_mass_properties(
                    prim_path,
                    isaaclab_sim.schemas.MassPropertiesCfg(mass=10.0),
                )
        
        objects_mesh_prims = sim_utils.get_all_matching_child_prims(
            objects.GetPath().pathString,
            predicate=lambda prim: prim.IsA(UsdGeom.Mesh),
        )

        for mesh_prim in objects_mesh_prims:
            if "exterior" in mesh_prim.GetName():
                continue
            print(f"Defining collision for mesh prim: {mesh_prim.GetPath().pathString}")
            isaaclab_sim.schemas.define_collision_properties(
                mesh_prim.GetPath(),
                isaaclab_sim.schemas.CollisionPropertiesCfg(collision_enabled=True),
            )
            mesh_collider = UsdPhysics.MeshCollisionAPI.Apply(mesh_prim)
            mesh_collider.CreateApproximationAttr().Set(objects_collision_approximation)
        
            if objects_collision_approximation == "convexDecomposition":
                collision_api = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(mesh_prim)
                collision_api.CreateHullVertexLimitAttr().Set(64)
                collision_api.CreateMaxConvexHullsAttr().Set(64)
                collision_api.CreateMinThicknessAttr().Set(0.001)
                collision_api.CreateShrinkWrapAttr().Set(True)
                collision_api.CreateErrorPercentageAttr().Set(0.1)
            elif objects_collision_approximation == "convexHull":
                collision_api = PhysxSchema.PhysxConvexHullCollisionAPI.Apply(mesh_prim)
                collision_api.CreateHullVertexLimitAttr().Set(64)
                collision_api.CreateMinThicknessAttr().Set(0.00001)
            elif objects_collision_approximation == "sdf":
                collision_api = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(mesh_prim)
                collision_api.CreateSdfResolutionAttr().Set(64)
            elif objects_collision_approximation == "none":
                collision_api = PhysxSchema.PhysxTriangleMeshCollisionAPI.Apply(mesh_prim)

    # structures = prim_utils.get_prim_at_path(f"{prim_path}/structure")
    structures = stage.GetPrimAtPath(f"{prim_path}/structure") 
    if structures and structures.IsValid():
        structure_mesh_prims = sim_utils.get_all_matching_child_prims(
            structures.GetPath().pathString,
            predicate=lambda prim: prim.IsA(UsdGeom.Mesh),
        )

        for mesh_prim in structure_mesh_prims:
            if "exterior" in mesh_prim.GetName():
                continue
            print(f"Defining collision for mesh prim: {mesh_prim.GetPath().pathString}")
            isaaclab_sim.schemas.define_collision_properties(
                mesh_prim.GetPath(),
                isaaclab_sim.schemas.CollisionPropertiesCfg(collision_enabled=True),
            )
            mesh_collider = UsdPhysics.MeshCollisionAPI.Apply(mesh_prim)
            mesh_collider.CreateApproximationAttr().Set(structure_collision_approximation)
        
            if structure_collision_approximation == "convexDecomposition":
                collision_api = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(mesh_prim)
                collision_api.CreateHullVertexLimitAttr().Set(64)
                collision_api.CreateMaxConvexHullsAttr().Set(64)
                collision_api.CreateMinThicknessAttr().Set(0.001)
                collision_api.CreateShrinkWrapAttr().Set(True)
                collision_api.CreateErrorPercentageAttr().Set(0.1)
            elif structure_collision_approximation == "convexHull":
                collision_api = PhysxSchema.PhysxConvexHullCollisionAPI.Apply(mesh_prim)
                collision_api.CreateHullVertexLimitAttr().Set(64)
                collision_api.CreateMinThicknessAttr().Set(0.00001)
            elif structure_collision_approximation == "sdf":
                collision_api = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(mesh_prim)
                collision_api.CreateSdfResolutionAttr().Set(128)
            elif structure_collision_approximation == "none":
                collision_api = PhysxSchema.PhysxTriangleMeshCollisionAPI.Apply(mesh_prim)


def main() -> None:
    sim_cfg = isaaclab_sim.SimulationCfg(
        dt=1.0 / 60.0,
        render_interval=1,
        device=args_cli.device,
        render=isaaclab_sim.RenderCfg(
            antialiasing_mode="Off",
            enable_dlssg=False,
            enable_dl_denoiser=False,
        ),
    )
    sim_cfg.use_fabric = not args_cli.disable_fabric
    sim = isaaclab_sim.SimulationContext(sim_cfg)

    cam_target = list(args_cli.robot_pos)
    cam_target[2] += 0.6
    sim.set_camera_view(eye=[cam_target[0] + 2.0, cam_target[1] + 2.0, cam_target[2] + 1.5], target=cam_target)

    scene_cfg = _make_scene_cfg()(num_envs=args_cli.num_envs, env_spacing=args_cli.env_spacing, replicate_physics=True)
    scene = InteractiveScene(scene_cfg)

    if _resolve_usd_path(args_cli.scene_usd):
        for i in range(args_cli.num_envs):
            prim_path = f"/World/envs/env_{i}/Scene"
            _set_rigid_body_and_colliders(prim_path)
    
    # _apply_solver_iteration_clamps(pos_iters=16, vel_iters=1)

    sim.reset()
    print("[INFO] Setup complete.")

    _reset_robot(scene)
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
