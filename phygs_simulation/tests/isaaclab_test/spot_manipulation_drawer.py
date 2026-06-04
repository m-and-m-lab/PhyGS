#!/usr/bin/env python3
"""Standalone IsaacLab smoke test for grasping a drawer with the clean Spot manipulation API."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from pathlib import Path
import sys
import time
import traceback
from typing import Any

import yaml


PHYGS_SIMULATION_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = PHYGS_SIMULATION_ROOT / "scripts"
if not SCRIPTS_ROOT.is_dir():
    raise FileNotFoundError(f"Expected phygs_simulation scripts directory at {SCRIPTS_ROOT}")
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from isaaclab.app import AppLauncher


DEFAULT_APP_CONFIG_PATH = Path(__file__).with_suffix(".yaml")
MANIPULATION_CONFIG_PATH = SCRIPTS_ROOT / "skills" / "manipulation" / "config" / "spot_arm_default.yaml"
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480


parser = argparse.ArgumentParser(description="Run one Spot drawer grasp smoke test.")
parser.add_argument(
    "--config",
    type=str,
    default=str(DEFAULT_APP_CONFIG_PATH),
    help="Path to the drawer grasp smoke-test YAML config.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import isaaclab.sim as isaaclab_sim
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.utils import configclass

from helpers.cam_utils import CameraRigConfig, load_camera_rig_config
from skills.manipulation import ManipulationState, SpotManipulationClient
from skills.manipulation import load_spot_manipulation_config as load_manipulation_api_config


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
    grasp_timeout_sec: float
    pre_grasp_delay_sec: float
    sensor_warmup_steps: int


@dataclass(frozen=True, slots=True)
class DrawerGraspSmokeConfig:
    robot: RobotSpawnConfig
    drawer: DrawerSpawnConfig
    request: RequestConfig
    scene: SceneConfig
    timeouts: TimeoutConfig


@dataclass(frozen=True, slots=True)
class ManipulationProfileFiles:
    robot_config_path: Path
    camera_rig_config_path: Path


@dataclass(frozen=True, slots=True)
class RobotAssetConfig:
    usd_path: str
    disable_gravity: bool
    fix_root_link: bool
    retract_joint_pos: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class CameraSensorSpec:
    name: str
    scene_key: str
    prim_path: str
    data_types: tuple[str, ...]


def load_drawer_grasp_smoke_config(path: str | Path) -> DrawerGraspSmokeConfig:
    config_path = _existing_path(path, base_dir=Path.cwd(), label="drawer grasp smoke config")
    root = _read_mapping(config_path, "drawer grasp smoke config")
    base_dir = config_path.parent

    robot = _required_mapping(root, "robot", "root")
    drawer = _required_mapping(root, "drawer", "root")
    request = _required_mapping(root, "request", "root")
    scene = _required_mapping(root, "scene", "root")
    timeouts = _required_mapping(root, "timeouts", "root")

    return DrawerGraspSmokeConfig(
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
            grasp_timeout_sec=_required_float(timeouts, "manipulation_sec", "timeouts"),
            pre_grasp_delay_sec=_required_float(timeouts, "pre_manipulation_delay_sec", "timeouts"),
            sensor_warmup_steps=_required_int(timeouts, "sensor_warmup_steps", "timeouts"),
        ),
    )


def load_profile_files(path: str | Path) -> ManipulationProfileFiles:
    profile_path = _existing_path(path, base_dir=Path.cwd(), label="manipulation API config")
    root = _read_mapping(profile_path, "manipulation API config")
    return ManipulationProfileFiles(
        robot_config_path=_required_existing_path(root, "robot_config_path", profile_path.parent, "manipulation API config"),
        camera_rig_config_path=_required_existing_path(
            root,
            "camera_rig_config_path",
            profile_path.parent,
            "manipulation API config",
        ),
    )


def load_robot_asset_config(path: str | Path) -> RobotAssetConfig:
    robot_path = _existing_path(path, base_dir=Path.cwd(), label="Spot robot config")
    root = _read_mapping(robot_path, "Spot robot config")
    retract_joint_pos = _required_mapping(root, "retract_joint_pos", "Spot robot config")
    return RobotAssetConfig(
        usd_path=str(_required_existing_path(root, "usd_path", robot_path.parent, "Spot robot config")),
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


class DrawerGraspSmokeTest:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.config = load_drawer_grasp_smoke_config(args.config)
        self.profile_files = load_profile_files(MANIPULATION_CONFIG_PATH)
        self.api_config = load_manipulation_api_config(MANIPULATION_CONFIG_PATH, device=args.device)
        self.robot_asset = load_robot_asset_config(self.profile_files.robot_config_path)
        self.camera_rig = load_camera_rig_config(self.profile_files.camera_rig_config_path)
        self.camera_specs = camera_sensor_specs(self.camera_rig)
        self._validate_camera_contract()

    def run(self) -> None:
        self._log_startup()

        sim = self._build_sim()
        scene = self._build_scene()
        sim.reset()

        robot = scene["robot"]
        self._sync_arm_targets_to_spawn(robot)
        cameras = resolve_scene_cameras(scene, self.camera_specs)
        client = self._create_manipulation_client(scene, robot, cameras)

        self._step_scene(sim, scene, self.config.timeouts.sensor_warmup_steps, "sensor warm-up")
        self._log(f"[AO-Grasp] Services ready: {client.ao_grasp.healthcheck()}")
        self._step_scene_for_seconds(
            sim,
            scene,
            self.config.timeouts.pre_grasp_delay_sec,
            "pre-grasp settle",
        )

        self._run_grasp_command(sim, scene, client)
        self._log("[RESULT] Spot drawer grasp smoke test completed successfully.")

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
        robot_asset = self.robot_asset
        robot_api = self.api_config.robot
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
                    usd_path=robot_asset.usd_path,
                    activate_contact_sensors=True,
                    rigid_props=isaaclab_sim.RigidBodyPropertiesCfg(disable_gravity=robot_asset.disable_gravity),
                    articulation_props=isaaclab_sim.ArticulationRootPropertiesCfg(fix_root_link=robot_asset.fix_root_link),
                ),
                init_state=ArticulationCfg.InitialStateCfg(
                    pos=config.robot.position,
                    rot=config.robot.rotation_wxyz,
                    joint_pos={".*_knee": -1.5, **dict(robot_asset.retract_joint_pos)},
                ),
                actuators={
                    "arm": ImplicitActuatorCfg(
                        joint_names_expr=list(robot_api.arm_joint_names),
                        effort_limit_sim=200.0,
                        stiffness=400.0,
                        damping=80.0,
                    ),
                    "gripper": ImplicitActuatorCfg(
                        joint_names_expr=list(robot_api.gripper_joint_names),
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

    def _create_manipulation_client(
        self,
        scene: InteractiveScene,
        robot: Any,
        cameras: Mapping[str, object],
    ) -> SpotManipulationClient:
        return SpotManipulationClient.create(
            robot=robot,
            scene=scene,
            cameras=cameras,
            device=self.args.device,
            config_path=MANIPULATION_CONFIG_PATH,
            stage_getter=get_current_stage,
            robot_reference_prim_path=self._env_prim_path(scene, "Robot"),
            log=self._log,
        )

    def _run_grasp_command(
        self,
        sim: isaaclab_sim.SimulationContext,
        scene: InteractiveScene,
        client: SpotManipulationClient,
    ) -> None:
        command_id = client.grasp(debug=self.config.request.debug)
        self._log(f"[Grasp] Started command_id={command_id}")

        deadline = time.monotonic() + self.config.timeouts.grasp_timeout_sec
        last_state: ManipulationState | None = None
        while time.monotonic() < deadline:
            if not simulation_app.is_running():
                raise RuntimeError("Simulation app stopped before the grasp reached a terminal state.")
            feedback = client.step()
            if feedback.state != last_state:
                self._log(f"[Grasp] state={feedback.state.value} message={feedback.message}")
                last_state = feedback.state

            scene.write_data_to_sim()
            sim.step()
            scene.update(sim.get_physics_dt())

            if feedback.is_terminal:
                if feedback.state != ManipulationState.DONE:
                    raise RuntimeError(
                        f"Grasp command {feedback.command_id} failed in state={feedback.state.value}: {feedback.message}"
                    )
                return

        raise TimeoutError(
            f"Grasp command {command_id} did not finish within {self.config.timeouts.grasp_timeout_sec:.2f}s."
        )

    def _sync_arm_targets_to_spawn(self, robot: Any) -> None:
        joint_names = (*self.api_config.robot.arm_joint_names, *self.api_config.robot.gripper_joint_names)
        joint_ids, resolved_names = robot.find_joints(list(joint_names), preserve_order=True)
        if tuple(resolved_names) != joint_names:
            raise RuntimeError(f"Expected joints {joint_names}, resolved {tuple(resolved_names)}.")

        spawn_joint_pos = robot.data.joint_pos[:, joint_ids].clone()
        robot.set_joint_position_target(spawn_joint_pos, joint_ids=joint_ids)
        self._log(f"[Sim] Synced arm targets for joints={tuple(resolved_names)}.")

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
        self._log(f"[Sim] Stepping {steps} frames for {label}.")
        for step_idx in range(steps):
            if not simulation_app.is_running():
                raise RuntimeError(f"Simulation app stopped during {label} at step {step_idx}/{steps}.")
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim.get_physics_dt())

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

    def _validate_camera_contract(self) -> None:
        rig_camera_names = set(self.camera_rig.camera_names)
        missing = [name for name in self.api_config.camera_names if name not in rig_camera_names]
        if missing:
            raise RuntimeError(f"Manipulation config references cameras missing from the rig: {missing}.")

    def _log_startup(self) -> None:
        ao_config = self.api_config.ao_grasp
        self._log("[Config] Spot drawer grasp smoke test")
        self._log(f"[Config] robot_usd={self.robot_asset.usd_path}")
        self._log(f"[Config] drawer_usd={self.config.drawer.usd_path}")
        self._log(f"[Config] cameras={self.api_config.camera_names}")
        self._log(
            "[Config] ao_grasp="
            f"pointscore={ao_config.pointscore_url}, "
            f"cgn={ao_config.cgn_url}, "
            f"max_depth_m={ao_config.max_depth_m}, "
            f"total_points={ao_config.total_points}"
        )
        self._log(f"[Config] curobo_robot_cfg={self.api_config.curobo.robot_cfg}")

    @staticmethod
    def _env_prim_path(scene: InteractiveScene, prim_name: str) -> str:
        return f"{scene.env_prim_paths[0]}/{prim_name.lstrip('/')}"

    @staticmethod
    def _log(message: str) -> None:
        print(message)


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
    DrawerGraspSmokeTest(args_cli).run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("[ABORTED] Spot drawer grasp smoke test interrupted by user.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"[FATAL] Spot drawer grasp smoke test failed: {exc}", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)
    finally:
        simulation_app.close()
