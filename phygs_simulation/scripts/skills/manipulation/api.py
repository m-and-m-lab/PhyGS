"""Public API types for camera-driven Spot manipulation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping

from helpers.cam_utils import DepthFilterConfig, ImageCrop

from .aograsp_client import AoGraspServiceConfig
from .curobo_interfaces import CuroboInterfaceConfig


DEFAULT_SPOT_MANIPULATION_CONFIG = "spot_arm_default"


class ManipulationState(str, Enum):
    """Lifecycle states for the camera-driven manipulation state machine."""

    IDLE = "idle"
    OPENING_GRIPPER = "opening_gripper"
    SETTLING_CAMERA = "settling_camera"
    PERCEIVING = "perceiving"
    PLANNING_PREGRASP = "planning_pregrasp"
    EXECUTING_PREGRASP = "executing_pregrasp"
    PLANNING_GRASP = "planning_grasp"
    EXECUTING_GRASP = "executing_grasp"
    CLOSING_GRIPPER = "closing_gripper"
    PULLING_BACK = "pulling_back"
    RELEASING = "releasing"
    DONE = "done"
    FAILED = "failed"

    @property
    def is_terminal(self) -> bool:
        return self in {ManipulationState.DONE, ManipulationState.FAILED}


@dataclass(frozen=True)
class ManipulationFeedback:
    """Current command status returned by ``SpotManipulationClient.step``."""

    command_id: int
    state: ManipulationState
    message: str = ""

    @property
    def is_terminal(self) -> bool:
        return self.state.is_terminal


@dataclass(frozen=True)
class SpotManipulationRobotConfig:
    """Robot names needed by the manipulation API."""

    arm_joint_names: tuple[str, ...]
    gripper_joint_names: tuple[str, ...]
    ee_body_name: str


@dataclass(frozen=True)
class SpotManipulationGripperConfig:
    """Spot claw targets and simple timing."""

    open_rad: float = -1.56
    closed_rad: float = -0.4
    tolerance_rad: float = 0.02
    open_timeout_steps: int = 180
    close_steps: int = 30
    camera_settle_steps: int = 10


@dataclass(frozen=True)
class SpotManipulationGraspConfig:
    """Pose transforms used to turn an AO-Grasp proposal into EE targets."""

    pregrasp_offset_m: float = 0.10
    pregrasp_axis: str = "z"
    pregrasp_sign: float = -1.0
    grasp_to_ee_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    grasp_to_ee_quat_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True)
class SpotManipulationPullConfig:
    """Base pull used for simplified articulated-object interaction."""

    backward_velocity_mps: float = 0.25
    steps: int = 40


@dataclass(frozen=True)
class SpotManipulationWorldConfig:
    """Static world collision settings for CuRobo USD collision mode."""

    only_paths: tuple[str, ...] = ("/World",)
    ignore_substring: tuple[str, ...] = ("Robot", "target", "curobo")
    robot_reference_prim_path: str | None = None


@dataclass(frozen=True)
class SpotManipulationConfig:
    """Complete config consumed by ``SpotManipulationClient``."""

    robot: SpotManipulationRobotConfig
    camera_names: tuple[str, ...]
    depth_filter: DepthFilterConfig
    crops_by_camera: Mapping[str, ImageCrop]
    camera_prim_paths: Mapping[str, str]
    ao_grasp: AoGraspServiceConfig
    curobo: CuroboInterfaceConfig
    gripper: SpotManipulationGripperConfig = SpotManipulationGripperConfig()
    grasp: SpotManipulationGraspConfig = SpotManipulationGraspConfig()
    pull: SpotManipulationPullConfig = SpotManipulationPullConfig()
    world: SpotManipulationWorldConfig = SpotManipulationWorldConfig()


@dataclass(frozen=True)
class SpotManipulationRobotHandles:
    """Resolved IsaacLab indices for the configured Spot arm."""

    arm_joint_ids: tuple[int, ...]
    gripper_joint_ids: tuple[int, ...]
    ee_body_id: int


def resolve_spot_manipulation_robot_handles(
    robot: Any,
    config: SpotManipulationRobotConfig,
) -> SpotManipulationRobotHandles:
    """Resolve configured joints and end-effector body on an IsaacLab articulation."""

    arm_ids, arm_names = robot.find_joints(list(config.arm_joint_names), preserve_order=True)
    gripper_ids, gripper_names = robot.find_joints(list(config.gripper_joint_names), preserve_order=True)
    ee_body_ids, ee_body_names = robot.find_bodies([config.ee_body_name], preserve_order=True)

    if tuple(arm_names) != tuple(config.arm_joint_names):
        raise RuntimeError(f"Resolved arm joints {tuple(arm_names)} do not match {config.arm_joint_names}.")
    if tuple(gripper_names) != tuple(config.gripper_joint_names):
        raise RuntimeError(f"Resolved gripper joints {tuple(gripper_names)} do not match {config.gripper_joint_names}.")
    if len(ee_body_ids) != 1 or tuple(ee_body_names) != (config.ee_body_name,):
        raise RuntimeError(f"Expected one EE body named {config.ee_body_name!r}, got {tuple(ee_body_names)}.")

    return SpotManipulationRobotHandles(
        arm_joint_ids=tuple(int(idx) for idx in arm_ids),
        gripper_joint_ids=tuple(int(idx) for idx in gripper_ids),
        ee_body_id=int(ee_body_ids[0]),
    )


class SpotManipulationClient:
    """User-facing Spot manipulation API.

    This class is intentionally thin: public calls live here, while the
    state machine and robotics implementation live in ``manipulation_skill``.
    """

    def __init__(
        self,
        *,
        _runtime: Any | None = None,
        robot: Any | None = None,
        scene: Any | None = None,
        cameras: Mapping[str, object] | None = None,
        config: SpotManipulationConfig | None = None,
        handles: SpotManipulationRobotHandles | None = None,
        ao_grasp: Any | None = None,
        curobo: Any | None = None,
        locomotion: Any | None = None,
        stage_getter: Callable[[], Any] | None = None,
    ) -> None:
        if _runtime is not None:
            self._runtime = _runtime
            return

        missing = [
            name
            for name, value in {
                "robot": robot,
                "scene": scene,
                "cameras": cameras,
                "config": config,
                "handles": handles,
                "ao_grasp": ao_grasp,
                "curobo": curobo,
            }.items()
            if value is None
        ]
        if missing:
            raise ValueError(f"Missing required SpotManipulationClient inputs: {missing}.")

        from .manipulation_skill import SpotManipulationRuntime

        self._runtime = SpotManipulationRuntime(
            robot=robot,
            scene=scene,
            cameras=cameras,
            config=config,
            handles=handles,
            ao_grasp=ao_grasp,
            curobo=curobo,
            locomotion=locomotion,
            stage_getter=stage_getter,
        )

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
    ) -> "SpotManipulationClient":
        """Create the manipulation client and all required runtime helpers."""

        from .manipulation_skill import SpotManipulationRuntime

        runtime = SpotManipulationRuntime.create(
            robot=robot,
            scene=scene,
            cameras=cameras,
            device=device,
            config_path=config_path,
            locomotion=locomotion,
            stage=stage,
            stage_getter=stage_getter,
            robot_reference_prim_path=robot_reference_prim_path,
            log=log,
        )
        return cls(_runtime=runtime)

    def grasp(self, *, debug: bool = False) -> int:
        """Start a camera-driven grasp command."""

        return self._runtime.grasp(debug=debug)

    def interact_articulated(self, *, debug: bool = False) -> int:
        """Start grasp, pull backward with locomotion, then release."""

        return self._runtime.interact_articulated(debug=debug)

    def release(self) -> int:
        """Start a gripper release command."""

        return self._runtime.release()

    def reset(self) -> None:
        """Return the client to idle and clear queued execution state."""

        self._runtime.reset()

    def stop(self) -> None:
        """Stop active manipulation and hold the robot at its current position."""

        self._runtime.stop()

    def step(self) -> ManipulationFeedback:
        """Advance the manipulation state machine by one control tick."""

        return self._runtime.step()

    @property
    def feedback(self) -> ManipulationFeedback:
        return self._runtime.feedback

    @property
    def state(self) -> ManipulationState:
        return self._runtime.state

    @property
    def is_active(self) -> bool:
        return self._runtime.is_active

    @property
    def last_error(self) -> str:
        return self._runtime.last_error

    @property
    def current_command_id(self) -> int:
        return self._runtime.current_command_id

    @property
    def robot(self) -> Any:
        return self._runtime.robot

    @property
    def scene(self) -> Any:
        return self._runtime.scene

    @property
    def cameras(self) -> Mapping[str, object]:
        return self._runtime.cameras

    @property
    def config(self) -> SpotManipulationConfig:
        return self._runtime.config

    @property
    def handles(self) -> SpotManipulationRobotHandles:
        return self._runtime.handles

    @property
    def ao_grasp(self) -> Any:
        return self._runtime.ao_grasp

    @property
    def curobo(self) -> Any:
        return self._runtime.curobo

    @property
    def locomotion(self) -> Any:
        return self._runtime.locomotion


__all__ = [
    "DEFAULT_SPOT_MANIPULATION_CONFIG",
    "ManipulationFeedback",
    "ManipulationState",
    "SpotManipulationClient",
    "SpotManipulationConfig",
    "SpotManipulationGraspConfig",
    "SpotManipulationGripperConfig",
    "SpotManipulationPullConfig",
    "SpotManipulationRobotConfig",
    "SpotManipulationRobotHandles",
    "SpotManipulationWorldConfig",
    "resolve_spot_manipulation_robot_handles",
]
