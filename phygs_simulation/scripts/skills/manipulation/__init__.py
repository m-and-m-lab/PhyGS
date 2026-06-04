"""Clean public API for camera-driven Spot manipulation in IsaacLab."""

from __future__ import annotations

from .aograsp_client import AoGraspClient, AoGraspDebugWriter, AoGraspServiceConfig, AoGraspServiceError, GraspPose
from .api import (
    DEFAULT_SPOT_MANIPULATION_CONFIG,
    ManipulationFeedback,
    ManipulationState,
    SpotManipulationClient,
    SpotManipulationConfig,
    SpotManipulationGraspConfig,
    SpotManipulationGripperConfig,
    SpotManipulationPullConfig,
    SpotManipulationRobotConfig,
    SpotManipulationRobotHandles,
    SpotManipulationWorldConfig,
    resolve_spot_manipulation_robot_handles,
)
from .curobo_interfaces import (
    CuroboDebugWriter,
    CuroboInputError,
    CuroboInterface,
    CuroboInterfaceConfig,
    CuroboInterfaceError,
    CuroboPlanningError,
    CuroboTriggerRequest,
    CuroboTriggerResult,
)


def load_spot_manipulation_config(*args, **kwargs):
    from .manipulation_skill import load_spot_manipulation_config as _load_spot_manipulation_config

    return _load_spot_manipulation_config(*args, **kwargs)

__all__ = [
    "AoGraspClient",
    "AoGraspDebugWriter",
    "AoGraspServiceConfig",
    "AoGraspServiceError",
    "CuroboDebugWriter",
    "CuroboInputError",
    "CuroboInterface",
    "CuroboInterfaceConfig",
    "CuroboInterfaceError",
    "CuroboPlanningError",
    "CuroboTriggerRequest",
    "CuroboTriggerResult",
    "DEFAULT_SPOT_MANIPULATION_CONFIG",
    "GraspPose",
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
    "load_spot_manipulation_config",
    "resolve_spot_manipulation_robot_handles",
]
