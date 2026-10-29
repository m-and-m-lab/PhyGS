# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Robot assembly from a robot description, and diagnostic arm control; acting APIs live under skills."""

from __future__ import annotations

import math
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from phygs.observations.localization import LocalizationEstimator
from phygs.observations.types import ObservationAccess
from phygs.recording.narration import Narration
from phygs.recording.timeline import CommandTimeline
from phygs.robots.posture import PostureController
from phygs.runtime.commands import CommandClient
from phygs.runtime.episode import EpisodeRuntime, EpisodeStatus, RunMode
from phygs.runtime.pipeline import (
    ObservationPipeline,
    PipelineConfig,
    ProcessingService,
    ServiceDeclaration,
)
from phygs.runtime.timing import duration_ticks
from phygs.skills.locomotion import api as locomotion_api

if TYPE_CHECKING:
    from isaaclab.scene import InteractiveScene
    from isaaclab.sim import SimulationContext

    from phygs.execution.locomotion.base import BaseController
    from phygs.execution.manipulation.planning import JointTrajectory
    from phygs.execution.navigation.interfaces import NavigationBackendFactory
    from phygs.observations.types import ObservationConfig, ObservationPacket
    from phygs.robots.description import RobotDescription
    from phygs.sim.state import IsaacLabWorld
    from phygs.skills.manipulation.api import HeldObject, ManipulationClient
    from phygs.skills.navigation.api import NavigationClient, NavigationConfig


class ArmClient(CommandClient):
    """Independent arm posture commands through the same ownership gate."""

    def __init__(
        self,
        runtime: EpisodeRuntime,
        controller: PostureController,
        world: IsaacLabWorld,
        limits: dict[str, tuple[float, float]],
        *,
        enabled: bool = True,
    ) -> None:
        super().__init__(runtime, controller.declaration.name)
        self._controller = controller
        self._world = world
        self._limits = limits
        self._enabled = enabled
        self.navigation: NavigationClient | None = None

    def command_joint_positions(self, positions: dict[str, float], *, duration: float = 1.0) -> int:
        """Interpolate requested arm angles [rad] over duration [s]; hold other arm joints."""
        if not self._enabled:
            raise PermissionError("Arm commands are not permitted by this task.")
        if self.navigation is not None and self.navigation.active:
            raise RuntimeError("Arm commands are prohibited while navigation uses the folded transit envelope.")
        ticks = duration_ticks(duration, self._runtime.physics_dt)
        if not positions or not set(positions).issubset(self._controller.declaration.joints):
            raise ValueError("Specify a nonempty subset of declared arm joints.")
        for joint, value in positions.items():
            lower, upper = self._limits[joint]
            if not math.isfinite(value) or not lower <= value <= upper:
                raise ValueError(f"Arm target {joint}={value} is outside [{lower}, {upper}] rad.")
        command_id = self._submit(ticks)
        measured = self._world.positions()
        self._controller.reset({joint: measured[joint] for joint in self._controller.declaration.joints})
        self._controller.target.update(positions)
        self._controller.duration = duration
        return command_id


# Base controller kinds of ``robot.yaml`` (``base.kind``); any other kind is a plugin ``module:Class``.
BASES = {
    "legged_policy": "phygs.execution.locomotion.controller:LocomotionController",
    "planar_joints": "phygs.execution.locomotion.planar:PlanarJointBase",
}


def base_controller_class(robot: RobotDescription) -> type[BaseController]:
    """The class of the robot's base controller (``base.kind``): it builds itself with
    ``from_robot(robot, articulation, *, physics_dt, source)`` and names the files its motion depends
    on with ``files(robot)``."""
    kind = robot.base.kind
    if kind not in BASES and ":" not in kind:
        raise ValueError(
            f"{robot.name}: unknown base kind {kind!r}; expected one of {sorted(BASES)} or 'module:Class'."
        )
    return robot.plugin(BASES.get(kind, kind))


def base_controller(robot: RobotDescription, articulation, *, physics_dt: float, source: RunMode) -> BaseController:
    """The robot's base controller, built on ``articulation``."""
    return base_controller_class(robot).from_robot(robot, articulation, physics_dt=physics_dt, source=source)


@dataclass
class RobotControl:
    """World-side setup with small locomotion and arm clients for agents."""

    runtime: EpisodeRuntime
    locomotion: locomotion_api.LocomotionClient
    arm: ArmClient
    robot: RobotDescription
    navigation: NavigationClient | None = None
    pipeline: ObservationPipeline | None = None
    manipulation: ManipulationClient | None = None
    # Skill steps with the episode time: printed, and kept for reports and video overlays.
    narration: Narration | None = None
    # Every command issued to the clients, as spans for video timelines.
    timeline: CommandTimeline | None = None

    @classmethod
    def create(
        cls,
        sim: SimulationContext,
        scene: InteractiveScene,
        robot: RobotDescription,
        *,
        mode: RunMode,
        observations: ObservationConfig | None = None,
        pipeline_config: PipelineConfig | None = None,
        processing_services: tuple[ProcessingService, ...] = (),
        observation_consumers: tuple[ServiceDeclaration, ...] = (),
        localization_estimator: LocalizationEstimator | None = None,
        navigation_config: NavigationConfig | None = None,
        navigation_backend_factory: NavigationBackendFactory | None = None,
        manipulation: bool = False,
        manipulation_pose_planner: Callable[
            [tuple[float, ...], HeldObject | None], JointTrajectory | Future[JointTrajectory]
        ]
        | None = None,
        manipulation_joint_planner: Callable[
            [list[float], HeldObject | None], JointTrajectory | Future[JointTrajectory]
        ]
        | None = None,
        allowed_actions: tuple[str, ...] = ("motion", "navigation", "arm"),
    ) -> RobotControl:
        """Bind ``robot`` (its description) to one initialized scene: its base controller, the arm
        posture client and, with ``manipulation``, the arm and gripper controller. The arm and the
        gripper start (and every reset returns them) at their configured start, ``arm.stow`` and
        ``initial_joints``: Isaac Lab spawns joints at the USD's own positions."""
        from phygs.sim.state import IsaacLabWorld

        if not allowed_actions or not set(allowed_actions) <= {"motion", "navigation", "arm", "manipulation"}:
            raise ValueError("Unsupported action interfaces.")
        if pipeline_config is not None:
            if observations is not None:
                raise ValueError("Choose pipeline_config or observations, not both.")
            observations = pipeline_config.capture
        factory = navigation_backend_factory
        if navigation_config is not None and factory is None:
            raise ValueError("Navigation configuration requires a navigation stack.")
        consumers = observation_consumers + (() if factory is None else (factory.declaration,))
        if observations is None and (consumers or processing_services or localization_estimator is not None):
            raise ValueError("Configured services require an observation pipeline.")
        if observations is not None:
            if pipeline_config is None:
                # Legacy trusted setup selected all captured products for the agent.
                access = ObservationAccess.from_config(observations)
                pipeline_config = PipelineConfig(
                    observations,
                    access,
                    access,
                    (),
                )
            ObservationPipeline.validate(
                pipeline_config,
                processors=processing_services,
                consumers=consumers,
                localization=localization_estimator,
            )
        articulation = scene["robot"]
        start = [articulation.joint_names.index(name) for name in (*robot.arm.joints, *robot.gripper.joints)]
        articulation.write_joint_position_to_sim_index(
            position=articulation.data.default_joint_pos.torch[:, start], joint_ids=start
        )
        scene.update(0.0)  # The world below snapshots this state for every reset.
        # A walking policy keeps its trained period at the scene's timestep, or rejects it.
        locomotion = base_controller(robot, articulation, physics_dt=sim.get_physics_dt(), source=mode)
        arm_names, gripper = robot.arm.joints, robot.gripper.joints
        posture = PostureController(arm_names, source=mode)
        from phygs.sim.fabric import FabricTransformWrites

        cameras = observations is not None and bool(observations.cameras)
        fabric_writes = FabricTransformWrites() if FabricTransformWrites.applies(sim, cameras) else None
        world = IsaacLabWorld(
            sim,
            scene,
            physics_sensors=("observation_imu",) if observations is not None and observations.imu else (),
            render_during_step=not cameras,
            fabric_writes=fabric_writes,
            gravity_compensation_joints=arm_names + gripper if manipulation else (),
        )
        observation_source = None
        pipeline = None
        if observations is not None:
            from phygs.sim.base import BaseLink
            from phygs.sim.observations import IsaacLabObservations

            base = BaseLink(articulation, robot.base_link)
            observation_source = IsaacLabObservations(
                scene, observations, sim.get_physics_dt(), base=base, fabric_writes=fabric_writes
            )
            pipeline = ObservationPipeline(
                observation_source,
                pipeline_config,
                processors=processing_services,
                consumers=consumers,
                localization=localization_estimator,
            )
        if factory is not None and navigation_config is None:
            from phygs.skills.navigation.api import NavigationConfig

            navigation_config = NavigationConfig(arm_posture=robot.arm.transit_posture)
        controller = None
        controllers = (locomotion, posture)
        limits = articulation.data.soft_joint_pos_limits.torch[0].tolist()
        joint_limits = {joint: tuple(limits[articulation.joint_names.index(joint)]) for joint in arm_names + gripper}
        if manipulation:
            from phygs.execution.manipulation.controller import ManipulationController, MotionConfig
            from phygs.sim.arm import WristPoseServo

            controller = ManipulationController(
                arm_names + gripper,
                world,
                joint_limits,
                MotionConfig(**robot.arm.motion),
                source=mode,
                pose_servo=WristPoseServo(articulation, arm_names, robot.arm.wrist_link),
                gripper=robot.gripper.primary,
                gripper_followers=dict(zip(gripper[1:], robot.gripper.ratios[1:], strict=True)),
            )
            controllers += (controller,)
        elif "manipulation" in allowed_actions:
            raise ValueError("Manipulation permission requires manipulation.")
        runtime = EpisodeRuntime(
            world,
            controllers,
            mode=mode,
            reset_targets=locomotion.standing_targets(),
            observations=pipeline,
            configuration={
                "robot": robot.summary(),
                "allowed_actions": allowed_actions,
                "manipulation": manipulation,
                "navigation_backend_factory": type(factory).__qualname__ if factory is not None else None,
                "navigation": asdict(navigation_config) if navigation_config is not None else None,
                "navigation_stack": factory.configuration if factory is not None else None,
            },
        )
        control = cls(
            runtime,
            locomotion_api.LocomotionClient(runtime, locomotion, enabled="motion" in allowed_actions),
            ArmClient(
                runtime, posture, world, {j: joint_limits[j] for j in arm_names}, enabled="arm" in allowed_actions
            ),
            robot,
            pipeline=pipeline,
        )
        if controller is not None:
            from phygs.skills.manipulation.api import ManipulationClient

            control.manipulation = ManipulationClient(
                runtime,
                controller,
                robot,
                enabled="manipulation" in allowed_actions,
                pose_planner=manipulation_pose_planner,
                joint_planner=manipulation_joint_planner,
            )
        if factory is not None:
            from phygs.skills.navigation.api import NavigationClient

            control.navigation = NavigationClient(
                runtime,
                control.locomotion,
                navigation_config,
                enabled="navigation" in allowed_actions,
                backend_factory=factory,
                observations=pipeline.view(factory.declaration.name),
            )
            control.arm.navigation = control.navigation
            if control.manipulation is not None:
                control.manipulation.navigation = control.navigation
            control.locomotion.bind_navigation(control.navigation)
        control.narration = Narration(lambda: runtime.time_seconds).attach()
        control.timeline = CommandTimeline(lambda: runtime.time_seconds)
        control.timeline.watch(control)
        return control

    def reset(self, *, settling_seconds: float = 1.0) -> None:
        """Restore an episode and settle in the standing posture for duration [s]."""
        ticks = duration_ticks(settling_seconds, self.runtime.physics_dt)
        if self.navigation is not None:
            self.navigation.reset()
        self.locomotion._end_episode()
        self.arm._end_episode()
        if self.manipulation is not None:
            self.manipulation._end_episode()
        self.runtime.reset(settling_ticks=ticks)
        if self.timeline is not None:
            self.timeline.clear()

    def observe(self) -> ObservationPacket:
        """Return timestamped, permitted observations without advancing simulation."""
        return self.runtime.observe()

    def product(self, name: str) -> Any:
        """Read a configured, agent-visible derived product in its native representation."""
        if self.pipeline is None:
            raise RuntimeError("No observation pipeline is configured.")
        return self.pipeline.product(name)

    def step(self) -> None:
        """Advance the environment by one physics tick; agents do not call this per command."""
        try:
            if self.navigation is not None:
                self.navigation.update()
        except Exception as exc:
            if self.runtime.status == EpisodeStatus.RUNNING:
                self.runtime.fail(str(exc))
            raise
        self.runtime.step()
        if self.manipulation is not None:
            self.manipulation.update()
        if self.timeline is not None:
            self.timeline.update()

    def close(self) -> None:
        """Release optional backend connections and stop any active navigation."""
        if self.narration is not None:
            self.narration.detach()
        if self.navigation is not None:
            self.navigation.close()
