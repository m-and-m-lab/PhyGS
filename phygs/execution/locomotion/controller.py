# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The ``legged_policy`` base: a learned walking policy owns the legs at runtime.

The policy wrapper is the robot's (``robot.yaml`` ``base.policy``, e.g. Spot's
:class:`~phygs.robots.spot.policy.SpotFlatTerrainPolicy`); it implements
:class:`WalkingPolicy`. This controller adds the command shaping (:mod:`.base`) and the stop.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import torch
import yaml

from phygs.execution.locomotion.base import DeadBand, Deceleration
from phygs.runtime.episode import ControllerDeclaration, JointTargets, RunMode
from phygs.runtime.timing import duration_ticks

if TYPE_CHECKING:
    from phygs.robots.description import RobotDescription


class WalkingPolicy(Protocol):
    """A walking policy wrapper: leg targets [rad] from a body velocity command [m/s, m/s, rad/s],
    computed without robot writes, and the gains of its standing and walking modes."""

    leg_joint_names: tuple[str, ...]
    control_dt: float
    stand_switch_seconds: float
    standing: bool
    stand_pose: torch.Tensor

    def initialize(self, robot) -> None: ...
    def reset(self) -> None: ...
    def compute_targets(self, dt: float, command: tuple[float, float, float]) -> torch.Tensor: ...
    def gain_targets(self) -> tuple[dict[str, float], dict[str, float]]: ...
    def stand_targets(self) -> torch.Tensor: ...


class LocomotionController:
    """Own the legs while preserving the exported policy's preprocessing and timing."""

    def __init__(
        self,
        policy: WalkingPolicy,
        *,
        dead_band: DeadBand,
        deceleration: Deceleration,
        stand_seconds: float,
        kinematics: str = "holonomic",
        lowest_speed_moves: dict | None = None,
        source: RunMode = RunMode.AUTONOMOUS,
    ) -> None:
        """``stand_seconds`` [s]: the stop interval after the command has slowed to zero, longer than
        the policy's latest stand switch. ``kinematics`` and ``lowest_speed_moves``: see
        :class:`~phygs.execution.locomotion.base.BaseController`."""
        if not stand_seconds > policy.stand_switch_seconds:
            raise ValueError(
                f"The stop must outlast the latest stand switch ({policy.stand_switch_seconds} s),"
                f" got {stand_seconds} s."
            )
        self.policy = policy
        self.dead_band = dead_band
        self.deceleration = deceleration
        self.kinematics = kinematics
        self.lowest_speed_moves = lowest_speed_moves
        self.declaration = ControllerDeclaration(
            "locomotion", policy.leg_joint_names, source, stop_ticks=duration_ticks(stand_seconds, policy.control_dt)
        )
        self.command = (0.0, 0.0, 0.0)
        self._limited = (0.0, 0.0, 0.0)
        self._gain_mode: bool | None = None

    @classmethod
    def load(
        cls,
        policy: WalkingPolicy,
        config_path: str | Path,
        *,
        kinematics: str = "holonomic",
        source: RunMode = RunMode.AUTONOMOUS,
    ) -> LocomotionController:
        """The controller with the dead band, deceleration, stop interval and lowest-speed moves of
        a locomotion settings file (``command_dead_band``, ``command_deceleration``,
        ``stop_stand_seconds``, optional ``lowest_speed_moves``)."""
        config = yaml.safe_load(Path(config_path).read_text())
        return cls(
            policy,
            dead_band=DeadBand.from_config(config["command_dead_band"]),
            deceleration=Deceleration.load(config_path),
            stand_seconds=float(config["stop_stand_seconds"]),
            kinematics=kinematics,
            lowest_speed_moves=config.get("lowest_speed_moves"),
            source=source,
        )

    @staticmethod
    def files(robot: RobotDescription) -> dict[str, Path]:
        """The files the walking depends on, by provenance key: the policy and its training configuration."""
        config = yaml.safe_load(robot.base.config.read_text())
        folder = robot.base.config.parent
        return {
            "policy_sha256": (folder / config["policy_path"]).resolve(strict=True),
            "policy_config_sha256": (folder / config["env_config_path"]).resolve(strict=True),
        }

    @classmethod
    def from_robot(
        cls, robot: RobotDescription, articulation, *, physics_dt: float, source: RunMode = RunMode.AUTONOMOUS
    ) -> LocomotionController:
        """The robot's walking policy (``base.policy``, settings ``base.config``) on ``articulation``."""
        if robot.base.config is None or "policy" not in robot.base.options:
            raise ValueError(f"{robot.name}: a legged_policy base needs base.policy and base.config.")
        policy = robot.plugin(robot.base.options["policy"])(robot.base.config, physics_dt=physics_dt)
        policy.initialize(articulation)
        return cls.load(policy, robot.base.config, kinematics=robot.base.kinematics, source=source)

    @property
    def sent_command(self) -> tuple[float, float, float]:
        """The commanded velocity [m/s, m/s, rad/s] after the dead band. The policy receives it
        once slowing down to it at the deceleration limit has finished (:attr:`applied_command`)."""
        return self.dead_band.apply(self.command)

    @property
    def applied_command(self) -> tuple[float, float, float]:
        """The command [m/s, m/s, rad/s] the policy received at the last step."""
        return self._limited

    def reset(self, positions: dict[str, float]) -> None:
        """Clear velocity commands and policy history; measured positions are in [rad]."""
        self.policy.reset()
        self.command = (0.0, 0.0, 0.0)
        self._limited = (0.0, 0.0, 0.0)
        self._gain_mode = None

    def stop(self) -> None:
        """Request zero velocity, retaining policy control during deceleration."""
        self.command = (0.0, 0.0, 0.0)

    def stop_duration_ticks(self) -> int:
        """Ticks to slow the policy command to zero at the deceleration limit, then ``stand_seconds``."""
        ramp = math.ceil(self.deceleration.ramp_seconds(self._limited) / self.policy.control_dt - 1e-9)
        return ramp + self.declaration.stop_ticks

    def step(self, dt: float) -> JointTargets:
        """Compute leg targets [rad] at physics interval [s], without robot writes."""
        self._limited = self.deceleration.apply(self._limited, self.sent_command, dt)
        values = self.policy.compute_targets(dt, self._limited)[0].tolist()
        positions = dict(zip(self.declaration.joints, values, strict=True))
        if self._gain_mode != self.policy.standing:
            self._gain_mode = self.policy.standing
            return self._with_gains(positions)
        return JointTargets(positions)

    def _with_gains(self, positions: dict[str, float]) -> JointTargets:
        """Leg targets [rad] with the gains of the policy's mode: PhysX drives while standing."""
        stiffness, damping = self.policy.gain_targets()
        if self.policy.standing:
            return JointTargets(positions, drive_stiffness=stiffness, drive_damping=damping)
        return JointTargets(positions, stiffness, damping)

    def hold_targets(self) -> dict[str, float]:
        """Leg setpoints [rad] the runtime keeps on release: the policy's standing targets while it
        stands, else none (the runtime then holds the measured positions).

        Holding the measured, load-deflected positions dropped the stance preload: the base sank by
        about load / stiffness after each stop (median 1.7 cm with 400 N·m/rad standing gains) and
        tipped after some left turns at 250 Hz (2026-10-16,
        output_check/20261016-2119_turn_tip_check).
        """
        if not self.policy.standing:
            return {}
        return dict(zip(self.declaration.joints, self.policy.stand_pose[0].tolist(), strict=True))

    def standing_targets(self) -> JointTargets:
        """Return the declared reset posture [rad] and standing gains."""
        values = self.policy.stand_targets()[0].tolist()
        return self._with_gains(dict(zip(self.declaration.joints, values, strict=True)))
