# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The ``planar_joints`` base: a base that moves on three velocity-driven joints."""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from phygs.execution.locomotion.base import DeadBand, Deceleration
from phygs.runtime.episode import ControllerDeclaration, JointTargets, RunMode
from phygs.runtime.timing import duration_ticks

if TYPE_CHECKING:
    from phygs.robots.description import RobotDescription


class PlanarJointBase:
    """A base on three joints under a fixed articulation root: ``x`` and ``y`` prismatic along the
    world axes [m], ``yaw`` revolute about the vertical [rad] (Isaac Lab's Ridgeback-Franka).

    Each step turns the body command [m/s, m/s, rad/s], after the dead band and the deceleration
    limit, into world-frame joint velocity targets at the measured yaw. The joints must be velocity
    driven (zero stiffness, high damping); their position targets are the measured positions.
    The base is holonomic. ``stop_seconds`` [s] is how long a stop keeps the joints after the
    command has slowed to zero.
    """

    def __init__(
        self,
        joints: tuple[str, str, str],
        read: Callable[[], tuple[float, float, float]],
        *,
        dead_band: DeadBand,
        deceleration: Deceleration,
        stop_seconds: float,
        physics_dt: float,
        lowest_speed_moves: dict | None = None,
        source: RunMode = RunMode.AUTONOMOUS,
    ) -> None:
        """``joints``: the x, y and yaw joints; ``read()`` returns their measured positions [m, m, rad]."""
        self.declaration = ControllerDeclaration(
            "locomotion", tuple(joints), source, stop_ticks=duration_ticks(stop_seconds, physics_dt)
        )
        self.kinematics = "holonomic"
        self.dead_band, self.deceleration = dead_band, deceleration
        self.lowest_speed_moves = lowest_speed_moves
        self.command = (0.0, 0.0, 0.0)
        self._limited = (0.0, 0.0, 0.0)
        self._read, self._dt = read, physics_dt

    @staticmethod
    def files(robot: RobotDescription) -> dict[str, Path]:
        """The files the base's motion depends on, by provenance key: its settings file."""
        return {"base_config_sha256": robot.base.config}

    @classmethod
    def from_robot(
        cls, robot: RobotDescription, articulation, *, physics_dt: float, source: RunMode = RunMode.AUTONOMOUS
    ) -> PlanarJointBase:
        """The base of ``robot.yaml``: ``base.joints`` ({x, y, yaw}) and ``base.config`` (the
        ``command_dead_band``, ``command_deceleration``, ``stop_seconds`` and optional
        ``lowest_speed_moves``)."""
        joints = robot.base.options.get("joints")
        if robot.base.config is None or not isinstance(joints, dict) or set(joints) != {"x", "y", "yaw"}:
            raise ValueError(f"{robot.name}: a planar_joints base needs base.joints {{x, y, yaw}} and base.config.")
        names = (joints["x"], joints["y"], joints["yaw"])
        ids, found = articulation.find_joints(list(names), preserve_order=True)
        if tuple(found) != names:
            raise ValueError(f"{robot.name}: base joints {names} resolved as {tuple(found)}.")

        def read() -> tuple[float, float, float]:
            return tuple(articulation.data.joint_pos.torch[0, ids].tolist())

        config = yaml.safe_load(robot.base.config.read_text())
        return cls(
            names,
            read,
            dead_band=DeadBand.from_config(config["command_dead_band"]),
            deceleration=Deceleration.load(robot.base.config),
            stop_seconds=float(config["stop_seconds"]),
            physics_dt=physics_dt,
            lowest_speed_moves=config.get("lowest_speed_moves"),
            source=source,
        )

    @property
    def sent_command(self) -> tuple[float, float, float]:
        """The commanded velocity [m/s, m/s, rad/s] after the dead band."""
        return self.dead_band.apply(self.command)

    @property
    def applied_command(self) -> tuple[float, float, float]:
        """The command [m/s, m/s, rad/s] the joints followed at the last step."""
        return self._limited

    def reset(self, positions: dict[str, float]) -> None:
        self.command = self._limited = (0.0, 0.0, 0.0)

    def stop(self) -> None:
        self.command = (0.0, 0.0, 0.0)

    def stop_duration_ticks(self) -> int:
        """Ticks to slow the command to zero at the deceleration limit, then ``stop_seconds``."""
        ramp = math.ceil(self.deceleration.ramp_seconds(self._limited) / self._dt - 1e-9)
        return ramp + self.declaration.stop_ticks

    def step(self, dt: float) -> JointTargets:
        """World-frame joint velocities [m/s, m/s, rad/s] for the body command at the measured yaw."""
        self._limited = self.deceleration.apply(self._limited, self.sent_command, dt)
        positions = self._read()
        v_x, v_y, v_rot = self._limited
        c, s = math.cos(positions[2]), math.sin(positions[2])
        joints = self.declaration.joints
        velocities = (c * v_x - s * v_y, s * v_x + c * v_y, v_rot)
        return JointTargets(dict(zip(joints, positions)), velocities=dict(zip(joints, velocities)))

    def hold_targets(self) -> dict[str, float]:
        """None: the runtime holds the measured joints at zero velocity."""
        return {}

    def standing_targets(self) -> JointTargets:
        """At reset the base stays where it is, still."""
        return JointTargets(dict(zip(self.declaration.joints, self._read())))
