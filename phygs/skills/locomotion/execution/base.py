# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The base controller contract and the command shaping every base shares.

A base controller moves the robot base from a body velocity command (v_x [m/s], v_y [m/s], v_rot
[rad/s]): a walking policy on the legs, or velocity targets on wheels or planar joints. The
locomotion client sets :attr:`BaseController.command`; the runtime steps the controller like any
other. A robot's ``robot.yaml`` names its controller kind (``base.kind``); every controller class
builds itself with ``from_robot(robot, articulation, *, physics_dt, source)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import yaml

from phygs.runtime.episode import ControllerDeclaration, JointTargets


@dataclass(frozen=True)
class DeadBand:
    """Velocity command dead band per axis, in the order (v_x [m/s], v_y [m/s], v_rot [rad/s]).

    A walking policy keeps standing for small commands. A command whose magnitude is at most
    ``zero`` is sent as 0, so the robot stands; one between ``zero`` and the smallest speed that
    moves in its direction (``lowest_positive``, or ``lowest_negative`` as a magnitude) is raised to
    that speed. Larger commands pass unchanged. A base without a dead band has all zeros.
    """

    zero: tuple[float, float, float]
    lowest_negative: tuple[float, float, float]
    lowest_positive: tuple[float, float, float]

    def __post_init__(self) -> None:
        for values in (self.zero, self.lowest_negative, self.lowest_positive):
            if len(values) != 3 or not all(math.isfinite(v) and v >= 0 for v in values):
                raise ValueError("Dead band values must be three finite, nonnegative numbers.")
        if any(z > min(n, p) for z, n, p in zip(self.zero, self.lowest_negative, self.lowest_positive)):
            raise ValueError("The zero band must not exceed the lowest walking speeds.")

    @classmethod
    def from_config(cls, band: dict) -> DeadBand:
        """From a ``command_dead_band`` section: ``{v_x: {zero, negative, positive}, v_y: ..., v_rot: ...}``."""
        axes = ("v_x", "v_y", "v_rot")
        return cls(
            tuple(float(band[a]["zero"]) for a in axes),
            tuple(float(band[a]["negative"]) for a in axes),
            tuple(float(band[a]["positive"]) for a in axes),
        )

    @classmethod
    def load(cls, path: str | Path) -> DeadBand:
        """The ``command_dead_band`` of a base settings file."""
        return cls.from_config(yaml.safe_load(Path(path).read_text())["command_dead_band"])

    def apply(self, command: tuple[float, float, float]) -> tuple[float, float, float]:
        """The command [m/s, m/s, rad/s] the base receives."""
        sent = []
        for value, zero, negative, positive in zip(command, self.zero, self.lowest_negative, self.lowest_positive):
            if abs(value) <= zero:
                value = 0.0
            elif value > 0:
                value = max(value, positive)
            else:
                value = min(value, -negative)
            sent.append(value)
        return tuple(sent)


@dataclass(frozen=True)
class Deceleration:
    """Slowing limit of the base command per axis, in the order (v_x [m/s²], v_y [m/s²],
    v_rot [rad/s²]). Speeding up is not limited; slowing down, a stop included, ramps the
    command at these rates instead of jumping."""

    rates: tuple[float, float, float]

    def __post_init__(self) -> None:
        if len(self.rates) != 3 or not all(math.isfinite(v) and v > 0 for v in self.rates):
            raise ValueError("Deceleration rates must be three finite, positive numbers.")

    @classmethod
    def load(cls, path: str | Path) -> Deceleration:
        """The ``command_deceleration`` of a base settings file."""
        rates = yaml.safe_load(Path(path).read_text())["command_deceleration"]
        return cls(tuple(float(rates[a]) for a in ("v_x", "v_y", "v_rot")))

    def apply(
        self, current: tuple[float, float, float], target: tuple[float, float, float], dt: float
    ) -> tuple[float, float, float]:
        """The command [m/s, m/s, rad/s] one interval dt [s] from ``current`` toward ``target``."""
        out = []
        for now, goal, rate in zip(current, target, self.rates):
            slowing = abs(goal) < abs(now) or goal * now < 0
            step = rate * dt
            out.append(now + math.copysign(step, goal - now) if slowing and abs(goal - now) > step else goal)
        return tuple(out)

    def ramp_seconds(self, current: tuple[float, float, float]) -> float:
        """Time [s] to slow ``current`` [m/s, m/s, rad/s] to zero."""
        return max(abs(value) / rate for value, rate in zip(current, self.rates))


class BaseController(Protocol):
    """Moves the base from a body velocity command; owns the base joints.

    ``kinematics`` is ``holonomic`` (v_x, v_y, v_rot) or ``differential`` (v_y must be 0).
    ``command`` is the latest body velocity [m/s, m/s, rad/s] asked; ``sent_command`` the same after
    the :attr:`dead_band`; ``applied_command`` what the base executed at the last step (after the
    deceleration limit). ``lowest_speed_moves`` is the measured distance [m] covered from rest at the
    lowest speed per direction and duration (``{seconds, forward, backward, left, right}``), which
    navigation's fine approach sizes its moves with, or None when the base moves as commanded.
    ``standing_targets`` is the posture the runtime settles at reset; ``hold_targets`` the set-points
    the runtime keeps on release (empty: hold the measured joints).
    """

    declaration: ControllerDeclaration
    kinematics: str
    command: tuple[float, float, float]
    dead_band: DeadBand
    lowest_speed_moves: dict | None

    @property
    def sent_command(self) -> tuple[float, float, float]: ...
    @property
    def applied_command(self) -> tuple[float, float, float]: ...
    def reset(self, positions: dict[str, float]) -> None: ...
    def step(self, dt: float) -> JointTargets: ...
    def stop(self) -> None: ...
    def stop_duration_ticks(self) -> int: ...
    def hold_targets(self) -> dict[str, float]: ...
    def standing_targets(self) -> JointTargets: ...
