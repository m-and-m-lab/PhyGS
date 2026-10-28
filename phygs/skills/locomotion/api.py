# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Public locomotion commands backed by runtime ownership and the robot's base controller."""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

from phygs.runtime.commands import CommandClient
from phygs.runtime.commands import CommandFeedback as CommandFeedback
from phygs.runtime.episode import EpisodeRuntime
from phygs.runtime.timing import duration_ticks

if TYPE_CHECKING:
    from phygs.execution.locomotion.base import BaseController
    from phygs.skills.navigation.api import NavigationClient

__all__ = ["LocomotionClient", "CommandFeedback"]

# Locomotion step narration (see phygs.tools.narration).
STEPS = logging.getLogger("phygs.skills.locomotion")


class LocomotionClient(CommandClient):
    """Submit body-frame velocities; runtime owns expiry and physics stepping."""

    def __init__(self, runtime: EpisodeRuntime, controller: BaseController, *, enabled: bool = True) -> None:
        super().__init__(runtime, controller.declaration.name)
        self._controller = controller
        self._enabled = enabled
        self._navigation: NavigationClient | None = None
        self._announced: tuple[float, float, float] | None = None  # The last velocity narrated.

    @property
    def controller(self) -> BaseController:
        """Trusted diagnostics for robot tests (environment-only; agents cannot reach it): the base
        controller (a walking one has its ``policy``)."""
        return self._controller

    def bind_navigation(self, navigation: NavigationClient) -> None:
        """Bind the navigation service during environment assembly.

        Direct motion remains unavailable until navigation finishes stopping.
        A client can be bound only once.
        """
        if self._navigation is not None:
            raise RuntimeError("Navigation is already bound to locomotion.")
        self._navigation = navigation

    def command_velocity(self, v_x: float, v_y: float, v_rot: float, *, duration: float = 0.5) -> int:
        """Command body velocity [m/s, m/s, rad/s] for duration [s].

        Updates preserve policy memory. Expiry decelerates before releasing the base. The robot's
        command dead band applies per axis (``command_dead_band`` in its base settings): a small
        command is sent as 0 and the robot stands; one below the lowest walking speed in its direction
        is raised to that speed. A differential base refuses ``v_y`` other than 0.
        """
        if not self._enabled:
            raise PermissionError("Direct motion is not permitted by this task.")
        self._require_direct_control()
        command_id = self._command_velocity(v_x, v_y, v_rot, duration=duration)
        command = (float(v_x), float(v_y), float(v_rot))
        if command != self._announced:  # Repeated updates of one velocity are not news.
            self._announced = command
            sent = self._controller.sent_command
            shaped = "" if sent == command else " (dead band: sent as {:.2f}, {:.2f}, {:.2f})".format(*sent)
            STEPS.info(
                f"command_velocity: v_x {v_x:.2f} m/s, v_y {v_y:.2f} m/s, v_rot {v_rot:.2f} rad/s for"
                f" {duration:.2f} s{shaped}; then slowing down"
            )
        return command_id

    def _command_velocity(self, v_x: float, v_y: float, v_rot: float, *, duration: float) -> int:
        command = (float(v_x), float(v_y), float(v_rot))
        if not all(math.isfinite(value) for value in command):
            raise ValueError("Velocity command must be finite.")
        if command[1] != 0.0 and self._controller.kinematics == "differential":
            raise ValueError("This robot's base is differential: it cannot move sideways (v_y must be 0).")
        command_id = self._submit(duration_ticks(duration, self._runtime.physics_dt))
        self._controller.command = command
        return command_id

    def stop(self) -> int:
        """Request controlled stopping and return its command ID."""
        if not self._enabled:
            raise PermissionError("Direct motion is not permitted by this task.")
        self._require_direct_control()
        STEPS.info("stop: controlled stop requested")
        self._announced = None
        return self._stop()

    def _require_direct_control(self) -> None:
        if self._navigation is not None and self._navigation.active:
            raise RuntimeError("Cancel navigation and wait for its controlled stop before direct motion.")

    def _stop(self) -> int:
        command_id = self._command_velocity(0.0, 0.0, 0.0, duration=0.5)
        self._runtime.request_stop(self._name)
        return command_id
