# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stable agent-facing navigation API over a selected execution backend."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import TYPE_CHECKING

from phygs.execution.navigation.interfaces import (
    NavigationBackend,
    NavigationBackendFactory,
    NavigationContext,
)
from phygs.execution.navigation.types import NavigationConfig as NavigationConfig
from phygs.execution.navigation.types import NavigationFeedback as NavigationFeedback
from phygs.execution.navigation.types import NavigationGoal as NavigationGoal
from phygs.execution.navigation.types import angle_error as angle_error
from phygs.runtime.episode import EpisodeStatus

if TYPE_CHECKING:
    from phygs.observations.types import ObservationReader
    from phygs.runtime.episode import EpisodeRuntime
    from phygs.skills.locomotion.api import LocomotionClient


__all__ = ["NavigationClient", "NavigationConfig", "NavigationGoal", "NavigationFeedback"]


class NavigationClient:
    """Delegate complete navigation operations while preserving the public API."""

    def __init__(
        self,
        runtime: EpisodeRuntime,
        locomotion: LocomotionClient,
        config: NavigationConfig = NavigationConfig(),
        *,
        enabled: bool = True,
        backend_factory: NavigationBackendFactory,
        observations: ObservationReader | None = None,
    ) -> None:
        self._runtime, self._enabled = runtime, enabled
        self._config = config
        self._observations = observations if observations is not None else runtime
        self._backend: NavigationBackend = backend_factory(
            NavigationContext(runtime, locomotion, self._observations, config)
        )

    @property
    def configuration(self) -> dict:
        """Return selected backend identity and experimental configuration."""
        return self._backend.configuration

    @property
    def active(self) -> bool:
        """Whether navigation is executing or stopping."""
        return self._backend.active

    @property
    def active_command(self) -> int | None:
        """The command executing or stopping now, or None."""
        return self._backend.active_command

    def navigate_to(
        self,
        x: float,
        y: float,
        heading: float | None = None,
        *,
        position_tolerance: float = 0.04,
        heading_tolerance: float = 0.04,
        timeout: float | None = None,
        frame: str | None = None,
    ) -> int:
        """Request position [m], optional heading [rad] and timeout [s].

        ``timeout`` None uses the configured default, which is no time budget: navigation runs
        until it succeeds, fails or is cancelled.

        ``frame`` defaults to the configured stack reference frame. A body-frame
        goal is frozen in that reference at submission time; private map frames
        without an explicit integration are rejected.
        """
        if not self._enabled:
            raise PermissionError("Navigation is not permitted by this task.")
        NavigationGoal(x, y, heading, position_tolerance, heading_tolerance)
        if self._runtime.status != EpisodeStatus.RUNNING:
            raise RuntimeError("Reset the environment before navigating.")
        reference = self._backend.configuration["goal_frame"]
        if frame == "body":
            pose = self._observations.observe().localization
            if pose is None or not pose.valid or pose.pose.target != reference:
                raise ValueError("Body-relative goals require valid localization in the stack frame.")
            if not 0 <= self._runtime.time_seconds - pose.stamp.time_seconds <= self._config.period:
                raise ValueError("Body-relative goal localization is stale.")
            matrix = pose.pose.matrix
            yaw = math.atan2(float(matrix[1, 0]), float(matrix[0, 0]))
            x, y = (
                float(matrix[0, 3]) + math.cos(yaw) * x - math.sin(yaw) * y,
                float(matrix[1, 3]) + math.sin(yaw) * x + math.cos(yaw) * y,
            )
            heading = None if heading is None else angle_error(yaw + heading, 0.0)
        elif frame is not None and frame != reference:
            raise ValueError(f"Unsupported goal frame {frame!r}; expected {reference!r} or 'body'.")
        return self._backend.navigate_to(
            x, y, heading, position_tolerance=position_tolerance, heading_tolerance=heading_tolerance, timeout=timeout
        )

    def explore(self, *, timeout: float | None = None) -> int:
        """Explore unexplored space until no frontier is left, the ``timeout`` [s] ends or the agent
        cancels; one command whose feedback names the frontier being approached.

        The backend picks frontiers (Nav2: explore_lite on its costmap) and walks to each in turn. It
        needs a map prior with unexplored space; a backend without exploration raises
        ``NotImplementedError``.
        """
        if not self._enabled:
            raise PermissionError("Navigation is not permitted by this task.")
        if self._runtime.status != EpisodeStatus.RUNNING:
            raise RuntimeError("Reset the environment before exploring.")
        explore = getattr(self._backend, "explore", None)
        if explore is None:
            raise NotImplementedError("This navigation backend cannot explore.")
        return explore(timeout=timeout)

    def feedback(self, command_id: int) -> NavigationFeedback:
        """Read normalized backend feedback without advancing physics."""
        return replace(self._backend.feedback(command_id), path=())

    @property
    def backend(self) -> NavigationBackend:
        """Trusted diagnostics for robot tests (environment-only; agents cannot reach it): the backend,
        e.g. ``backend.feedback(id)`` with the private planned path."""
        return self._backend

    def cancel(self, command_id: int) -> None:
        """Cancel a goal and request controlled stopping."""
        self._backend.cancel(command_id)

    def reset(self) -> None:
        """Cancel external work and archive old commands before a world reset."""
        self._backend.reset()

    def update(self) -> None:
        """Environment-owned backend tick; agents do not call this method."""
        self._backend.update()

    def close(self) -> None:
        """Terminate active execution and release backend connections."""
        if self._runtime.status == EpisodeStatus.RUNNING and self.active:
            self._runtime.emergency_stop("Navigation client closed")
        self._backend.close()
