# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Backend contract for complete navigation operations, independent of agent code."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from phygs.execution.navigation.types import NavigationConfig, NavigationFeedback
from phygs.observations.types import ObservationReader
from phygs.runtime.pipeline import ServiceDeclaration

if TYPE_CHECKING:
    from phygs.runtime.episode import EpisodeRuntime
    from phygs.skills.locomotion.api import LocomotionClient


class NavigationBackend(Protocol):
    """Own navigation execution; never step physics or bypass runtime command authority.

    Goals use the configured fixed reference frame [m, rad]. IDs are unique across resets. Reset must
    cancel external work and discard stale commands before another episode starts.
    Update must stop external execution when the episode is no longer running.
    """

    @property
    def configuration(self) -> dict: ...
    @property
    def active(self) -> bool: ...
    @property
    def active_command(self) -> int | None: ...
    def navigate_to(
        self,
        x: float,
        y: float,
        heading: float | None = None,
        *,
        position_tolerance: float = 0.04,
        heading_tolerance: float = 0.04,
        timeout: float | None = None,
    ) -> int: ...
    def feedback(self, command_id: int) -> NavigationFeedback: ...
    def cancel(self, command_id: int) -> None: ...
    def reset(self) -> None: ...
    def update(self) -> None: ...
    def close(self) -> None: ...


@dataclass(frozen=True)
class NavigationContext:
    """World-side execution and permitted observations; no mandatory map type."""

    runtime: EpisodeRuntime
    locomotion: LocomotionClient
    observations: ObservationReader
    config: NavigationConfig


class NavigationBackendFactory(Protocol):
    """Own stack-specific dependencies and declare all indirect information access."""

    @property
    def configuration(self) -> dict: ...

    @property
    def declaration(self) -> ServiceDeclaration: ...

    def __call__(self, context: NavigationContext) -> NavigationBackend: ...
