# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Privileged hints: ground truth from the simulation, each call refused without its grant.

Agents reach these through ``session.hints`` (on the main thread, audited). The scene object
supplies the ground truth; it is never given to the agent.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from phygs.episodes.session import ArticulatedPart, TargetHint


class HintSource(Protocol):
    """What a loaded scene supplies to the hints (ground truth now)."""

    def object_box(self, name: str) -> tuple[Any, Any]: ...
    def placement(self, name: str) -> tuple[str | None, str | None]: ...
    def part(self, name: str) -> ArticulatedPart: ...


class PrivilegedHints:
    """``target()`` needs object_targets; ``part()`` needs scene_hints; ``boxes()`` needs known_map."""

    GRANTS = {
        "target": "object_targets",
        "part": "scene_hints",
        "boxes": "known_map",
    }

    def __init__(
        self,
        scene: HintSource,
        target: str | None,
        grants: tuple[str, ...],
        *,
        boxes: Callable[[], dict[str, tuple[Any, Any]]],
    ) -> None:
        self._scene, self._target, self._grants, self._boxes = scene, target, frozenset(grants), boxes

    def grant(self, member: str) -> list[str]:
        """The grant a member reads (for the audit log)."""
        return [self.GRANTS[member]] if member in self.GRANTS else []

    def _require(self, member: str) -> None:
        grant = self.GRANTS[member]
        if grant not in self._grants:
            raise PermissionError(f"hints.{member}() needs the {grant} grant, which this episode does not give.")

    def target(self) -> TargetHint:
        """The target's scene name, its box now, and where it started (``inside`` a part or ``on`` furniture)."""
        self._require("target")
        if self._target is None:
            raise RuntimeError("This episode has no target.")
        center, size = self._scene.object_box(self._target)
        inside, on = self._scene.placement(self._target)
        return TargetHint(self._target, tuple(map(float, center)), tuple(map(float, size)), inside, on)

    def part(self, name: str) -> ArticulatedPart:
        """An articulated part now: handle, opening direction, travel limits, current opening [m]."""
        self._require("part")
        return self._scene.part(name)

    def boxes(self) -> dict[str, tuple[tuple[float, ...], tuple[float, ...]]]:
        """Known obstacle boxes around the robot now (the arm planner's): collider -> (center, size)
        [m], world frame."""
        self._require("boxes")
        return {
            name: (tuple(map(float, center)), tuple(map(float, size))) for name, (center, size) in self._boxes().items()
        }
