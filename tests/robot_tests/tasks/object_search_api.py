# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Small agent-facing endpoint for grounded object-search declarations."""

from __future__ import annotations

from collections.abc import Callable

from phygs_robot_tests.tasks.object_search import TargetDeclaration

from phygs.observations.types import Stamp


class ObjectSearchClient:
    """Submit a target pixel from a permitted camera capture."""

    def __init__(self, submit: Callable[[TargetDeclaration], str], *, camera: str) -> None:
        if not camera:
            raise ValueError("Object-search client requires a camera name.")
        self._submit = submit
        self._camera = camera

    def declare_target(self, stamp: Stamp, pixel_x: int, pixel_y: int) -> str:
        """Declare a target pixel from one exact camera capture."""
        return self._submit(TargetDeclaration(self._camera, stamp, pixel_x, pixel_y))
