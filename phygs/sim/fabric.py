# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PhysX pose writes to Fabric, kept only for the steps that rendering reads."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.sim import SimulationContext


class FabricTransformWrites:
    """Switch PhysX's per-step body pose writes to Fabric on or off.

    Only rendering reads these poses. With cameras and no viewport or visualizer, rendering
    happens only at camera capture ticks, but PhysX still writes every body's pose each step.
    The observation source keeps the writes for the :attr:`STEPS_BEFORE_CAPTURE` steps before
    each capture; the world keeps them on from restore through settling. Physics does not read
    Fabric.
    """

    SETTING = "/physics/fabricUpdateTransformations"
    STEPS_BEFORE_CAPTURE = 2
    """Written steps per capture. A child prim of a body, such as a mounted camera, follows the
    body's write one step late; with one written step its pose was one capture old."""

    def __init__(self) -> None:
        import carb

        self._settings = carb.settings.get_settings()
        self._enabled = bool(self._settings.get(self.SETTING))

    @staticmethod
    def applies(sim: SimulationContext, cameras: bool) -> bool:
        """Return whether only camera captures render: cameras, no GUI and no visualizer."""
        return cameras and not sim.has_gui and not sim.has_active_visualizers()

    @property
    def enabled(self) -> bool:
        return self._enabled

    @classmethod
    def before_capture(cls, tick: int, capture_ticks: int) -> bool:
        """Return whether the step after ``tick`` is one of the last before a capture tick."""
        return (tick + cls.STEPS_BEFORE_CAPTURE) % capture_ticks < cls.STEPS_BEFORE_CAPTURE

    def set(self, enabled: bool) -> None:
        """Write body poses to Fabric in the following physics steps, or not."""
        if enabled != self._enabled:
            self._settings.set_bool(self.SETTING, enabled)
            self._enabled = enabled
