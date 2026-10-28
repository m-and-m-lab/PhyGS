# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Independent joint-position interpolation for diagnostic arm control."""

from phygs.runtime.episode import ControllerDeclaration, RunMode


class PostureController:
    """Interpolate from measured joints to a requested posture over a fixed duration."""

    def __init__(self, joints: tuple[str, ...], *, source: RunMode = RunMode.AUTONOMOUS) -> None:
        self.declaration = ControllerDeclaration("arm", joints, source)
        self.initial: dict[str, float] = {}
        self.target: dict[str, float] = {}
        self.elapsed = 0.0
        self.duration = 1.0

    def reset(self, positions: dict[str, float]) -> None:
        """Initialize references from current measured joint angles [rad]."""
        self.initial = positions.copy()
        self.target = positions.copy()
        self.elapsed = 0.0

    def step(self, dt: float) -> dict[str, float]:
        """Return interpolated joint targets [rad] after interval [s]."""
        self.elapsed += dt
        fraction = min(self.elapsed / self.duration, 1.0)
        return {joint: start + fraction * (self.target[joint] - start) for joint, start in self.initial.items()}
