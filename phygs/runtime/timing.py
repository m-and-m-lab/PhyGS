# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Exact conversion between command durations and runtime clock ticks."""

import math


def duration_ticks(seconds: float, dt: float) -> int:
    """Convert a positive duration [s] to exact physics ticks of length [s]."""
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("Duration must be positive and finite.")
    ticks = round(seconds / dt)
    if ticks < 1 or not math.isclose(ticks * dt, seconds, rel_tol=0, abs_tol=1e-10):
        raise ValueError(f"Duration {seconds} must be an integer multiple of physics_dt={dt}.")
    return ticks
