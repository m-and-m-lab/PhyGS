# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ROS 2 availability check that imports nothing heavy, so launchers can run it before Kit starts.

Importing NumPy (or torch) before Kit starts leaves OpenBLAS worker threads running when
Kit's telemetry forks, and the fork handler crashes the process (exit 139).
"""

import importlib.util


def require_ros2() -> None:
    """Fail before simulation startup when optional ROS bindings are unavailable."""
    missing = [
        name
        for name in ("rclpy", "nav2_msgs", "tf2_ros", "rosgraph_msgs", "lifecycle_msgs")
        if importlib.util.find_spec(name) is None
    ]
    if missing:
        raise RuntimeError("Nav2 backend requires a sourced ROS 2 Jazzy environment; missing: " + ", ".join(missing))
