# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The fake robot's actuator plugin: a file of the robot's folder, loaded by path."""


def actuator_cfgs(physics_dt: float) -> dict:
    return {"all": dict(physics_dt=physics_dt)}
