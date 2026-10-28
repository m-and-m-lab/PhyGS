# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The Ridgeback-Franka's actuator groups (``robot.yaml`` ``actuators``): Isaac Lab's own configuration
(velocity-driven base joints, implicit position drives on the arm and the fingers), with the plain
Franka's finger gains."""

from __future__ import annotations

import copy


def actuator_cfgs(physics_dt: float) -> dict:
    """Isaac Lab's ``RIDGEBACK_FRANKA_PANDA_CFG`` actuators, the fingers with ``FRANKA_PANDA_CFG``'s
    gains (2e3 N/m, 1e2 N·s/m): at the Ridgeback configuration's 1e5 N/m a closing finger reached its
    200 N limit and shot a 4 cm box out of the grasp (table_reach, 2026-10-28). Implicit drives do
    not depend on ``physics_dt`` [s]."""
    from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG
    from isaaclab_assets.robots.ridgeback_franka import RIDGEBACK_FRANKA_PANDA_CFG

    actuators = copy.deepcopy(RIDGEBACK_FRANKA_PANDA_CFG.actuators)
    actuators["panda_hand"] = copy.deepcopy(FRANKA_PANDA_CFG.actuators["panda_hand"])
    return actuators
