# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared room and circulation dimensions [m] for the revised Bear Loop house."""

BATHROOMS = (
    dict(name="primary", level=1, bounds=(3.6, 1.7, 6.9, 4.65), vanity=(5.05, 4.2), toilet=(4.2, 4.2)),
    dict(name="guest", level=1, bounds=(12.35, 3.1, 14.45, 4.85), vanity=(13.8, 4.45), toilet=(12.85, 4.4)),
    dict(name="east", level=1, bounds=(19.2, 1.95, 20.9, 3.9), vanity=(20.35, 3.45), toilet=(19.65, 3.45)),
    dict(name="ground", level=0, bounds=(18.7, 4.1, 21.65, 5.7), vanity=(20.85, 5.25), toilet=(19.85, 5.25)),
)

# Walking strips are kept clear even while furniture is closed. Doorways are tested separately.
WALKING_STRIPS = (
    ("foyer_west_of_stairs", 0, (6.87, 1.05, 7.82, 4.60)),
    ("foyer_east_of_stairs", 0, (10.80, 0.9, 11.65, 5.65)),
    ("upper_stair_hall", 1, (10.80, 0.9, 12.20, 5.65)),
    ("upper_cross_hall", 1, (11.75, 4.95, 16.30, 6.00)),
    ("pool_to_bunk", 1, (19.90, 6.35, 20.75, 8.55)),
)
