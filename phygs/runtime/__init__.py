# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Runtime: what owns time and joints in an episode: the integer clock, exclusive joint ownership,
command bookkeeping, reset and emergency stop (:mod:`.episode`, :mod:`.commands`, :mod:`.timing`),
and the observation pipeline (:mod:`.pipeline`)."""
