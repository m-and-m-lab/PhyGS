# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deprecated alias of :mod:`phygs.skills.composite`, where the composite skills moved on 2026-11-05.

Importing ``phygs.composite`` or one of its modules (``skills``, ``tuning``, ``geometry``) still
works and warns; import from ``phygs.skills.composite`` instead. The alias goes in a later release.
"""

import sys
import warnings

from phygs.skills import composite as _composite
from phygs.skills.composite import TUNING, CompositeSkills, Outcome, Tuning, geometry, skills, tuning

__all__ = ["TUNING", "CompositeSkills", "Outcome", "Tuning"]

for _name, _module in (("geometry", geometry), ("skills", skills), ("tuning", tuning)):
    sys.modules[f"{__name__}.{_name}"] = _module  # The old module paths resolve to the moved modules.

warnings.warn(
    "phygs.composite moved to phygs.skills.composite; import it from there.", DeprecationWarning, stacklevel=2
)
del _composite
