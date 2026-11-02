# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The check before the agent starts: the settled world is the one the spec describes.

After the reset's settle the robot must stand (its base upright, above its fallen height), every
scene body must be at rest where the episode placed it, and every drawer at the opening the spec
declares (closed unless the spec's ``parts`` open it). A failed check is
a setup error: the World reports it (``status: setup_error``) and never starts the agent, so a broken
scene or episode is fixed, not scored. The limits are the ``start_check`` settings
(``world/environment.yaml``).
"""

from __future__ import annotations

LIMITS = ("body_speed_max_mps", "body_shift_max_m", "upright_min_cos", "drawer_open_max_m")
NAMED = 5  # Bodies named in a problem line; the rest are counted.


def check(truth, fallen_height: float | None, limits: dict, openings: dict[str, float] | None = None) -> dict:
    """``{passed, problems, measured}`` for the settled world. ``truth`` is the World's ``Truth``;
    ``fallen_height`` the robot's [m] (None: it cannot fall); ``openings`` the drawers the spec
    opens [m]. A drawer may be ``drawer_open_max_m`` [m] from its opening (0 for the others); None
    leaves drawers unchecked."""
    missing = set(LIMITS) - set(limits)
    if missing:
        raise ValueError(f"start_check lacks {sorted(missing)} (world/environment.yaml).")
    base = truth.base_pose()
    height, up = float(base[2, 3]), float(base[2, 2])
    motion = truth.body_motion()
    declared, openings = dict(openings or {}), truth.part_openings()  # The spec's openings, and the ones now.
    problems = []
    if up < limits["upright_min_cos"]:
        problems.append(f"the robot is tilted: its base's up axis has cos {up:.3f} < {limits['upright_min_cos']}")
    if fallen_height is not None and height < fallen_height:
        problems.append(f"the robot is down: its base is at {height:.3f} m, below {fallen_height} m")
    moving = sorted(name for name, (speed, _) in motion.items() if speed > limits["body_speed_max_mps"])
    if moving:
        problems.append(f"{len(moving)} bodies still move (over {limits['body_speed_max_mps']} m/s): {_names(moving)}")
    moved = sorted(name for name, (_, shift) in motion.items() if shift > limits["body_shift_max_m"])
    if moved:
        problems.append(f"{len(moved)} bodies left their places (over {limits['body_shift_max_m']} m): {_names(moved)}")
    if limits["drawer_open_max_m"] is not None:
        limit = limits["drawer_open_max_m"]
        open_ = sorted(name for name, opening in openings.items() if name not in declared and opening > limit)
        if open_:
            problems.append(f"{len(open_)} drawers are open (over {limit} m): {_names(open_)}")
        off = sorted(name for name, opening in declared.items() if abs(openings.get(name, 0.0) - opening) > limit)
        if off:
            problems.append(f"{len(off)} drawers are not at the spec's opening (off by over {limit} m): {_names(off)}")
    fastest = max(motion, key=lambda name: motion[name][0], default=None)
    farthest = max(motion, key=lambda name: motion[name][1], default=None)
    widest = max(openings, key=openings.get, default=None)
    measured = dict(
        base_height_m=round(height, 3),
        base_up_cos=round(up, 4),
        bodies=len(motion),
        fastest_body=fastest,
        fastest_mps=round(motion[fastest][0], 4) if fastest else 0.0,
        farthest_body=farthest,
        farthest_shift_m=round(motion[farthest][1], 4) if farthest else 0.0,
        widest_drawer=widest,
        widest_opening_m=round(openings[widest], 4) if widest else 0.0,
    )
    return dict(passed=not problems, problems=problems, measured=measured, limits=dict(limits))


def _names(names: list[str]) -> str:
    more = len(names) - NAMED
    return ", ".join(names[:NAMED]) + (f" and {more} more" if more > 0 else "")
