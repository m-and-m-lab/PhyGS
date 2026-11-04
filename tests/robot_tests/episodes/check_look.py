# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""look_at's sweep: open the drawer the target starts in (the composite skill, from its stance) or,
with the target on a table, stay where the robot stands; then look at the target from each side of
look_at's sweep. Each side reached keeps an observation: enough sides must be reached, the look
camera must see the target with depth in at least one of them, and the arm ends stowed.

The target's box is read through the target hint the spec grants (the plan runs on, so ground truth
cannot be read then); what a view sees is measured on that view's own observation.

    ./isaaclab.sh -p scripts/PhyGS/tests/robot_tests/episodes/check_look.py \
        --spec scripts/PhyGS/tests/robot_tests/configs/episodes/dev_house_drawer.yaml \
        --output scripts/PhyGS/output_check/$(date +%Y%m%d-%H%M)_look_house
"""

from __future__ import annotations

import numpy as np
from phygs_robot_tests.episodes import episode

from phygs.composite import CompositeSkills


def samples(center, size) -> np.ndarray:
    """13 points on a box (center, size) [m]: a 3x3 grid on its top, the centres of its 4 sides."""
    c, s = np.asarray(center, dtype=float), np.asarray(size, dtype=float)
    top = [c + [dx * s[0] * 0.4, dy * s[1] * 0.4, s[2] / 2] for dx in (-1, 0, 1) for dy in (-1, 0, 1)]
    sides = [c + [sx * s[0] / 2, sy * s[1] / 2, 0.0] for sx, sy in ((1, 0), (-1, 0), (0, 1), (0, -1))]
    return np.array(top + sides)


def seen_points(observation, camera: str, points: np.ndarray, tolerance_m: float) -> int:
    """How many ``points`` [m] the camera's depth in ``observation`` returns within ``tolerance_m``."""
    found = observation.cameras[camera]
    depth, valid = found.depth.cpu().numpy(), found.depth_valid.cpu().numpy()
    camera_from_world = np.linalg.inv(observation.transform("world", found.frame, found.stamp).cpu().numpy())
    local = (camera_from_world @ np.column_stack((points, np.ones(len(points)))).T).T[:, :3]
    pixels = (found.intrinsics.cpu().numpy() @ (local / local[:, 2:3]).T).T
    count = 0
    for (u, v, _), z in zip(pixels, local[:, 2], strict=True):
        col, row = int(round(u)), int(round(v))
        if z > 0 and 0 <= row < depth.shape[0] and 0 <= col < depth.shape[1]:
            count += bool(valid[row, col] and abs(depth[row, col] - z) <= tolerance_m)
    return count


def plan(session, seen: dict) -> None:
    """Open the target's drawer if it is in one, then look at the target; ``seen`` keeps the views
    reached and the target's points each one sees."""
    skills = CompositeSkills(session)
    target = session.hints.target()
    if target.inside is not None:
        skills.open_drawer(session.hints.part(target.inside))
        target = session.hints.target()  # The open drawer moved it.
    looked = skills.look_at(target.center)
    points, camera = samples(target.center, target.size), session.robot.look_camera
    tolerance = episode.SETTINGS["look_depth_tolerance_m"]
    seen["views"] = {str(look.view): seen_points(look.observation, camera, points, tolerance) for look in skills.looks}
    seen["out_of_reach"] = looked.measured["out_of_reach"]


def main() -> None:
    check, args, spec = episode.start(__doc__, "dev_house_drawer")
    settings = episode.SETTINGS

    def body() -> None:
        seen = {}
        with episode.world(args, spec) as world:
            report = world.run(lambda session: plan(session, seen), name="check_look.plan")
            episode.agent_checks(check, report)
            views = seen.get("views", {})
            least = settings["look_min_views"]
            check.measure("views_reached", len(views), len(views) >= least, f"fewer than {least} views reached")
            best = max(views.values(), default=0)
            least = settings["look_min_points"]
            check.measure("target_points_seen", views, best >= least, f"no view saw {least} of the target's 13 points")
            stowed = world.truth().stow_error_rad()
            check.measure("stow_error_rad", round(stowed, 4), stowed <= settings["stow_tolerance_rad"], "not stowed")

    check.run(body)


if __name__ == "__main__":
    main()
