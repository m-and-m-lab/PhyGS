# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Table to drawer: open the drawer, pick the target from the table, carry it stowed, set it
upright on the drawer's open floor and push the drawer shut. At the end the target is in the
closed drawer, upright.

It runs under a retrieve spec (dev_grid_table) for its scene and target; the retrieve evaluator does
not describe this task, so its verdict is marked not applicable.

    ./isaaclab.sh -p scripts/PhyGS/tests/robot_tests/episodes/check_drawer_place.py \
        --output scripts/PhyGS/output_check/$(date +%Y%m%d-%H%M)_drawer_place
"""

from __future__ import annotations

import numpy as np
from phygs_robot_tests.episodes import episode

from phygs.composite import CompositeSkills


def plan(session, name: str, seen: dict) -> None:
    """Put the target into the drawer ``name`` (the task names it) and close it; ``seen`` keeps the
    place spot [m]."""
    skills = CompositeSkills(session)
    target = session.hints.target()
    skills.open_drawer(session.hints.part(name))
    skills.pick(target.center, target.size, name=target.name)
    part = session.hints.part(name)
    spot = seen["spot"] = tuple(skills.place_spot(part, target.size).tolist())
    skills.place_upright(spot, part=part)
    session.sleep(1.0)  # The released target settles.
    skills.close_drawer(session.hints.part(name))
    episode.stow(session)


def main() -> None:
    check, args, spec = episode.start(__doc__, "dev_grid_table")
    settings = episode.SETTINGS

    def body() -> None:
        name, seen = settings["place_drawer"], {}
        with episode.world(args, spec) as world:
            report = world.run(lambda session: plan(session, name, seen), name="check_drawer_place.plan")
            episode.agent_checks(check, report)
            check.data["retrieve_evaluation"] = "not_applicable"
            truth = world.truth()
            target = truth.object_pose(spec.task["target"])
            # The drawer carries the target when it closes: its offset from the spot is mostly the stroke.
            offset = [round(v, 3) for v in (target[:3, 3] - np.asarray(seen["spot"])).tolist()]
            inside = truth.inside(spec.task["target"], name)
            check.measure("target_in_drawer", dict(offset_from_place_spot_m=offset), inside, "not in the drawer")
            tilt = float(np.degrees(np.arccos(np.clip(target[2, 2], -1.0, 1.0))))
            check.measure("target_tilt_deg", round(tilt, 1), tilt <= 10.0, "the target is not upright")
            closed = truth.part(name).opening
            check.measure("closed_m", round(closed, 4), closed <= settings["closed_max_m"], "the drawer did not close")

    check.run(body)


if __name__ == "__main__":
    main()
